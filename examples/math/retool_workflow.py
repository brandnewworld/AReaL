"""Multi-turn hermes tool-call workflow for retool-dapo training.

Patterned after examples/tir/tir_workflow.py — but the trigger is the literal
hermes pair `<tool_call> {...json...} </tool_call>` (matching what
verl-0519/recipe/retool/local_subprocess_tool_config.yaml sets up), and the
only tool is `code_interpreter` dispatched via examples.math.retool_tool.

For each rollout sample:
  1. Send current input_ids to the engine with stop=["</tool_call>"].
  2. Append generated tokens to the trajectory (loss_mask=1, real versions).
  3. If the assistant emitted a complete <tool_call>...</tool_call> block,
     parse the JSON, run the code, inject
     `\\n<tool_response>\\n{out}\\n</tool_response>\\n` back into the context
     (loss_mask=0, versions=-1). Repeat up to max_turns or max_total_tokens.
  4. If parsing fails or the tool errors, inject an error <tool_response> with
     the same masking — the reward function's format gating will handle it.
  5. Hand the full transcript to retool_reward_fn for the trajectory reward.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from typing import Any

import aiofiles
import aiofiles.os
import colorama
import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    dataclass,
    field,
)
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors

from examples.math.retool_tool import _truncate, run_code_interpreter

logger = logging.getLogger("Retool workflow")

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


@dataclass
class RetoolConfig:
    max_turns: int = field(default=8)
    max_total_tokens: int = field(default=18432)
    tool_timeout_s: int = field(default=10)
    tool_memory_limit_mb: int = field(default=1024)
    tool_max_output_chars: int = field(default=4096)
    tool_max_response_length_chars: int = field(default=1024)
    enable_thinking: bool = field(default=True)


@dataclass
class RetoolGRPOConfig(GRPOConfig):
    retool: RetoolConfig = field(default_factory=RetoolConfig)


class RetoolWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        retool_config: RetoolConfig,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
    ):
        super().__init__()
        self.reward_fn = reward_fn
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.cfg = retool_config
        self.rollout_stat_scope = rollout_stat_scope
        self.dump_dir = dump_dir
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def arun_episode(self, engine: InferenceEngine, data: dict[str, Any]):
        prompt_ids = self.tokenizer.apply_chat_template(
            data["messages"],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=self.cfg.enable_thinking,
        )
        prompt_str = self.tokenizer.decode(prompt_ids)
        n_samples = self.gconfig.n_samples
        results = await asyncio.gather(
            *[
                self._run_one_episode(engine, prompt_ids, prompt_str, data)
                for _ in range(n_samples)
            ]
        )

        if self.dump_dir is not None:
            version = engine.get_version()
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex
            file_path = os.path.join(dump_path, f"{qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
                for i, (_, p, c, r, sl, turns) in enumerate(results):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}, seqlen: {sl}, turns: {turns}, reward: {r}.",
                            f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{p}{colorama.Style.RESET_ALL}",
                            f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{c}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    await f.write(info + "\n")

        return concat_padded_tensors([res[0] for res in results])

    async def _run_one_episode(
        self,
        engine: InferenceEngine,
        prompt_ids: list[int],
        prompt_str: str,
        data: dict[str, Any],
    ):
        rid = uuid.uuid4().hex
        seq: list[int] = list(prompt_ids)
        logprobs: list[float] = [0.0] * len(seq)
        loss_mask: list[int] = [0] * len(seq)
        versions: list[int] = [-1] * len(seq)
        completion_str = ""
        turn = 0
        stopped_normally = False

        while turn < self.cfg.max_turns:
            remaining = self.cfg.max_total_tokens - len(seq)
            if remaining <= 0:
                break
            gconfig = self.gconfig.new(
                n_samples=1,
                max_new_tokens=min(self.gconfig.max_new_tokens, remaining),
                stop=(self.gconfig.stop or []) + [TOOL_CALL_CLOSE],
            )
            req = ModelRequest(
                rid=rid,
                input_ids=seq,
                gconfig=gconfig,
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)

            seq.extend(resp.output_tokens)
            logprobs.extend(resp.output_logprobs)
            loss_mask.extend([1] * resp.output_len)
            versions.extend(resp.output_versions)
            decoded = self.tokenizer.decode(resp.output_tokens)
            completion_str += decoded

            open_count = decoded.count(TOOL_CALL_OPEN)
            close_count = decoded.count(TOOL_CALL_CLOSE)

            # If backend stripped the stop string but we did see an open tag,
            # re-append the close tag so downstream parsing and the reward's
            # format gating see a balanced trajectory.
            if open_count > close_count:
                close_ids = self.tokenizer.encode(
                    TOOL_CALL_CLOSE, add_special_tokens=False
                )
                seq.extend(close_ids)
                logprobs.extend([0.0] * len(close_ids))
                loss_mask.extend([1] * len(close_ids))
                versions.extend(
                    [resp.output_versions[-1] if resp.output_versions else -1]
                    * len(close_ids)
                )
                completion_str += TOOL_CALL_CLOSE
                decoded += TOOL_CALL_CLOSE

            if TOOL_CALL_OPEN not in decoded:
                stopped_normally = True
                break

            matches = TOOL_CALL_RE.findall(decoded)
            if not matches:
                # <tool_call> without a parseable JSON body; let the reward fn
                # mark format_invalid and stop here.
                break

            tool_output_text = await self._dispatch_tool(matches[-1].strip())
            response_block = (
                f"\n<tool_response>\n{tool_output_text}\n</tool_response>\n"
            )
            response_ids = self.tokenizer.encode(
                response_block, add_special_tokens=False
            )

            budget = self.cfg.max_total_tokens - len(seq)
            if budget <= 0:
                break
            if len(response_ids) > budget:
                response_ids = response_ids[:budget]

            seq.extend(response_ids)
            logprobs.extend([0.0] * len(response_ids))
            loss_mask.extend([0] * len(response_ids))
            versions.extend([-1] * len(response_ids))
            completion_str += self.tokenizer.decode(response_ids)
            turn += 1

        # Hard cap at max_total_tokens.
        if len(seq) > self.cfg.max_total_tokens:
            seq = seq[: self.cfg.max_total_tokens]
            logprobs = logprobs[: self.cfg.max_total_tokens]
            loss_mask = loss_mask[: self.cfg.max_total_tokens]
            versions = versions[: self.cfg.max_total_tokens]

        completion_ids = seq[len(prompt_ids):]
        reward = await self.async_reward_fn(
            prompt_str,
            completion_str,
            prompt_ids,
            completion_ids,
            **data,
        )
        reward = float(reward)

        stats_tracker.get(self.rollout_stat_scope).scalar(
            reward=reward,
            num_turns=turn,
            stopped_normally=int(stopped_normally),
            seq_len=len(seq),
        )

        res = dict(
            input_ids=torch.tensor(seq).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            rewards=torch.tensor([reward]),
        )
        return res, prompt_str, completion_str, reward, len(seq), turn

    async def _dispatch_tool(self, payload: str) -> str:
        try:
            call = json.loads(payload)
        except json.JSONDecodeError as exc:
            return f"Error: malformed tool_call JSON: {exc}"
        if not isinstance(call, dict):
            return "Error: tool_call must be a JSON object"
        if call.get("name") != "code_interpreter":
            return f"Error: unsupported tool name {call.get('name')!r}"
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            return "Error: arguments must be a JSON object"
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return "Error: arguments.code must be a non-empty string"

        try:
            output, _metrics = await asyncio.to_thread(
                run_code_interpreter,
                code,
                timeout_s=self.cfg.tool_timeout_s,
                memory_limit_mb=self.cfg.tool_memory_limit_mb,
                max_output_chars=self.cfg.tool_max_output_chars,
            )
        except Exception as exc:  # pragma: no cover — sandbox failures
            return f"Error: tool execution raised {type(exc).__name__}: {exc}"

        truncated, _ = _truncate(output, self.cfg.tool_max_response_length_chars)
        return truncated
