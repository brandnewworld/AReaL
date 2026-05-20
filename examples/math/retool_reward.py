"""Reward function for retool-dapo training in AReaL.

Semantically aligned with verl-0519/recipe/retool/retool_tool_required.py and
verl-0519/verl/utils/reward_score/math_dapo.py:
  - Minerva-style "Answer: ..." extraction over the last 300 chars.
  - Tool-call format gating: malformed / unbalanced / spam / mismatched
    response counts collapse the trajectory to -1.
  - No-tool trajectories collapse to -1.
  - Otherwise: base_score (+1 / -1) + 0.1 * min(n_tool, MAX_REWARDED_TOOL_CALLS).
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
MAX_REWARDED_TOOL_CALLS = 4
MAX_ALLOWED_TOOL_CALLS = 8


# ---------------------------------------------------------------------------
# math_dapo (Minerva) scoring — ported verbatim from
# verl-0519/verl/utils/reward_score/math_dapo.py
# ---------------------------------------------------------------------------

_SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

_REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def _last_boxed_only_string(s: str) -> Optional[str]:
    idx = s.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    open_count = 0
    while i < len(s):
        if s[i] == "{":
            open_count += 1
        if s[i] == "}":
            open_count -= 1
            if open_count == 0:
                right_brace_idx = i
                break
        i += 1
    return s[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def _remove_boxed(s: str) -> str:
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


def _normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]
    for before, after in _SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in _REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def _is_correct_minerva(solution_str: str, gt: str) -> tuple[bool, str]:
    pattern = r"(?i)Answer\s*:\s*([^\n]+)"
    matches = re.findall(pattern, solution_str)
    extracted_answer = matches[-1] if matches else "[INVALID]"
    pred = _normalize_final_answer(extracted_answer)
    gt_norm = _normalize_final_answer(gt)
    return (pred == gt_norm), pred


def _math_dapo_compute_score(solution_str: str, ground_truth: str) -> dict[str, Any]:
    # Match verl: only look at the tail of the trajectory.
    tail = solution_str[-300:]
    correct, pred = _is_correct_minerva(tail, ground_truth)
    return {
        "score": 1.0 if correct else -1.0,
        "acc": correct,
        "pred": pred,
    }


# ---------------------------------------------------------------------------
# Tool-call format gating — ported from retool_tool_required.py
# ---------------------------------------------------------------------------


def _extract_valid_tool_calls(solution_str: str) -> list[dict[str, Any]]:
    valid_calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(solution_str):
        payload = match.group(1).strip()
        try:
            call = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(call, dict):
            continue
        if call.get("name") != "code_interpreter":
            continue
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            continue
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            continue
        valid_calls.append(call)
    return valid_calls


def _tool_stats(solution_str: str) -> dict[str, Any]:
    raw_tool_calls = solution_str.count("<tool_call>")
    raw_tool_call_closes = solution_str.count("</tool_call>")
    raw_tool_responses = solution_str.count("<tool_response>")
    valid_tool_calls = len(_extract_valid_tool_calls(solution_str))

    malformed_tool_calls = raw_tool_calls - valid_tool_calls
    tool_spam = raw_tool_calls > MAX_ALLOWED_TOOL_CALLS
    response_mismatch = valid_tool_calls > raw_tool_responses + 1
    unbalanced_tags = raw_tool_calls != raw_tool_call_closes
    format_invalid = raw_tool_calls > 0 and (
        malformed_tool_calls > 0 or response_mismatch or unbalanced_tags or tool_spam
    )

    return {
        "raw_tool_calls": raw_tool_calls,
        "valid_tool_calls": valid_tool_calls,
        "raw_tool_responses": raw_tool_responses,
        "malformed_tool_calls": max(0, malformed_tool_calls),
        "format_invalid": format_invalid,
        "tool_spam": tool_spam,
    }


def compute_score(solution_str: str, ground_truth: str) -> dict[str, Any]:
    base = _math_dapo_compute_score(solution_str, ground_truth)
    stats = _tool_stats(solution_str)
    n_tool = int(stats["valid_tool_calls"])

    if stats["format_invalid"]:
        score = -1.0
    elif n_tool == 0:
        score = -1.0
    else:
        score = base["score"] + 0.1 * min(n_tool, MAX_REWARDED_TOOL_CALLS)

    acc_tool_required = bool(
        base.get("acc") and not stats["format_invalid"] and n_tool > 0
    )
    return {
        "score": score,
        "acc": base.get("acc"),
        "acc_tool_required": float(acc_tool_required),
        "pred": base.get("pred") or "",
        "n_tool": n_tool,
        "raw_tool_calls": int(stats["raw_tool_calls"]),
        "raw_tool_responses": int(stats["raw_tool_responses"]),
        "malformed_tool_calls": int(stats["malformed_tool_calls"]),
        "tool_format_invalid": bool(stats["format_invalid"]),
        "tool_spam": bool(stats["tool_spam"]),
    }


def retool_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    ground_truth: str,
    **kwargs: Any,
) -> float:
    """AReaL-style reward signature; returns a single scalar."""
    result = compute_score(completions, ground_truth)
    return float(result["score"])
