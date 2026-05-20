"""Local subprocess Python code interpreter for the retool experiment.

Ported from verl-0519/recipe/retool/local_subprocess_tool.py — keeps the same
sandbox semantics (timeout, RLIMIT_AS / RLIMIT_FSIZE / RLIMIT_NOFILE,
print_last_expression, output truncation) but exposes a single sync function
that the AReaL workflow can call via asyncio.to_thread.
"""

from __future__ import annotations

import ast
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_PYTHON_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE
)


def extract_python_code(raw_code: str) -> str:
    if not isinstance(raw_code, str):
        raise ValueError(f"code must be a string, got {type(raw_code).__name__}")
    match = _PYTHON_FENCE_RE.search(raw_code)
    if match:
        return match.group(1).strip()
    return raw_code


def print_last_expression(code: str) -> str:
    lines = code.splitlines()
    for line_idx in range(len(lines) - 1, -1, -1):
        stripped = lines[line_idx].strip()
        if stripped:
            break
    else:
        return code

    if stripped.startswith("print("):
        return code

    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return code

    if not parsed.body or not isinstance(parsed.body[-1], ast.Expr):
        return code

    indent = lines[line_idx][: len(lines[line_idx]) - len(lines[line_idx].lstrip())]
    lines[line_idx] = f"{indent}print({stripped})"
    trailing_newline = "\n" if code.endswith("\n") else ""
    return "\n".join(lines) + trailing_newline


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        raise ValueError(f"max_output_chars must be positive, got {max_chars}")
    if len(text) <= max_chars:
        return text, False
    half = max_chars // 2
    return text[:half] + "\n...<truncated>...\n" + text[-half:], True


def _set_child_limits(
    memory_limit_mb: int | None,
    cpu_time_limit_s: int,
    file_size_limit_mb: int,
) -> None:
    if cpu_time_limit_s <= 0:
        raise ValueError(f"cpu_time_limit_s must be positive, got {cpu_time_limit_s}")
    if file_size_limit_mb <= 0:
        raise ValueError(f"file_size_limit_mb must be positive, got {file_size_limit_mb}")

    resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit_s, cpu_time_limit_s))
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (file_size_limit_mb * 1024 * 1024, file_size_limit_mb * 1024 * 1024),
    )
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if memory_limit_mb is not None:
        if memory_limit_mb <= 0:
            raise ValueError(
                f"memory_limit_mb must be positive when set, got {memory_limit_mb}"
            )
        memory_limit_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))


def run_code_interpreter(
    code: str,
    *,
    timeout_s: int = 10,
    memory_limit_mb: int | None = 1024,
    max_output_chars: int = 4096,
    file_size_limit_mb: int = 16,
) -> tuple[str, dict[str, Any]]:
    """Execute Python code in a sandboxed subprocess.

    Returns (output_text, metrics). On error, output_text contains a
    human-readable description of stdout/stderr; metrics carries exit_code,
    timed_out, duration_s, etc.
    """
    code = print_last_expression(extract_python_code(code))
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="areal-retool-") as tmpdir:
        script_path = Path(tmpdir) / "main.py"
        script_path.write_text(code, encoding="utf-8")

        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=tmpdir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
                preexec_fn=lambda: _set_child_limits(
                    memory_limit_mb=memory_limit_mb,
                    cpu_time_limit_s=max(timeout_s + 1, 1),
                    file_size_limit_mb=file_size_limit_mb,
                ),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            stderr = exc.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            elapsed = time.monotonic() - started
            combined, truncated = _truncate(stdout + stderr, max_output_chars)
            output = f"Execution timed out after {timeout_s}s."
            if combined:
                output += f"\nPartial output:\n{combined}"
            return output, {
                "exit_code": None,
                "timed_out": True,
                "duration_s": elapsed,
                "stdout_chars": len(stdout),
                "stderr_chars": len(stderr),
                "truncated": truncated,
            }

    elapsed = time.monotonic() - started
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    truncated_stdout, stdout_truncated = _truncate(stdout.strip(), max_output_chars)
    truncated_stderr, stderr_truncated = _truncate(stderr.strip(), max_output_chars)
    metrics = {
        "exit_code": completed.returncode,
        "timed_out": False,
        "duration_s": elapsed,
        "stdout_chars": len(stdout),
        "stderr_chars": len(stderr),
        "truncated": stdout_truncated or stderr_truncated,
    }
    if completed.returncode == 0:
        return truncated_stdout, metrics

    error_parts = [f"Python execution failed with exit_code={completed.returncode}."]
    if truncated_stdout:
        error_parts.append(f"STDOUT:\n{truncated_stdout}")
    if truncated_stderr:
        error_parts.append(f"STDERR:\n{truncated_stderr}")
    return "\n".join(error_parts), metrics
