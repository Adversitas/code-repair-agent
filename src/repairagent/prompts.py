"""Chat prompts and code extraction."""

from __future__ import annotations

import re

SYSTEM = (
    "You are a careful Python programmer. Answer with one complete Python code block "
    "that defines the requested function. Do not print anything, do not write tests, "
    "and keep comments and docstrings short."
)

FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def task_prompt(text: str, visible: tuple[str, ...]) -> str:
    tests = "\n".join(visible)
    return f"{text}\nYour code must pass this test:\n{tests}"


def repair_prompt(report: str) -> str:
    return (
        "I ran your code against the test. Report:\n"
        f"{report}\n"
        "Fix the code. Answer with the full corrected code block."
    )


def blind_repair_prompt() -> str:
    """Ablation: the model is told it failed but gets no test output."""
    return "Your code is not correct. Fix the code. Answer with the full corrected code block."


def extract_code(reply: str) -> str:
    """Return the longest fenced code block, or the raw reply if there is none."""
    blocks = FENCE.findall(reply)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    # unterminated fence (generation cut by max_new_tokens)
    if "```" in reply:
        tail = reply.split("```", 1)[1]
        tail = re.sub(r"^(?:python|py)?\s*\n", "", tail, flags=re.IGNORECASE)
        return tail.strip() + "\n"
    return reply.strip() + "\n"
