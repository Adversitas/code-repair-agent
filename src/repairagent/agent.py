"""Generate -> execute -> repair loop, run in lock-step over a batch of tasks.

Policies (same budget of ``k`` generations per task, the agent may stop early
once the *visible* test passes):

* ``single``   one greedy answer, no tool use.
* ``resample`` up to k independent answers (1 greedy + k-1 sampled), keep the
               first that passes the visible test.
* ``repair``   greedy answer, then up to k-1 repair turns where the model gets
               the sandbox test report (failed assertion, actual value, error).
* ``repair_t`` same as ``repair`` but repair turns are sampled (temperature),
               so the model cannot resubmit the identical program by default.
* ``blind``    ablation of ``repair``: same turns, but the model is only told
               "your code is not correct" with no test output.

The final answer is scored on all MBPP assertions (hidden tests).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .data import Task
from .prompts import SYSTEM, blind_repair_prompt, extract_code, repair_prompt, task_prompt
from .sandbox import ExecResult, feedback, run_tests

POLICIES = ("single", "resample", "repair", "repair_t", "blind")


@dataclass
class Episode:
    task: Task
    messages: list[dict] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)
    visible: list[ExecResult] = field(default_factory=list)
    tokens: int = 0
    repeats: int = 0
    done: bool = False

    @property
    def final_code(self) -> str:
        for code, res in zip(self.codes, self.visible):
            if res.all_passed:
                return code
        return self.codes[-1] if self.codes else ""


def _first_messages(task: Task) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task_prompt(task.text, task.visible)},
    ]


def _conv_chars(e: Episode) -> int:
    return sum(len(m["content"]) for m in e.messages)


def _chunks(seq, n, char_budget=None):
    """Fixed-size batches, or (with ``char_budget``) length-sorted batches whose
    padded size ``len(batch) * longest_conversation`` stays under the budget, so
    long multi-turn repair conversations do not run the GPU out of memory."""
    if char_budget is None:
        for i in range(0, len(seq), n):
            yield seq[i : i + n]
        return
    chunk, longest = [], 0
    for e in sorted(seq, key=_conv_chars):
        size = _conv_chars(e)
        if chunk and (len(chunk) >= n or (len(chunk) + 1) * max(longest, size) > char_budget):
            yield chunk
            chunk, longest = [], 0
        chunk.append(e)
        longest = max(longest, size)
    if chunk:
        yield chunk


def run_policy(
    llm,
    tasks: list[Task],
    policy: str,
    k: int = 4,
    batch_size: int = 16,
    max_new_tokens: int = 384,
    temperature: float = 0.8,
    workers: int = 8,
    timeout: float = 5.0,
    char_budget: int | None = None,
    progress=None,
) -> list[dict]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    budget = 1 if policy == "single" else k
    eps = [Episode(t, _first_messages(t)) for t in tasks]
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for turn in range(budget):
            active = [e for e in eps if not e.done]
            if not active:
                break
            sampled = policy in ("resample", "repair_t") and turn > 0
            for chunk_i, chunk in enumerate(_chunks(active, batch_size, char_budget)):
                if policy == "resample":
                    convs = [_first_messages(e.task) for e in chunk]
                else:
                    convs = [e.messages for e in chunk]
                gens = llm.generate(
                    convs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if sampled else 0.0,
                    seed=1000 * turn + chunk_i,
                )
                codes = [extract_code(g.text) for g in gens]
                results = list(
                    pool.map(
                        lambda ec: run_tests(ec[1], list(ec[0].task.visible), ec[0].task.setup, timeout),
                        zip(chunk, codes),
                    )
                )
                for e, g, code, res in zip(chunk, gens, codes, results):
                    if e.codes and code.strip() == e.codes[-1].strip():
                        e.repeats += 1
                    e.codes.append(code)
                    e.visible.append(res)
                    e.tokens += g.new_tokens
                    if res.all_passed or turn == budget - 1:
                        e.done = True
                        continue
                    if policy in ("repair", "repair_t", "blind"):
                        blind = policy == "blind"
                        user = blind_repair_prompt() if blind else repair_prompt(feedback(res))
                        e.messages = e.messages + [
                            {"role": "assistant", "content": g.text},
                            {"role": "user", "content": user},
                        ]
            if progress:
                progress(policy, turn, sum(e.done for e in eps), len(eps))
        finals = list(
            pool.map(
                lambda e: run_tests(e.final_code, list(e.task.hidden), e.task.setup, timeout),
                eps,
            )
        )
        firsts = list(
            pool.map(
                lambda e: run_tests(e.codes[0], list(e.task.hidden), e.task.setup, timeout),
                eps,
            )
        )
    finally:
        pool.shutdown()
    rows = []
    for e, fin, first in zip(eps, finals, firsts):
        rows.append(
            {
                "task_id": e.task.task_id,
                "policy": policy,
                "generations": len(e.codes),
                "repeats": e.repeats,
                "new_tokens": e.tokens,
                "visible_pass": any(r.all_passed for r in e.visible),
                "hidden_pass": fin.all_passed,
                "first_hidden_pass": first.all_passed,
                "first_failure": e.visible[0].failure_kind,
                "final_failure": fin.failure_kind,
            }
        )
    return rows
