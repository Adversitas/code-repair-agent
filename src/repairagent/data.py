"""MBPP task loading.

The agent only ever sees the *first* assertion of each task (it fixes the
function name and signature). All three original MBPP assertions are used to
score the final answer, so a policy that over-fits the visible test is caught.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    task_id: int
    text: str
    tests: tuple[str, ...]
    setup: str = ""

    @property
    def visible(self) -> tuple[str, ...]:
        return self.tests[:1]

    @property
    def hidden(self) -> tuple[str, ...]:
        return self.tests


def load_tasks(path: str | Path, limit: int | None = None) -> list[Task]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks.append(
                Task(
                    task_id=int(row["task_id"]),
                    text=row["text"].strip(),
                    tests=tuple(row["test_list"]),
                    setup=row.get("test_setup_code", "") or "",
                )
            )
            if limit and len(tasks) >= limit:
                break
    return tasks
