import json
from pathlib import Path

import pytest

from repairagent.agent import run_policy
from repairagent.cli import main, render
from repairagent.data import Task, load_tasks
from repairagent.llm import ScriptedChat
from repairagent.metrics import paired_bootstrap, summarize
from repairagent.prompts import extract_code

DATA = Path(__file__).resolve().parents[1] / "data" / "mbpp_test.jsonl"

TASK = Task(
    1,
    "Write a function to double a number.",
    ("assert double(2) == 4", "assert double(5) == 10", "assert double(0) == 0"),
)
GOOD = "```python\ndef double(x):\n    return 2 * x\n```"
BAD = "```python\ndef double(x):\n    return x + 2\n```"  # passes the visible test only


def test_extract_code_variants():
    assert extract_code("text\n```python\nx = 1\n```\nmore") == "x = 1\n"
    assert extract_code("```\na = 1\n```") == "a = 1\n"
    assert extract_code("```python\ny = 2\n") == "y = 2\n"  # cut generation
    assert extract_code("z = 3") == "z = 3\n"


def test_load_tasks():
    tasks = load_tasks(DATA, limit=5)
    assert len(tasks) == 5
    assert all(len(t.tests) == 3 and t.visible == t.tests[:1] for t in tasks)


def test_visible_overfit_is_caught():
    rows = run_policy(ScriptedChat(lambda c: BAD), [TASK], "repair", k=3)
    r = rows[0]
    assert r["visible_pass"] and not r["hidden_pass"]
    assert r["generations"] == 1  # stopped early: the agent only sees the visible test


def test_repair_uses_feedback():
    seen = []

    def reply(conv):
        seen.append(conv[-1]["content"])
        return (
            GOOD
            if "your function returned" in conv[-1]["content"]
            else "```python\ndef double(x):\n    return x\n```"
        )

    rows = run_policy(ScriptedChat(reply), [TASK], "repair", k=3)
    assert rows[0]["hidden_pass"] and rows[0]["generations"] == 2
    assert "your function returned 2" in seen[-1]


def test_blind_gets_no_test_output():
    msgs = []

    def reply(conv):
        msgs.append(conv[-1]["content"])
        return "```python\ndef double(x):\n    return x\n```"

    rows = run_policy(ScriptedChat(reply), [TASK], "blind", k=3)
    assert rows[0]["generations"] == 3
    assert rows[0]["repeats"] == 2
    assert all("returned" not in m for m in msgs)


def test_resample_restarts_conversation():
    lens = []

    def reply(conv):
        lens.append(len(conv))
        return "```python\ndef double(x):\n    return 0\n```"

    run_policy(ScriptedChat(reply), [TASK], "resample", k=3)
    assert lens == [2, 2, 2]


def test_single_budget_and_unknown_policy():
    llm = ScriptedChat(lambda c: "no code")
    rows = run_policy(llm, [TASK], "single", k=4)
    assert rows[0]["generations"] == 1 and rows[0]["first_failure"] in ("syntax", "import")
    with pytest.raises(ValueError):
        run_policy(llm, [TASK], "nope")


def test_metrics_and_bootstrap():
    rows = []
    for i in range(20):
        rows.append(
            {
                "task_id": i,
                "policy": "a",
                "hidden_pass": i < 15,
                "visible_pass": True,
                "first_hidden_pass": i < 10,
                "generations": 2,
                "repeats": 0,
                "new_tokens": 5,
                "first_failure": "pass",
            }
        )
        rows.append(
            {
                "task_id": i,
                "policy": "b",
                "hidden_pass": i < 10,
                "visible_pass": i < 10,
                "first_hidden_pass": i < 10,
                "generations": 1,
                "repeats": 0,
                "new_tokens": 3,
                "first_failure": "pass",
            }
        )
    s = summarize(rows)
    assert s["a"]["hidden_pass"] == 0.75 and s["a"]["overfit_rate"] == 0.25
    d = paired_bootstrap(rows, "a", "b")
    assert d["diff"] == pytest.approx(0.25) and d["lo"] <= 0.25 <= d["hi"] and d["lo"] > 0
    for r in rows:
        r["model"] = "m"
    assert "| a |" in render(rows)


def test_smoke_cli(capsys):
    assert main(["smoke", "--limit", "6"]) == 0
    assert "repair" in capsys.readouterr().out


def test_report_cli(tmp_path):
    p = tmp_path / "r.jsonl"
    row = {
        "task_id": 1,
        "policy": "single",
        "hidden_pass": True,
        "visible_pass": True,
        "first_hidden_pass": True,
        "generations": 1,
        "repeats": 0,
        "new_tokens": 4,
        "first_failure": "pass",
        "model": "m",
    }
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "o.md"
    assert main(["report", str(p), "--out", str(out)]) == 0
    assert "single" in out.read_text(encoding="utf-8")


def test_char_budget_chunks():
    from repairagent.agent import Episode, _chunks

    eps = [Episode(TASK, [{"role": "user", "content": "x" * n}]) for n in (10, 500, 20, 400, 30)]
    chunks = list(_chunks(eps, 8, char_budget=1000))
    assert sorted(len(m["content"]) for c in chunks for e in c for m in e.messages) == [10, 20, 30, 400, 500]
    for c in chunks:
        assert len(c) * max(len(e.messages[0]["content"]) for e in c) <= 1000 or len(c) == 1
    assert [len(c) for c in _chunks(eps, 2)] == [2, 2, 1]
