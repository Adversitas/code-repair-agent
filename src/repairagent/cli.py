"""Command line: ``repairagent run | report | smoke``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .agent import POLICIES, run_policy
from .data import load_tasks
from .metrics import paired_bootstrap, summarize

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "mbpp_test.jsonl"


def _log(policy, turn, done, n):
    print(f"  [{policy}] turn {turn + 1}: {done}/{n} episodes finished", flush=True)


def cmd_run(a) -> int:
    from .llm import HFChat

    tasks = load_tasks(a.data, a.limit)
    llm = HFChat(a.model)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for pol in a.policies:
            t0 = time.time()
            rows = run_policy(
                llm,
                tasks,
                pol,
                k=a.k,
                batch_size=a.batch_size,
                max_new_tokens=a.max_new_tokens,
                char_budget=a.char_budget,
                progress=_log,
            )
            secs = time.time() - t0
            for r in rows:
                r["model"] = a.model
                r["seconds_policy"] = round(secs, 1)
                f.write(json.dumps(r) + "\n")
            f.flush()
            s = summarize(rows)[pol]
            print(f"{pol}: hidden pass {s['hidden_pass']:.3f} in {secs:.0f}s", flush=True)
    return 0


def _read(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def render(rows: list[dict]) -> str:
    models = sorted({r.get("model", "?") for r in rows})
    lines = []
    for m in models:
        rs = [r for r in rows if r.get("model", "?") == m]
        summ = summarize(rs)
        lines += [
            f"### {m} ({next(iter(summ.values()))['n']} MBPP tasks)",
            "",
            "| policy | hidden pass | visible pass | first-try pass | overfit "
            "| gens/task | repeat rate | new tokens/task |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for pol in [p for p in POLICIES if p in summ] + sorted(set(summ) - set(POLICIES)):
            s = summ[pol]
            lines.append(
                f"| {pol} | {s['hidden_pass']:.3f} | {s['visible_pass']:.3f} | "
                f"{s['first_try_hidden_pass']:.3f} | {s['overfit_rate']:.3f} | "
                f"{s['mean_generations']:.2f} | {s['repeat_rate']:.3f} | {s['mean_new_tokens']:.0f} |"
            )
        lines += ["", "Paired differences in hidden pass rate (95% bootstrap CI):", ""]
        for a_, b_ in [
            ("repair", "single"),
            ("resample", "single"),
            ("repair_t", "single"),
            ("repair_t", "resample"),
            ("repair", "blind"),
        ]:
            if a_ in summ and b_ in summ:
                d = paired_bootstrap(rs, a_, b_)
                lines.append(f"- {a_} - {b_}: {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
        lines += ["", "First-attempt failure types (single policy):", ""]
        if "single" in summ:
            ff = summ["single"]["first_failure"]
            lines.append(", ".join(f"{k}: {v}" for k, v in sorted(ff.items())))
        lines.append("")
    return "\n".join(lines)


def cmd_report(a) -> int:
    text = render(_read(a.inputs))
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_smoke(a) -> int:
    """End-to-end run with a scripted model (no GPU): checks the loop and sandbox."""
    from .llm import ScriptedChat

    tasks = load_tasks(a.data, a.limit)
    gold = {}
    with open(a.data, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gold[row["text"].strip()] = row["code"]

    def reply(conv):
        # wrong on the first turn, gold code once it has been given a test report
        text = conv[1]["content"].split("\nYour code must pass")[0]
        if len(conv) == 2:
            return "```python\ndef nothing():\n    return None\n```"
        return f"```python\n{gold[text]}\n```"

    rows = []
    for pol in POLICIES:
        rows += run_policy(ScriptedChat(reply), tasks, pol, k=2, batch_size=4)
    for r in rows:
        r["model"] = "scripted"
    print(render(rows))
    s = summarize(rows)
    ok = s["repair"]["hidden_pass"] > s["single"]["hidden_pass"]
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="repairagent", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run policies with a local Hugging Face model")
    r.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    r.add_argument("--data", default=str(DEFAULT_DATA))
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    r.add_argument("--k", type=int, default=4)
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--max-new-tokens", type=int, default=384)
    r.add_argument(
        "--char-budget", type=int, default=None, help="cap batch_size * longest prompt (chars) to avoid OOM"
    )
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_run)
    rep = sub.add_parser("report", help="aggregate run files into a markdown table")
    rep.add_argument("inputs", nargs="+")
    rep.add_argument("--out")
    rep.set_defaults(fn=cmd_report)
    s = sub.add_parser("smoke", help="scripted end-to-end check, no model needed")
    s.add_argument("--data", default=str(DEFAULT_DATA))
    s.add_argument("--limit", type=int, default=12)
    s.set_defaults(fn=cmd_smoke)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
