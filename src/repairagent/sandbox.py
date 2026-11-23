"""Run untrusted generated code against assertions in a separate, time-limited process.

This is an isolation *convenience*, not a security boundary: the child is a
fresh interpreter in isolated mode (``-I``), in an empty temporary directory,
with a minimal environment and a wall-clock timeout. For untrusted users run it
inside a container or a VM with no network and no secrets.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass

RUNNER = r"""
import json, sys, traceback
src = open("solution.py", encoding="utf-8").read()
tests = json.load(open("tests.json", encoding="utf-8"))
setup = tests["setup"]
out = {"stage": "ok", "results": []}
g = {"__name__": "solution"}
try:
    exec(compile(src, "solution.py", "exec"), g)
    if setup:
        exec(setup, g)
except BaseException as e:
    out["stage"] = "syntax" if isinstance(e, SyntaxError) else "import"
    out["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()[-600:]
    print(json.dumps(out)); sys.exit(0)
for t in tests["asserts"]:
    r = {"assert": t, "passed": False}
    try:
        exec(t, g)
        r["passed"] = True
    except AssertionError:
        r["kind"] = "wrong_answer"
        call = tests["calls"].get(t)
        if call:
            try:
                r["actual"] = repr(eval(call, g))[:300]
            except BaseException as e:
                r["actual"] = "raised " + type(e).__name__
    except BaseException as e:
        r["kind"] = "runtime_error"
        r["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()[-400:]
    out["results"].append(r)
print(json.dumps(out))
"""


def left_call(assertion: str) -> str | None:
    """For ``assert f(x) == y`` return the source of ``f(x)`` (used to report actual values)."""
    try:
        node = ast.parse(assertion).body[0]
    except SyntaxError:
        return None
    if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
        left = node.test.left
        if isinstance(left, ast.Call):
            return ast.unparse(left)
    return None


@dataclass
class ExecResult:
    stage: str  # ok | syntax | import | timeout
    passed: list[bool]
    details: list[dict]
    error: str = ""

    @property
    def all_passed(self) -> bool:
        return self.stage == "ok" and all(self.passed)

    @property
    def failure_kind(self) -> str:
        if self.stage != "ok":
            return self.stage
        kinds = [d.get("kind") for d in self.details if not d["passed"]]
        if not kinds:
            return "pass"
        return "runtime_error" if "runtime_error" in kinds else "wrong_answer"


def run_tests(code: str, asserts: list[str], setup: str = "", timeout: float = 5.0) -> ExecResult:
    calls = {a: c for a in asserts if (c := left_call(a))}
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(d, "tests.json"), "w", encoding="utf-8") as f:
            json.dump({"asserts": asserts, "setup": setup, "calls": calls}, f)
        env = {"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", textwrap.dedent(RUNNER)],
                cwd=d,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecResult("timeout", [False] * len(asserts), [], f"timed out after {timeout}s")
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        err = (proc.stderr or "no output")[-400:]
        return ExecResult("import", [False] * len(asserts), [], err)
    if out["stage"] != "ok":
        return ExecResult(out["stage"], [False] * len(asserts), [], out.get("error", ""))
    res = out["results"]
    return ExecResult("ok", [r["passed"] for r in res], res)


def feedback(result: ExecResult) -> str:
    """Human-readable test report given back to the model."""
    if result.stage == "timeout":
        return "The code did not finish within the time limit (infinite loop or too slow)."
    if result.stage in ("syntax", "import"):
        return f"The code failed before any test ran:\n{result.error}"
    lines = []
    for d in result.details:
        if d["passed"]:
            lines.append(f"PASSED: {d['assert']}")
        elif d.get("kind") == "runtime_error":
            lines.append(f"ERROR:  {d['assert']}\n        {d['error']}")
        else:
            got = f"\n        your function returned {d['actual']}" if "actual" in d else ""
            lines.append(f"FAILED: {d['assert']}{got}")
    return "\n".join(lines)
