from repairagent.sandbox import feedback, left_call, run_tests

ASSERTS = ["assert add(1, 2) == 3", "assert add(2, 2) == 4"]


def test_pass():
    r = run_tests("def add(a, b):\n    return a + b\n", ASSERTS)
    assert r.all_passed and r.failure_kind == "pass"


def test_wrong_answer_reports_actual_value():
    r = run_tests("def add(a, b):\n    return a - b\n", ASSERTS)
    assert not r.all_passed
    assert r.failure_kind == "wrong_answer"
    assert r.details[0]["actual"] == "-1"
    assert "your function returned -1" in feedback(r)


def test_runtime_error():
    r = run_tests("def add(a, b):\n    return math.floor(a) + b\n", ASSERTS)
    assert r.failure_kind == "runtime_error"
    assert "NameError" in feedback(r)


def test_syntax_error():
    r = run_tests("def add(a, b)\n    return a + b\n", ASSERTS)
    assert r.stage == "syntax"
    assert "failed before any test ran" in feedback(r)


def test_timeout():
    r = run_tests("def add(a, b):\n    while True:\n        pass\n", ASSERTS, timeout=1.5)
    assert r.stage == "timeout"
    assert not r.all_passed


def test_setup_code_runs():
    r = run_tests("def f():\n    return X\n", ["assert f() == 5"], setup="X = 5")
    assert r.all_passed


def test_left_call():
    assert left_call("assert f(1, [2]) == 3") == "f(1, [2])"
    assert left_call("assert x") is None
    assert left_call("not python (") is None
