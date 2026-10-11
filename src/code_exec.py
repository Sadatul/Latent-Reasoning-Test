"""
Run model-generated code against tests, in a subprocess with time and memory
limits. Used by run_eval.py.

Three problem formats (see DATASET.md):
  functional  LiveCodeBench LeetCode: call Solution().<func_name>(*args)
  stdin       LiveCodeBench AtCoder/Codeforces: full program, stdin -> stdout
  humaneval   HumanEval+: function + EvalPlus `check(candidate)` harness

Every scorer returns a dict:
  status        pass | wrong_answer | runtime_error | timeout
  passed        bool
  num_tests     number of tests (HumanEval+: 1, the whole check() call)
  num_passed    tests passed before the first failure
  first_failure None, or {test_index, input, expected, got, error}

This executes untrusted code. The limits stop runaway programs, not a
malicious one; run it on a machine where that is acceptable.
"""

import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

MEMORY_LIMIT_BYTES = 4 * 1024**3
MAX_FIELD_CHARS = 2000  # truncate inputs/outputs stored in first_failure

# Imports LiveCodeBench prepends to every solution.
LCB_IMPORT_HEADER = """\
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
from typing import *
from collections import *
from heapq import *
from bisect import *
from itertools import *
from functools import *
from math import *
sys.setrecursionlimit(50000)
"""

# Runs inside the subprocess. argv: mode, code_path, tests_path, result_path,
# per-test timeout. Writes one JSON result to result_path, so whatever the
# model's code prints cannot corrupt the result.
_RUNNER = r'''
import json, signal, sys, traceback, io

mode, code_path, tests_path, result_path, timeout = sys.argv[1:6]
timeout = float(timeout)

def save(result):
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)

class _Timeout(BaseException):
    pass

def _alarm(signum, frame):
    raise _Timeout()

signal.signal(signal.SIGALRM, _alarm)

def err_tail():
    return traceback.format_exc()[-1500:]

src = open(code_path, encoding="utf-8").read()
real_stdout = sys.stdout

if mode == "humaneval":
    sys.stdout = io.StringIO()
    try:
        exec(compile(src, "solution.py", "exec"), {"__name__": "__solution__"})
        save({"status": "pass", "num_passed": 1})
    except AssertionError:
        save({"status": "wrong_answer", "num_passed": 0, "error": err_tail()})
    except BaseException:
        save({"status": "runtime_error", "num_passed": 0, "error": err_tail()})
    sys.exit(0)

# functional (LeetCode)
func_name = json.loads(open(tests_path, encoding="utf-8").readline())["func_name"]
tests = [json.loads(l) for l in open(tests_path, encoding="utf-8").readlines()[1:]]
ns = {"__name__": "__solution__"}
sys.stdout = io.StringIO()
try:
    signal.setitimer(signal.ITIMER_REAL, timeout)
    exec(compile(src, "solution.py", "exec"), ns)
    signal.setitimer(signal.ITIMER_REAL, 0)
    method = getattr(ns["Solution"](), func_name)
except _Timeout:
    save({"status": "timeout", "num_passed": 0, "error": "timeout while loading code"})
    sys.exit(0)
except BaseException:
    signal.setitimer(signal.ITIMER_REAL, 0)
    save({"status": "runtime_error", "num_passed": 0, "error": err_tail()})
    sys.exit(0)

for i, t in enumerate(tests):
    args = [json.loads(line) for line in t["input"].split("\n")]
    expected = json.loads(t["output"])
    sys.stdout = io.StringIO()
    try:
        signal.setitimer(signal.ITIMER_REAL, timeout)
        got = method(*args)
        signal.setitimer(signal.ITIMER_REAL, 0)
    except _Timeout:
        save({"status": "timeout", "num_passed": i, "test_index": i, "error": f"exceeded {timeout}s"})
        sys.exit(0)
    except BaseException:
        signal.setitimer(signal.ITIMER_REAL, 0)
        save({"status": "runtime_error", "num_passed": i, "test_index": i, "error": err_tail()})
        sys.exit(0)
    try:
        got_json = json.loads(json.dumps(got))
    except (TypeError, ValueError):
        got_json = repr(got)
    if got_json != expected:
        save({"status": "wrong_answer", "num_passed": i, "test_index": i, "got": repr(got)})
        sys.exit(0)

save({"status": "pass", "num_passed": len(tests)})
'''


def _limit_resources():
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _run(cmd, cwd, timeout, stdin_text=None):
    """Run cmd; return (returncode, stdout, stderr, timed_out). Kills the whole process group on timeout."""
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=_limit_resources, start_new_session=True,
    )
    try:
        out, err = proc.communicate(
            input=stdin_text.encode("utf-8") if stdin_text is not None else None, timeout=timeout)
        return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), False
    except subprocess.TimeoutExpired:
        _kill(proc)
        return None, "", "", True
    except BaseException:  # e.g. Ctrl+C: don't leave the child running
        _kill(proc)
        raise


def _kill(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.communicate()


def _clip(value):
    if value is None:
        return None
    s = value if isinstance(value, str) else json.dumps(value)
    return s if len(s) <= MAX_FIELD_CHARS else s[:MAX_FIELD_CHARS] + f"... [{len(s) - MAX_FIELD_CHARS} more chars]"


def _result(status, num_tests, num_passed, failure=None):
    return {
        "status": status,
        "passed": status == "pass",
        "num_tests": num_tests,
        "num_passed": num_passed,
        "first_failure": failure,
    }


def _run_runner(mode, code, tests_lines, timeout, total_timeout):
    with tempfile.TemporaryDirectory(prefix="eval_") as tmp:
        tmp = Path(tmp)
        (tmp / "runner.py").write_text(_RUNNER, encoding="utf-8")
        (tmp / "solution.py").write_text(code, encoding="utf-8")
        (tmp / "tests.jsonl").write_text("\n".join(tests_lines) + "\n", encoding="utf-8")
        result_path = tmp / "result.json"
        rc, _out, err, timed_out = _run(
            [sys.executable, "runner.py", mode, "solution.py", "tests.jsonl", str(result_path), str(timeout)],
            cwd=tmp, timeout=total_timeout)
        if timed_out:
            return {"status": "timeout", "num_passed": 0, "error": f"process exceeded {total_timeout}s"}
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        # Crashed hard (segfault, MemoryError at import, os._exit, ...).
        return {"status": "runtime_error", "num_passed": 0,
                "error": f"exit code {rc}; stderr: {err[-1500:]}"}


def score_functional(code, tests, func_name, test_timeout=6.0):
    """LiveCodeBench LeetCode problem. tests: list of {"input", "output", "testtype"}."""
    lines = [json.dumps({"func_name": func_name})] + [json.dumps(t) for t in tests]
    total = min(10 + test_timeout * len(tests), 600)
    r = _run_runner("functional", LCB_IMPORT_HEADER + "\n" + code, lines, test_timeout, total)
    failure = None
    if r["status"] != "pass":
        i = r.get("test_index")
        t = tests[i] if i is not None else None
        failure = {
            "test_index": i,
            "input": _clip(t["input"]) if t else None,
            "expected": _clip(t["output"]) if t else None,
            "got": _clip(r.get("got")),
            "error": r.get("error"),
        }
    return _result(r["status"], len(tests), r["num_passed"], failure)


def _outputs_match(got, expected):
    """LiveCodeBench-style comparison: per line, ignore surrounding whitespace and
    trailing blank lines; if the text differs, accept numerically equal tokens."""
    g = [l.strip() for l in got.rstrip().split("\n")]
    e = [l.strip() for l in expected.rstrip().split("\n")]
    if g == e:
        return True
    if len(g) != len(e):
        return False
    for gl, el in zip(g, e):
        gt, et = gl.split(), el.split()
        if len(gt) != len(et):
            return False
        for a, b in zip(gt, et):
            if a == b:
                continue
            try:
                if abs(float(a) - float(b)) > 1e-6:
                    return False
            except ValueError:
                return False
    return True


def score_stdin(code, tests, test_timeout=6.0):
    """LiveCodeBench AtCoder/Codeforces problem. One subprocess per test; stops at the first failure."""
    with tempfile.TemporaryDirectory(prefix="eval_") as tmp:
        (Path(tmp) / "solution.py").write_text(LCB_IMPORT_HEADER + "\n" + code, encoding="utf-8")
        for i, t in enumerate(tests):
            rc, out, err, timed_out = _run(
                [sys.executable, "solution.py"], cwd=tmp, timeout=test_timeout, stdin_text=t["input"])
            if timed_out:
                status, error = "timeout", f"exceeded {test_timeout}s"
            elif rc != 0:
                status, error = "runtime_error", err[-1500:]
            elif not _outputs_match(out, t["output"]):
                status, error = "wrong_answer", None
            else:
                continue
            failure = {
                "test_index": i,
                "input": _clip(t["input"]),
                "expected": _clip(t["output"]),
                "got": _clip(out) if not timed_out else None,
                "error": error,
            }
            return _result(status, len(tests), i, failure)
    return _result("pass", len(tests), len(tests))


def build_humaneval_program(row, code):
    """prompt + `pass` stub defines the prompt's imports and helper functions;
    the model's code then (re)defines the entry point; then the EvalPlus tests."""
    return (row["prompt"] + "    pass\n\n"
            + code + "\n\n"
            + row["test"] + "\n\n"
            + f"check({row['entry_point']})\n")


def score_humaneval(row, code, timeout=60.0):
    program = build_humaneval_program(row, code)
    r = _run_runner("humaneval", program, [], timeout, timeout)
    failure = None
    if r["status"] != "pass":
        failure = {"test_index": None, "input": None, "expected": None, "got": None, "error": r.get("error")}
    return _result(r["status"], 1, r["num_passed"], failure)
