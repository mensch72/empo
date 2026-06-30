#!/usr/bin/env python3
"""Rerun a logged experiment script and compare its output (STUB).

The canonical "artifact" for an `experiment`-type task: take the script the
agent claims to have run, run it again, and confirm it reproduces the recorded
result. Reproducibility is the proof.

This is a wired-in STUB: it runs --script and checks the exit code, and if
--expect-substr is given, that the substring appears in stdout. Extend it to
compare against a stored expected-output file / numeric tolerance as the project
needs.

Usage:
    python3 verify/rerun_experiment.py --script experiments/foo.py
    python3 verify/rerun_experiment.py --script experiments/foo.py \
        --expect-substr "converged"
"""
import argparse
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--script", required=True, help="path to the rerunnable script")
    p.add_argument("--expect-substr", default=None,
                   help="substring that must appear in stdout for a pass")
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    try:
        proc = subprocess.run(
            [sys.executable, args.script],
            capture_output=True, text=True, timeout=args.timeout,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not run {args.script}: {exc}", file=sys.stderr)
        return 2

    if proc.returncode != 0:
        print(f"FAIL: {args.script} exited {proc.returncode}\n{proc.stderr}",
              file=sys.stderr)
        return 1

    if args.expect_substr and args.expect_substr not in proc.stdout:
        print(f"FAIL: {args.expect_substr!r} not in output of {args.script}",
              file=sys.stderr)
        return 1

    print(f"PASS: {args.script} reran cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
