#!/usr/bin/env python3
"""Numeric / symbolic verification helper.

Exit 0 IFF the expression evaluates (symbolically, via sympy) to the expected
value. This is the canonical "artifact" for a `math`-type task: a rerunnable
check that mechanically proves the claim.

Usage:
    python3 verify/check_numeric.py --expr "2+2" --expected 4
    python3 verify/check_numeric.py --expr "sin(pi/2)" --expected 1
    python3 verify/check_numeric.py --expr "diff(x**2, x)" --expected "2*x"
"""
import argparse
import sys

import sympy


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expr", required=True, help="sympy expression to evaluate")
    p.add_argument("--expected", required=True, help="expected value (sympy-parsed)")
    args = p.parse_args()

    try:
        got = sympy.sympify(args.expr)
        want = sympy.sympify(args.expected)
        # simplify(got - want) == 0  is the robust equality test in sympy
        equal = sympy.simplify(got - want) == 0
    except Exception as exc:  # noqa: BLE001 - report any parse/eval failure
        print(f"FAIL: could not evaluate: {exc}", file=sys.stderr)
        return 2

    if equal:
        print(f"PASS: {args.expr} == {args.expected}  (= {got})")
        return 0
    print(f"FAIL: {args.expr} = {got}, expected {want}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
