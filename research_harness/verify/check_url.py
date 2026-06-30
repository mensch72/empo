#!/usr/bin/env python3
"""Confirm a fetched source actually exists (STUB-ish, but functional).

The canonical "artifact" for a `lit`-type task: the cited source resolves. This
does an HTTP(S) HEAD/GET and exits 0 on a 2xx/3xx status. It does NOT judge
relevance — that is a human call.

Usage:
    python3 verify/check_url.py --url https://arxiv.org/abs/2306.00001
"""
import argparse
import sys
import urllib.request


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--timeout", type=int, default=30)
    args = p.parse_args()

    req = urllib.request.Request(args.url, method="GET",
                                 headers={"User-Agent": "research-harness/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            code = resp.getcode()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {args.url} did not resolve: {exc}", file=sys.stderr)
        return 1

    if 200 <= code < 400:
        print(f"PASS: {args.url} -> HTTP {code}")
        return 0
    print(f"FAIL: {args.url} -> HTTP {code}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
