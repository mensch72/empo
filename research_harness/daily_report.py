#!/usr/bin/env python3
"""daily_report.py -- summarise the last 24h of harness activity.

Reads the state files and writes a short markdown digest to
reports/YYYY-MM-DD.md: tasks done, tasks blocked, and items awaiting human
judgement. Intended to be run once a day (cron) or on demand.

Usage:
    python3 daily_report.py            # window = last 24h, today's date
    python3 daily_report.py --hours 48
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def parse_ts(s: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc)
    except (ValueError, TypeError):
        return None


def read_log(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "_schema" in obj:
            continue
        rows.append(obj)
    return rows


# count "## task:" blocks by status in task_queue.md
def count_statuses(path: Path) -> dict:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    cur = None
    for line in path.read_text().splitlines():
        if line.startswith("## task:"):
            cur = True
        elif cur and line.strip().startswith("status:"):
            st = line.split(":", 1)[1].strip()
            counts[st] = counts.get(st, 0) + 1
            cur = None
    return counts


def tail_recent(path: Path, since: dt.datetime, max_items: int = 20) -> list[str]:
    """Pull bullet entries whose leading [timestamp] is within the window."""
    if not path.exists():
        return []
    text = path.read_text()
    # split on top-level "- [" bullets
    items = re.split(r"\n(?=- \[)", text)
    out = []
    for it in items:
        m = re.match(r"- \[([^\]]+)\]", it.strip())
        if not m:
            continue
        ts = parse_ts(m.group(1))
        # date-only timestamps (open_questions) -> keep if today-ish; be lenient
        if ts is None or ts >= since:
            out.append(it.strip())
    return out[-max_items:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=int, default=24)
    args = ap.parse_args()

    cfg = load_cfg()
    state = ROOT / cfg["paths"]["state_dir"]
    reports = ROOT / cfg["paths"]["reports_dir"]
    reports.mkdir(parents=True, exist_ok=True)

    now = dt.datetime.now(dt.timezone.utc)
    since = now - dt.timedelta(hours=args.hours)

    log = read_log(state / "experiment_log.jsonl")
    recent = [r for r in log if (parse_ts(r.get("timestamp", "")) or now) >= since]
    done = [r for r in recent if r.get("passed")]
    failed = [r for r in recent if not r.get("passed")]
    cost = sum(r.get("cost_usd") or 0 for r in recent)

    statuses = count_statuses(state / "task_queue.md")
    blocked_entries = tail_recent(state / "needs_human.md", since)

    lines = []
    lines.append(f"# Research harness digest -- {now:%Y-%m-%d}")
    lines.append(f"_window: last {args.hours}h (since {since:%Y-%m-%d %H:%MZ})_\n")

    lines.append("## At a glance")
    lines.append(f"- iterations logged in window: **{len(recent)}** "
                 f"({len(done)} passed, {len(failed)} failed)")
    lines.append(f"- estimated claude cost in window: **${cost:.4f}**")
    qsummary = ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())) or "none"
    lines.append(f"- task_queue.md status counts: {qsummary}\n")

    lines.append("## Tasks completed (verified)")
    if done:
        for r in done:
            lines.append(f"- `{r.get('task_id')}` ({r.get('type')}) "
                         f"-- artifact: `{r.get('command')}` "
                         f"-- log: `{r.get('result_path')}`")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Tasks blocked")
    if failed:
        for r in failed:
            lines.append(f"- `{r.get('task_id')}` ({r.get('type')}) "
                         f"-- {r.get('note','')}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Awaiting human (needs_human.md)")
    if blocked_entries:
        for e in blocked_entries:
            first = e.splitlines()[0]
            lines.append(f"- {first.lstrip('- ')}")
    else:
        lines.append("- (nothing new in window)")
    lines.append("")

    out_path = reports / f"{now:%Y-%m-%d}.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
