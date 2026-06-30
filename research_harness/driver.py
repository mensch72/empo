#!/usr/bin/env python3
"""driver.py -- the autonomous research-harness loop.

Each iteration:
  a. read state/task_queue.md, pick the highest-priority `todo` task;
  b. build the iteration prompt and invoke `claude` headless (bounded);
  c. run the task's `verification` command;
  d. pass  -> mark `done`, append to findings.md;
     fail  -> mark `blocked`, log why to needs_human.md  (never `done` w/o artifact);
  e. drain state/new_tasks.md (follow-ups the iteration produced) into the queue.

Bounds (from config.yaml): max iterations per run, per-iteration budget +
wall-clock timeout (the claude CLI has no --max-turns flag, verified via
`claude --help`), and a hard wall-clock ceiling for the whole run. Any bound hit
=> loud exit.

Usage:
    python3 driver.py                 # run the loop in real mode
    python3 driver.py --dry-run       # print prompt + planned action, no claude
    python3 driver.py --max-iters 1   # override config bound for this run
    python3 driver.py --once          # exactly one iteration
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def loud(msg: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{msg}\n{bar}\n", flush=True)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    return cfg


def p(cfg: dict, key: str) -> Path:
    """Resolve a configured path (relative to harness root)."""
    return ROOT / cfg["paths"][key]


# --------------------------------------------------------------------------- #
# task_queue.md parsing / serialisation
# --------------------------------------------------------------------------- #
TASK_HEADER = re.compile(r"^##\s*task:\s*(?P<id>\S+)\s*$", re.MULTILINE)
KEYS = ("type", "status", "priority", "description", "verification", "note")


def parse_tasks(text: str) -> tuple[str, list[dict]]:
    """Return (header_comment, [task dicts]) preserving order."""
    matches = list(TASK_HEADER.finditer(text))
    header = text[: matches[0].start()] if matches else text
    tasks: list[dict] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        task = {"id": m.group("id")}
        for line in block.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, _, v = line.partition(":")
            k = k.strip().lower()
            if k in KEYS:
                task[k] = v.strip()
        task.setdefault("status", "todo")
        task.setdefault("priority", "100")
        tasks.append(task)
    return header, tasks


def serialise_tasks(header: str, tasks: list[dict]) -> str:
    out = [header.rstrip("\n"), ""]
    for t in tasks:
        out.append(f"## task: {t['id']}")
        for k in ("type", "status", "priority", "description", "verification", "note"):
            if k in t and t[k] != "":
                out.append(f"{k}: {t[k]}")
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def write_tasks(path: Path, header: str, tasks: list[dict]) -> None:
    path.write_text(serialise_tasks(header, tasks))


def pick_task(tasks: list[dict]) -> dict | None:
    todo = [t for t in tasks if t.get("status") == "todo"]
    if not todo:
        return None
    # lowest priority number first; stable for ties (queue order)
    return min(todo, key=lambda t: _int(t.get("priority"), 100))


def _int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# prompt building
# --------------------------------------------------------------------------- #
def build_prompt(cfg: dict, task: dict) -> str:
    tmpl = (p(cfg, "prompts_dir") / "iteration.md").read_text()
    repl = {
        "{{task_id}}": task["id"],
        "{{task_type}}": task.get("type", "?"),
        "{{task_description}}": task.get("description", ""),
        "{{task_verification}}": task.get("verification", ""),
    }
    for k, v in repl.items():
        tmpl = tmpl.replace(k, v)
    return tmpl


# --------------------------------------------------------------------------- #
# claude invocation
# --------------------------------------------------------------------------- #
def invoke_claude(cfg: dict, prompt: str, task_id: str) -> dict:
    """Run headless claude. Returns a dict with raw json + saved run_path."""
    c = cfg["claude"]
    runs = p(cfg, "runs_dir")
    runs.mkdir(parents=True, exist_ok=True)
    run_path = runs / f"{task_id}-{int(time.time())}.json"

    cmd = [
        c["binary"], "-p", prompt,
        "--output-format", "json",
        "--model", str(c["model"]),
        "--max-budget-usd", str(c["per_iteration_budget_usd"]),
    ]
    if cfg["permissions"]["skip_permissions"]:
        loud("!! WARNING: skip_permissions=ON -> passing "
             "--dangerously-skip-permissions. The agent can run arbitrary "
             "commands UNATTENDED. Only safe in a disposable sandbox.")
        cmd.append("--dangerously-skip-permissions")

    timeout = int(c["per_iteration_timeout_sec"])
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
        )
        raw = proc.stdout
        run_path.write_text(raw or proc.stderr)
        data = json.loads(raw) if raw.strip() else {}
        return {
            "run_path": str(run_path.relative_to(ROOT)),
            "is_error": bool(data.get("is_error", proc.returncode != 0)),
            "num_turns": data.get("num_turns"),
            "cost_usd": data.get("total_cost_usd"),
            "result": data.get("result", ""),
        }
    except subprocess.TimeoutExpired:
        run_path.write_text(f"TIMEOUT after {timeout}s")
        return {"run_path": str(run_path.relative_to(ROOT)), "is_error": True,
                "num_turns": None, "cost_usd": None,
                "result": f"timeout after {timeout}s"}
    except Exception as exc:  # noqa: BLE001
        run_path.write_text(f"ERROR: {exc}")
        return {"run_path": str(run_path.relative_to(ROOT)), "is_error": True,
                "num_turns": None, "cost_usd": None, "result": f"error: {exc}"}


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def run_verification(cfg: dict, command: str) -> tuple[bool, str]:
    if not command:
        return False, "no verification command on task (cannot prove completion)"
    try:
        proc = subprocess.run(
            command, cwd=ROOT, shell=True, capture_output=True, text=True,
            timeout=int(cfg["claude"]["per_iteration_timeout_sec"]),
        )
    except subprocess.TimeoutExpired:
        return False, "verification timed out"
    out = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, out


# --------------------------------------------------------------------------- #
# state-file logging
# --------------------------------------------------------------------------- #
def append(path: Path, text: str) -> None:
    with path.open("a") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def log_experiment(cfg: dict, row: dict) -> None:
    append(p(cfg, "state_dir") / "experiment_log.jsonl", json.dumps(row))


def record_finding(cfg: dict, task: dict, run_path: str) -> None:
    entry = (
        f"\n- [{utcnow()}] {task['id']} ({task.get('type','?')}): "
        f"{task.get('description','')}\n"
        f"    artifact: {task.get('verification','')}\n"
        f"    run_log:  {run_path}"
    )
    append(p(cfg, "state_dir") / "findings.md", entry)


def record_block(cfg: dict, task: dict, reason: str) -> None:
    entry = (
        f"\n- [{utcnow()}] {task['id']}: BLOCKED -- {task.get('description','')}\n"
        f"    context: {reason}"
    )
    append(p(cfg, "state_dir") / "needs_human.md", entry)


def drain_new_tasks(cfg: dict, header: str, tasks: list[dict]) -> list[dict]:
    """Move any task blocks from new_tasks.md into the live queue."""
    nt_path = p(cfg, "state_dir") / "new_tasks.md"
    if not nt_path.exists():
        return tasks
    nt_text = nt_path.read_text()
    nt_header, new = parse_tasks(nt_text)
    accepted = []
    existing_ids = {t["id"] for t in tasks}
    for t in new:
        if not t.get("verification"):
            print(f"  drain: rejecting follow-up {t['id']} (no verification)")
            continue
        if t["id"] in existing_ids:
            print(f"  drain: skipping duplicate follow-up id {t['id']}")
            continue
        t.setdefault("status", "todo")
        accepted.append(t)
        existing_ids.add(t["id"])
    if accepted:
        print(f"  drain: enqueued {len(accepted)} follow-up task(s): "
              f"{[t['id'] for t in accepted]}")
        tasks = tasks + accepted
    # reset inbox to just its header
    nt_path.write_text(nt_header.rstrip("\n") + "\n")
    return tasks


# --------------------------------------------------------------------------- #
# one iteration
# --------------------------------------------------------------------------- #
def iteration(cfg: dict, dry_run: bool) -> str:
    """Run one iteration. Returns a short status string for the loop."""
    queue = p(cfg, "state_dir") / "task_queue.md"
    header, tasks = parse_tasks(queue.read_text())
    task = pick_task(tasks)
    if task is None:
        return "EMPTY"

    print(f"-> picked task {task['id']} (type={task.get('type')}, "
          f"priority={task.get('priority')})")
    prompt = build_prompt(cfg, task)

    if dry_run:
        loud("DRY RUN -- prompt that WOULD be sent to claude:")
        print(prompt)
        loud("DRY RUN -- planned action:")
        print(f"  invoke: claude -p <prompt> --output-format json "
              f"--model {cfg['claude']['model']} "
              f"--max-budget-usd {cfg['claude']['per_iteration_budget_usd']}"
              + (" --dangerously-skip-permissions"
                 if cfg['permissions']['skip_permissions'] else ""))
        print(f"  then verify: {task.get('verification')}")
        print(f"  pass -> mark done + findings.md ; fail -> blocked + needs_human.md")
        return "DRYRUN"

    # mark in_progress
    task["status"] = "in_progress"
    write_tasks(queue, header, tasks)

    # b. invoke claude
    run = invoke_claude(cfg, prompt, task["id"])
    print(f"   claude: error={run['is_error']} turns={run['num_turns']} "
          f"cost=${run['cost_usd']} log={run['run_path']}")

    # c. verify (independent of what claude reported -- the artifact is the truth)
    passed, vout = run_verification(cfg, task.get("verification", ""))
    print(f"   verification {'PASSED' if passed else 'FAILED'}: "
          f"{vout.splitlines()[-1] if vout else ''}")

    # d. update status -- NEVER done without a passing artifact
    if passed:
        task["status"] = "done"
        record_finding(cfg, task, run["run_path"])
    else:
        task["status"] = "blocked"
        reason = (f"verification command failed. output: {vout[:500]} | "
                  f"claude error={run['is_error']}, result="
                  f"{str(run['result'])[:200]}")
        task["note"] = "blocked: verification failed (see needs_human.md)"
        record_block(cfg, task, reason)

    # persist status + experiment log
    write_tasks(queue, header, tasks)
    log_experiment(cfg, {
        "timestamp": utcnow(),
        "task_id": task["id"],
        "type": task.get("type"),
        "command": task.get("verification", ""),
        "result_path": run["run_path"],
        "passed": passed,
        "num_turns": run["num_turns"],
        "cost_usd": run["cost_usd"],
        "note": vout.splitlines()[-1] if vout else "",
    })

    # e. enqueue follow-ups produced this iteration
    header, tasks = parse_tasks(queue.read_text())
    tasks = drain_new_tasks(cfg, header, tasks)
    write_tasks(queue, header, tasks)

    return "DONE" if passed else "BLOCKED"


# --------------------------------------------------------------------------- #
# loop
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompt and planned action; do NOT call claude")
    ap.add_argument("--once", action="store_true", help="run exactly one iteration")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="override config max_iterations_per_run for this run")
    args = ap.parse_args()

    cfg = load_config()
    max_iters = args.max_iters if args.max_iters is not None \
        else int(cfg["bounds"]["max_iterations_per_run"])
    if args.once:
        max_iters = 1
    hard_ceiling = int(cfg["bounds"]["hard_wall_clock_sec"])

    if cfg["permissions"]["skip_permissions"]:
        loud("!! skip_permissions is ON for this run (see config.yaml).")

    start = time.time()
    print(f"driver start {utcnow()}  max_iters={max_iters}  "
          f"hard_ceiling={hard_ceiling}s  dry_run={args.dry_run}")

    completed = 0
    for i in range(1, max_iters + 1):
        elapsed = time.time() - start
        if elapsed > hard_ceiling:
            loud(f"HARD WALL-CLOCK CEILING HIT ({elapsed:.0f}s > "
                 f"{hard_ceiling}s) after {completed} iteration(s). Exiting.")
            return 3
        print(f"\n--- iteration {i}/{max_iters} "
              f"(elapsed {elapsed:.0f}s) ---")
        status = iteration(cfg, args.dry_run)
        if status == "EMPTY":
            loud("No `todo` tasks left in the queue. Nothing to do. Exiting.")
            return 0
        completed += 1
        if args.dry_run:
            break

    loud(f"MAX ITERATIONS REACHED ({completed}/{max_iters}). Exiting cleanly. "
         f"Re-run to continue the queue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
