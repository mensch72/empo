# Autonomous research harness

A small driver loop that repeatedly invokes **headless Claude Code** to grind
through well-scoped research subtasks (literature search, running/logging
experiments, refactoring, checking algebra against a spec) for an AI-safety
theory project — unattended for days — while **parking every conceptual
judgement call for the human**.

The governing rule is **"no artifact, no progress"**: a task counts as done only
when a *checkable artifact* proves it (a sympy/numeric check, a rerunnable logged
script, a fetched source). The driver decides done/blocked by **running the
task's `verification` command**, not by trusting what the agent says.

> This harness lives in `research_harness/` (a subdirectory) so it does not
> collide with the host project's own `README.md` and files at the repo root.

## Layout

```
research_harness/
  driver.py            # the loop (pick task -> claude -> verify -> record)
  daily_report.py      # 24h markdown digest into reports/
  config.yaml          # caps, model, paths, permission-skip toggle
  prompts/iteration.md # per-iteration prompt template (the hard rules)
  verify/              # verification helpers the driver calls
    check_numeric.py   #   math claim   -> sympy equality check
    rerun_experiment.py#   empirical    -> rerun a logged script (stub)
    check_url.py       #   literature   -> confirm a source resolves
  state/
    task_queue.md      # ordered tasks (id/type/status/priority/verification)
    findings.md        # append-only log of VERIFIED results
    open_questions.md  # tractable open questions
    needs_human.md     # fork-in-the-road decisions + blocked obstacles
    experiment_log.jsonl # one JSON object per iteration
    new_tasks.md       # follow-up inbox; driver drains it into task_queue.md
    runs/              # saved raw claude run JSON, one per iteration
  reports/             # daily digests, reports/YYYY-MM-DD.md
```

## Prerequisites

- `claude` CLI on PATH (verified against **v2.1.197**), authenticated.
- Python 3.11+, with `pyyaml` and `sympy` (`pip install pyyaml sympy`).

## Seed the task queue

Add task blocks to `state/task_queue.md`. Each block **must** include a
`verification:` line — a shell command (run from the harness root) that exits 0
only when a real artifact confirms the result:

```
## task: lit-002
type: lit
status: todo
priority: 2
description: Find the canonical reference for the maximum-entropy RL objective.
verification: python3 verify/check_url.py --url https://arxiv.org/abs/1702.08165
```

Valid `type`: `lit | experiment | math | refactor | writing`.
Valid `status`: `todo | in_progress | done | blocked`.
`priority` is an integer — **lower = more urgent**.

## Run one iteration

```bash
cd research_harness
python3 driver.py --once          # pick the top todo, run it, verify, record
python3 driver.py --dry-run       # print the prompt + planned action, no claude
```

## Run the loop

```bash
python3 driver.py                 # up to bounds.max_iterations_per_run
python3 driver.py --max-iters 50  # override the per-run cap
```

The driver exits **loudly** when any bound is hit:
- `max_iterations_per_run` (per run),
- `per_iteration_budget_usd` + `per_iteration_timeout_sec` (per claude call),
- `hard_wall_clock_sec` (whole run).

It is designed to be re-run (e.g. from cron) to keep draining the queue.

## Interacting with a running loop

The loop is file-driven and checks for control signals **between iterations**, so
nothing you do can corrupt an in-flight task — the current iteration always
finishes first.

**See what it's doing (read-only, safe anytime):**
```bash
python3 driver.py status
```
Prints the queue status counts, the in-progress task (if any), the next `todo`,
the last few logged iterations, and whether a STOP is pending. It only reads the
state files — it never touches the loop.

**Stop it gracefully:**
```bash
touch state/STOP        # driver finishes the current iteration, then exits
# ... or, if you have the terminal / pid:
kill -TERM <pid>        # same; Ctrl-C works too (second Ctrl-C force-kills)
```
After a STOP-file stop, **delete the sentinel before re-running**:
```bash
rm state/STOP
```

**Steer the work (also between iterations):**
- Add/re-prioritise/cancel tasks by editing `state/task_queue.md` (lower
  `priority` = picked sooner; set `status: blocked` to skip one). Edit only while
  the loop is paused/stopped or between iterations — edits made *during* an
  iteration are overwritten when the driver writes the task's result back.
- Append follow-up tasks to `state/new_tasks.md` (the append-only inbox); the
  driver drains them into the queue and won't clobber your additions.

There is intentionally **no live mid-task control** (no socket/REPL): an
unattended driver that shells out to a blocking `claude` call can only be steered
at iteration boundaries. The STOP file is the mechanism that works for remote /
headless runs where you can't send Ctrl-C.

## Read the digest

```bash
python3 daily_report.py           # writes reports/YYYY-MM-DD.md (last 24h)
python3 daily_report.py --hours 48
```

## Unattended runs & the permission toggle

A truly unattended agent needs to run tools without a human approving each one.
The `claude` CLI gates that behind `--dangerously-skip-permissions`. The driver
will pass it **only** when `permissions.skip_permissions: true` in `config.yaml`,
and it prints a loud warning every run when on.

**It is OFF by default.** Turn it on only in a disposable, network-isolated
sandbox where the agent running arbitrary commands unattended is acceptable.

## Design notes / choices made

- **No `--max-turns` in the CLI.** `claude --help` (v2.1.197) exposes no
  turn-limit flag. So "per-iteration turn cap" is implemented as a **budget cap**
  (`--max-budget-usd`) plus a **wall-clock `subprocess` timeout**
  (`per_iteration_timeout_sec`). The raw run JSON still records `num_turns` for
  visibility. If a future CLI adds a turn flag, wire it in `invoke_claude()`.
- **Verification is the source of truth.** The driver ignores the agent's
  self-assessment; a task becomes `done` only if its `verification` command exits
  0. This is what makes "no artifact, no progress" enforceable.
- **Judgement calls are parked, not made.** The iteration prompt forbids the
  agent from resolving conceptual/design questions; those go to `needs_human.md`.
  Blocked tasks also land there with context.
- **Follow-ups via an inbox.** The agent appends new subtasks to
  `state/new_tasks.md`; the driver drains them into the queue (rejecting any
  without a `verification:` line, and duplicate ids).
- **Placement.** Put in `research_harness/` to avoid clobbering the host repo's
  root files; all paths in `config.yaml` are relative to this directory.
- **Verify helpers are minimal/stubs.** `check_numeric.py` is fully functional;
  `rerun_experiment.py` and `check_url.py` are deliberately thin — extend them
  per the project's real experiments and citation rules.
