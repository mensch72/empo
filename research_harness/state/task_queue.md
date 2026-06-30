<!--
SCHEMA: task_queue.md  (ordered, machine-parsed by driver.py)

Everything above the first "## task:" header is the schema comment and is
preserved on rewrite. Each task is a block introduced by a header line:

    ## task: <id>

followed by `key: value` lines (one per line). Recognised keys:

    type:          one of  lit | experiment | math | refactor | writing
    status:        one of  todo | in_progress | done | blocked
    priority:      integer; LOWER number = higher priority (1 is most urgent)
    description:   one-line human description of the subtask
    verification:  REQUIRED. A shell command the driver runs to prove the task
                   is complete. It MUST exit 0 only when a checkable artifact
                   confirms the result (a numeric/sympy check, a rerunnable
                   logged script, a fetched source). No passing command => the
                   task can never be marked done. This is the "no artifact, no
                   progress" gate.
    note:          (optional) free text; the driver appends failure reasons here

The driver picks the highest-priority `todo` task each iteration, runs the
iteration prompt through headless claude, then runs `verification`. Pass =>
status becomes `done` and a line is appended to findings.md. Fail => status
becomes `blocked` and the obstacle is logged to needs_human.md.

Follow-up tasks discovered during an iteration are appended by the agent to
state/new_tasks.md and drained into this file by the driver.
-->

## task: dummy-001
type: math
status: done
priority: 1
description: Sanity-check the harness: confirm 2 + 2 == 4 via a numeric script.
verification: python3 verify/check_numeric.py --expr "2+2" --expected 4
