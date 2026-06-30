<!--
SCHEMA: findings.md  (append-only log of VERIFIED results)

Never edit or delete existing entries; the driver only appends. A line is
written here only after a task's `verification` command exited 0. Each entry:

    - [<UTC timestamp>] <task_id> (<type>): <description>
        artifact: <path or command that proves it>
        run_log:  <path to the saved claude run JSON>

If there is no artifact, there is no finding. Conceptual conclusions do not
belong here until something checkable backs them.
-->

- [2026-06-30T20:34:15Z] dummy-001 (math): Sanity-check the harness: confirm 2 + 2 == 4 via a numeric script.
    artifact: python3 verify/check_numeric.py --expr "2+2" --expected 4
    run_log:  state/runs/dummy-001-1782851625.json
