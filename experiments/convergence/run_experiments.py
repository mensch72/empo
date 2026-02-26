#!/usr/bin/env python3
"""
Parallel experiment launcher for phase2_robot_policy_demo.py.

Runs one process per map, streams output to per-run log files,
and respects a worker-count limit to avoid CPU thrashing.

Usage:
    # Run on a list of maps (each gets its own process + log file)
    python experiments/convergence_assessment/run_experiments.py key_bearer trolley small_trolley

    # Limit to 2 concurrent runs (default: number of CPU cores)
    python experiments/convergence_assessment/run_experiments.py key_bearer trolley --workers 2

    # Pass extra flags through to the demo script
    python experiments/convergence_assessment/run_experiments.py key_bearer trolley --quick --seed 0

    # Read maps from a file (one per line, # comments ignored)
    python experiments/convergence_assessment/run_experiments.py --maps-file my_maps.txt --steps 50000
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from generate_experiments import SIZES


REPO_ROOT = Path(__file__).parent.parent.parent
SCRIPT_PATH = REPO_ROOT / 'examples' / 'phase2' / 'phase2_robot_policy_demo.py'
LOGS_ROOT = REPO_ROOT / 'outputs' / 'convergence' / 'logs'

def run_instance(map_name: str, log_dir: Path, seed: int) -> dict:
    """Run a single demo process, streaming output to a log file."""
    safe_name = map_name.replace('/', '_').replace('.yaml', '')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'{safe_name}_{timestamp}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(SCRIPT_PATH),
        '--world', map_name,
        '--seed', str(seed),  # unique seed per run
        '--rollouts', '1',
    ]

    print(f'[{safe_name}] Starting  → {log_path.relative_to(REPO_ROOT)}')
    start = time.monotonic()

    with open(log_path, 'w') as log_file:
        log_file.write(f'# Command: {" ".join(cmd)}\n')
        log_file.write(f'# Started: {datetime.now().isoformat()}\n\n')
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(REPO_ROOT),
        )
        returncode = proc.wait()

    elapsed = time.monotonic() - start
    status = 'OK' if returncode == 0 else f'FAILED (exit {returncode})'
    print(f'[{safe_name}] {status}  ({elapsed:.0f}s)')

    return {'map': map_name, 'log': str(log_path), 'returncode': returncode, 'elapsed': elapsed}


def main():
    maps = []
    for w, h in SIZES:
        maps.append(f'convergence/freeing{w}x{h}')

    num_workers = 4 # os.cpu_count()
    workers = min(num_workers, len(maps))
    print(f'Running {len(maps)} experiment(s), up to {workers} concurrent')

    results = []
    log_dir = LOGS_ROOT
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_instance, map_name, log_dir, i): map_name
            for i, map_name in enumerate(maps)
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                map_name = futures[future]
                print(f'[{map_name}] ERROR: {exc}')
                results.append({'map': map_name, 'returncode': -1, 'elapsed': 0, 'log': ''})

    # Summary
    print()
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    ok = [r for r in results if r['returncode'] == 0]
    failed = [r for r in results if r['returncode'] != 0]
    for r in sorted(results, key=lambda r: r['map']):
        status = 'OK' if r['returncode'] == 0 else f"FAILED ({r['returncode']})"
        print(f"  {status:20s}  {r['elapsed']:6.0f}s  {r['map']}")
    print()
    print(f'{len(ok)}/{len(results)} succeeded')
    if failed:
        print('Failed runs:')
        for r in failed:
            print(f"  {r['map']}  log: {r['log']}")
        sys.exit(1)


if __name__ == '__main__':
    main()
