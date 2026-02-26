#!/usr/bin/env python3
"""
Generate freeing-challenge maps for convergence experiments.

Clears multigrid_worlds/convergence/ and writes one YAML per size.

Layout (W columns × H rows):

  Row 0:    We We We ... We We We      ← top wall
  Row 1:    We We We ... Ay We We      ← human row: all walls except Ay at col W-3
  Row 2:    We .. ..  ... Ro .. We     ← rock at col W-3; col W-2 (right) empty for push
  Row 3..   We .. ..  ... .. .. We     ← open space
  Row H-2:  We Ae ..  ... .. .. We     ← robot at col 1
  Row H-1:  We We We  ... We We We     ← bottom wall

Usage:
    python experiments/convergence_assessment/generate_experiment_maps.py 7x5 9x7 11x9
    python experiments/convergence_assessment/generate_experiment_maps.py 7x5 9x7 --dry-run
"""

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = REPO_ROOT / 'multigrid_worlds' / 'convergence'


def generate_freeing_maps():
    min_width, max_width = 5, 10
    min_height, max_height = 4, 10

    def generate_freeing_map(width: int, height: int):
        rows = []
        rows.append(' '.join(['We'] * width))
        rows.append(' '.join('Ay' if x == width - 3 else 'We' for x in range(width)))
        row2 = []
        for x in range(width):
            if x == 0 or x == width - 1:
                row2.append('We')
            elif x == 1 and height == 4:
                row2.append('Ae')  # for 5x4, put robot here since row 3 doesn't exist
            elif x == width - 3:
                row2.append('Ro')
            else:
                row2.append('..')
        rows.append(' '.join(row2))

        for _ in range(3, height - 2):
            rows.append(' '.join(['We'] + ['..'] * (width - 2) + ['We']))

        if height > 4:
            rows.append(' '.join(['We', 'Ae'] + ['..'] * (width - 3) + ['We']))

        rows.append(' '.join(['We'] * width))
        return '\n'.join('  ' + row for row in rows)

    for width in range(min_width, max_width):
        for height in range(min_height, max_height):
            generated_map = generate_freeing_map(width, height)

            goals = [(width - 3, 1)]
            for y in range(2, height - 1):
                for x in range(1, width - 1):
                    goals.append((x, y))

            out_path = OUTPUT_DIR / f'freeing{width}x{height}.yaml'
            out_path.write_text(build_yaml(generated_map, width, height, goals))

    print("Generated freeing maps for widths {}-{} and heights {}-{} in {}/".format(
        min_width, max_width - 1, min_height, max_height - 1, OUTPUT_DIR.relative_to(REPO_ROOT)))

            
def build_yaml(map: str, width: int, height: int, goals: list[tuple[int, int]]) -> str:
    goals_lines = '\n'.join(f'  - "{x},{y}"' for x, y in goals)
    return f"""\
# Freeing challenge {width}x{height}
# Robot (grey) must push the rock left to free the trapped human (yellow).
metadata:
  name: "Freeing {width}x{height}"
  description: "The human (yellow) is trapped in the top-right corner behind a rock. The robot (grey) must navigate to the right of the rock and push it left to free the human."
  author: "convergence_assessment/generate_experiments.py"
  version: "1.0.0"
  created: "{date.today().isoformat()}"
  category: "convergence"
  tags: ["robot", "human", "rock", "freeing", "convergence"]

map: |
{map}

max_steps: {max(10, (width - 1) * (height - 1))}
can_push_rocks: "e"

possible_goals:
{goals_lines}
"""


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print(f'Cleared {OUTPUT_DIR.relative_to(REPO_ROOT)}/')
    OUTPUT_DIR.mkdir(parents=True)

    generate_freeing_maps()


if __name__ == '__main__':
    main()
