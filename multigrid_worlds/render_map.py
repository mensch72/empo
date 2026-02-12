#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "multigrid"))

from gym_multigrid.multigrid import MultiGridEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a MultiGrid YAML map to a single PNG frame."
    )
    parser.add_argument(
        "config",
        help="Path to YAML map (e.g. multigrid_worlds/puzzles/core_cases_compact.yaml)",
    )
    parser.add_argument(
        "--tile-size", type=int, default=32, help="Tile size in pixels"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Defaults to <config>.png",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    env = MultiGridEnv(config_file=str(config_path))
    if hasattr(env, "seed"):
        try:
            env.seed(args.seed)
        except Exception:
            pass
    env.reset()

    img = env.render(mode="rgb_array", highlight=False, tile_size=args.tile_size)
    out_path = Path(args.out) if args.out else config_path.with_suffix(".png")

    try:
        from PIL import Image

        Image.fromarray(img).save(out_path)
    except Exception:
        import imageio.v2 as imageio

        imageio.imwrite(out_path, img)

    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
