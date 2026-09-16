#!/usr/bin/env python3
"""Write a synthetic corpus so the pipeline can be exercised without BLIP.

The generated arrays follow the layout the ``cache`` backend expects, so the
rest of the pipeline behaves exactly as it does on real data:

    demo_data/manifest.json
    demo_data/feats/<video_id>.npy
    demo_data/captions/<video_id>.json
    demo_data/text_feats/<video_id>.npy

Usage::

    python scripts/make_demo_data.py --out demo_data --videos 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saip.synthetic import write_synthetic_corpus          # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="demo_data")
    parser.add_argument("--videos", type=int, default=8)
    parser.add_argument("--segments", type=int, default=5)
    parser.add_argument("--frames-per-segment", type=int, default=24)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    manifest = write_synthetic_corpus(
        args.out, n_videos=args.videos, n_segments=args.segments,
        frames_per_segment=args.frames_per_segment, fps=args.fps, seed=args.seed)
    print(f"wrote {args.videos} synthetic videos under {args.out}")
    print(f"manifest: {manifest}")
    print("run the pipeline with:")
    print(f"  python -m saip --manifest {manifest} "
          f"--feat-dir {args.out}/feats --caption-dir {args.out}/captions "
          f"--text-feat-dir {args.out}/text_feats --out-dir {args.out}/out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
