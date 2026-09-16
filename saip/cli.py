"""Command-line entry point.

    python -m saip --manifest manifest.json --feat-dir feats \
                   --caption-dir captions --text-feat-dir text_feats \
                   --dataset activitynet --out-dir out

Anything not exposed as a dedicated flag can be set with ``--set``::

    python -m saip --config configs/activitynet.yaml \
                   --set bcnet.max_iters=10 --set sfs.num_events=20 \
                   --ablate uniq --ablate conf

Configuration precedence is ``defaults < --config < explicit flags < --set``.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from typing import Any, Optional, Sequence

from .config import SAIPConfig
from .pipeline import run_pipeline

_ABLATION_NAMES = ("uniq", "density", "narr", "conf")


def _coerce(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, (list, tuple)):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return type(current)(_coerce(p, current[0]) if current else p for p in parts)
    return value


def apply_override(cfg: SAIPConfig, assignment: str) -> None:
    """Set ``section.field=value`` on a configuration object."""
    if "=" not in assignment:
        raise SystemExit(f"--set expects section.field=value, got {assignment!r}")
    path, raw = assignment.split("=", 1)
    parts = path.strip().split(".")
    if len(parts) != 2:
        raise SystemExit(f"--set expects section.field=value, got {assignment!r}")

    section_name, field_name = parts
    if not hasattr(cfg, section_name):
        raise SystemExit(f"unknown configuration section {section_name!r}")
    section = getattr(cfg, section_name)
    allowed = {f.name: f for f in fields(section)}
    if field_name not in allowed:
        raise SystemExit(f"unknown field {section_name}.{field_name}")
    current = getattr(section, field_name)
    setattr(section, field_name, _coerce(raw, current))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saip",
        description="Summarization-aware iterative pseudo-labeling for dense "
                    "video captioning (Sections 3.1-3.4).")
    parser.add_argument("--config", default=None,
                        help="YAML or JSON configuration file")
    parser.add_argument("--manifest", default=None, help="video inventory JSON")
    parser.add_argument("--out-dir", default=None, help="output directory")
    parser.add_argument("--dataset", default=None,
                        choices=["activitynet", "charades"],
                        help="layout of the pseudo-label dataset")

    features = parser.add_argument_group("features")
    features.add_argument("--backend", default=None,
                          choices=["cache", "blip", "mock"])
    features.add_argument("--feat-dir", default=None)
    features.add_argument("--caption-dir", default=None)
    features.add_argument("--text-feat-dir", default=None)
    features.add_argument("--video-root", default=None)
    features.add_argument("--device", default=None)

    run = parser.add_argument_group("pipeline")
    run.add_argument("--fps", type=float, default=None,
                     help="frame sampling rate gamma (video fps / stride)")
    run.add_argument("--num-events", type=int, default=None,
                     help="K, pseudo-label events per video")
    run.add_argument("--pool-min", type=int, default=None)
    run.add_argument("--pool-max", type=int, default=None)
    run.add_argument("--max-iters", type=int, default=None,
                     help="ceiling on the refinement rounds of Eq. (17)")
    run.add_argument("--epochs", type=int, default=None)
    run.add_argument("--lr", type=float, default=None)
    run.add_argument("--delta", type=int, default=None,
                     help="boundary search radius Delta, in frames")
    run.add_argument("--ablate", action="append", default=[], metavar="DIM",
                     choices=_ABLATION_NAMES,
                     help="zero one SFS dimension (repeatable)")
    run.add_argument("--set", action="append", default=[], metavar="SEC.FIELD=VAL",
                     help="generic configuration override (repeatable)")
    run.add_argument("--quiet", action="store_true")
    return parser


def build_config(argv: Optional[Sequence[str]] = None) -> SAIPConfig:
    args = build_parser().parse_args(argv)
    cfg = SAIPConfig.from_file(args.config) if args.config else SAIPConfig()

    explicit = {
        ("features", "backend"): args.backend,
        ("features", "feat_dir"): args.feat_dir,
        ("features", "caption_dir"): args.caption_dir,
        ("features", "text_feat_dir"): args.text_feat_dir,
        ("features", "video_root"): args.video_root,
        ("features", "device"): args.device,
        ("candidates", "fps"): args.fps,
        ("candidates", "pool_min"): args.pool_min,
        ("candidates", "pool_max"): args.pool_max,
        ("sfs", "num_events"): args.num_events,
        ("bcnet", "max_iters"): args.max_iters,
        ("bcnet", "epochs"): args.epochs,
        ("bcnet", "lr"): args.lr,
        ("bcnet", "delta"): args.delta,
    }
    for (section, name), value in explicit.items():
        if value is not None:
            setattr(getattr(cfg, section), name, value)

    if args.manifest:
        cfg.manifest = args.manifest
    if args.out_dir:
        cfg.output.out_dir = args.out_dir
    if args.dataset:
        cfg.output.dataset = args.dataset
    cfg.ablate_dimensions = list(args.ablate)
    for assignment in args.set:
        apply_override(cfg, assignment)

    cfg._quiet = args.quiet          # type: ignore[attr-defined]
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = build_config(argv)
    quiet = getattr(cfg, "_quiet", False)
    run_pipeline(cfg, manifest=cfg.manifest, verbose=not quiet)
    return 0


if __name__ == "__main__":                       # pragma: no cover
    raise SystemExit(main())
