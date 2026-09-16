"""Reading the video inventory and writing the pseudo-label dataset.

Section 3.4 ends with the output of the final pseudo-label set ``L*``.  Two
layouts are supported, both of which downstream DVC code consumes directly:

``activitynet``
    ``train_pseudo.json``::

        {"<video_id>": {"duration": 185.2,
                        "timestamps": [[0.13, 0.42], ...],   # normalised to [0, 1]
                        "sentences":  ["a person ...", ...]}}

``charades``
    ``charades_sta_train_pseudo.txt``, one line per event::

        <video_id> <start_sec> <end_sec>##<description>

Frame indices are converted with the sampling rate gamma: frame ``t`` covers the
half-open interval ``[t / gamma, (t + 1) / gamma)``.  Timestamps in the
ActivityNet layout are then divided by the video duration, which keeps events
that sit at the very end of a video inside ``[0, 1]`` even when the recorded
duration and the sampled frame count disagree slightly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


def load_manifest(path: str | Path) -> List[dict]:
    """Read a video inventory.

    Accepts either ``{"videos": [...]}`` or a bare list.  Every record needs an
    ``id``; ``duration`` (seconds) and ``category`` are optional but recommended
    -- Section 3.4(1) needs the category, and the ActivityNet layout needs the
    duration.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data.get("videos") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("manifest must be {'videos': [...]} or a JSON list")

    cleaned: List[dict] = []
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not record.get("id"):
            raise ValueError(f"manifest entry {index} has no 'id'")
        video_id = str(record["id"])
        if video_id in seen:
            raise ValueError(f"duplicate video id {video_id!r} in manifest")
        seen.add(video_id)
        cleaned.append({
            "id": video_id,
            "duration": float(record["duration"]) if record.get("duration") else None,
            "category": record.get("category"),
            "path": record.get("path"),
        })
    return cleaned


def frame_span_to_seconds(start_frame: int, end_frame: int, fps: float) -> tuple:
    """Frame span -> ``(start, end)`` in seconds, end exclusive."""
    fps = max(float(fps), 1e-9)
    return start_frame / fps, (end_frame + 1) / fps


def build_activitynet_records(units: Iterable[object],
                              fps: float) -> Dict[str, dict]:
    """Assemble the ``train_pseudo.json`` payload from the finished units.

    Each unit must expose ``video_id``, ``duration`` and ``selected``.
    """
    records: Dict[str, dict] = {}
    for unit in units:
        if not getattr(unit, "selected", None):
            continue
        duration = getattr(unit, "duration", None) or \
            len(unit.features) / max(fps, 1e-9)
        duration = max(float(duration), 1e-9)
        spans = []
        for event in unit.selected:
            start, end = frame_span_to_seconds(event.s, event.e, fps)
            start = float(np.clip(start / duration, 0.0, 1.0))
            end = float(np.clip(end / duration, 0.0, 1.0))
            if end > start:
                spans.append((start, end, event.caption))
        # Written in chronological order, as in the ActivityNet annotations.
        spans.sort(key=lambda span: span[0])
        if spans:
            records[unit.video_id] = {
                "duration": round(duration, 3),
                "timestamps": [[round(s, 4), round(e, 4)] for s, e, _ in spans],
                "sentences": [caption for _, _, caption in spans],
            }
    return records


def build_charades_lines(units: Iterable[object], fps: float) -> List[str]:
    """Assemble the Charades-style ``<vid> <start> <end>##<description>`` lines."""
    lines: List[str] = []
    for unit in units:
        for event in getattr(unit, "selected", []):
            start, end = frame_span_to_seconds(event.s, event.e, fps)
            lines.append(f"{unit.video_id} {start:.2f} {end:.2f}##"
                         f"{event.caption.strip()}")
    return lines


def write_pseudo_labels(units: Sequence[object],
                        dataset: str,
                        out_dir: str | Path,
                        fps: float) -> Path:
    """Write the pseudo-label dataset and return the path of the main file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "activitynet":
        records = build_activitynet_records(units, fps)
        target = out_dir / "train_pseudo.json"
        target.write_text(json.dumps(records, indent=2, ensure_ascii=False),
                          encoding="utf-8")
        return target
    if dataset == "charades":
        lines = build_charades_lines(units, fps)
        target = out_dir / "charades_sta_train_pseudo.txt"
        target.write_text("\n".join(lines) + ("\n" if lines else ""),
                          encoding="utf-8")
        return target
    raise ValueError(f"unknown dataset layout {dataset!r}; "
                     "expected 'activitynet' or 'charades'")


def write_event_scores(units: Sequence[object],
                       out_dir: str | Path,
                       fps: float) -> Path:
    """Dump the per-event SFS breakdown, i.e. the ablation axis of Section 4.6.

    The file records the four raw dimension scores as well as the SFS of every
    event that entered the final pseudo-label set, which is what makes the
    ``w/o S_uniq`` / ``w/o S_density`` / ... rows reproducible by zeroing one
    weight in the configuration and rerunning.
    """
    payload = {}
    for unit in units:
        payload[unit.video_id] = [
            {**event.to_dict(fps), "type_id": int(event.type_id)}
            for event in getattr(unit, "selected", [])
        ]
    target = Path(out_dir) / "pseudo_label_scores.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    return target


def write_json(payload: Mapping, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    return target
