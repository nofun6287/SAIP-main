"""Synthetic corpus generator.

**This module exists only so that the pipeline can be executed end to end on a
laptop, without a GPU, without the BLIP checkpoints, and without ActivityNet.**
It is what ``tests/`` and ``scripts/make_demo_data.py`` use; none of the numbers
reported in the paper come from it.

Each synthetic video is a sequence of latent semantic segments: every segment has
its own direction in feature space, the frame features are that direction plus
noise, and the caption vocabulary is indexed by segment, so that

* the cosine-distance trajectory of Eq. (2) has a peak exactly at a segment
  change, which the peak detection of Section 3.1 should find;
* ``cos(q_i, h_t)`` is high inside the segment the description came from and low
  outside it, which is what Eq. (10) measures;
* events from the same segment type land in the same K-means cluster of
  Section 3.4.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import FeatureProvider

#: Caption templates, one per latent segment index.  Kept short and generic on
#: purpose: they only have to be distinct strings for the text-feature step.
SEGMENT_TEMPLATES = [
    "a person is opening a box and taking the object out",
    "a person is pouring water into a glass on the table",
    "a person is walking across the room towards the door",
    "a person is cutting vegetables on a wooden board",
    "a person is wiping the table with a cloth",
    "a person is putting the plate into the cupboard",
    "a person is riding a bicycle along the street",
    "a person is writing something in a notebook",
]


def _video_rng(video_id: str, seed: int) -> np.random.RandomState:
    digest = hashlib.sha1(f"{seed}:{video_id}".encode("utf-8")).digest()
    return np.random.RandomState(int.from_bytes(digest[:4], "little"))


def generate_video(video_id: str,
                   dim: int = 768,
                   n_segments: int = 5,
                   frames_per_segment: int = 24,
                   noise: float = 0.08,
                   seed: int = 7,
                   ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Return ``(frame features (T, d), per-frame captions, per-frame text feats)``."""
    rng = _video_rng(video_id, seed)
    T = n_segments * frames_per_segment
    basis = rng.randn(len(SEGMENT_TEMPLATES), dim).astype(np.float32)
    basis /= np.linalg.norm(basis, axis=1, keepdims=True) + 1e-8
    segment_of = rng.randint(0, len(SEGMENT_TEMPLATES), size=n_segments)

    features, captions, text_features = [], [], []
    for frame in range(T):
        segment = frame // frames_per_segment
        direction = basis[segment_of[segment]]
        features.append(direction + rng.normal(0, noise, dim))
        captions.append(SEGMENT_TEMPLATES[segment_of[segment]])
        # The text embedding of a caption sits near the visual direction of its
        # own segment, which is what makes Eq. (10) discriminate.
        text_features.append(direction + rng.normal(0, 0.05, dim))
    return (np.asarray(features, dtype=np.float32), captions,
            np.asarray(text_features, dtype=np.float32))


class SyntheticFeatures(FeatureProvider):
    """In-memory provider used when ``FeatureConfig.backend == "mock"``."""

    def __init__(self, dim: int = 768, seed: int = 7,
                 n_segments: int = 5, frames_per_segment: int = 24) -> None:
        self.dim = dim
        self.seed = seed
        self.n_segments = n_segments
        self.frames_per_segment = frames_per_segment
        self._cache: Dict[str, Tuple[np.ndarray, List[str], np.ndarray]] = {}

    def _video(self, video_id: str):
        if video_id not in self._cache:
            self._cache[video_id] = generate_video(
                video_id, dim=self.dim, n_segments=self.n_segments,
                frames_per_segment=self.frames_per_segment, seed=self.seed)
        return self._cache[video_id]

    def frame_features(self, video_id: str) -> Optional[np.ndarray]:
        return self._video(video_id)[0]

    def captions_for(self, video_id: str, frame_indices: Sequence[int]):
        _, captions, text_features = self._video(video_id)
        picked_captions = [captions[i] if 0 <= i < len(captions) else ""
                           for i in frame_indices]
        picked_features = np.stack(
            [text_features[i] if 0 <= i < len(text_features)
             else np.zeros(self.dim, dtype=np.float32) for i in frame_indices])
        return picked_captions, picked_features

    def release(self, video_id: str) -> None:
        self._cache.pop(video_id, None)


def write_synthetic_corpus(root: str | Path,
                           n_videos: int = 8,
                           categories: Sequence[str] = ("cooking", "sports"),
                           dim: int = 768,
                           n_segments: int = 5,
                           frames_per_segment: int = 24,
                           fps: float = 3.0,
                           seed: int = 7) -> Path:
    """Write a demo corpus in the layout of the ``cache`` backend.

    Produces ``manifest.json``, ``feats/``, ``captions/`` and ``text_feats/``
    under ``root`` and returns the manifest path.
    """
    root = Path(root)
    (root / "feats").mkdir(parents=True, exist_ok=True)
    (root / "captions").mkdir(parents=True, exist_ok=True)
    (root / "text_feats").mkdir(parents=True, exist_ok=True)

    videos = []
    for index in range(n_videos):
        video_id = f"demo_{index:04d}"
        features, captions, text_features = generate_video(
            video_id, dim=dim, n_segments=n_segments,
            frames_per_segment=frames_per_segment, seed=seed)
        np.save(root / "feats" / f"{video_id}.npy", features)
        np.save(root / "text_feats" / f"{video_id}.npy", text_features)
        (root / "captions" / f"{video_id}.json").write_text(
            json.dumps(captions, ensure_ascii=False), encoding="utf-8")
        videos.append({
            "id": video_id,
            "category": categories[index % len(categories)],
            "duration": round(len(features) / fps, 3),
            "path": None,
        })

    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"videos": videos}, indent=2,
                                   ensure_ascii=False), encoding="utf-8")
    return manifest
