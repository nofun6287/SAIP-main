"""Section 3.1 -- over-generation of candidate events.

The stage deliberately keeps far more proposals than the final pseudo-label set
needs, and leaves the quality judgement to the SFS stage, which has global
information about the whole pool.

    Eq. (1)  h_t = BLIP_enc(f_t) in R^768          -- done by saip.features
    Eq. (2)  d_t = 1 - cos(h_t, h_{t+1})           -- frame_distance()
    Eq. (3)  f_bar = mean of h_t over the event    -- CandidateEvent.fbar
             proposed from every pair of boundary candidates spanning >= 2 s

The module is pure NumPy and has no model dependency, so it can be unit-tested
on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import CandidateConfig


@dataclass
class CandidateEvent:
    """One candidate event m_i of Section 3.1.

    ``s`` and ``e`` are inclusive frame indices into the feature sequence.  All
    frame indices in this package are inclusive on both ends, so an event has
    ``e - s + 1`` frames.
    """

    uid: int                       # stable identity of the proposal within its video
    s: int
    e: int
    mid: int                       # (s + e) // 2, the frame the description is read from
    fbar: np.ndarray               # Eq. (3) mean BLIP feature of the segment, shape (d,)
    var: float                     # Eq. (8) numerator, (1/n) * sum_t ||h_t - fbar||^2
    n_frames: int

    caption: str = ""              # pseudo-description c_i, filled by saip.features
    text_feat: Optional[np.ndarray] = None   # q_i, the BLIP text embedding of caption
    type_id: int = -1              # event type discovered by saip.calibration (Section 3.4)

    # SFS dimension scores -- raw values first, then the normalised values that
    # actually enter Eq. (4).
    s_uniq: float = 0.0
    s_density: float = 0.0
    s_narr: float = 0.0
    s_conf: float = 0.0
    s_uniq_n: float = 0.0
    s_density_n: float = 0.0
    s_narr_n: float = 0.0
    s_conf_n: float = 0.0
    sfs: float = 0.0

    def __post_init__(self) -> None:
        self.s = int(self.s)
        self.e = int(self.e)
        self.mid = int(self.mid)

    @property
    def span(self) -> int:
        """Number of frames covered by the event."""
        return self.e - self.s + 1

    def duration(self, fps: float) -> float:
        """Duration in seconds; the denominator convention of Eq. (8) is applied
        by :func:`saip.sfs.score_s_density`, not here."""
        return (self.e - self.s) / max(fps, 1e-9)

    def recompute_statistics(self, features: np.ndarray) -> None:
        """Refresh ``fbar`` and ``var`` after the boundaries have moved.

        Section 3.3.2 re-localises the boundaries of the candidate events, which
        invalidates both quantities because they are defined over the segment.
        """
        seg = features[self.s:self.e + 1]
        if seg.shape[0] == 0:
            return
        self.fbar = seg.mean(axis=0)
        # Eq. (8) numerator: (1/n) * sum_t ||h_t - h_bar||^2.  The squared norm is
        # summed over the 768 feature dimensions, hence .sum(axis=1) and not a
        # plain .mean() over the array.
        self.var = float(np.mean(np.sum((seg - self.fbar) ** 2, axis=1)))
        self.n_frames = int(seg.shape[0])
        self.mid = (self.s + self.e) // 2

    def to_dict(self, fps: float, include_scores: bool = True) -> dict:
        out = {
            "start_frame": self.s,
            "end_frame": self.e,
            "start_sec": round(self.s / max(fps, 1e-9), 3),
            "end_sec": round((self.e + 1) / max(fps, 1e-9), 3),
            "caption": self.caption,
        }
        if include_scores:
            out.update({
                "sfs": round(self.sfs, 6),
                "s_uniq": round(self.s_uniq, 6),
                "s_density": round(self.s_density, 6),
                "s_narr": round(self.s_narr, 6),
                "s_conf": round(self.s_conf, 6),
            })
        return out


# ---------------------------------------------------------------------------
# Eq. (2): cosine distance between adjacent frames
# ---------------------------------------------------------------------------
def frame_distance(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Eq. (2): ``d_t = 1 - cos(h_t, h_{t+1})``, of length ``T - 1``."""
    h = np.asarray(features, dtype=np.float32)
    h = h / (np.linalg.norm(h, axis=-1, keepdims=True) + eps)
    cos = np.clip((h[:-1] * h[1:]).sum(axis=-1), -1.0, 1.0)
    return (1.0 - cos).astype(np.float32)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Edge-padded moving average; suppresses single-frame noise before peak
    detection.  IMPLEMENTATION CHOICE -- not specified by the paper."""
    if window <= 1 or x.size == 0:
        return x
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:x.size]


def detect_boundaries(diff: np.ndarray,
                      min_gap: int,
                      thresh_k: float,
                      smooth_window: int = 3) -> List[int]:
    """Locate the time positions at which visual semantics change significantly.

    A frame index ``t`` is kept when the smoothed distance trajectory has a
    local maximum at ``t`` that exceeds ``mean + thresh_k * std``; peaks closer
    together than ``min_gap`` frames are suppressed in favour of the stronger
    one.  ``min_gap`` is the 2-second lower bound on the span of an event.
    """
    if diff.size < 3:
        return []
    d = _moving_average(np.asarray(diff, dtype=np.float32), smooth_window)
    sigma = float(d.std())
    if sigma < 1e-9:
        return []
    thresh = float(d.mean()) + thresh_k * sigma

    peaks = [t for t in range(1, d.size - 1)
             if d[t] > d[t - 1] and d[t] >= d[t + 1] and d[t] > thresh]
    if not peaks:
        return []

    # Non-maximum suppression, strongest peak first.
    order = np.argsort(d[np.asarray(peaks)])[::-1]
    kept: List[int] = []
    for idx in order:
        p = peaks[int(idx)]
        if all(abs(p - q) >= min_gap for q in kept):
            kept.append(p)
    return sorted(kept)


def pairwise_spans(boundaries: Sequence[int],
                   t_last: int,
                   min_span: int) -> List[Tuple[int, int]]:
    """Every pair ``(s, e)`` with ``e > s`` and a span of at least ``min_span``.

    The first and the last frame of the video act as sentinel boundaries: the
    start and the end of the video are semantic boundaries by construction, and
    an event running up to them must be reachable.  IMPLEMENTATION CHOICE -- the
    paper says "every pair of boundary candidates" without naming the sentinels.
    """
    b = sorted({0, t_last, *(int(x) for x in boundaries)})
    return [(b[i], b[j])
            for i in range(len(b))
            for j in range(i + 1, len(b))
            if b[j] - b[i] + 1 >= min_span]


def _make_event(uid: int, features: np.ndarray, s: int, e: int) -> CandidateEvent:
    seg = features[s:e + 1]
    fbar = seg.mean(axis=0)
    var = float(np.mean(np.sum((seg - fbar) ** 2, axis=1)))
    return CandidateEvent(uid=uid, s=s, e=e, mid=(s + e) // 2,
                          fbar=fbar.astype(np.float32), var=var,
                          n_frames=int(seg.shape[0]))


def build_candidates(features: np.ndarray,
                     cfg: Optional[CandidateConfig] = None,
                     ) -> Tuple[List[CandidateEvent], dict]:
    """Build the candidate pool of Section 3.1 for one video.

    Returns the pool and a metadata dict (number of boundary candidates, the
    peak threshold actually used, whether the pool had to be trimmed).
    """
    cfg = cfg or CandidateConfig()
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"features must be (T, d), got shape {features.shape}")

    T = int(features.shape[0])
    min_span = max(int(round(cfg.min_span_sec * cfg.fps)), 2)
    meta = {"T": T, "n_boundaries": 0, "thresh_k": cfg.peak_thresh_k,
            "trimmed": False, "padded": False}
    if T < min_span + 1:
        return [], meta

    diff = frame_distance(features)
    t_last = T - 1

    # Relax the peak threshold step by step until the pool reaches pool_min.
    chosen: List[CandidateEvent] = []
    for factor in cfg.peak_thresh_factors:
        thresh_k = cfg.peak_thresh_k * float(factor)
        boundaries = detect_boundaries(diff, min_gap=min_span, thresh_k=thresh_k,
                                       smooth_window=cfg.peak_smooth_win)
        spans = pairwise_spans(boundaries, t_last, min_span)
        pool = [_make_event(i, features, s, e) for i, (s, e) in enumerate(spans)]
        chosen, meta["n_boundaries"], meta["thresh_k"] = pool, len(boundaries), thresh_k
        if len(pool) >= cfg.pool_min:
            break

    if not chosen:
        return [], meta

    # Upper bound of the pool.
    if len(chosen) > cfg.pool_max:
        if cfg.trim_by_variance:
            chosen.sort(key=lambda c: c.var, reverse=True)
        chosen = chosen[:cfg.pool_max]
        meta["trimmed"] = True

    # Lower bound.  Reached only when the trajectory carries too few peaks; the
    # pool is padded with evenly spaced spans so that downstream stages always
    # see at least pool_min proposals.
    if len(chosen) < cfg.pool_min:
        chosen = _pad_pool(features, chosen, cfg.pool_min, min_span)
        meta["padded"] = True

    for new_uid, event in enumerate(chosen):
        event.uid = new_uid
        event.mid = (event.s + event.e) // 2
    chosen.sort(key=lambda c: c.s)
    return chosen, meta


def _pad_pool(features: np.ndarray,
              pool: List[CandidateEvent],
              target: int,
              min_span: int) -> List[CandidateEvent]:
    """IMPLICIT fallback: pad an under-sized pool with evenly spaced spans."""
    T = int(features.shape[0])
    existing = {(c.s, c.e) for c in pool}
    grid = sorted({int(x) for x in np.linspace(0, T - 1, num=12).round()})
    next_uid = max((c.uid for c in pool), default=-1) + 1
    padded = list(pool)
    for i in range(len(grid)):
        for j in range(i + 1, len(grid)):
            if len(padded) >= target:
                return padded
            s, e = grid[i], grid[j]
            if e - s + 1 < min_span or (s, e) in existing:
                continue
            padded.append(_make_event(next_uid, features, s, e))
            next_uid += 1
    return padded
