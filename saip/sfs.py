"""Section 3.2 -- SFS scoring (Eqs. 4-10) and greedy event selection.

    Eq. (4)   SFS = 0.25 S_uniq + 0.25 S_density + 0.25 S_narr + 0.25 S_conf
    Eq. (5)   S_uniq    = 1 - max_{j != i} cos(f_i, f_j)
    Eq. (6)   S_uniq    = 1 - max_{j != i} [ cos(f_i, f_j) * phi(dt_ij) ]
    Eq. (7)   phi(dt)   piecewise linear, 1.0 -> 0.3 between dt = 0.05 and 0.25
    Eq. (8)   S_density = [ (1/n) sum_t ||h_t - h_bar||^2 ] / [ (e - s) / gamma ]
    Eq. (9)   S_narr    = P(type | category) * (1 - |mid / T - 0.5|)
    Eq. (10)  S_conf    = mean in-segment alignment - mean out-of-segment alignment

Two conventions are worth stating explicitly, because they are easy to get
wrong and both feed Eq. (4):

* ``T`` in Eqs. (7) and (9) is the number of frames of the **video**, not of the
  candidate pool.  The normalised temporal distance ``dt`` and the temporal
  centrality term are both defined against the whole video.

* Eq. (4) combines the four dimensions directly, so they must share a scale.
  Section 3.2 asserts the range of each one (``S_uniq`` in [0, 1], the position
  factor in [0.5, 1], ``S_conf`` "normalised to the interval [0, 1]") and
  Section 3.4(1) normalises ``S_density`` across videos of a category.  The
  concrete mapping used here is a per-pool min-max rescaling, selected by
  ``SFSConfig.normalize``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .candidates import CandidateEvent
from .config import SFSConfig


# ---------------------------------------------------------------------------
# Eq. (7): piecewise-linear temporal modulation factor
# ---------------------------------------------------------------------------
def temporal_modulation(dt: float, cfg: Optional[SFSConfig] = None) -> float:
    """``phi(dt)`` of Eq. (7).

    ``dt`` is the normalised temporal distance ``|mid_i - mid_j| / T``.
    """
    cfg = cfg or SFSConfig()
    if dt < cfg.phi_near:
        return 1.0
    if dt <= cfg.phi_far:
        return cfg.phi_intercept - cfg.phi_slope * dt
    return cfg.phi_floor


def _phi_array(dt: np.ndarray, cfg: SFSConfig) -> np.ndarray:
    """Vectorised Eq. (7); identical to :func:`temporal_modulation` elementwise."""
    return np.where(
        dt < cfg.phi_near, 1.0,
        np.where(dt <= cfg.phi_far,
                 cfg.phi_intercept - cfg.phi_slope * dt,
                 cfg.phi_floor)).astype(np.float32)


# ---------------------------------------------------------------------------
# Eqs. (5) and (6): semantic uniqueness
# ---------------------------------------------------------------------------
def score_s_uniq(events: Sequence[CandidateEvent],
                 video_len: int,
                 cfg: Optional[SFSConfig] = None) -> np.ndarray:
    """Eqs. (5) and (6), computed pairwise over the whole candidate pool."""
    cfg = cfg or SFSConfig()
    n = len(events)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.ones(1, dtype=np.float32)

    f = np.stack([np.asarray(e.fbar, dtype=np.float32) for e in events])
    f = f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-8)
    cosine = np.clip(f @ f.T, -1.0, 1.0, out=None)

    mids = np.asarray([e.mid for e in events], dtype=np.float32)
    dt = np.abs(mids[:, None] - mids[None, :]) / max(float(video_len), 1.0)
    phi = _phi_array(dt, cfg)
    np.fill_diagonal(phi, 0.0)          # exclude j == i from the max

    worst = (cosine * phi).max(axis=-1)
    return np.clip(1.0 - worst, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Eq. (8): information density
# ---------------------------------------------------------------------------
def score_s_density(events: Sequence[CandidateEvent],
                    fps: float,
                    cfg: Optional[SFSConfig] = None) -> np.ndarray:
    """Eq. (8), using ``var`` as computed by :mod:`saip.candidates`.

    ``fps`` is the sampling rate gamma of Eq. (8), i.e.
    :attr:`saip.config.CandidateConfig.fps`; it is passed in rather than
    duplicated in the SFS configuration so that the frame span and the duration
    can never drift apart.

    The denominator is ``(e_i - s_i) / gamma``, exactly as printed in the paper.
    Setting ``cfg.density_closed_span`` divides by the number of frames actually
    summed in the numerator instead, ``(e_i - s_i + 1) / gamma``.
    """
    cfg = cfg or SFSConfig()
    out = np.zeros(len(events), dtype=np.float32)
    for i, e in enumerate(events):
        span = e.e - e.s + (1 if cfg.density_closed_span else 0)
        seconds = span / max(fps, 1e-9)
        out[i] = e.var / max(seconds, 1e-9)
    return out


# ---------------------------------------------------------------------------
# Eq. (9): narrative role
# ---------------------------------------------------------------------------
def score_s_narr(events: Sequence[CandidateEvent],
                 video_len: int,
                 stats=None,
                 category_id: Optional[int] = None) -> np.ndarray:
    """Eq. (9): ``P(type(m_i) | category(V)) * (1 - |mid / T - 0.5|)``.

    ``T`` is the length of the video in feature frames, as stated in the paper.
    When no cross-video statistics are available yet (the very first round, or
    a single-video corpus) the class-conditional probability degenerates to 1
    and only the temporal-centrality factor remains.
    """
    total = max(float(video_len), 1.0)
    out = np.zeros(len(events), dtype=np.float32)
    for i, e in enumerate(events):
        centrality = 1.0 - abs(e.mid / total - 0.5)
        prior = 1.0
        if stats is not None:
            prior = stats.p_type_given_category(e, category_id)
        out[i] = prior * centrality
    return out


# ---------------------------------------------------------------------------
# Eq. (10): cross-modal confidence
# ---------------------------------------------------------------------------
def score_s_conf(events: Sequence[CandidateEvent],
                 features: np.ndarray,
                 eps: float = 1e-8) -> np.ndarray:
    """Eq. (10): main-lobe minus side-lobe cross-modal alignment.

    ``q_i`` is the BLIP text embedding of the pseudo-description of the event
    and ``h_t`` the frame features; both live in the same 768-d space.  Events
    without a description yet score 0.
    """
    h = np.asarray(features, dtype=np.float32)
    h = h / (np.linalg.norm(h, axis=-1, keepdims=True) + eps)
    T = h.shape[0]
    out = np.zeros(len(events), dtype=np.float32)

    for i, e in enumerate(events):
        if e.text_feat is None:
            continue
        q = np.asarray(e.text_feat, dtype=np.float32)
        q = q / (np.linalg.norm(q) + eps)
        alignment = h @ q                                   # (T,)

        lo, hi = max(e.s, 0), min(e.e, T - 1)
        n_in = hi - lo + 1
        n_out = T - n_in
        if n_in <= 0 or n_out <= 0:
            continue
        inside = alignment[lo:hi + 1].mean()
        outside = (alignment.sum() - alignment[lo:hi + 1].sum()) / n_out
        out[i] = float(inside - outside)
    return out


# ---------------------------------------------------------------------------
# Eq. (4): combination
# ---------------------------------------------------------------------------
def _minmax(x: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1] over the current candidate pool."""
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def normalise_dimensions(raw: Sequence[np.ndarray], mode: str) -> List[np.ndarray]:
    if mode == "none":
        return [np.asarray(x, dtype=np.float32) for x in raw]
    if mode == "minmax":
        return [_minmax(x) for x in raw]
    raise ValueError(f"unknown SFSConfig.normalize {mode!r}; "
                     "expected 'minmax' or 'none'")


def compute_sfs(events: List[CandidateEvent],
                features: np.ndarray,
                video_len: int,
                fps: float,
                cfg: Optional[SFSConfig] = None,
                stats=None,
                category_id: Optional[int] = None,
                ) -> np.ndarray:
    """Evaluate Eq. (4) for every event in the pool, in place.

    ``fps`` is the sampling rate gamma of Eq. (8), ``video_len`` the frame count
    ``T`` that Eqs. (7) and (9) normalise against.  ``stats`` is a
    :class:`saip.calibration.CorpusStats`; when it carries a per-category mean
    density, Section 3.4(1) calibration is applied to ``S_density`` before
    Eq. (4).  The raw Eq. (8) value stays in ``event.s_density`` so that the
    calibration step itself can read it back.
    """
    cfg = cfg or SFSConfig()
    if not events:
        return np.zeros(0, dtype=np.float32)

    s_uniq = score_s_uniq(events, video_len, cfg)
    s_density = score_s_density(events, fps, cfg)
    s_narr = score_s_narr(events, video_len, stats, category_id)
    s_conf = score_s_conf(events, features)

    calibrated = s_density
    if stats is not None and category_id is not None:
        calibrated = stats.calibrate_density(s_density, category_id)

    normalised = normalise_dimensions(
        [s_uniq, calibrated, s_narr, s_conf], cfg.normalize)

    weights = np.asarray(cfg.weights, dtype=np.float32)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("SFSConfig.weights must contain at least one "
                         "non-zero weight")
    weights = weights / total
    sfs = sum(w * n for w, n in zip(weights, normalised))

    for i, e in enumerate(events):
        e.s_uniq, e.s_density = float(s_uniq[i]), float(s_density[i])
        e.s_narr, e.s_conf = float(s_narr[i]), float(s_conf[i])
        e.s_uniq_n, e.s_density_n = float(normalised[0][i]), float(normalised[1][i])
        e.s_narr_n, e.s_conf_n = float(normalised[2][i]), float(normalised[3][i])
        e.sfs = float(sfs[i])
    return np.asarray(sfs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Section 3.2.5: greedy selection
# ---------------------------------------------------------------------------
def temporal_iou(s1: int, e1: int, s2: int, e2: int) -> float:
    """One-dimensional temporal IoU over closed frame intervals."""
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    union = max(e1, e2) - min(s1, s2) + 1
    return inter / max(union, 1)


def greedy_select(events: Sequence[CandidateEvent],
                  k: int,
                  tioi_thresh: float = 0.5) -> List[CandidateEvent]:
    """Pick the K highest-scoring events, dropping anything they overlap.

    Each step takes the remaining candidate with the largest SFS and removes
    every other candidate whose temporal IoU with it exceeds ``tioi_thresh``
    (Section 3.2.5). Fixed-score ranking with overlap suppression is a heuristic;
    the cardinality-constrained submodular approximation guarantee does not apply.
    """
    pool = list(events)
    selected: List[CandidateEvent] = []
    while pool and len(selected) < k:
        best = max(range(len(pool)), key=lambda i: pool[i].sfs)
        chosen = pool.pop(best)
        selected.append(chosen)
        pool = [e for e in pool
                if temporal_iou(chosen.s, chosen.e, e.s, e.e) <= tioi_thresh]
    return selected
