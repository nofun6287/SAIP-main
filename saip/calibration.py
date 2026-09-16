"""Section 3.4 -- cross-video statistical calibration.

Three quantities are re-aggregated after every round of refinement:

(1) the mean information density of the pseudo-label events of each video
    category, used to normalise ``S_density`` and remove baseline differences
    between categories;
(2) the occurrence frequency and the mean SFS of each event type inside each
    category, from which event types are confirmed either as *core events*
    (high frequency, high SFS) or as *background events* (low frequency, low
    SFS);
(3) the mean number of pseudo-label events of the videos of each category, used
    to check that the choice of K is reasonable.

The class-conditional probability that feeds Eq. (9) is the **fraction of the
videos of a category that contain an event of a given type**, exactly as the
example in the paper describes it ("'chopping' occurs in about 75% of videos,
P = 0.75").  It is therefore not a distribution over types that sums to one;
several types can each reach a high value, and the multiplicative form of
Eq. (9) is what keeps a rare event from outranking a common one.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .candidates import CandidateEvent
from .config import CalibrationConfig


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def _auto_k(embeddings: np.ndarray,
            k_min: int,
            k_max: int,
            tolerance: float = 1e-3) -> int:
    """Pick the cluster count with the best silhouette coefficient.

    Section 3.2.3 specifies the silhouette criterion and leaves the search range
    open; ``CalibrationConfig.type_k_range`` / ``cat_k_range`` hold it.  Scores
    within ``tolerance`` of the best are treated as tied so that a flat plateau
    does not automatically select the largest ``k``.
    """
    x = _l2(embeddings)
    upper = min(k_max, len(x) - 1)
    scores: List[Tuple[int, float]] = []
    for k in range(k_min, upper + 1):
        if k >= len(x):
            break
        labels = KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(x)
        try:
            scores.append((k, float(silhouette_score(x, labels))))
        except ValueError:                       # fewer distinct labels than k
            continue
    if not scores:
        return max(1, min(k_min, len(x)))
    best = max(score for _, score in scores)
    return min(k for k, score in scores if score >= best - tolerance)


class CorpusStats:
    """Corpus-level statistics of one refinement round.

    The object is rebuilt from the current pseudo-labels after every round, so
    no information leaks in from the ground truth at any point.
    """

    def __init__(self, cfg: Optional[CalibrationConfig] = None) -> None:
        self.cfg = cfg or CalibrationConfig()
        self.event_kmeans: Optional[KMeans] = None
        self.category_kmeans: Optional[KMeans] = None
        self.category_vocab: Optional[Dict[str, int]] = None
        self.n_event_types: int = 0
        self.n_categories: int = 0
        #: P(type | category), shape (n_categories, n_event_types).
        self.p_type_given_category_table: Optional[np.ndarray] = None
        #: Mean Eq. (8) density of the pseudo-label events of each category.
        self.category_mean_density: Optional[np.ndarray] = None
        #: Diagnostics produced by :meth:`describe`.
        self.report: dict = {}

    # -------------------------------------------------------- event types
    def fit_event_types(self, text_embeddings: np.ndarray) -> None:
        """K-means over the BLIP text embeddings of the pseudo-descriptions."""
        emb = np.asarray(text_embeddings, dtype=np.float32)
        if emb.ndim != 2 or len(emb) < self.cfg.min_samples:
            return
        k = self.cfg.n_event_types or _auto_k(emb, self.cfg.type_k_range[0],
                                              self.cfg.type_k_range[1],
                                              self.cfg.silhouette_tolerance)
        k = max(1, min(k, len(emb)))
        self.event_kmeans = KMeans(n_clusters=k, n_init=10,
                                   random_state=42).fit(_l2(emb))
        self.n_event_types = k

    def predict_event_types(self, events: Sequence[CandidateEvent]) -> None:
        """Write the type label of every event into ``CandidateEvent.type_id``."""
        if self.event_kmeans is None:
            for e in events:
                e.type_id = 0
            self.n_event_types = max(self.n_event_types, 1)
            return
        usable = [(i, e) for i, e in enumerate(events) if e.text_feat is not None]
        if usable:
            emb = _l2(np.stack([np.asarray(e.text_feat, dtype=np.float32)
                                for _, e in usable]))
            labels = self.event_kmeans.predict(emb)
            for (_, event), label in zip(usable, labels):
                event.type_id = int(label)
        for e in events:
            if e.text_feat is None:
                e.type_id = 0

    # -------------------------------------------------------- categories
    def assign_categories(self, videos: Sequence[object]) -> None:
        """Give every video a category id.

        The manifest category is used when present (ActivityNet ships one);
        otherwise the video-level mean feature is clustered, as Section 3.2.3
        describes.  ``videos`` must expose ``category``, ``mean_feature`` and a
        writable ``category_id`` attribute.
        """
        from_manifest = self.cfg.use_manifest_categories and any(
            getattr(v, "category", None) for v in videos)
        if from_manifest:
            vocab = sorted({v.category for v in videos if getattr(v, "category", None)})
            self.category_vocab = {name: i for i, name in enumerate(vocab)}
            for v in videos:
                v.category_id = self.category_vocab.get(v.category, 0)
            self.n_categories = max(1, len(vocab))
            return

        embeddings = np.stack([np.asarray(v.mean_feature, dtype=np.float32)
                               for v in videos])
        if len(embeddings) >= self.cfg.min_samples:
            k = self.cfg.n_categories or _auto_k(
                embeddings, self.cfg.cat_k_range[0], self.cfg.cat_k_range[1],
                self.cfg.silhouette_tolerance)
            k = max(1, min(k, len(embeddings)))
            self.category_kmeans = KMeans(n_clusters=k, n_init=10,
                                          random_state=42).fit(_l2(embeddings))
            labels = self.category_kmeans.predict(_l2(embeddings))
            self.n_categories = k
        else:
            labels = np.zeros(len(embeddings), dtype=int)
            self.n_categories = 1
        for video, label in zip(videos, labels):
            video.category_id = int(label)

    # ------------------------------------------- P(type | category), Eq. (9)
    def fit_priors(self, videos: Sequence[object]) -> None:
        """Estimate the class-conditional probability of Eq. (9).

        ``P(type = t | category = c)`` is the Laplace-smoothed fraction of the
        videos of category ``c`` that contain at least one pseudo-label event of
        type ``t``.
        """
        n_types = max(self.n_event_types, 1)
        n_cats = max(self.n_categories, 1)
        videos_per_cat = np.zeros(n_cats)
        videos_with = np.zeros((n_cats, n_types))

        for video in videos:
            cat = int(getattr(video, "category_id", 0))
            if not 0 <= cat < n_cats:
                continue
            videos_per_cat[cat] += 1.0
            for type_id in {int(e.type_id) for e in video.selected}:
                if 0 <= type_id < n_types:
                    videos_with[cat, type_id] += 1.0

        alpha = self.cfg.laplace_alpha
        self.p_type_given_category_table = (
            (videos_with + alpha) / (videos_per_cat[:, None] + alpha * n_types))

    def p_type_given_category(self, event: CandidateEvent,
                              category_id: Optional[int]) -> float:
        """Look up Eq. (9)'s first factor; 1.0 before any statistics exist."""
        table = self.p_type_given_category_table
        if table is None or category_id is None:
            return 1.0
        cat = int(category_id)
        if not 0 <= cat < table.shape[0]:
            return 1.0
        type_id = int(getattr(event, "type_id", 0))
        if not 0 <= type_id < table.shape[1]:
            return 1.0
        return float(table[cat, type_id])

    # --------------------------------------- Section 3.4(1): density means
    def fit_density_means(self, videos: Sequence[object]) -> None:
        """Mean Eq. (8) density of the pseudo-label events of each category."""
        n_cats = max(self.n_categories, 1)
        totals = np.zeros(n_cats)
        counts = np.zeros(n_cats)
        for video in videos:
            cat = int(getattr(video, "category_id", 0))
            if not 0 <= cat < n_cats:
                continue
            for event in video.selected:
                totals[cat] += float(event.s_density)
                counts[cat] += 1.0
        # Categories without any event keep a mean of 0, which would blow the
        # division up; fall back to the global mean for them.
        global_mean = totals.sum() / counts.sum() if counts.sum() > 0 else 1.0
        means = np.where(counts > 0, totals / np.maximum(counts, 1.0), global_mean)
        self.category_mean_density = np.maximum(means, 1e-8).astype(np.float32)

    def calibrate_density(self, density: np.ndarray,
                          category_id: Optional[int]) -> np.ndarray:
        """Section 3.4(1): divide ``S_density`` by the category mean."""
        if self.category_mean_density is None or category_id is None:
            return np.asarray(density, dtype=np.float32)
        cat = int(category_id)
        if not 0 <= cat < len(self.category_mean_density):
            return np.asarray(density, dtype=np.float32)
        return (np.asarray(density, dtype=np.float32)
                / self.category_mean_density[cat])

    # ------------------------------------------------------- Section 3.4(2)-(3)
    def describe(self, videos: Sequence[object]) -> dict:
        """Core/background event types and the event count per category.

        The paper defines core events as "high-frequency with a high SFS" and
        background events as "low-frequency with a low SFS" without fixing a
        cut-off; the quantiles in ``CalibrationConfig`` provide one, and every
        raw number is reported so the labelling can be recomputed by hand.
        """
        n_types = max(self.n_event_types, 1)
        n_cats = max(self.n_categories, 1)
        videos_per_cat = np.zeros(n_cats)
        events_per_video: Dict[int, List[int]] = {c: [] for c in range(n_cats)}
        videos_with = np.zeros((n_cats, n_types), dtype=int)
        sfs_sum = np.zeros((n_cats, n_types))
        sfs_count = np.zeros((n_cats, n_types))

        for video in videos:
            cat = int(getattr(video, "category_id", 0))
            if not 0 <= cat < n_cats:
                continue
            videos_per_cat[cat] += 1.0
            events_per_video[cat].append(len(video.selected))
            for type_id in {int(e.type_id) for e in video.selected}:
                if 0 <= type_id < n_types:
                    videos_with[cat, type_id] += 1

        for video in videos:
            cat = int(getattr(video, "category_id", 0))
            if not 0 <= cat < n_cats:
                continue
            for event in video.selected:
                type_id = int(event.type_id)
                if 0 <= type_id < n_types:
                    sfs_sum[cat, type_id] += float(event.sfs)
                    sfs_count[cat, type_id] += 1.0

        frequency = np.where(videos_per_cat[:, None] > 0,
                             videos_with / np.maximum(videos_per_cat[:, None], 1.0),
                             0.0)
        mean_sfs = np.where(sfs_count > 0, sfs_sum / np.maximum(sfs_count, 1.0),
                            np.nan)

        def _quantile(values: np.ndarray, q: float, fallback: float) -> float:
            finite = values[np.isfinite(values)]
            return float(np.quantile(finite, q)) if finite.size else fallback

        freq_cut = _quantile(frequency, self.cfg.core_freq_quantile, 0.0)
        sfs_cut = _quantile(mean_sfs, self.cfg.core_sfs_quantile, 0.0)

        types: List[dict] = []
        for cat in range(n_cats):
            for type_id in range(n_types):
                if videos_with[cat, type_id] == 0:
                    continue
                freq = float(frequency[cat, type_id])
                score = mean_sfs[cat, type_id]
                if not np.isfinite(score):
                    label = "unobserved"
                elif freq >= freq_cut and score >= sfs_cut:
                    label = "core"
                elif freq < freq_cut and score < sfs_cut:
                    label = "background"
                else:
                    label = "mixed"
                types.append({
                    "category_id": cat,
                    "event_type": type_id,
                    "n_videos": int(videos_with[cat, type_id]),
                    "n_events": int(sfs_count[cat, type_id]),
                    "frequency": round(freq, 4),
                    "mean_sfs": None if not np.isfinite(score) else round(float(score), 4),
                    "label": label,
                })

        per_category = [{
            "category_id": cat,
            "name": self.category_name(cat),
            "n_videos": int(videos_per_cat[cat]),
            "mean_events_per_video": (
                round(float(np.mean(events_per_video[cat])), 3)
                if events_per_video[cat] else None),
        } for cat in range(n_cats)]

        self.report = {
            "n_event_types": int(n_types),
            "n_categories": int(n_cats),
            "core_frequency_cut": round(freq_cut, 4),
            "core_sfs_cut": round(sfs_cut, 4),
            "event_types": types,
            "categories": per_category,
        }
        return self.report

    def category_name(self, category_id: int) -> Optional[str]:
        if not self.category_vocab:
            return None
        for name, idx in self.category_vocab.items():
            if idx == category_id:
                return name
        return None

    # ------------------------------------------------------------------ fit
    def fit(self, videos: Sequence[object],
            text_embeddings: Optional[np.ndarray] = None) -> "CorpusStats":
        """Run the whole Section 3.4 calibration on the current pseudo-labels."""
        if text_embeddings is None:
            text_embeddings = np.stack(
                [np.asarray(e.text_feat, dtype=np.float32)
                 for v in videos for e in v.selected if e.text_feat is not None]
            ) if any(e.text_feat is not None for v in videos for e in v.selected) else None

        if text_embeddings is not None:
            self.fit_event_types(text_embeddings)
        self.assign_categories(videos)
        for video in videos:
            self.predict_event_types(video.selected)
        self.fit_priors(videos)
        self.fit_density_means(videos)
        self.describe(videos)
        return self
