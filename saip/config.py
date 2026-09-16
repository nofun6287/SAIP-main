"""Configuration objects for the SAIP pipeline.

Every hyper-parameter is declared once here, with the value used in the paper as
the default.  A configuration can be built from a YAML/JSON file, from a plain
dict, or from the command line (see :mod:`saip.cli`); the precedence is
``defaults < config file < command line``.

Where the paper does not pin a value down (peak-detection thresholds, the
mechanism used to keep the candidate pool inside ``[pool_min, pool_max]``, and
the quantiles that separate core from background event types), the choice made
here is marked ``IMPLEMENTATION CHOICE`` and explained in ``docs/paper_mapping.md``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin, get_type_hints


def _coerce_value(value: Any, hint: Any) -> Any:
    """Turn a YAML/JSON value into the shape its annotation asks for.

    YAML has no tuple type, so a field declared as a tuple arrives as a list;
    the rest of the code compares and unpacks these by position, and the
    round-trip through ``to_dict`` should give back what was read.
    """
    if get_origin(hint) is tuple and isinstance(value, (list, tuple)):
        return tuple(value)
    return value


@dataclass
class CandidateConfig:
    """Section 3.1 -- over-generation of candidate events."""

    #: Sampling rate gamma of the frame-level features, in frames per second
    #: (paper: keyframes are extracted at 3 fps; Eq. 8 uses gamma to turn a
    #: frame span into a duration in seconds).
    fps: float = 3.0

    #: Minimum span of a candidate event.  The paper writes "no less than
    #: 2 seconds (i.e. 6 frames)" at 3 fps.
    min_span_sec: float = 2.0

    #: Lower / upper bound of the candidate pool size M (paper: M = 20 ~ 80).
    pool_min: int = 20
    pool_max: int = 80

    #: IMPLEMENTATION CHOICE -- peak detection on the cosine-distance trajectory.
    #: A frame is a boundary candidate when it is a local maximum, lies above
    #: ``mean + peak_thresh_k * std``, and is at least one minimum span away
    #: from every stronger peak (non-maximum suppression).
    peak_smooth_win: int = 3
    peak_thresh_k: float = 0.8

    #: IMPLEMENTATION CHOICE -- the threshold is relaxed by these factors, in
    #: order, until the candidate pool reaches ``pool_min``.  The paper states
    #: only the target pool size, not how it is reached.
    peak_thresh_factors: Tuple[float, ...] = (1.0, 0.5, 0.25)

    #: IMPLEMENTATION CHOICE -- when the pool exceeds ``pool_max`` it is trimmed
    #: by descending segment variance, i.e. by the S_density proxy of Eq. (8).
    #: Set to ``False`` to keep the first ``pool_max`` candidates in start-time
    #: order instead.
    trim_by_variance: bool = True

    seed: int = 42


@dataclass
class SFSConfig:
    """Section 3.2 -- SFS scoring (Eqs. 4-10) and greedy selection."""

    #: Eq. (4): the four dimensions enter with equal weight.
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

    #: How the four dimensions are placed on a common scale before Eq. (4) is
    #: evaluated.  ``"minmax"`` maps each dimension to [0, 1] over the candidate
    #: pool of the current video; ``"none"`` uses the raw values.
    #: See ``docs/paper_mapping.md`` for why a normalisation step is needed for
    #: Eq. (4) to be meaningful, and which sentences of Section 3.2 license it.
    normalize: str = "minmax"

    #: Eq. (7): piecewise-linear temporal modulation factor phi(dt).
    phi_near: float = 0.05
    phi_far: float = 0.25
    phi_floor: float = 0.3
    phi_slope: float = 3.5
    phi_intercept: float = 1.175

    #: Eq. (8) denominator ``(e_i - s_i) / gamma``, exactly as printed.
    #: Set to ``True`` to divide by the number of frames actually summed in the
    #: numerator, ``(e_i - s_i + 1) / gamma``.
    density_closed_span: bool = False

    #: Section 3.2.5: candidates overlapping a selected event by more than this
    #: temporal IoU are discarded (paper: tIoU > 0.5).
    tioi_thresh: float = 0.5

    #: Size of the pseudo-label set produced for each video.  The paper states
    #: the candidate pool is "roughly 2-8 times the target K" with M = 20 ~ 80,
    #: which fixes K = 10; see ``docs/paper_mapping.md``.
    num_events: int = 10


@dataclass
class BCNetConfig:
    """Section 3.3 -- boundary calibration network and iterative refinement."""

    #: Eqs. (11)-(14).
    conv_kernel: int = 3
    d_attn: int = 256
    n_heads: int = 4
    hidden: int = 128

    #: Section 3.3.2(1).
    epochs: int = 3
    lr: float = 1e-3
    batch_size: int = 32

    #: The paper writes ``L_total = L_BCE + L_DIoU``, i.e. lambda = 1.
    lambda_diou: float = 1.0

    #: Positives are the 2K boundary frames out of T, so an unweighted BCE can
    #: collapse to the all-negative solution on long videos.  The paper's
    #: Eq. (15) is unweighted, which is the default; ``"auto"`` switches on the
    #: ``neg/pos`` positive-class weight of the PyTorch implementation.
    bce_pos_weight: str = "none"

    #: Search radius Delta, in frames, of the boundary re-localisation in both
    #: Eq. (16) and Section 3.3.2(2).
    delta: int = 4

    #: Temperature of the differentiable soft-argmax used to obtain the
    #: predicted boundary of Eq. (16) during training.
    softmax_temperature: float = 0.5

    #: Maximum number of refinement rounds.  Table 3 of the paper reports
    #: convergence after T = 6 rounds, so the default ceiling is set above it
    #: and Eq. (17) stops the loop earlier in practice.
    max_iters: int = 8

    #: Eq. (17).
    #: How the intersection of the two pseudo-label sets is decided.
    #: ``"tioi"`` matches an event of round t with an event of round t+1 when
    #: their temporal IoU reaches ``sfs.tioi_thresh``, which is the usual reading
    #: of the Jaccard similarity of two sets of temporal intervals;
    #: ``"identity"`` requires the very same candidate proposal to be selected
    #: again, which turns J into a binary "did the selection freeze" flag and
    #: makes it largely redundant with the displacement term.  The paper writes
    #: the formula without fixing the matching rule.
    jaccard_mode: str = "tioi"
    delta_jaccard: float = 0.98
    delta_boundary_frames: float = 0.5

    #: Re-decode the description of an event from its new middle frame after a
    #: boundary update.  Off by default: it needs the BLIP caption decoder.
    refresh_captions: bool = False

    seed: int = 42


@dataclass
class CalibrationConfig:
    """Section 3.4 -- cross-video statistical calibration."""

    #: Laplace smoothing for P(type | category); the paper specifies smoothing
    #: but not its strength.
    laplace_alpha: float = 0.5

    #: Number of event types / video categories.  ``None`` selects the value in
    #: ``type_k_range`` / ``cat_k_range`` with the best silhouette coefficient,
    #: as described in Section 3.2.3.
    n_event_types: Optional[int] = None
    n_categories: Optional[int] = None
    type_k_range: Tuple[int, int] = (3, 16)
    cat_k_range: Tuple[int, int] = (3, 24)

    #: When several k are within this silhouette margin of the best one the
    #: smallest of them wins.  The criterion is flat over a range of k whenever
    #: the embeddings contain near-duplicate captions, and picking the largest k
    #: of a flat region fragments one event type into many.
    silhouette_tolerance: float = 1e-3

    #: Minimum number of samples before a K-means fit is attempted.
    min_samples: int = 5

    #: IMPLEMENTATION CHOICE -- Section 3.4(2) says event types that are
    #: "high-frequency with a high SFS" are core events and those that are
    #: "low-frequency with a low SFS" are background events, without giving a
    #: cut-off.  A type is labelled ``core`` when both its occurrence frequency
    #: within the class and its mean SFS are at or above these within-class
    #: quantiles, and ``background`` when both are below them.
    core_freq_quantile: float = 0.5
    core_sfs_quantile: float = 0.5

    #: Video categories are taken from the manifest when it carries a
    #: ``category`` field (ActivityNet does), and discovered by clustering the
    #: video-level mean features otherwise.
    use_manifest_categories: bool = True


@dataclass
class FeatureConfig:
    """How frame features, frame captions and text features are obtained."""

    #: ``"cache"``  read pre-extracted arrays from disk (recommended: the
    #:               feature extractor runs once on a GPU machine),
    #: ``"blip"``   run the frozen BLIP models in-process,
    #: ``"mock"``   synthesise a corpus without any model, for smoke tests only.
    backend: str = "cache"

    #: Directory holding ``<video_id>.npy`` with the (T, 768) frame features.
    feat_dir: Optional[str] = None
    #: Directory holding ``<video_id>.npy`` with the (T, 768) text features of
    #: the per-frame captions.
    text_feat_dir: Optional[str] = None
    #: Directory holding ``<video_id>.json`` with the per-frame captions.
    caption_dir: Optional[str] = None
    #: Directory holding the decoded videos, used by the ``blip`` backend.
    video_root: Optional[str] = None

    #: Model identifiers, by name.  The weights themselves are not distributed
    #: with this repository.
    itm_model: str = "blip-itm-base-coco"
    caption_model: str = "blip-caption-large-coco"
    itm_ckpt: Optional[str] = None
    caption_ckpt: Optional[str] = None
    device: str = "cuda"
    batch_size: int = 128

    #: Frame sampling: the video is decoded at ``video_fps / stride`` fps, the
    #: value that must match ``CandidateConfig.fps``.
    stride: int = 8
    input_size: int = 384

    #: Number of caption samples drawn per frame; the first one is used.
    num_stnc: int = 1


@dataclass
class OutputConfig:
    """Pseudo-label dataset that is written to disk."""

    #: ``"activitynet"`` writes ``train_pseudo.json`` in the ActivityNet
    #: Captions layout, ``"charades"`` writes ``charades_sta_train_pseudo.txt``.
    dataset: str = "activitynet"
    out_dir: str = "out"

    #: Also dump the per-event SFS breakdown, which is what the ablation study
    #: of Section 4.6 switches on and off.
    write_event_scores: bool = True

    #: Also dump the Section 3.4(2)-(3) diagnostics.
    write_calibration_report: bool = True


@dataclass
class SAIPConfig:
    """Top-level configuration."""

    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    sfs: SFSConfig = field(default_factory=SFSConfig)
    bcnet: BCNetConfig = field(default_factory=BCNetConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    #: Video inventory: ``{"videos": [{"id": ..., "duration": ..., ...}, ...]}``.
    manifest: Optional[str] = None

    #: Ablation switches for Section 4.6: dropping a dimension sets its weight
    #: to zero, and ``sfs_only_conf`` reproduces the SPL-equivalent scoring.
    ablate_dimensions: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SAIPConfig":
        """Build a configuration from a nested dict, ignoring unknown keys.

        The annotations are resolved with :func:`typing.get_type_hints` rather
        than read off ``dataclasses.fields`` directly, because
        ``from __future__ import annotations`` turns every annotation into a
        string; a naive ``field.type`` lookup would not recognise the nested
        sections and would silently leave them as plain dicts.
        """
        hints = get_type_hints(cls)
        kwargs: Dict[str, Any] = {}
        for key, value in (data or {}).items():
            if key not in hints:
                continue
            hint = hints[key]
            if isinstance(value, dict) and hasattr(hint, "__dataclass_fields__"):
                sub = get_type_hints(hint)
                clean = {k: _coerce_value(v, sub[k])
                         for k, v in value.items() if k in sub}
                kwargs[key] = hint(**clean)
            else:
                kwargs[key] = _coerce_value(value, hint)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str | Path) -> "SAIPConfig":
        """Load a YAML (``.yml``/``.yaml``) or JSON configuration file."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - dependency hint
                raise RuntimeError(
                    "Reading a YAML configuration requires PyYAML "
                    "(pip install pyyaml), or use a .json file instead."
                ) from exc
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------- ablations
    def apply_ablations(self) -> "SAIPConfig":
        """Zero the SFS weights named in :attr:`ablate_dimensions`.

        Recognised names are ``uniq``, ``density``, ``narr`` and ``conf``, which
        reproduce the first four rows of the ablation table in Section 4.6.
        """
        if not self.ablate_dimensions:
            return self
        index = {"uniq": 0, "density": 1, "narr": 2, "conf": 3}
        weights = list(self.sfs.weights)
        for name in self.ablate_dimensions:
            key = name.strip().lower()
            if key not in index:
                raise ValueError(
                    f"unknown ablation dimension {name!r}; "
                    f"expected one of {sorted(index)}")
            weights[index[key]] = 0.0
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        self.sfs.weights = tuple(weights)
        return self
