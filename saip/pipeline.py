"""End-to-end orchestration of Sections 3.1-3.4.

    load inventory
      for each video:
        Section 3.1   frame features -> candidate pool (over-generation)
        Section 3.2   descriptions -> SFS -> greedy selection -> L^(0)
      Section 3.4     cross-video calibration on L^(0)
      Section 3.3     repeat until Eq. (17) holds:
                        train the boundary calibration network on L^(t)
                        re-localise the boundaries of the candidate pool
                        refresh the segment statistics of every candidate
                        recompute SFS, re-run greedy selection  -> L^(t+1)
                        re-aggregate the Section 3.4 statistics
                      stop on Jaccard > 0.98 and mean shift < 0.5 frames
      Section 3.4     write the final pseudo-label dataset L*

The loop mirrors the paper's ``Phi: L^(t) -> L^(t+1)`` one to one.  Two points
are worth spelling out because they are where a re-implementation usually drifts:
the boundaries that are corrected are those of the **whole candidate pool**, not
only of the currently selected events; and the previous round is snapshotted by
value, since the events are mutated in place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .calibration import CorpusStats
from .candidates import CandidateEvent, build_candidates
from .config import SAIPConfig
from .dataset_io import (load_manifest, write_event_scores, write_json,
                         write_pseudo_labels)
from .features import FeatureProvider, build_provider
from .sfs import compute_sfs, greedy_select


@dataclass
class VideoUnit:
    """Everything known about one video while the pipeline runs."""

    video_id: str
    duration: float
    features: np.ndarray                       # (T, d), Eq. (1)
    category: Optional[str] = None
    category_id: int = 0
    pool: List[CandidateEvent] = field(default_factory=list)
    selected: List[CandidateEvent] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def mean_feature(self) -> np.ndarray:
        """Video-level mean feature, used to cluster categories in Section 3.2.3."""
        return self.features.mean(axis=0)

    @property
    def length(self) -> int:
        return int(self.features.shape[0])


@dataclass
class RoundLog:
    """One line of the iteration history, i.e. one row of the paper's Table 3."""

    round_index: int
    loss: float
    jaccard: float
    displacement: float
    n_events: int
    stopped: bool = False

    def as_dict(self) -> dict:
        return {"round": self.round_index,
                "train_loss": round(self.loss, 6),
                "jaccard": round(self.jaccard, 6),
                "mean_boundary_shift_frames": round(self.displacement, 4),
                "n_pseudo_events": self.n_events,
                "converged": self.stopped}


class SAIPPipeline:
    """The whole annotation-free pipeline."""

    def __init__(self, cfg: SAIPConfig, provider: Optional[FeatureProvider] = None,
                 verbose: bool = True) -> None:
        self.cfg = cfg.apply_ablations()
        cfg.features.sampling_fps = cfg.candidates.fps
        self.provider = provider or build_provider(cfg.features)
        self.verbose = verbose
        self.units: List[VideoUnit] = []
        self.stats: Optional[CorpusStats] = None
        self.history: List[RoundLog] = []
        self.notes: List[str] = []

    # ------------------------------------------------------------------ utils
    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    # ------------------------------------------------------- Section 3.1 + 3.2
    def _build_pool(self, record: dict) -> Optional[VideoUnit]:
        video_id = record["id"]
        features = self.provider.frame_features(video_id)
        if features is None and hasattr(self.provider, "set_video"):
            features = self.provider.set_video(video_id, record.get("path"))
        if features is None or features.shape[0] < 8:
            self.notes.append(f"{video_id}: no usable frame features, skipped")
            return None
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError(f"{video_id}: features must be a finite (T, d) array")

        pool, meta = build_candidates(features, self.cfg.candidates)
        if not pool:
            self.notes.append(f"{video_id}: empty candidate pool, skipped")
            return None

        captions, text_features = self.provider.captions_for(
            video_id, [event.mid for event in pool])
        for index, event in enumerate(pool):
            event.caption = captions[index] if index < len(captions) else ""
            if not event.caption:
                raise ValueError(f"{video_id}: missing caption at frame {event.mid}")
            if text_features is not None and index < len(text_features):
                event.text_feat = text_features[index]
                if event.text_feat.shape != (features.shape[1],) or not np.isfinite(event.text_feat).all():
                    raise ValueError(f"{video_id}: invalid shared-space text feature")
            else:
                raise ValueError(f"{video_id}: missing text features")

        duration = record.get("duration") or features.shape[0] / max(
            self.cfg.candidates.fps, 1e-9)
        return VideoUnit(video_id=video_id, duration=float(duration),
                         features=features, category=record.get("category"),
                         pool=pool, meta=meta)

    def _reselect(self, unit: VideoUnit) -> None:
        """Recompute SFS over the pool of one video and run greedy selection."""
        if self.stats is not None:
            self.stats.predict_event_types(unit.pool)
        compute_sfs(unit.pool, unit.features, unit.length,
                    fps=self.cfg.candidates.fps,
                    cfg=self.cfg.sfs, stats=self.stats,
                    category_id=unit.category_id)
        unit.selected = greedy_select(unit.pool, self.cfg.sfs.num_events,
                                      self.cfg.sfs.tioi_thresh)

    def _refresh_descriptions(self, unit: VideoUnit) -> None:
        """Re-read the description of every pool event from its new middle frame."""
        captions, text_features = self.provider.captions_for(
            unit.video_id, [event.mid for event in unit.pool])
        for index, event in enumerate(unit.pool):
            if index < len(captions) and captions[index]:
                event.caption = captions[index]
            if text_features is not None and index < len(text_features):
                event.text_feat = text_features[index]

    # ------------------------------------------------------------- Section 3.2
    def build_initial_pseudo_labels(self) -> None:
        """Round 0: SFS over the raw candidate pool gives ``L^(0)``."""
        self._log("[saip] Section 3.1-3.2: candidate pools and SFS selection")
        for unit in self.units:
            self._reselect(unit)
        self._log(f"[saip] L^(0): {len(self.units)} videos, "
                  f"{self.event_count()} pseudo-label events")

    # ------------------------------------------------------------- Section 3.4
    def calibrate(self) -> None:
        self.stats = CorpusStats(self.cfg.calibration)
        self.stats.fit(self.units)
        for unit in self.units:
            self._reselect(unit)

    # ------------------------------------------------------------- Section 3.3
    def _refine_round(self, index: int):
        """One application of ``Phi``; returns the round log or ``None``."""
        from .bcnet import (BoundaryCalibrationNet, RoundSnapshot,
                            convergence_metrics, has_converged,
                            relocalise_boundaries, train_boundary_net)

        snapshots = {u.video_id: RoundSnapshot.of(u.selected) for u in self.units}
        data = [(u.features, u.selected) for u in self.units if u.selected]
        if not data:
            return None

        import torch
        torch.manual_seed(self.cfg.bcnet.seed + index)
        model = BoundaryCalibrationNet(feat_dim=int(self.units[0].features.shape[1]),
                                       cfg=self.cfg.bcnet)
        losses = train_boundary_net(model, data, cfg=self.cfg.bcnet,
                                    device=self.cfg.features.device,
                                    verbose=False)

        import torch
        model.eval()
        with torch.no_grad():
            for unit in self.units:
                tensor = torch.from_numpy(unit.features.astype(np.float32))[None]
                probability = torch.sigmoid(
                    model(tensor.to(self.cfg.features.device))[0, :, 0]
                ).cpu().numpy()
                # Section 3.3.2(2): the boundaries of the candidate pool are
                # corrected, then the segment statistics are refreshed because
                # both f_bar of Eq. (3) and the variance of Eq. (8) are defined
                # over the segment.
                relocalise_boundaries(unit.pool, probability,
                                      delta=self.cfg.bcnet.delta,
                                      min_span=self._min_span_frames())
                for event in unit.pool:
                    event.recompute_statistics(unit.features)
                if self.cfg.bcnet.refresh_captions:
                    self._refresh_descriptions(unit)

        # Recompute SFS on the corrected pool and re-run the greedy rule.
        for unit in self.units:
            self._reselect(unit)
        self.stats.fit(self.units)
        for unit in self.units:
            self._reselect(unit)

        jaccard = 1.0
        displacement = 0.0
        for unit in self.units:
            j, d = convergence_metrics(snapshots[unit.video_id], unit.selected,
                                       mode=self.cfg.bcnet.jaccard_mode,
                                       tioi_thresh=self.cfg.sfs.tioi_thresh)
            jaccard = min(jaccard, j)
            displacement = max(displacement, d)
        stopped = has_converged(jaccard, displacement, self.cfg.bcnet)
        return RoundLog(round_index=index, loss=float(losses[-1]) if losses else 0.0,
                        jaccard=jaccard, displacement=displacement,
                        n_events=self.event_count(), stopped=stopped)

    def _min_span_frames(self) -> int:
        cfg = self.cfg.candidates
        return max(int(round(cfg.min_span_sec * cfg.fps)), 2)

    def refine(self) -> None:
        """Section 3.3: iterate until Eq. (17) holds."""
        if self.cfg.bcnet.max_iters <= 0:
            self.notes.append("iterative refinement disabled (max_iters = 0)")
            return
        if not _torch_available():
            self.notes.append(
                "PyTorch is not importable, so the boundary calibration network "
                "was skipped and the SFS-only solution L^(0) is reported. "
                "Install torch to run Section 3.3.")
            self._log("[saip] " + self.notes[-1])
            return

        self._log(f"[saip] Section 3.3: iterative refinement "
                  f"(max {self.cfg.bcnet.max_iters} rounds, "
                  f"delta_J = {self.cfg.bcnet.delta_jaccard}, "
                  f"delta_b = {self.cfg.bcnet.delta_boundary_frames} frames)")
        for index in range(1, self.cfg.bcnet.max_iters + 1):
            log = self._refine_round(index)
            if log is None:
                break
            self.history.append(log)
            self._log(f"[saip]   round {index}: loss={log.loss:.4f} "
                      f"J={log.jaccard:.4f} shift={log.displacement:.2f} frames"
                      f"{'  -> Eq. (17) satisfied, stop' if log.stopped else ''}")
            if log.stopped:
                break
        else:
            self.notes.append(
                f"Eq. (17) was not satisfied within {self.cfg.bcnet.max_iters} "
                "rounds; the last round is reported. Raise bcnet.max_iters if "
                "the iteration is expected to run longer.")

    # ------------------------------------------------------------------ driver
    def event_count(self) -> int:
        return sum(len(u.selected) for u in self.units)

    def run(self, manifest: Optional[str] = None) -> List[VideoUnit]:
        manifest = manifest or self.cfg.manifest
        if not manifest:
            raise ValueError("no manifest given (SAIPConfig.manifest or run(manifest=...))")

        records = load_manifest(manifest)
        self._log(f"[saip] manifest: {len(records)} videos")
        started = time.time()
        for record in records:
            unit = self._build_pool(record)
            if unit is not None:
                self.units.append(unit)
            self.provider.release(record["id"])
        if not self.units:
            raise RuntimeError("no video produced a usable candidate pool")
        self._log(f"[saip] candidate pools built in {time.time() - started:.1f}s "
                  f"({sum(len(u.pool) for u in self.units)} candidates)")

        self.build_initial_pseudo_labels()
        self.calibrate()
        self.refine()
        return self.units

    # ---------------------------------------------------------------- output
    def write_outputs(self, out_dir: Optional[str] = None) -> Dict[str, str]:
        cfg = self.cfg.output
        target = Path(out_dir or cfg.out_dir)
        paths = {"pseudo_labels": str(write_pseudo_labels(
            self.units, cfg.dataset, target, self.cfg.candidates.fps))}

        if cfg.write_event_scores:
            paths["event_scores"] = str(write_event_scores(
                self.units, target, self.cfg.candidates.fps))
        if cfg.write_calibration_report and self.stats is not None:
            payload = {
                "calibration": self.stats.report,
                "iteration": [log.as_dict() for log in self.history],
                "converged": bool(self.history and self.history[-1].stopped),
                "config": self.cfg.to_dict(),
                "notes": self.notes,
            }
            paths["report"] = str(write_json(payload, target / "saip_report.json"))
        return paths


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def run_pipeline(cfg: SAIPConfig, manifest: Optional[str] = None,
                 verbose: bool = True) -> SAIPPipeline:
    """Convenience wrapper: build, run and write the outputs."""
    pipeline = SAIPPipeline(cfg, verbose=verbose)
    pipeline.run(manifest=manifest)
    paths = pipeline.write_outputs()
    if verbose:
        for name, path in paths.items():
            print(f"[saip] wrote {name}: {path}")
    return pipeline
