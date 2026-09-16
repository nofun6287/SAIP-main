"""Unit tests and an end-to-end smoke test.

    pytest -q                       # all tests
    python tests/test_pipeline.py   # end-to-end only, no pytest needed

Everything runs on CPU with synthetic data; no BLIP checkpoint and no video file
is required.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saip.bcnet import (BoundaryCalibrationNet, RoundSnapshot,  # noqa: E402
                        build_boundary_labels, convergence_metrics, diou_loss,
                        has_converged, relocalise_boundaries)
from saip.calibration import CorpusStats                            # noqa: E402
from saip.candidates import build_candidates, frame_distance        # noqa: E402
from saip.config import SAIPConfig                                  # noqa: E402
from saip.sfs import (compute_sfs, greedy_select, score_s_conf,     # noqa: E402
                      score_s_uniq, temporal_iou, temporal_modulation)
from saip.synthetic import generate_video, write_synthetic_corpus   # noqa: E402


# ---------------------------------------------------------------------------
# Section 3.1
# ---------------------------------------------------------------------------
def test_frame_distance_matches_eq2():
    """Eq. (2): identical directions give 0, opposite directions give 2."""
    h = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
                 dtype=np.float32)
    d = frame_distance(h)
    assert d.shape == (3,)
    assert abs(float(d[0]) - 0.0) < 1e-6
    assert abs(float(d[1]) - 2.0) < 1e-6
    assert abs(float(d[2]) - 1.0) < 1e-6


def test_candidate_pool_respects_bounds_and_spans():
    features, _, _ = generate_video("unit-video", dim=64, n_segments=5,
                                    frames_per_segment=20)
    cfg = SAIPConfig()
    pool, meta = build_candidates(features, cfg.candidates)
    assert pool, "candidate pool should not be empty"
    assert cfg.candidates.pool_min <= len(pool) <= cfg.candidates.pool_max
    min_span = round(cfg.candidates.min_span_sec * cfg.candidates.fps)
    for event in pool:
        assert 0 <= event.s <= event.e < features.shape[0]
        assert event.span >= min_span
        assert event.mid == (event.s + event.e) // 2
        # Eq. (3): the mean feature really is the mean of the segment.
        assert np.allclose(event.fbar, features[event.s:event.e + 1].mean(axis=0),
                           atol=1e-5)
        # Eq. (8) numerator: (1/n) * sum_t ||h_t - f_bar||^2, summed over dims.
        expected = np.mean(np.sum((features[event.s:event.e + 1] - event.fbar) ** 2,
                                  axis=1))
        assert abs(event.var - float(expected)) < 1e-3


def test_candidate_uids_are_unique():
    features, _, _ = generate_video("uid-video", dim=32, n_segments=4,
                                    frames_per_segment=16)
    pool, _ = build_candidates(features, SAIPConfig().candidates)
    assert len({e.uid for e in pool}) == len(pool)


# ---------------------------------------------------------------------------
# Section 3.2
# ---------------------------------------------------------------------------
def test_phi_matches_eq7():
    """Eq. (7) is continuous at both knots and flat outside them."""
    assert temporal_modulation(0.0) == 1.0
    assert abs(temporal_modulation(0.05) - 1.0) < 1e-9
    assert abs(temporal_modulation(0.25) - 0.3) < 1e-9
    assert abs(temporal_modulation(0.15) - (1.175 - 3.5 * 0.15)) < 1e-9
    assert temporal_modulation(1.0) == 0.3
    xs = np.linspace(0, 0.4, 400)
    ys = [temporal_modulation(float(x)) for x in xs]
    assert max(abs(np.diff(ys))) < 0.02, "phi should have no jump"


def test_sfs_is_bounded_and_greedy_returns_k():
    features, captions, text_features = generate_video(
        "sfs-video", dim=64, n_segments=5, frames_per_segment=20)
    cfg = SAIPConfig()
    pool, _ = build_candidates(features, cfg.candidates)
    for index, event in enumerate(pool):
        event.caption = captions[event.mid]
        event.text_feat = text_features[event.mid]

    scores = compute_sfs(pool, features, features.shape[0],
                         fps=cfg.candidates.fps, cfg=cfg.sfs)
    assert np.all(np.isfinite(scores))
    assert scores.min() >= -1e-6 and scores.max() <= 1.0 + 1e-6
    for event in pool:
        assert 0.0 <= event.s_uniq <= 1.0
        assert event.s_uniq_n == 0.0 or 0.0 <= event.s_uniq_n <= 1.0

    selected = greedy_select(pool, k=6, tioi_thresh=cfg.sfs.tioi_thresh)
    assert len(selected) == 6
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            assert temporal_iou(selected[i].s, selected[i].e,
                                selected[j].s, selected[j].e) <= cfg.sfs.tioi_thresh
    # Greedy keeps the best available score at every step.
    assert selected[0].sfs >= max(e.sfs for e in pool) - 1e-9


def test_s_uniq_and_s_conf_follow_the_definitions():
    """Eq. (5): duplicated events must be penalised; Eq. (10): a matching
    description must beat a mismatching one."""
    features, captions, text_features = generate_video(
        "def-video", dim=64, n_segments=4, frames_per_segment=20)
    cfg = SAIPConfig()
    pool, _ = build_candidates(features, cfg.candidates)
    for index, event in enumerate(pool):
        event.caption = captions[event.mid]
        event.text_feat = text_features[event.mid]

    uniq = score_s_uniq(pool, features.shape[0], cfg.sfs)
    assert np.all(uniq >= 0) and np.all(uniq <= 1)

    # A description taken from the middle of its own event aligns better inside
    # than outside, so Eq. (10) must be positive for most candidates.
    conf = score_s_conf(pool, features)
    assert np.mean(conf > 0) > 0.6

    # Replacing every description with one from an unrelated segment removes the
    # main-lobe advantage.
    unrelated = pool[0].text_feat
    for event in pool:
        event.text_feat = unrelated
    shifted = score_s_conf(pool, features)
    assert np.mean(shifted > 0) <= np.mean(conf > 0)


# ---------------------------------------------------------------------------
# Section 3.3
# ---------------------------------------------------------------------------
def test_boundary_labels_are_the_event_endpoints():
    features, _, _ = generate_video("label-video", dim=32, n_segments=3,
                                    frames_per_segment=12)
    pool, _ = build_candidates(features, SAIPConfig().candidates)
    events = pool[:3]
    length = features.shape[0]
    y = build_boundary_labels(events, length)
    positives = np.flatnonzero(y)
    assert sorted(positives.tolist()) == sorted(
        {e.s for e in events} | {e.e for e in events})
    assert set(np.unique(y)).issubset({0.0, 1.0})


def test_diou_loss_is_differentiable_and_finite():
    import torch
    features, _, _ = generate_video("diou-video", dim=32, n_segments=3,
                                    frames_per_segment=12)
    pool, _ = build_candidates(features, SAIPConfig().candidates)
    events = pool[:3]
    logits = torch.zeros(1, features.shape[0], 1, requires_grad=True)
    loss = diou_loss(logits, [events], delta=4, lengths=[features.shape[0]])
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_relocalise_moves_boundaries_and_reports_shift():
    features, _, _ = generate_video("reloc-video", dim=32, n_segments=4,
                                    frames_per_segment=16)
    pool, _ = build_candidates(features, SAIPConfig().candidates)
    event = pool[0]
    before = (event.s, event.e)
    probability = np.zeros(features.shape[0])
    probability[min(before[0] + 2, features.shape[0] - 1)] = 1.0
    probability[min(before[1] - 2, features.shape[0] - 1)] = 1.0
    shift = relocalise_boundaries([event], probability, delta=4, min_span=2)
    assert shift > 0
    assert (event.s, event.e) != before
    assert event.mid == (event.s + event.e) // 2


def test_eq17_convergence_uses_both_conditions():
    features, _, _ = generate_video("conv-video", dim=32, n_segments=3,
                                    frames_per_segment=12)
    pool, _ = build_candidates(features, SAIPConfig().candidates)
    selected = greedy_select(pool, 4)
    snapshot = RoundSnapshot.of(selected)

    jaccard, displacement = convergence_metrics(snapshot, selected)
    assert jaccard == 1.0 and displacement == 0.0
    assert has_converged(jaccard, displacement)

    # Nudge every start by two frames: the sets still match, but each event now
    # contributes (|ds| + |de|) / 2 = 1 frame, above the 0.5-frame bound of
    # Eq. (17), so the second condition fails.
    for event in selected:
        event.s += 2
    jaccard, displacement = convergence_metrics(snapshot, selected)
    assert jaccard == 1.0
    assert abs(displacement - 1.0) < 1e-9
    assert not has_converged(jaccard, displacement)

    # Matching by identity instead of by temporal overlap: a selected event that
    # was not in the previous round lowers the Jaccard similarity.
    selected[0].uid = max(e.uid for e in selected) + 1
    identity_j, _ = convergence_metrics(snapshot, selected, mode="identity")
    assert identity_j < 1.0
    assert has_converged(1.0, 0.0)


def test_boundary_net_forward_and_training_step():
    import torch
    features, _, _ = generate_video("net-video", dim=48, n_segments=3,
                                    frames_per_segment=12)
    cfg = SAIPConfig()
    pool, _ = build_candidates(features, cfg.candidates)
    model = BoundaryCalibrationNet(feat_dim=48, cfg=cfg.bcnet)
    out = model(torch.from_numpy(features)[None])
    assert out.shape == (1, features.shape[0], 1)

    from saip.bcnet import train_boundary_net
    history = train_boundary_net(model, [(features, pool[:4])], cfg=cfg.bcnet,
                                 device="cpu", verbose=False)
    assert len(history) == cfg.bcnet.epochs
    assert all(math.isfinite(value) for value in history)


# ---------------------------------------------------------------------------
# Section 3.4
# ---------------------------------------------------------------------------
def test_p_type_given_category_is_a_video_frequency():
    """Eq. (9)'s prior is the fraction of the class's videos containing the
    type, so several types can be high at once and no row sums to one."""
    stats = CorpusStats()
    stats.n_event_types = 3
    stats.n_categories = 1

    class _Event:
        def __init__(self, type_id):
            self.type_id = type_id

    class _Video:
        def __init__(self, category_id, types):
            self.category_id = category_id
            self.selected = [_Event(t) for t in types]

    videos = [_Video(0, [0, 1]), _Video(0, [0]), _Video(0, [0, 2]), _Video(0, [0])]
    stats.fit_priors(videos)
    table = stats.p_type_given_category_table
    assert table.shape == (1, 3)
    # Type 0 occurs in 4 of 4 videos -> (4 + 0.5) / (4 + 1.5), i.e. near 1 and
    # far above the 4 / 7 = 0.57 a normalised distribution over types would give.
    assert table[0, 0] > 0.8
    assert table[0, 1] < table[0, 0]
    assert not np.isclose(table.sum(), 1.0)


def test_density_calibration_divides_by_the_category_mean():
    stats = CorpusStats()
    stats.category_mean_density = np.array([2.0, 4.0], dtype=np.float32)
    out = stats.calibrate_density(np.array([4.0, 4.0], dtype=np.float32), 0)
    assert np.allclose(out, [2.0, 2.0])


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_pipeline_end_to_end(tmp_path=None):
    if tmp_path is not None:
        _run_end_to_end(Path(tmp_path))
        return
    # Without pytest there is no tmp_path fixture, so make (and remove) one.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="saip_test_") as tmp:
        _run_end_to_end(Path(tmp))


def _run_end_to_end(root: Path):
    manifest = write_synthetic_corpus(root, n_videos=4, dim=64, n_segments=4,
                                      frames_per_segment=20)
    out_dir = root / "out"

    cfg = SAIPConfig()
    cfg.manifest = str(manifest)
    cfg.candidates.pool_min, cfg.candidates.pool_max = 20, 60
    cfg.sfs.num_events = 6
    cfg.bcnet.max_iters = 2
    cfg.bcnet.epochs = 1
    cfg.features.backend = "cache"
    cfg.features.feat_dir = str(root / "feats")
    cfg.features.caption_dir = str(root / "captions")
    cfg.features.text_feat_dir = str(root / "text_feats")
    cfg.features.device = "cpu"
    cfg.output.out_dir = str(out_dir)

    from saip.pipeline import run_pipeline
    pipeline = run_pipeline(cfg, verbose=False)

    assert len(pipeline.units) == 4
    assert pipeline.event_count() == 4 * 6
    assert len(pipeline.history) >= 1

    payload = json.loads((out_dir / "train_pseudo.json").read_text())
    assert len(payload) == 4
    for record in payload.values():
        assert len(record["timestamps"]) == 6
        assert len(record["sentences"]) == 6
        starts = [span[0] for span in record["timestamps"]]
        assert starts == sorted(starts), "events must be written in time order"
        for start, end in record["timestamps"]:
            assert 0.0 <= start < end <= 1.0

    report = json.loads((out_dir / "saip_report.json").read_text())
    assert report["calibration"]["n_categories"] >= 1
    assert isinstance(report["converged"], bool)
    assert (out_dir / "pseudo_label_scores.json").is_file()


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def test_config_files_build_nested_sections():
    """A YAML configuration must produce configuration objects, not dicts.

    Regression test: the dataclass annotations are strings under
    ``from __future__ import annotations``, so a naive ``field.type`` lookup in
    ``SAIPConfig.from_dict`` leaves every nested section as a plain dict, and
    the first ``--config`` run then fails on ``'dict' object has no attribute
    'feat_dir'``.
    """
    from saip.config import (BCNetConfig, CalibrationConfig, CandidateConfig,
                             FeatureConfig, OutputConfig, SFSConfig)

    expected = {"candidates": CandidateConfig, "sfs": SFSConfig,
                "bcnet": BCNetConfig, "calibration": CalibrationConfig,
                "features": FeatureConfig, "output": OutputConfig}
    root = Path(__file__).resolve().parents[1]

    for name in ("activitynet", "charades"):
        cfg = SAIPConfig.from_file(root / "configs" / f"{name}.yaml")
        for section, klass in expected.items():
            assert isinstance(getattr(cfg, section), klass), (name, section)
        # YAML has no tuple type; the tuple fields must come back as tuples.
        assert isinstance(cfg.sfs.weights, tuple)
        assert len(cfg.sfs.weights) == 4
        assert isinstance(cfg.candidates.peak_thresh_factors, tuple)
        assert isinstance(cfg.calibration.type_k_range, tuple)
        assert cfg.output.dataset == name

    activitynet = SAIPConfig.from_file(root / "configs" / "activitynet.yaml")
    assert activitynet.features.backend == "cache"
    assert activitynet.features.feat_dir == "feats"
    assert activitynet.sfs.num_events == 10
    # Unknown keys are ignored rather than raising.
    assert SAIPConfig.from_dict({"nonsense": 1, "sfs": {"unknown": 2}}).sfs.weights \
        == (0.25, 0.25, 0.25, 0.25)

    # The CLI has to be able to override a field of a loaded file.
    from saip.cli import apply_override
    apply_override(activitynet, "features.feat_dir=/tmp/feats")
    apply_override(activitynet, "sfs.weights=0,0,0,1")
    assert activitynet.features.feat_dir == "/tmp/feats"
    assert activitynet.sfs.weights == (0.0, 0.0, 0.0, 1.0)


def test_ablations_zero_and_renormalise():
    cfg = SAIPConfig()
    cfg.ablate_dimensions = ["uniq", "conf"]
    cfg.apply_ablations()
    assert cfg.sfs.weights == (0.0, 0.5, 0.5, 0.0)
    assert SAIPConfig().apply_ablations().sfs.weights == (0.25, 0.25, 0.25, 0.25)


if __name__ == "__main__":
    test_frame_distance_matches_eq2()
    test_candidate_pool_respects_bounds_and_spans()
    test_candidate_uids_are_unique()
    test_phi_matches_eq7()
    test_sfs_is_bounded_and_greedy_returns_k()
    test_s_uniq_and_s_conf_follow_the_definitions()
    test_boundary_labels_are_the_event_endpoints()
    test_diou_loss_is_differentiable_and_finite()
    test_relocalise_moves_boundaries_and_reports_shift()
    test_eq17_convergence_uses_both_conditions()
    test_boundary_net_forward_and_training_step()
    test_p_type_given_category_is_a_video_frequency()
    test_density_calibration_divides_by_the_category_mean()
    test_pipeline_end_to_end()
    test_config_files_build_nested_sections()
    test_ablations_zero_and_renormalise()
    print("all checks passed")
