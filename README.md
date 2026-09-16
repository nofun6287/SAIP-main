# SAIP: Summarization-Aware Iterative Pseudo-Labeling

Reference implementation of the pseudo-label generation pipeline of

> **Summarization-Aware Iterative Pseudo-Labeling for Dense Video Captioning**

SAIP turns *raw video with no human annotation* into a dense video captioning
training set. Each video is described by a set of temporal events, each event
carrying a natural-language sentence, in the same layout as the ActivityNet
Captions and Charades annotations that downstream models consume.

The repository covers Sections 3.1-3.4 of the paper, i.e. everything up to — and
including — the pseudo-label dataset. Training of the downstream models (PDVC,
Vid2Seq) is **not** part of this repository.

```
 raw video ──► BLIP features ──► over-generated candidate pool        §3.1
                                        │
                                        ▼
                        SFS scoring  +  greedy selection  ──► L(0)   §3.2
                                        │
                                        ▼
                cross-video statistical calibration                  §3.4
                                        │
                                        ▼
             boundary calibration network, re-localise, re-score     §3.3
             repeat until Eq. (17) is satisfied  ──► L*             §3.3
                                        │
                                        ▼
                    pseudo-label dataset (train_pseudo.json)         §3.4
```

---

## 1. What is in the box

| Path | Contents |
|---|---|
| `saip/candidates.py` | **Section 3.1** — Eq. (2) frame distance, peak detection, pairwise event proposals, pool size control |
| `saip/sfs.py` | **Section 3.2** — Eqs. (4)-(10), the four SFS dimensions, and the greedy selection of Section 3.2.5 |
| `saip/bcnet.py` | **Section 3.3** — Eqs. (11)-(14) network, Eqs. (15)-(16) losses, re-localisation, Eq. (17) stopping rule |
| `saip/calibration.py` | **Section 3.4** — category calibration, `P(type \| category)` and the core/background event analysis |
| `saip/pipeline.py` | the full loop, `L(0) → L(1) → … → L*` |
| `saip/features.py` | BLIP adapters and video decoding; the interface that produces `h_t`, `c_t` and `q_i` |
| `saip/dataset_io.py` | manifest reader and the two pseudo-label dataset writers |
| `saip/synthetic.py` | **test-only** synthetic corpus generator (see the warning in the module docstring) |
| `docs/paper_mapping.md` | equation-by-equation map from the paper to this code, including every implementation choice and every place where the paper is ambiguous |
| `configs/` | ready-to-edit YAML configurations |
| `tests/` | unit tests plus an end-to-end CPU smoke test |

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Only `numpy`, `scikit-learn` and `PyYAML` are needed to run Sections 3.1, 3.2 and
3.4. `torch` is needed for the boundary calibration network of Section 3.3; if it
is missing the pipeline still runs and reports the SFS-only labels `L(0)`, with a
note saying so. `ffmpeg-python` and the BLIP sources are needed only if you
extract features in-process (the `blip` backend) instead of reading them from disk.

## 3. Quick start — no downloads, no GPU

```bash
python scripts/make_demo_data.py --out demo_data --videos 8

python -m saip \
    --manifest demo_data/manifest.json \
    --feat-dir demo_data/feats \
    --caption-dir demo_data/captions \
    --text-feat-dir demo_data/text_feats \
    --out-dir demo_data/out \
    --device cpu
```

This writes `demo_data/out/train_pseudo.json`, `pseudo_label_scores.json` and
`saip_report.json`. The demo corpus is synthetic and is there to prove that the
code path runs, not to produce numbers.

## 4. Running on real data

### 4.1 The three cached arrays (recommended)

Feature extraction is a one-off GPU job, so the pipeline reads its inputs from
disk.  For every video id, provide:

```
feats/<video_id>.npy         float32 (T, 768)   h_t, Eq. (1), one frame every `stride` of the video
captions/<video_id>.json     list[str], length T   c_t, the description of each frame
text_feats/<video_id>.npy    float32 (T, 768)   q_i of Eq. (10), the BLIP text embedding of c_t
```

`captions/<video_id>.json` may also hold a list of lists (one per captioning
sample); the first sample of each frame is used.

The arrays come from the two frozen BLIP models:

| Quantity | Model |
|---|---|
| `h_t`, `q_i` | BLIP retrieval, `blip-itm-base-coco` (ViT-B, 384×384) |
| `c_t` | BLIP captioning, `blip-caption-large-coco` |

The weights are **not** distributed here.  Point `FeatureConfig.itm_ckpt` and
`FeatureConfig.caption_ckpt` at your local copies and use `--backend blip` to run
the models in-process, or extract them with your own script and use the default
`cache` backend.

### 4.2 Manifest

```json
{"videos": [
  {"id": "v_abc123",
   "duration": 185.2,
   "category": "Sports",
   "path": "/data/activitynet/videos/v_abc123.mp4"}
]}
```

`duration` and `category` are optional but recommended: Section 3.4(1)
normalises `S_density` per category and the ActivityNet writer uses the duration
to normalise timestamps.  Without a `category` the video-level mean features are
clustered instead, as Section 3.2.3 describes.

### 4.3 Run

```bash
python -m saip --config configs/activitynet.yaml \
    --manifest /data/activitynet/manifest_train.json \
    --out-dir runs/activitynet
```

Any single value can be overridden without editing the file:

```bash
python -m saip --config configs/activitynet.yaml \
    --num-events 20 --max-iters 10 --set candidates.fps=2.0
```

`--set` takes a `section.field=value` pair whose section and field names are the
ones used in the YAML file (`candidates`, `sfs`, `bcnet`, `calibration`,
`features`, `output`).

Charades uses the same code with `--dataset charades`, which writes
`charades_sta_train_pseudo.txt` (`<video_id> <start> <end>##<description>`).

## 5. Output

| File | Contents |
|---|---|
| `train_pseudo.json` | `{video_id: {duration, timestamps: [[s, e], …] normalised to [0, 1], sentences: [...]}}`, chronological |
| `charades_sta_train_pseudo.txt` | one `vid start end##description` line per event |
| `pseudo_label_scores.json` | the four raw SFS dimensions and the SFS of every emitted event — this is what the ablation of Section 4.6 switches |
| `saip_report.json` | the round-by-round history of Eq. (17), the Section 3.4(2)-(3) statistics, the resolved configuration, and any notes (for example, that torch was unavailable) |

## 6. Key hyper-parameters

All of them live in `saip/config.py` and in `configs/*.yaml`, with the paper's
value as the default.

| Setting | Default | Source |
|---|---|---|
| `candidates.fps` (γ) | 3.0 | Section 3.1, "keyframes at 3 fps" |
| `candidates.min_span_sec` | 2.0 | Section 3.1, "no less than 2 seconds (i.e. 6 frames)" |
| `candidates.pool_min` / `pool_max` (M) | 20 / 80 | Section 3.1 |
| `sfs.weights` | `0.25` each | Eq. (4) |
| `sfs.phi_near` / `phi_far` / `phi_floor` | 0.05 / 0.25 / 0.3 | Eq. (7) |
| `sfs.tioi_thresh` | 0.5 | Section 3.2.5 |
| `sfs.num_events` (K) | 10 | see `docs/paper_mapping.md` |
| `bcnet.d_attn` / `n_heads` / `conv_kernel` / `hidden` | 256 / 4 / 3 / 128 | Eqs. (11)-(14) |
| `bcnet.lambda_diou` | 1.0 | `L_total = L_BCE + L_DIoU` |
| `bcnet.delta` (Δ) | 4 frames | Eq. (16), Section 3.3.2(2) |
| `bcnet.delta_jaccard` / `delta_boundary_frames` | 0.98 / 0.5 | Eq. (17) |
| `bcnet.max_iters` | 8 | ceiling above the T = 6 of Table 3 |

## 7. Ablations

The rows of the ablation table are configuration switches:

```bash
python -m saip --config configs/activitynet.yaml --ablate uniq     # w/o S_uniq
python -m saip --config configs/activitynet.yaml --ablate density  # w/o S_density
python -m saip --config configs/activitynet.yaml --ablate narr     # w/o S_narr
python -m saip --config configs/activitynet.yaml --ablate conf     # w/o S_conf
```

Zeroing one weight and renormalising the rest is the direct implementation of
"remove this dimension from Eq. (4)".  The `S_conf`-only configuration, which the
paper identifies with SPL, is `--set sfs.weights=0,0,0,1`.

## 8. Tests

```bash
pytest -q                        # 16 tests, CPU only, a few seconds
python tests/test_pipeline.py    # same checks without pytest
```

The suite covers Eq. (2), the pool bounds and the Eq. (3) mean feature, Eq. (7)
at both knots, the range of the SFS dimensions, the greedy rule, the Eq. (15)
labels, the differentiability of the Eq. (16) loss, the Eq. (17) stopping rule,
the document-frequency form of `P(type | category)`, the loading of the shipped
YAML configurations, and a full run of the pipeline on a synthetic corpus.

## 9. Citation

```bibtex
@article{saip,
  title  = {Summarization-Aware Iterative Pseudo-Labeling for Dense Video Captioning},
  author = {...},
  year   = {...}
}
```

## 10. License

MIT — see [LICENSE](LICENSE).
