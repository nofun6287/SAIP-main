# Paper to code, equation by equation

This document is the contract between the paper and this repository.  For every
equation it names the symbol in the code, and for every place where the paper
leaves something open it records which choice was made and why.  Anything marked
**IMPLEMENTATION CHOICE** is not derivable from the paper and should be stated
as such if it is ever described in writing.

Notation: frame indices are inclusive on both ends, so an event `[s, e]` covers
`e - s + 1` frames.  `T` is the number of sampled frames of the **video**, and
`γ` (`candidates.fps`) the sampling rate.

---

## Section 3.1 — over-generation of candidate events

| Paper | Code | Note |
|---|---|---|
| \(h_t = \mathrm{BLIP}_{enc}(f_t) \in \mathbb{R}^{768}\) (1) | `saip.features.BlipModels.frame_features` | ViT-B retrieval model, 384×384, features taken from `visual_encoder` + `vision_proj` |
| \(c_t\), description of a frame | `saip.features.BlipModels.frame_captions` | captioning model, `sample=True, max_length=20, min_length=5` |
| \(d_t = 1 - \cos(h_t, h_{t+1})\) (2) | `saip.candidates.frame_distance` | |
| peak detection on \(d_t\) | `saip.candidates.detect_boundaries` | **IMPLEMENTATION CHOICE**: local maximum of the 3-frame moving average, above `mean + k·std`, then non-maximum suppression at the 2 s spacing; `k = 0.8` relaxed to `0.4` and `0.2` until the pool reaches `M_min` |
| every pair \((b_i, b_j)\), \(b_j > b_i\), span ≥ 2 s | `saip.candidates.pairwise_spans` | **IMPLEMENTATION CHOICE**: frame 0 and frame `T-1` are added as sentinel boundaries, since the start and the end of a video are boundaries by construction |
| \(\bar f = \frac{1}{e-s}\sum_{t=s}^{e} h_t\) (3) | `CandidateEvent.fbar`, `CandidateEvent.recompute_statistics` | the sum runs over `e - s + 1` frames while the printed denominator is `e - s`; the code uses the **exact mean of the closed interval** |
| \(M = 20 \sim 80\) | `CandidateConfig.pool_min/pool_max` | **IMPLEMENTATION CHOICE**: an over-large pool is trimmed by descending segment variance (the \(S_{density}\) proxy), an under-full one is padded with evenly spaced spans |
| \(M\) is "roughly 2-8 times the target \(K\)" | `SFSConfig.num_events = 10` | 20/10 = 2 and 80/10 = 8, so the sentence fixes **K = 10**. The paper never prints K explicitly |

## Section 3.2 — SFS scoring and event selection

| Paper | Code |
|---|---|
| \(\mathrm{SFS} = 0.25 S_{uniq} + 0.25 S_{density} + 0.25 S_{narr} + 0.25 S_{conf}\) (4) | `saip.sfs.compute_sfs` |
| \(S_{uniq} = 1 - \max_{j \ne i} \cos(\bar f_i, \bar f_j)\) (5) | `saip.sfs.score_s_uniq` |
| \(\overline{S_{uniq}} = 1 - \max_{j \ne i}[\cos(\bar f_i,\bar f_j)\varphi(\Delta t_{ij})]\) (6) | `saip.sfs.score_s_uniq` |
| \(\varphi\) piecewise linear (7) | `saip.sfs.temporal_modulation` / `_phi_array` |
| \(\Delta t_{ij} = \lvert \mathrm{mid}(m_i) - \mathrm{mid}(m_j)\rvert / T\) | `score_s_uniq(..., video_len)` — `video_len` is `T` of the video, **not** the largest midpoint of the pool |
| \(S_{density} = \frac{\frac1n \sum_t \lVert h_t - \bar h_i\rVert^2}{(e_i - s_i)/\gamma}\) (8) | `saip.sfs.score_s_density` |
| the numerator is a squared **norm** | `CandidateEvent.var` sums over the 768 dimensions; a plain `.mean()` over the array would silently divide by 768 |
| the denominator is printed as \((e_i-s_i)/\gamma\) | kept as printed; `SFSConfig.density_closed_span` switches to the count that matches the numerator, \((e_i-s_i+1)/\gamma\) |
| \(S_{narr} = P(\mathrm{type}(m_i)\mid\mathrm{category}(V)) \cdot (1 - \lvert \frac{\mathrm{mid}}{T} - 0.5\rvert)\) (9) | `saip.sfs.score_s_narr` |
| "mid(m_i) is the midpoint of the event and T the total duration of the video" | `video_len` again — using the pool maximum instead would make the position factor depend on how the pool was trimmed |
| \(P(\mathrm{type}\mid\mathrm{category})\) | `saip.calibration.CorpusStats.fit_priors`; the paper's worked example ("chopping occurs in about 75% of videos, P = 0.75") makes this the **fraction of the videos of the category that contain the type**, Laplace-smoothed, not a distribution over types |
| "K-means on the text embeddings … number of clusters by the silhouette coefficient" | `fit_event_types`, `_auto_k`; search range is `type_k_range`, and the smallest k within `silhouette_tolerance` of the best wins so that a flat plateau does not select the largest k |
| \(S_{conf}\) = mean inside \(- \) mean outside (10) | `saip.sfs.score_s_conf`; the two means are divided by the number of frames actually summed on each side, which is what makes the printed denominators \((e_i-s_i)\) and \(T-(e_i-s_i)\) correct |
| "the values of \(S_{conf}\) are normalised to the interval \([0,1]\)" | the normalisation step below |
| greedy: take the highest SFS, drop everything with tIoU > 0.5 (3.2.5) | `saip.sfs.greedy_select`, `temporal_iou` |

**On the normalisation step.**  Eq. (4) adds the four dimensions directly, so they
have to share a scale: \(S_{uniq} \in [0,1]\) by Eq. (5), the position factor of
Eq. (9) is stated to lie in \([0.5, 1]\), \(S_{conf}\) is stated to be normalised
to \([0,1]\), and \(S_{density}\) is calibrated across videos by Section 3.4(1).
The concrete mapping used here is a **min-max rescaling over the candidate pool of
the current video** (`SFSConfig.normalize = "minmax"`); `"none"` uses the raw
values.  Min-max is per video, so it is unaffected by how many videos are in the
corpus.  The paper does not name this operation; a reader implementing Eq. (4)
literally from the raw quantities would get a different ranking, because the
per-frame variance of Eq. (8) is orders of magnitude larger than the other three
terms and \(S_{conf}\) can be negative.

## Section 3.3 — iterative refinement

| Paper | Code |
|---|---|
| \(\mathrm{Conv1D}_1\) + GELU + LayerNorm (11), \(\mathrm{Conv1D}_2\) (12), kernel 3, stride 1 | `saip.bcnet.BoundaryCalibrationNet` (`conv1`, `conv2`, `norm1`, `norm2`) |
| \(H^{(3)} = \mathrm{LayerNorm}(H^{(2)} + \mathrm{MHSA}(H^{(2)}))\) (13), \(h = 4\), \(d_{attn} = 256\) | `proj` maps the 768-d conv output into the 256-d attention subspace first; Eq. (13) adds the attention output back onto \(H^{(2)}\), which is only well defined once both are 256-d |
| \(b = \sigma(W_2 \cdot \mathrm{GELU}(W_1 H^{(3)} + c_1) + c_2)\) (14), \(W_1 \in \mathbb{R}^{256 \times 128}\) | `head` |
| \(\mathcal{L}_{BCE} = -\frac1T \sum [y \log b + (1-y)\log(1-b)]\) (15) | `saip.bcnet.bce_loss` with `bce_pos_weight = "none"`, i.e. **unweighted**, exactly as printed |
| "the start and end frame positions of every event are positive samples" | `build_boundary_labels` |
| \(\mathcal{L}_{DIoU} = \frac1K \sum_k [1 - \mathrm{IoU} + \rho^2/c^2]\) (16) | `saip.bcnet.diou_loss`, one term per event, divided by the number of events |
| "(s\*_k, e\*_k) is the position with the highest confidence within [s_k − Δ, e_k + Δ]" | read as the neighbourhood of **each** boundary separately, which is the rule Section 3.3.2(2) states in words: `[s_k − Δ, s_k + Δ]` for the start and `[e_k − Δ, e_k + Δ]` for the end |
| "the position with the highest confidence" | at training time the position is read out with a temperature-scaled **soft-argmax** so that the loss is differentiable in the network output; at inference, `relocalise_boundaries` uses the plain arg-max of the same window with `bcnet.delta` |
| \(\rho^2 / c^2\) | in one dimension \(\rho\) is the centre distance and \(c\) the length of the smallest enclosing interval |
| \(\mathcal{L}_{total} = \mathcal{L}_{BCE} + \mathcal{L}_{DIoU}\) | `lambda_diou = 1.0` |
| (2) "the boundaries of the candidate events in the candidate pool are corrected" | the **whole pool** is re-localised, not only the events currently selected; the segment statistics \(f_{bar}\) and the variance are then recomputed, because both are defined over the segment |
| "the SFS scores of the candidate events are then updated … forming \(\mathcal{L}^{(t+1)}\)" | SFS recomputed on the corrected pool, then the greedy rule of Section 3.2.5 re-run |
| \(J(\mathcal{L}^{(t)}, \mathcal{L}^{(t+1)}) > 0.98\) and \(\overline{\Delta b}^{(t)} < 0.5\) frames (17) | `convergence_metrics`, `has_converged` |
| \(J = \lvert A \cap B\rvert / \lvert A \cup B\rvert\) | two readings are supported.  `jaccard_mode = "tioi"` (default) matches an event of one round to an event of the other when their tIoU reaches `tioi_thresh`, which is the usual Jaccard similarity of two sets of temporal intervals; `"identity"` requires the same proposal to be re-selected, which makes J a binary "did the selection freeze" flag.  With the default K = 10 the threshold 0.98 is only met when all ten events match either way |
| "averaged over the events retained in both rounds and over the start and end time stamps" | each matched pair contributes \((\lvert \Delta s\rvert + \lvert \Delta e\rvert)/2\) |
| the previous round has to be compared by value | `RoundSnapshot` copies `(s, e)` out; the events are mutated in place by the re-localisation, so keeping references would compare an object with itself and always report zero displacement |

## Section 3.4 — cross-video statistical calibration and output

| Paper | Code |
|---|---|
| (1) per-category mean density, used to normalise \(S_{density}\) | `CorpusStats.fit_density_means`, `calibrate_density`, applied inside `compute_sfs` |
| (2) occurrence frequency and mean SFS per event type, core vs background | `CorpusStats.describe`; **IMPLEMENTATION CHOICE**: the paper gives no cut-off, so a type is `core` when both quantities are at or above the within-class quantiles (`core_freq_quantile`, `core_sfs_quantile`), `background` when both are below, and `mixed` otherwise.  Every raw number is written to `saip_report.json` so the labelling can be redone by hand |
| (3) mean number of pseudo-label events per category, "to verify the reasonableness of the choice of K" | `describe` → `categories[].mean_events_per_video` |
| "the final pseudo-label dataset \(\mathcal{L}^*\)" | `saip.dataset_io.write_pseudo_labels` |
| timestamps normalised to \([0,1]\) (ActivityNet layout) | `build_activitynet_records` divides by the recorded duration, which keeps an event ending at the last frame inside the interval even when the duration and the sampled frame count disagree |
| Charades layout | `build_charades_lines` |

## Values the paper prints that the code had to interpret

1. **K = 10.**  Section 3.1 says the pool of 20-80 candidates is "roughly 2-8
   times the target \(K\)", which fixes \(K = 10\); the paper never states K
   directly.  Set `sfs.num_events` if your run used something else — it changes
   the size of every pseudo-label set and hence the density of the output.
2. **T = 6 rounds.**  Table 3 reports convergence after six rounds, so
   `bcnet.max_iters` defaults to 8 and Eq. (17) stops the loop earlier.
   If a run reaches the ceiling, `saip_report.json` says so instead of pretending
   to have converged.
3. **The refinement threshold \(\delta_b = 0.5\) frames.**  The displacement is
   measured between consecutive rounds, so a run that reaches the ceiling has
   simply not stopped moving yet.
