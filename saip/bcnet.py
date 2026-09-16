"""Section 3.3 -- temporal boundary calibration network and iterative refinement.

Architecture (Eqs. 11-14)
------------------------
Two 1-D convolutions (kernel 3, stride 1), each followed by GELU and layer
normalisation, then one multi-head self-attention layer with ``h = 4`` heads and
``d_attn = 256`` over the whole frame sequence, with a residual connection, and
finally a two-layer head ``256 -> 128 -> 1`` whose output is squashed by a
sigmoid into a per-frame boundary confidence ``b_t``.

Training (Eqs. 15-16)
---------------------
``L_total = L_BCE + lambda * L_DIoU``.  The paper writes the sum without a
coefficient, so ``lambda = 1`` is the default.  Eq. (15) is an unweighted binary
cross-entropy over the T frames of the video; Eq. (16) averages one DIoU term per
event, over the whole predicted box ``(s*, e*)``.

Refinement (Section 3.3.2)
--------------------------
Boundaries of the **candidate events of the pool** are re-localised by taking the
highest-confidence frame inside a window of radius ``Delta`` around each original
boundary; the SFS scores are then recomputed on the corrected pool and the greedy
rule of Section 3.2.5 forms the next pseudo-label set.

Stopping rule (Eq. 17)
----------------------
``J(L^(t), L^(t+1)) > 0.98`` and the mean boundary displacement below 0.5 frames.

Importing this module requires PyTorch.  Everything else in the package runs
without it, so the pipeline degrades to the SFS-only solution ``L^(0)`` when
torch is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .candidates import CandidateEvent
from .config import BCNetConfig
from .sfs import temporal_iou


class BoundaryCalibrationNet(nn.Module):
    """Eqs. (11)-(14)."""

    def __init__(self, feat_dim: int = 768, cfg: Optional[BCNetConfig] = None) -> None:
        super().__init__()
        cfg = cfg or BCNetConfig()
        self.cfg = cfg
        self.conv1 = nn.Conv1d(feat_dim, feat_dim, kernel_size=cfg.conv_kernel,
                               padding=cfg.conv_kernel // 2)
        self.conv2 = nn.Conv1d(feat_dim, feat_dim, kernel_size=cfg.conv_kernel,
                               padding=cfg.conv_kernel // 2)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        # Eq. (13) adds the attention output back onto the conv output, which is
        # only well defined once both live in the d_attn = 256 subspace the
        # paper fixes for the attention layer.
        self.proj = nn.Linear(feat_dim, cfg.d_attn)
        self.attn = nn.MultiheadAttention(cfg.d_attn, cfg.n_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(cfg.d_attn)
        # Eq. (14): W_1 in R^{256 x 128}, W_2 in R^{128 x 1}.
        self.head = nn.Sequential(
            nn.Linear(cfg.d_attn, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, 1))

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``x``: ``(B, T, d)`` -> boundary logits ``(B, T, 1)``.

        ``padding_mask`` is ``(B, T)`` with 1 on real frames and 0 on padding.
        """
        key_padding = None
        if padding_mask is not None:
            key_padding = ~padding_mask.bool()      # True marks a padded position

        h = x.transpose(1, 2)                                        # (B, d, T)
        h = self.norm1(F.gelu(self.conv1(h)).transpose(1, 2))        # Eq. (11)
        h = self.norm2(F.gelu(self.conv2(h.transpose(1, 2))).transpose(1, 2))
        h = self.proj(h)                                             # (B, T, d_attn)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding)
        h = self.norm3(h + attn_out)                                 # Eq. (13)
        return self.head(h)                                          # Eq. (14)


# ---------------------------------------------------------------------------
# Supervision and losses
# ---------------------------------------------------------------------------
def build_boundary_labels(events: Sequence[CandidateEvent],
                          length: int,
                          pos_margin: int = 0) -> np.ndarray:
    """Binary target of Eq. (15): the start and end frame of each event is 1."""
    y = np.zeros(length, dtype=np.float32)
    for event in events:
        for position in (event.s, event.e):
            if 0 <= position < length:
                lo = max(0, position - pos_margin)
                hi = min(length - 1, position + pos_margin)
                y[lo:hi + 1] = 1.0
    return y


def bce_loss(logits: torch.Tensor,
             targets: torch.Tensor,
             mask: torch.Tensor,
             pos_weight_mode: str = "none") -> torch.Tensor:
    """Eq. (15), averaged over the valid frames of the batch.

    ``pos_weight_mode`` is ``"none"`` for the loss exactly as printed in the
    paper, or ``"auto"`` to weight the 2K positive boundary frames by
    ``#negatives / #positives``.
    """
    pos_weight = None
    if pos_weight_mode == "auto":
        positives = float((targets * mask).sum().item())
        negatives = float((mask * (1.0 - targets)).sum().item())
        pos_weight = torch.tensor(max(negatives / max(positives, 1.0), 1.0),
                                  device=logits.device, dtype=torch.float32)
    elif pos_weight_mode != "none":
        raise ValueError(f"unknown bce_pos_weight {pos_weight_mode!r}")

    loss = F.binary_cross_entropy_with_logits(logits, targets,
                                              reduction="none",
                                              pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def _soft_argmax_window(probabilities: torch.Tensor,
                        lo: torch.Tensor,
                        hi: torch.Tensor,
                        temperature: float) -> torch.Tensor:
    """Differentiable arg-max of ``probabilities`` inside ``[lo, hi]``.

    Eq. (16) takes the position of highest confidence; a plain ``argmax`` would
    carry no gradient, so the position is read out as a temperature-scaled
    softmax over the window.  As the temperature goes to zero this converges to
    the arg-max used at inference time in :func:`relocalise_boundaries`.
    """
    width = int((hi - lo).max().item()) + 1
    offsets = torch.arange(width, device=probabilities.device)
    index = lo[:, None] + offsets[None, :]
    valid = index <= hi[:, None]
    index = index.clamp(max=probabilities.shape[1] - 1)
    window = probabilities.gather(1, index)
    weights = F.softmax(torch.log(window.clamp(min=1e-6)) / temperature, dim=-1)
    weights = weights * valid.to(weights.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return (weights * index.to(weights.dtype)).sum(dim=-1)


def diou_loss(logits: torch.Tensor,
              events_list: Sequence[Sequence[CandidateEvent]],
              delta: int,
              temperature: float = 0.5,
              lengths: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Eq. (16): ``(1/K) * sum_k [1 - IoU + rho^2 / c^2]``.

    ``(s*_k, e*_k)`` is the highest-confidence position inside the ``+- Delta``
    window around each original boundary, which is the same neighbourhood rule
    Section 3.3.2(2) uses for re-localisation.
    """
    probabilities = torch.sigmoid(logits.squeeze(-1))                 # (B, T)
    T = probabilities.shape[1]
    if lengths is None:
        lengths = [T] * probabilities.shape[0]

    starts, ends, rows = [], [], []
    for b, events in enumerate(events_list):
        valid_len = int(lengths[b]) if b < len(lengths) else T
        for event in events:
            s = min(max(int(event.s), 0), valid_len - 1)
            e = min(max(int(event.e), 0), valid_len - 1)
            starts.append((s, max(0, s - delta), min(valid_len - 1, s + delta)))
            ends.append((e, max(0, e - delta), min(valid_len - 1, e + delta)))
            rows.append(b)
    if not starts:
        return torch.zeros((), device=logits.device, dtype=torch.float32)

    row_index = torch.tensor(rows, device=logits.device, dtype=torch.long)
    probs = probabilities[row_index]                                  # (N, T)

    def _window(spec):
        centre = torch.tensor([c for c, _, _ in spec], device=logits.device,
                              dtype=torch.float32)
        lo = torch.tensor([l for _, l, _ in spec], device=logits.device,
                          dtype=torch.long)
        hi = torch.tensor([h for _, _, h in spec], device=logits.device,
                          dtype=torch.long)
        return centre, _soft_argmax_window(probs, lo, hi, temperature)

    s_gt, s_pred = _window(starts)
    e_gt, e_pred = _window(ends)

    inter = torch.clamp(torch.minimum(e_gt, e_pred) - torch.maximum(s_gt, s_pred) + 1.0,
                        min=0.0)
    union = torch.maximum(e_gt, e_pred) - torch.minimum(s_gt, s_pred) + 1.0
    iou = inter / union.clamp(min=1e-6)
    # rho: distance between the centres; c: diagonal of the enclosing box.
    rho_sq = (((s_gt + e_gt) - (s_pred + e_pred)) / 2.0) ** 2
    c_sq = (torch.maximum(e_gt, e_pred) - torch.minimum(s_gt, s_pred)) ** 2
    per_event = (1.0 - iou) + rho_sq / c_sq.clamp(min=1e-6)
    return per_event.sum() / max(len(starts), 1)


def train_boundary_net(model: nn.Module,
                       data: Sequence[Tuple[np.ndarray, Sequence[CandidateEvent]]],
                       cfg: Optional[BCNetConfig] = None,
                       device: str = "cpu",
                       verbose: bool = False) -> List[float]:
    """Train ``M^(t)`` on the pseudo-label set ``L^(t)`` (Section 3.3.2(1)).

    ``data`` is a sequence of ``(frame features (T, d), events)`` pairs.  The
    caller's sequence is not reordered.
    """
    cfg = cfg or BCNetConfig()
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    order = np.arange(len(data))
    history: List[float] = []

    for epoch in range(cfg.epochs):
        rng = np.random.RandomState(cfg.seed + epoch)
        rng.shuffle(order)
        epoch_loss, batches = 0.0, 0
        for start in range(0, len(order), cfg.batch_size):
            chunk = [data[i] for i in order[start:start + cfg.batch_size]]
            if not chunk:
                continue
            length = max(int(feats.shape[0]) for feats, _ in chunk)
            dim = int(chunk[0][0].shape[-1])
            feats = torch.zeros(len(chunk), length, dim, device=device)
            mask = torch.zeros(len(chunk), length, device=device)
            targets = torch.zeros(len(chunk), length, device=device)
            lengths, events_list = [], []
            for b, (video_feats, events) in enumerate(chunk):
                t = int(video_feats.shape[0])
                feats[b, :t] = torch.from_numpy(
                    np.asarray(video_feats, dtype=np.float32))
                mask[b, :t] = 1.0
                targets[b, :t] = torch.from_numpy(build_boundary_labels(events, t))
                lengths.append(t)
                events_list.append(events)

            logits = model(feats, padding_mask=mask)
            loss = bce_loss(logits.squeeze(-1), targets, mask, cfg.bce_pos_weight)
            loss = loss + cfg.lambda_diou * diou_loss(
                logits, events_list, delta=cfg.delta,
                temperature=cfg.softmax_temperature, lengths=lengths)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item())
            batches += 1

        average = epoch_loss / max(batches, 1)
        history.append(average)
        if verbose:
            print(f"[bcnet] epoch {epoch + 1}/{cfg.epochs} loss={average:.4f}")
    return history


# ---------------------------------------------------------------------------
# Section 3.3.2(2): boundary re-localisation
# ---------------------------------------------------------------------------
def relocalise_boundaries(events: Sequence[CandidateEvent],
                          boundary_prob: np.ndarray,
                          delta: int,
                          min_span: int = 2) -> float:
    """Move every boundary to the highest-confidence frame inside ``+- delta``.

    Returns the mean displacement in frames, counting the start and the end
    separately.  Events whose two new boundaries would collapse to a span
    shorter than ``min_span`` frames are left untouched.
    """
    probability = np.asarray(boundary_prob, dtype=np.float64).reshape(-1)
    length = probability.shape[0]
    if length == 0 or not events:
        return 0.0

    shifts: List[float] = []
    for event in events:
        lo = max(0, event.s - delta)
        hi = min(length - 1, event.s + delta)
        new_s = lo + int(np.argmax(probability[lo:hi + 1]))
        lo = max(0, event.e - delta)
        hi = min(length - 1, event.e + delta)
        new_e = lo + int(np.argmax(probability[lo:hi + 1]))
        if new_s > new_e:
            new_s, new_e = new_e, new_s
        if new_e - new_s + 1 < min_span:
            continue
        shifts.append(abs(new_s - event.s) + abs(new_e - event.e))
        event.s, event.e = new_s, new_e
        event.mid = (new_s + new_e) // 2
    return float(np.mean(shifts)) if shifts else 0.0


# ---------------------------------------------------------------------------
# Eq. (17): stopping rule
# ---------------------------------------------------------------------------
@dataclass
class RoundSnapshot:
    """The pseudo-label set of one round, detached from the live event objects.

    The events are mutated in place by the refinement, so the previous round has
    to be copied out by value; keeping references would compare an object with
    itself and always report zero displacement.
    """

    ids: frozenset
    bounds: Dict[int, Tuple[int, int]]

    @classmethod
    def of(cls, events: Sequence[CandidateEvent]) -> "RoundSnapshot":
        return cls(ids=frozenset(e.uid for e in events),
                   bounds={e.uid: (e.s, e.e) for e in events})


def _greedy_match(previous: RoundSnapshot,
                  current: Sequence[CandidateEvent],
                  tioi_thresh: float) -> List[Tuple[CandidateEvent, Tuple[int, int]]]:
    """Match current events to previous ones by temporal IoU, greedily.

    Pairs at or above ``tioi_thresh`` are considered in descending order of IoU;
    each event of either round is used at most once.  A greedy pass is the
    standard way to match two sets of temporal intervals and needs no assignment
    solver.
    """
    candidates = []
    for event in current:
        for uid, (old_s, old_e) in previous.bounds.items():
            iou = temporal_iou(event.s, event.e, old_s, old_e)
            if iou >= tioi_thresh:
                candidates.append((iou, event, uid))
    candidates.sort(key=lambda item: item[0], reverse=True)

    matched: List[Tuple[CandidateEvent, Tuple[int, int]]] = []
    used_events, used_previous = set(), set()
    for _, event, uid in candidates:
        if id(event) in used_events or uid in used_previous:
            continue
        used_events.add(id(event))
        used_previous.add(uid)
        matched.append((event, previous.bounds[uid]))
    return matched


def convergence_metrics(previous: RoundSnapshot,
                        current: Sequence[CandidateEvent],
                        mode: str = "tioi",
                        tioi_thresh: float = 0.5,
                        ) -> Tuple[float, float]:
    """Eq. (17): Jaccard similarity and mean boundary displacement.

    ``mode`` selects the matching rule, see
    :attr:`saip.config.BCNetConfig.jaccard_mode`.  The displacement is averaged
    over the events retained in both rounds and over the start and the end of
    each event, so each matched pair contributes
    ``(|ds| + |de|) / 2``.
    """
    current_ids = {e.uid for e in current}

    if mode == "identity":
        union = previous.ids | current_ids
        jaccard = len(previous.ids & current_ids) / len(union) if union else 1.0
        pairs = [(e, previous.bounds[e.uid]) for e in current
                 if e.uid in previous.bounds]
    elif mode == "tioi":
        pairs = _greedy_match(previous, current, tioi_thresh)
        union = len(previous.bounds) + len(current) - len(pairs)
        jaccard = len(pairs) / union if union else 1.0
    else:
        raise ValueError(f"unknown jaccard_mode {mode!r}; "
                         "expected 'tioi' or 'identity'")

    shifts = [(abs(event.s - old_s) + abs(event.e - old_e)) / 2.0
              for event, (old_s, old_e) in pairs]
    displacement = float(np.mean(shifts)) if shifts else 0.0
    return float(jaccard), displacement


def has_converged(jaccard: float,
                  displacement: float,
                  cfg: Optional[BCNetConfig] = None) -> bool:
    """Both conditions of Eq. (17) must hold."""
    cfg = cfg or BCNetConfig()
    return (jaccard > cfg.delta_jaccard
            and displacement < cfg.delta_boundary_frames)
