"""Intrinsic evaluation in seconds. Temporal agreement is not caption correctness."""
from collections import Counter
import re
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def iou_matrix(predicted, reference):
    p = np.asarray(predicted, dtype=float).reshape(-1, 2)
    r = np.asarray(reference, dtype=float).reshape(-1, 2)
    intersection = np.maximum(0, np.minimum(p[:, None, 1], r[None, :, 1]) -
                              np.maximum(p[:, None, 0], r[None, :, 0]))
    union = (p[:, 1]-p[:, 0])[:, None] + (r[:, 1]-r[:, 0])[None, :] - intersection
    return intersection / np.maximum(union, 1e-12)


def token_f1(a, b):
    a, b = (Counter(re.findall(r"\w+", s.lower())) for s in (a, b))
    overlap = sum((a & b).values())
    return 2 * overlap / max(sum(a.values()) + sum(b.values()), 1)


def evaluate(predictions, references, video_ids, thresholds=(.3, .5, .7)):
    rows = []
    for vid in video_ids:
        gt = references[vid]
        pred = predictions.get(vid, {'timestamps': [], 'sentences': []})
        if len(pred['timestamps']) != len(pred['sentences']):
            raise ValueError(f'Misaligned captions: {vid}')
        duration = pred.get('duration', gt['duration'])
        for start, end in pred['timestamps']:
            if not (0 <= start < end <= duration + .001):
                raise ValueError(f'Invalid seconds timestamp for {vid}: {(start,end)}')
        # Different descriptions of the same interval are references, not separate events.
        intervals = sorted(set(tuple(t) for t in gt['timestamps']))
        refs = [[s for t, s in zip(gt['timestamps'], gt['sentences']) if tuple(t) == span]
                for span in intervals]
        ious = iou_matrix(pred['timestamps'], intervals)
        row = dict(id=vid, n_pred=len(ious), n_ref=len(intervals),
                   duration_difference_seconds=float(duration-gt['duration']),
                   duplicate_caption_fraction=1-len(set(pred['sentences']))/max(len(pred['sentences']),1))
        best = ious.max(axis=0) if len(ious) else np.zeros(len(intervals))
        row['mean_best_iou'] = float(best.mean()) if len(best) else 0.
        for t in thresholds:
            hit = (ious >= t)
            match = maximum_bipartite_matching(csr_matrix(hit), perm_type='column') if hit.size else []
            tp = sum(int(j >= 0) for j in match)
            row[f'tp@{t}'] = tp
            row[f'recovered@{t}'] = int((best >= t).sum())
        lexical = []
        for i, caption in enumerate(pred['sentences']):
            js = np.where(ious[i] >= .5)[0]
            lexical.append(max((token_f1(caption,r) for j in js for r in refs[j]),default=0.))
        row['localized_token_f1_sum'] = sum(lexical)
        rows.append(row)
    npred = sum(r['n_pred'] for r in rows)
    nref = sum(r['n_ref'] for r in rows)
    summary = dict(videos=len(rows), predicted_events=npred, unique_reference_intervals=nref,
        mean_best_iou_macro=float(np.mean([r['mean_best_iou'] for r in rows])),
        localized_token_f1=sum(r['localized_token_f1_sum'] for r in rows)/max(npred,1),
        duplicate_caption_fraction_macro=float(np.mean([r['duplicate_caption_fraction'] for r in rows])),
        protocol='Seconds; exact duplicate reference spans merged; maximum-cardinality one-to-one temporal matching. Token F1 is lexical overlap at IoU>=0.5, unmatched predictions score zero; not CIDEr, METEOR or a semantic correctness metric.')
    for t in thresholds:
        tp = sum(r[f'tp@{t}'] for r in rows)
        summary[f'precision@{t}'] = tp/max(npred,1)
        summary[f'recall@{t}'] = tp/max(nref,1)
        summary[f'f1@{t}'] = 2*tp/max(npred+nref,1)
        summary[f'coverage_recall@{t}'] = sum(r[f'recovered@{t}'] for r in rows)/max(nref,1)
    rng = np.random.default_rng(42)
    values = np.array([r['mean_best_iou'] for r in rows])
    boots = [values[rng.integers(0,len(values),len(values))].mean() for _ in range(2000)]
    summary['mean_best_iou_macro_ci95'] = np.quantile(boots,[.025,.975]).tolist()
    return dict(summary=summary,per_video=rows)
