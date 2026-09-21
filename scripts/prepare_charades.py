"""Select a deterministic raw-video subset; annotations only define split membership."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', required=True)
    p.add_argument('--video-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--count', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    annotation_path = Path(a.annotations)
    ids = sorted(json.loads(annotation_path.read_text(encoding='utf-8')))
    root = Path(a.video_root)
    eligible = [v for v in ids if (root / (v + '.mp4')).is_file()]
    chosen = sorted(random.Random(a.seed).sample(eligible, a.count))
    records = []
    for vid in chosen:
        path = root / (vid + '.mp4')
        cap = cv2.VideoCapture(str(path))
        fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps <= 0 or n <= 0:
            raise ValueError(f'Invalid video: {path}')
        records.append(dict(id=vid, path=str(path.resolve()), duration=n / fps))
    payload = dict(videos=records, sampling=dict(seed=a.seed, count=a.count,
        eligible=len(eligible), annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        protocol='Sorted split IDs, existing MP4 files, Python random.sample; durations from video headers. No sentences or boundaries used.'))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'Wrote {len(records)} videos to {out}', flush=True)


if __name__ == '__main__':
    main()
