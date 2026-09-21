"""Create an annotation-free ActivityNet manifest from a local video directory."""
import argparse
import json
from pathlib import Path
import cv2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video-root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--ids',help='Optional JSON list of split IDs; no captions or timestamps')
    a=p.parse_args()
    paths=sorted(Path(a.video_root).glob('*.mp4'))
    ids=None
    if a.ids:
        ids=set(json.loads(Path(a.ids).read_text(encoding='utf-8')))
    records=[]
    for path in paths:
        if ids is not None and path.stem not in ids: continue
        cap=cv2.VideoCapture(str(path))
        fps,n=cap.get(cv2.CAP_PROP_FPS),cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps<=0 or n<=0: raise ValueError(f'Invalid video: {path}')
        records.append(dict(id=path.stem,duration=n/fps,path=str(path.resolve())))
    if not records: raise ValueError('No matching videos')
    if ids is not None and set(r['id'] for r in records)!=ids:
        raise ValueError('Requested split contains missing videos; no silent exclusion is allowed')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(videos=records),indent=2),encoding='utf-8')
    print(f'Wrote {len(records)} videos to {out}')


if __name__=='__main__': main()
