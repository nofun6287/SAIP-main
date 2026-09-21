"""Run cached SAIP, write seconds-based JSON and evaluate frozen predictions."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('OMP_NUM_THREADS','1')
import torch
from saip.config import SAIPConfig
from saip.pipeline import SAIPPipeline
from saip.dataset_io import build_activitynet_records, write_json
from saip.evaluation import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--annotations')
    p.add_argument('--config',default='configs/activitynet.yaml')
    p.add_argument('--device',default='cuda')
    p.add_argument('--variant',choices=['full','initial','alignment','no_uniq','no_density','no_narr','no_conf'],default='full')
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg=SAIPConfig.from_file(a.config)
    cfg.manifest=a.manifest
    cfg.features.device=a.device
    cfg.features.feat_dir=str(Path(a.cache)/'feats')
    cfg.features.caption_dir=str(Path(a.cache)/'captions')
    cfg.features.text_feat_dir=str(Path(a.cache)/'text_feats')
    cfg.output.out_dir=a.out
    if a.variant=='initial': cfg.bcnet.max_iters=0
    if a.variant=='alignment': cfg.sfs.weights=(0,0,0,1)
    if a.variant.startswith('no_'): cfg.ablate_dimensions=[a.variant[3:]]
    manifest=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    meta=json.loads((Path(a.cache)/'extraction.json').read_text(encoding='utf-8'))
    if meta.get('caption_scope') == 'candidates':
        if json.loads(json.dumps(vars(cfg.candidates))) != meta['candidates']:
            raise ValueError('Candidate-only cache requires the same proposal configuration')
        if cfg.bcnet.refresh_captions:
            raise ValueError('Caption refresh requires the all-frame cache')
    if meta['fps']!=cfg.candidates.fps:
        raise ValueError('Sampling FPS differs between cache and pipeline')
    for r in manifest['videos']:
        if not (Path(a.cache)/'metadata'/(r['id']+'.json')).exists():
            raise ValueError(f"Incomplete extraction: {r['id']}")
    pipeline=SAIPPipeline(cfg)
    pipeline.run()
    if len(pipeline.units)!=len(manifest['videos']):
        raise RuntimeError('Some videos were skipped; do not report a complete experiment')
    pipeline.write_outputs()
    predictions=build_activitynet_records(pipeline.units,cfg.candidates.fps)
    write_json(predictions,Path(a.out)/'train_pseudo.json')
    write_json(dict(variant=a.variant,video_ids=[r['id'] for r in manifest['videos']],
                   timestamp_unit='seconds',extraction=meta),Path(a.out)/'experiment.json')
    if a.annotations:
        references=json.loads(Path(a.annotations).read_text(encoding='utf-8'))
        result=evaluate(predictions,references,[r['id'] for r in manifest['videos']])
        write_json(result,Path(a.out)/'evaluation.json')
        print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':
    main()
