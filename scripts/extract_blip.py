"""Offline, resumable BLIP retrieval/caption cache extraction at fixed sampling FPS."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault('USE_TF', '0')
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import BertTokenizer, BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def frames(path, fps):
    cap = cv2.VideoCapture(str(path))
    native = cap.get(cv2.CAP_PROP_FPS)
    if native <= 0:
        raise ValueError(f'Invalid video: {path}')
    idx, sample = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= round(sample * native / fps):
            yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            sample += 1
        idx += 1
    cap.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'blip-source', 'retrieval-checkpoint', 'caption-model', 'out'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--fps', type=float, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int)
    p.add_argument('--precision', choices=['float32','float16'], default='float32')
    p.add_argument('--caption-scope', choices=['all','candidates'], default='all')
    p.add_argument('--config', default='configs/activitynet.yaml')
    a = p.parse_args()
    if a.fps <= 0 or a.batch_size <= 0:
        p.error('--fps and --batch-size must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    sys.path.insert(0, str(Path(a.blip_source).resolve()))
    import models.blip as blip
    def tokenizer():
        tok = BertTokenizer.from_pretrained(a.caption_model, local_files_only=True)
        tok.add_special_tokens({'bos_token': '[DEC]'})
        tok.add_special_tokens({'additional_special_tokens': ['[ENC]']})
        tok.enc_token_id = tok.convert_tokens_to_ids('[ENC]')
        return tok
    blip.init_tokenizer = tokenizer
    from models.blip_itm import blip_itm
    retrieval = blip_itm(pretrained=a.retrieval_checkpoint, image_size=384, vit='base',
        med_config=str(Path(a.blip_source) / 'configs/med_config.json')).to(a.device).eval()
    processor = BlipProcessor.from_pretrained(a.caption_model, local_files_only=True)
    captioner = BlipForConditionalGeneration.from_pretrained(a.caption_model, local_files_only=True).to(a.device).eval()
    if a.precision == 'float16':
        retrieval.half()
        captioner.half()
    transform = transforms.Compose([transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    out = Path(a.out)
    for name in ('feats', 'captions', 'text_feats', 'metadata'):
        (out / name).mkdir(parents=True, exist_ok=True)
    config = dict(fps=a.fps, retrieval_checkpoint=str(Path(a.retrieval_checkpoint).resolve()),
        caption_model=str(Path(a.caption_model).resolve()), sampling='nearest native frame at t=k/fps',
        generation=dict(do_sample=False,num_beams=1,max_new_tokens=25), feature_dim=256)
    if a.precision != 'float32':
        config['precision'] = a.precision
    if a.caption_scope == 'candidates':
        from saip.config import SAIPConfig
        from saip.candidates import build_candidates
        candidate_cfg = SAIPConfig.from_file(a.config).candidates
        if candidate_cfg.fps != a.fps:
            raise ValueError('Candidate configuration FPS mismatch')
        config['caption_scope'] = 'candidates'
        config['candidates'] = vars(candidate_cfg)
    meta = out / 'extraction.json'
    if meta.exists() and json.loads(meta.read_text()) != json.loads(json.dumps(config)):
        raise ValueError('Cache configuration mismatch; use a new output directory')
    meta.write_text(json.dumps(config, indent=2), encoding='utf-8')
    records = json.loads(Path(a.manifest).read_text())['videos']
    if a.limit:
        records = records[:a.limit]
    started = time.time()
    with torch.inference_mode():
        for index, record in enumerate(records):
            vid = record['id']
            marker = out / 'metadata' / (vid + '.json')
            if marker.exists():
                continue
            if a.device.startswith('cuda'):
                torch.cuda.empty_cache()
            images = list(frames(record['path'], a.fps))
            if not images:
                raise ValueError(f'No frames: {vid}')
            visual, texts, captions = [], [], []
            for start in range(0,len(images),a.batch_size):
                chunk = images[start:start+a.batch_size]
                tensor = torch.stack([transform(im) for im in chunk]).to(a.device, dtype=next(retrieval.parameters()).dtype)
                visual.append(retrieval.vision_proj(retrieval.visual_encoder(tensor)[:,0]).cpu().numpy())
            features = np.concatenate(visual).astype('float32')
            indices = list(range(len(images)))
            if a.caption_scope == 'candidates':
                pool, _ = build_candidates(features,candidate_cfg)
                indices = sorted(set(e.mid for e in pool))
            for start in range(0,len(indices),a.batch_size):
                chunk = [images[i] for i in indices[start:start+a.batch_size]]
                inputs = processor(images=chunk, return_tensors='pt').to(a.device)
                inputs['pixel_values'] = inputs['pixel_values'].to(dtype=next(captioner.parameters()).dtype)
                tokens = captioner.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=25)
                sentences = processor.batch_decode(tokens, skip_special_tokens=True)
                captions.extend([s.strip() for s in sentences])
                enc = retrieval.tokenizer(sentences,padding='max_length',truncation=True,max_length=35,return_tensors='pt').to(a.device)
                z = retrieval.text_encoder(enc.input_ids,attention_mask=enc.attention_mask,return_dict=True,mode='text')
                texts.append(retrieval.text_proj(z.last_hidden_state[:,0]).cpu().numpy())
            text_array=np.zeros((len(images),256),dtype='float32')
            text_array[indices]=np.concatenate(texts).astype('float32')
            caption_array=['']*len(images)
            for i,caption in zip(indices,captions): caption_array[i]=caption
            np.save(out/'feats'/(vid+'.npy'),features)
            np.save(out/'text_feats'/(vid+'.npy'),text_array)
            (out/'captions'/(vid+'.json')).write_text(json.dumps(caption_array),encoding='utf-8')
            marker.write_text(json.dumps(dict(id=vid,frames=len(images),caption_frames=indices,fps=a.fps,batch_size=a.batch_size)),encoding='utf-8')
            print(f'[{index+1}/{len(records)}] {vid}: {len(images)} frames, {len(indices)} captions; elapsed {time.time()-started:.1f}s',flush=True)


if __name__ == '__main__':
    main()
