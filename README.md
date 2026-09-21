# SAIP: Summarization-Aware Iterative Pseudo-Labeling

SAIP generates temporal interval–sentence pseudo-labels from raw videos, then
evaluates their temporal coverage and lexical agreement against separate references.
**ActivityNet Captions is the public target and default configuration.**
Generation does not use target-video human boundaries, descriptions or categories;
the frozen BLIP models inherit external pretraining supervision.

The pipeline includes BLIP extraction, candidate generation, four-component SFS
selection, corpus statistics, iterative boundary calibration and seconds-based
JSON export. Downstream PDVC/Vid2Seq training is outside this repository.

## Reproduce

See [the complete setup and commands](docs/reproduce_activitynet.md),
[paper-to-code mapping](docs/paper_mapping.md), and [中文说明](README.zh-CN.md).

```bash
pip install -r requirements-reproduce.txt
python scripts/prepare_activitynet.py --video-root /data/activitynet/videos --ids /data/activitynet/train_ids.json --out runs/activitynet/manifest.json
python scripts/extract_blip.py --manifest runs/activitynet/manifest.json --blip-source /models/BLIP --retrieval-checkpoint /models/model_base_retrieval_coco.pth --caption-model /models/blip-caption --out runs/activitynet/cache --config configs/activitynet.yaml --caption-scope candidates --precision float16 --batch-size 4
python scripts/run_experiment.py --manifest runs/activitynet/manifest.json --cache runs/activitynet/cache --out runs/activitynet/saip --config configs/activitynet.yaml
python scripts/evaluate_pseudo_labels.py --manifest runs/activitynet/manifest.json --predictions runs/activitynet/saip/train_pseudo.json --annotations /data/activitynet/reference.json --out runs/activitynet/saip/evaluation.json
python -m pytest -q
```

Replace all paths with local inputs. Checkpoints, videos and annotations are not
included. The caption model directory must contain Transformers-compatible BLIP
weights and tokenizer/processor files. Use the same candidate configuration for
extraction and generation. Candidate-only caching requires caption refresh off.

## Outputs and interpretation

- `train_pseudo.json`: video IDs, durations, timestamps **in seconds**, sentences.
- `saip_report.json`: resolved configuration, iteration history and convergence status.
- `experiment.json`: extraction settings and complete video inventory.
- `evaluation.json`: per-video results and aggregate one-to-one temporal P/R/F1,
  unrestricted coverage, mean best IoU and localized token F1.

Token F1 is a lexical diagnostic, not CIDEr or semantic correctness. Convergence
means label stability, not accuracy; a run reaching its iteration limit explicitly
reports that it has not converged. SFS selection is a greedy heuristic without a
submodular approximation guarantee. Category density scaling cancels under the
default within-video min–max normalization.

The optional Charades adapter was used for a 500-video engineering check only.
It does not replace the paper’s ActivityNet experiments or validate its downstream
scores. Public reproduction starts from `configs/activitynet.yaml`; see the guide
for hardware, model variants, seeds and limitations.

## License

Source: [MIT](LICENSE). Obtain datasets and pretrained models separately under
their respective licenses. This release does not redistribute those assets.
