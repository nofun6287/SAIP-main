"""Model access layer: frame features, frame captions and text features.

The three quantities of Section 3.1-3.2 are produced by two frozen BLIP models:

======================  =========================================  ================
quantity                paper                                      model
======================  =========================================  ================
``h_t`` (Eq. 1)         visual feature of a keyframe              BLIP retrieval
``c_t``                 description of a keyframe                 BLIP captioning
``q_i`` (Eq. 10)        text embedding of a description           BLIP retrieval
======================  =========================================  ================

Large-scale pseudo-labelling is usually run once on a GPU machine, so this
package defaults to the **cache** backend: the arrays are written to disk by the
feature extractor and read back here.  The ``blip`` backend runs the models
in-process when the checkpoints are available locally; the model weights are not
distributed with this repository, only their identifiers.

Recommended cache layout::

    feats/<video_id>.npy        float32 (T, 768)          Eq. (1)
    captions/<video_id>.json    list[str], one per frame  c_t
    text_feats/<video_id>.npy   float32 (T, 768)          q_i of c_t

The caption file may also hold a list of lists (one entry per sampling of the
decoder); the first sample of each frame is used, matching ``num_stnc = 1``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import FeatureConfig

#: Identifiers of the models this package expects, by name.  ``vit`` and
#: ``image_size`` are the arguments the corresponding BLIP builders take.
MODEL_REGISTRY = {
    "blip-itm-base-coco": {"builder": "blip_itm", "vit": "base", "image_size": 384},
    "blip-itm-large-coco": {"builder": "blip_itm", "vit": "large", "image_size": 384},
    "blip-caption-base-coco": {"builder": "blip_decoder", "vit": "base",
                               "image_size": 384},
    "blip-caption-large-coco": {"builder": "blip_decoder", "vit": "large",
                                "image_size": 384},
}


class ModelUnavailable(RuntimeError):
    """Raised when a model or a piece of cached data cannot be loaded."""


# ---------------------------------------------------------------------------
# Video decoding
# ---------------------------------------------------------------------------
def _parse_frame_rate(value: str) -> float:
    """Parse ffprobe's ``"30000/1001"`` frame-rate strings safely."""
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            rate = float(numerator) / float(denominator)
            return rate if rate > 0 else 0.0
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def probe_video(path: str) -> dict:
    try:
        import ffmpeg
    except ImportError as exc:                                # pragma: no cover
        raise ModelUnavailable(
            "ffmpeg-python is required to decode videos "
            "(pip install ffmpeg-python, plus a system ffmpeg binary).") from exc

    try:
        info = ffmpeg.probe(str(path))
    except Exception as exc:
        raise ModelUnavailable(f"ffprobe failed on {path}: {exc}") from exc

    stream = next((s for s in info.get("streams", [])
                   if s.get("codec_type") == "video"), None)
    if stream is None:
        raise ModelUnavailable(f"no video stream in {path}")
    fps = _parse_frame_rate(stream.get("avg_frame_rate", "0"))
    if fps <= 0:
        fps = _parse_frame_rate(stream.get("r_frame_rate", "0"))
    return {"height": int(stream["height"]), "width": int(stream["width"]),
            "fps": fps, "nb_frames": int(stream.get("nb_frames", 0) or 0)}


def decode_frames(path: str,
                  stride: int = 8,
                  input_size: int = 384,
                  centre_crop: bool = True) -> np.ndarray:
    """Decode a video to ``(T, 3, input_size, input_size)`` uint8 frames.

    One frame every ``stride`` is kept, so the resulting frame rate is
    ``video_fps / stride`` -- the value that must be passed as
    ``CandidateConfig.fps``.
    """
    try:
        import ffmpeg
    except ImportError as exc:                                # pragma: no cover
        raise ModelUnavailable(
            "ffmpeg-python is required to decode videos "
            "(pip install ffmpeg-python, plus a system ffmpeg binary).") from exc

    info = probe_video(path)
    if info["fps"] <= 0:
        raise ModelUnavailable(f"could not determine the frame rate of {path}")

    height, width = info["height"], info["width"]
    if height >= width:
        out_h, out_w = int(height * input_size / width), input_size
    else:
        out_h, out_w = input_size, int(width * input_size / height)

    stream = (ffmpeg.input(str(path))
              .filter("fps", fps=info["fps"] / max(stride, 1))
              .filter("scale", out_w, out_h))
    if centre_crop:
        stream = stream.crop(int((out_w - input_size) / 2.0),
                             int((out_h - input_size) / 2.0),
                             input_size, input_size)

    raw, _ = (stream.output("pipe:", format="rawvideo", pix_fmt="rgb24")
              .run(capture_stdout=True, quiet=True))
    frame_bytes = input_size * input_size * 3
    count = len(raw) // frame_bytes
    if count == 0:
        raise ModelUnavailable(f"no frames decoded from {path}")
    frames = np.frombuffer(raw[:count * frame_bytes], np.uint8)
    frames = frames.reshape(count, input_size, input_size, 3)
    return np.ascontiguousarray(frames.transpose(0, 3, 1, 2))


def resolve_video_path(video_id: str,
                       recorded_path: Optional[str],
                       video_root: Optional[str]) -> Optional[str]:
    if recorded_path and Path(recorded_path).is_file():
        return recorded_path
    if video_root:
        for suffix in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
            candidate = Path(video_root) / f"{video_id}{suffix}"
            if candidate.is_file():
                return str(candidate)
    return None


# ---------------------------------------------------------------------------
# BLIP in-process backend
# ---------------------------------------------------------------------------
class BlipModels:
    """Lazily constructed pair of frozen BLIP models."""

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        self._itm = None
        self._decoder = None
        self._preprocess = None

    # -- construction ------------------------------------------------------
    def _torch(self):
        try:
            import torch
            from torchvision import transforms
            from torchvision.transforms.functional import InterpolationMode
        except ImportError as exc:                            # pragma: no cover
            raise ModelUnavailable(
                "the 'blip' backend needs torch and torchvision "
                "(pip install torch torchvision).") from exc
        if self._preprocess is None:
            self._preprocess = transforms.Compose([
                transforms.Resize((self.cfg.input_size, self.cfg.input_size),
                                  interpolation=InterpolationMode.BICUBIC),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711)),
            ])
        return torch

    def _build(self, model_name: str, checkpoint: Optional[str]):
        spec = MODEL_REGISTRY.get(model_name)
        if spec is None:
            raise ModelUnavailable(
                f"unknown model {model_name!r}; known identifiers: "
                f"{sorted(MODEL_REGISTRY)}")
        if not checkpoint or not Path(checkpoint).is_file():
            raise ModelUnavailable(
                f"checkpoint for {model_name!r} not found at {checkpoint!r}. "
                "The BLIP weights are not distributed with this repository; "
                "download them and set FeatureConfig.itm_ckpt / caption_ckpt, "
                "or use the 'cache' backend.")
        try:
            if spec["builder"] == "blip_itm":
                from models.blip_itm import blip_itm as builder
            else:
                from models.blip import blip_decoder as builder
        except ImportError as exc:
            raise ModelUnavailable(
                "the LAVIS BLIP sources are not importable; add the directory "
                "holding models/blip.py and models/blip_itm.py to PYTHONPATH."
            ) from exc
        return builder(pretrained=checkpoint, image_size=spec["image_size"],
                       vit=spec["vit"])

    @property
    def itm(self):
        if self._itm is None:
            torch = self._torch()
            self._itm = self._build(self.cfg.itm_model, self.cfg.itm_ckpt) \
                .to(self.cfg.device).eval()
            torch.set_grad_enabled(True)
        return self._itm

    @property
    def decoder(self):
        if self._decoder is None:
            self._decoder = self._build(self.cfg.caption_model,
                                        self.cfg.caption_ckpt) \
                .to(self.cfg.device).eval()
        return self._decoder

    # -- extraction --------------------------------------------------------
    def frame_features(self, frames: np.ndarray) -> np.ndarray:
        """Eq. (1): ``(T, 3, S, S)`` uint8 -> ``(T, 768)`` float32."""
        torch = self._torch()
        model = self.itm
        out: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(frames), self.cfg.batch_size):
                batch = frames[start:start + self.cfg.batch_size]
                tensor = self._preprocess(
                    torch.from_numpy(np.ascontiguousarray(batch)).float() / 255.0
                ).to(self.cfg.device)
                encoded = model.visual_encoder(tensor)
                out.append(model.vision_proj(encoded[:, 0, :]).cpu().numpy())
        if not out:
            return np.zeros((0, 768), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)

    def frame_captions(self, frames: np.ndarray,
                       frame_indices: Sequence[int]) -> List[str]:
        torch = self._torch()
        model = self.decoder
        selected = frames[np.asarray(frame_indices, dtype=int)]
        captions = [""] * len(selected)
        with torch.no_grad():
            for start in range(0, len(selected), self.cfg.batch_size):
                batch = selected[start:start + self.cfg.batch_size]
                tensor = self._preprocess(
                    torch.from_numpy(np.ascontiguousarray(batch)).float() / 255.0
                ).to(self.cfg.device)
                generated = model.generate(tensor, sample=True, max_length=20,
                                           min_length=5)
                for offset, text in enumerate(generated):
                    captions[start + offset] = str(text).strip()
        return captions

    def text_features(self, sentences: Sequence[str]) -> np.ndarray:
        """``q_i`` of Eq. (10), in the same space as the frame features."""
        torch = self._torch()
        model = self.itm
        if not len(sentences):
            return np.zeros((0, 768), dtype=np.float32)
        out: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(sentences), self.cfg.batch_size):
                chunk = list(sentences[start:start + self.cfg.batch_size])
                tokens = model.tokenizer(chunk, padding="max_length",
                                         truncation=True, max_length=35,
                                         return_tensors="pt").to(self.cfg.device)
                encoded = model.text_encoder(tokens.input_ids,
                                             attention_mask=tokens.attention_mask,
                                             return_dict=True, mode="text")
                out.append(model.text_proj(
                    encoded.last_hidden_state[:, 0, :]).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class FeatureProvider:
    """Interface used by the pipeline."""

    def frame_features(self, video_id: str) -> Optional[np.ndarray]:
        raise NotImplementedError

    def captions_for(self, video_id: str,
                     frame_indices: Sequence[int]) -> Tuple[List[str], Optional[np.ndarray]]:
        raise NotImplementedError

    def release(self, video_id: str) -> None:
        """Drop any cached frames held for ``video_id``."""


class CachedFeatures(FeatureProvider):
    """Read the three arrays the feature extractor wrote to disk."""

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        for name in ("feat_dir", "caption_dir", "text_feat_dir"):
            if getattr(cfg, name) is None:
                raise ModelUnavailable(
                    f"the 'cache' backend needs FeatureConfig.{name}; "
                    "see README.md for the expected layout.")
        self._caption_cache: dict = {}

    def _path(self, directory: str, video_id: str, suffix: str) -> Path:
        return Path(directory) / f"{video_id}{suffix}"

    def frame_features(self, video_id: str) -> Optional[np.ndarray]:
        path = self._path(self.cfg.feat_dir, video_id, ".npy")
        if not path.is_file():
            return None
        return np.load(str(path)).astype(np.float32)

    def _captions(self, video_id: str) -> List:
        if video_id not in self._caption_cache:
            path = self._path(self.cfg.caption_dir, video_id, ".json")
            if not path.is_file():
                self._caption_cache[video_id] = []
            else:
                self._caption_cache[video_id] = json.loads(
                    path.read_text(encoding="utf-8"))
        return self._caption_cache[video_id]

    def captions_for(self, video_id: str,
                     frame_indices: Sequence[int]) -> Tuple[List[str], Optional[np.ndarray]]:
        stored = self._captions(video_id)
        captions: List[str] = []
        for index in frame_indices:
            if 0 <= index < len(stored):
                entry = stored[index]
                value = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
                captions.append(str(value).strip())
            else:
                captions.append("")

        text_features = None
        path = self._path(self.cfg.text_feat_dir, video_id, ".npy")
        if path.is_file():
            array = np.load(str(path)).astype(np.float32)
            if array.ndim == 3:                 # (T, num_stnc, d)
                array = array[:, 0, :]
            if array.ndim == 2:
                text_features = np.stack(
                    [array[i] if 0 <= i < len(array) else np.zeros(array.shape[1],
                                                                    dtype=np.float32)
                     for i in frame_indices])
        return captions, text_features

    def release(self, video_id: str) -> None:
        self._caption_cache.pop(video_id, None)


class BlipFeatures(FeatureProvider):
    """Decode the video and run the frozen BLIP models in-process."""

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        if not cfg.video_root:
            raise ModelUnavailable("the 'blip' backend needs FeatureConfig.video_root")
        self.models = BlipModels(cfg)
        self._frames: dict = {}

    def _frames_for(self, video_id: str, recorded_path: Optional[str]) -> np.ndarray:
        if video_id not in self._frames:
            path = resolve_video_path(video_id, recorded_path, self.cfg.video_root)
            if path is None:
                raise ModelUnavailable(f"video file for {video_id!r} not found")
            self._frames[video_id] = decode_frames(path, stride=self.cfg.stride,
                                                   input_size=self.cfg.input_size)
        return self._frames[video_id]

    def frame_features(self, video_id: str) -> Optional[np.ndarray]:
        return None                      # driven by the pipeline through `set_video`

    def set_video(self, video_id: str, recorded_path: Optional[str] = None) -> np.ndarray:
        """Decode the video once and return its frame features."""
        frames = self._frames_for(video_id, recorded_path)
        return self.models.frame_features(frames)

    def captions_for(self, video_id: str,
                     frame_indices: Sequence[int]) -> Tuple[List[str], Optional[np.ndarray]]:
        frames = self._frames.get(video_id)
        if frames is None:
            return [""] * len(frame_indices), None
        valid = [i for i in frame_indices if 0 <= i < len(frames)]
        captions = [""] * len(frame_indices)
        if valid:
            decoded = self.models.frame_captions(frames, valid)
            for position, text in zip(valid, decoded):
                captions[frame_indices.index(position)] = text
        filled = [c for c in captions if c]
        text_features = None
        if filled:
            features = self.models.text_features(captions)
            text_features = features
        return captions, text_features

    def release(self, video_id: str) -> None:
        self._frames.pop(video_id, None)


def build_provider(cfg: FeatureConfig) -> FeatureProvider:
    """Factory for the backend named in ``cfg.backend``."""
    if cfg.backend == "cache":
        return CachedFeatures(cfg)
    if cfg.backend == "blip":
        return BlipFeatures(cfg)
    if cfg.backend == "mock":
        from .synthetic import SyntheticFeatures
        return SyntheticFeatures()
    raise ModelUnavailable(
        f"unknown FeatureConfig.backend {cfg.backend!r}; "
        "expected 'cache', 'blip' or 'mock'")
