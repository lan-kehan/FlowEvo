"""PickScore scoring wrapper with local-model friendly defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


class PickScoreBatchScorer:
    """Score generated images against prompts with PickScore v1."""

    def __init__(
        self,
        model_path: str | Path = "PickScore_v1",
        processor_path: str | Path | None = None,
        device: str = "cuda",
        dtype: str = "float32",
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device
        self.dtype = _torch_dtype(torch, dtype)
        if str(device).startswith("cpu") and self.dtype in {torch.float16, torch.bfloat16}:
            self.dtype = torch.float32
        model_name = str(model_path)
        processor_name = str(processor_path or model_path)

        self.processor = AutoProcessor.from_pretrained(
            processor_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).eval()
        self.model = self.model.to(device=device, dtype=self.dtype)

    def __call__(self, prompts: Sequence[str], images: Sequence[object]) -> list[float]:
        if len(prompts) != len(images):
            raise ValueError("prompts and images must have the same length")

        image_inputs = self.processor(
            images=list(images),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = self.processor(
            text=list(prompts),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {key: value.to(self.device) for key, value in image_inputs.items()}
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}

        with self.torch.no_grad():
            image_embs = self._feature_tensor(self.model.get_image_features(**image_inputs))
            image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

            text_embs = self._feature_tensor(self.model.get_text_features(**text_inputs))
            text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

            logit_scale = self.model.logit_scale.exp()
            scores = logit_scale * (text_embs @ image_embs.T).diag()
            return (scores / 26.0).detach().float().cpu().tolist()

    def _feature_tensor(self, features):
        if hasattr(features, "pooler_output"):
            return features.pooler_output
        if isinstance(features, dict) and "pooler_output" in features:
            return features["pooler_output"]
        if isinstance(features, (tuple, list)):
            return features[0]
        return features


def _torch_dtype(torch: object, dtype: str):
    normalized = str(dtype).lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "no"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")
