# -*- coding: utf-8 -*-
"""ResNet 임베더. torch/app.models 는 여기서만 import 한다(테스트에서는 set_embedder 로 가짜를 꽂을 수 있다)."""
from typing import List, Optional, Protocol

import numpy as np
from PIL import Image


class EmbedderLike(Protocol):
    def embed(self, rgb_list: List[np.ndarray], chunk: int = 32) -> np.ndarray: ...


class Embedder:
    def __init__(self):
        import torch                                   # noqa: F401  (지연 import)
        from app.config import DEVICE
        from app.models import load_resnet_model, resnet_preprocess
        self._torch = torch
        self._device = DEVICE
        self._pre = resnet_preprocess
        self.model = load_resnet_model()
        try:
            self.model.eval()
        except Exception:
            pass

    def embed(self, rgb_list: List[np.ndarray], chunk: int = 32) -> np.ndarray:
        torch = self._torch
        outs = []
        for i in range(0, len(rgb_list), chunk):
            part = rgb_list[i:i + chunk]
            tensors = [self._pre(Image.fromarray(np.ascontiguousarray(a))) for a in part]
            with torch.no_grad():
                try:
                    o = self.model(torch.stack(tensors).to(self._device))
                    o = o.reshape(o.shape[0], -1).cpu().numpy()
                except Exception:
                    o = np.stack([self.model(t.unsqueeze(0).to(self._device)).reshape(-1).cpu().numpy() for t in tensors])
            outs.append(o.astype(np.float32))
        return np.concatenate(outs, axis=0)


_EMBEDDER: Optional[EmbedderLike] = None


def get_embedder() -> EmbedderLike:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER


def set_embedder(e: Optional[EmbedderLike]) -> None:
    """테스트/오프라인 평가용: 임베더 교체."""
    global _EMBEDDER
    _EMBEDDER = e
