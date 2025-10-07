from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from ai_finance.config import get_settings


class EmbeddingEncoder:
    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.embed.model_name
        self.model = SentenceTransformer(self.model_name)
        self.dimension = settings.embed.dimension

    def encode(self, texts: Iterable[str], normalize: bool = True) -> np.ndarray:
        sentences: List[str] = [t if t is not None else "" for t in texts]
        vectors = self.model.encode(sentences, convert_to_numpy=True, normalize_embeddings=normalize)
        return vectors


