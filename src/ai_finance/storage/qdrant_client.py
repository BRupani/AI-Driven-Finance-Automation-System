from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from ai_finance.config import get_settings


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        https=settings.qdrant.use_https,
    )


def ensure_collection(dimension: int) -> None:
    settings = get_settings()
    client = get_qdrant_client()
    collections = client.get_collections().collections
    names = {c.name for c in collections}
    if settings.qdrant.collection_name not in names:
        client.create_collection(
            collection_name=settings.qdrant.collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )


def upsert_vectors(ids: List[str], vectors: np.ndarray, payloads: List[dict]) -> None:
    settings = get_settings()
    client = get_qdrant_client()
    points = [
        PointStruct(id=str(ids[i]), vector=vectors[i].tolist(), payload=payloads[i])
        for i in range(len(ids))
    ]
    client.upsert(collection_name=settings.qdrant.collection_name, points=points)


def search_similar(
    query_vector: np.ndarray,
    top_k: int = 10,
    hotel_name: Optional[str] = None,
) -> List[Tuple[str, float, dict]]:
    settings = get_settings()
    client = get_qdrant_client()
    qfilter: Optional[Filter] = None
    if hotel_name:
        qfilter = Filter(must=[FieldCondition(key="hotel_name", match=MatchValue(value=hotel_name))])
    result = client.search(
        collection_name=settings.qdrant.collection_name,
        query_vector=query_vector.tolist(),
        limit=top_k,
        query_filter=qfilter,
        with_payload=True,
        score_threshold=None,
    )
    output: List[Tuple[str, float, dict]] = []
    for r in result:
        output.append((str(r.id), float(r.score), dict(r.payload or {})))
    return output


