from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ai_finance.config import get_settings
from ai_finance.embedding.encoder import EmbeddingEncoder
from ai_finance.search.opensearch_client import search_bm25
from ai_finance.storage.qdrant_client import search_similar


@dataclass
class MatchCandidate:
    invoice_id: str
    bm25_score: float
    vector_score: float
    blended_score: float
    payload: Dict


def _within_date_window(source: Dict, candidate: Dict, window_days: int) -> bool:
    def to_dt(v: Optional[str]) -> Optional[datetime]:
        if not v:
            return None
        try:
            return datetime.fromisoformat(str(v).split("T")[0])
        except Exception:
            return None

    s_in = to_dt(source.get("check_in_date"))
    s_out = to_dt(source.get("check_out_date"))
    c_in = to_dt(candidate.get("check_in_date"))
    c_out = to_dt(candidate.get("check_out_date"))
    window = timedelta(days=window_days)
    checks: List[bool] = []
    if s_in and c_in:
        checks.append(abs((s_in - c_in).days) <= window.days)
    if s_out and c_out:
        checks.append(abs((s_out - c_out).days) <= window.days)
    return all(checks) if checks else True


def compute_blended_score(bm25_score: float, vector_score: float, alpha: float) -> float:
    # BM25 is higher-is-better, cosine similarity is higher-is-better (already normalized)
    return alpha * vector_score + (1.0 - alpha) * bm25_score


def multistage_match(
    query_text: str,
    source_doc: Dict,
    hotel_name: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    encoder: Optional[EmbeddingEncoder] = None,
) -> List[MatchCandidate]:
    settings = get_settings()

    # Stage 1: BM25 candidate retrieval
    bm25_hits = search_bm25(
        query=query_text,
        top_k=settings.pipeline.top_k_bm25,
        hotel_name=hotel_name,
        date_from=date_from,
        date_to=date_to,
    )

    # Stage 2: Vector similarity on candidates (or full) using encoder
    enc = encoder or EmbeddingEncoder()
    # Build query vector from salient fields
    text_fields = [
        source_doc.get("guest_name", ""),
        source_doc.get("hotel_name", ""),
        source_doc.get("hotel_address", ""),
        source_doc.get("notes", ""),
    ]
    query_vec = enc.encode([" ".join([t for t in text_fields if t])])[0]

    # Option A: Direct vector search in Qdrant with optional hotel_name filter
    vec_hits = search_similar(query_vector=query_vec, top_k=settings.pipeline.top_k_vector, hotel_name=hotel_name)
    vec_scores: Dict[str, Tuple[float, Dict]] = {vid: (score, payload) for vid, score, payload in vec_hits}

    # Stage 3: Blend scores and apply rule-based checks
    alpha = settings.pipeline.blend_alpha
    window_days = settings.pipeline.date_window_days
    candidates: List[MatchCandidate] = []
    for inv_id, bm25_score, bm25_src in bm25_hits:
        vscore, vpayload = vec_scores.get(inv_id, (0.0, {}))
        payload = {**bm25_src, **vpayload}
        if not _within_date_window(source_doc, payload, window_days):
            continue
        blended = compute_blended_score(bm25_score, vscore, alpha)
        candidates.append(
            MatchCandidate(
                invoice_id=inv_id,
                bm25_score=bm25_score,
                vector_score=vscore,
                blended_score=blended,
                payload=payload,
            )
        )

    candidates.sort(key=lambda c: c.blended_score, reverse=True)
    return candidates


