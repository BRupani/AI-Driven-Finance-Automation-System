from __future__ import annotations

from typing import Dict, List

import numpy as np

from ai_finance.config import get_settings
from ai_finance.embedding.encoder import EmbeddingEncoder
from ai_finance.ingestion.s3_ingest import load_invoices_from_s3
from ai_finance.search.opensearch_client import ensure_index, index_documents
from ai_finance.storage.qdrant_client import ensure_collection, upsert_vectors


def build_documents(df) -> List[Dict]:
    docs: List[Dict] = []
    for _, row in df.iterrows():
        doc = {
            "invoice_id": str(row.get("invoice_id", "")),
            "guest_name": row.get("guest_name"),
            "hotel_name": row.get("hotel_name"),
            "hotel_address": row.get("hotel_address"),
            "notes": row.get("notes"),
            "check_in_date": row.get("check_in_date"),
            "check_out_date": row.get("check_out_date"),
        }
        docs.append(doc)
    return docs


def build_texts_for_embedding(documents: List[Dict]) -> List[str]:
    texts: List[str] = []
    for d in documents:
        parts = [
            d.get("guest_name") or "",
            d.get("hotel_name") or "",
            d.get("hotel_address") or "",
            d.get("notes") or "",
        ]
        texts.append(" ".join([p for p in parts if p]))
    return texts


def run_index_pipeline(s3_prefix: str | None = None) -> int:
    settings = get_settings()
    df = load_invoices_from_s3(prefix=s3_prefix)
    if df.empty:
        return 0

    documents = build_documents(df)

    # OpenSearch
    ensure_index()
    index_documents(documents)

    # Qdrant
    encoder = EmbeddingEncoder()
    ensure_collection(encoder.dimension)
    texts = build_texts_for_embedding(documents)
    vectors = encoder.encode(texts)
    ids = [d["invoice_id"] for d in documents]
    upsert_vectors(ids=ids, vectors=vectors, payloads=documents)
    return len(documents)


