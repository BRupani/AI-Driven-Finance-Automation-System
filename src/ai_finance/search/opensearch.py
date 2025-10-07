from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from opensearchpy import OpenSearch

from ai_finance.config import get_settings


def get_opensearch() -> OpenSearch:
    s = get_settings()
    auth = (s.opensearch.username, s.opensearch.password) if s.opensearch.username else None
    client = OpenSearch(
        hosts=[{"host": s.opensearch.host.replace("http://", "").replace("https://", ""), "port": s.opensearch.port}],
        http_auth=auth,
        use_ssl=s.opensearch.host.startswith("https://"),
        verify_certs=False,
        ssl_show_warn=False,
        timeout=60,
    )
    return client


def ensure_index() -> None:
    s = get_settings()
    client = get_opensearch()
    if not client.indices.exists(index=s.opensearch.index_name):
        body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                "analysis": {
                    "analyzer": {
                        "folding": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "invoice_id": {"type": "keyword"},
                    "guest_name": {"type": "text", "analyzer": "folding"},
                    "hotel_name": {"type": "text", "analyzer": "folding"},
                    "hotel_address": {"type": "text", "analyzer": "folding"},
                    "notes": {"type": "text", "analyzer": "folding"},
                    "check_in_date": {"type": "date", "format": "yyyy-MM-dd||strict_date_optional_time"},
                    "check_out_date": {"type": "date", "format": "yyyy-MM-dd||strict_date_optional_time"}
                }
            }
        }
        client.indices.create(index=s.opensearch.index_name, body=body)


def index_documents(documents: List[Dict[str, Any]]) -> None:
    s = get_settings()
    client = get_opensearch()
    actions = []
    for doc in documents:
        _id = doc.get("invoice_id")
        actions.append({"index": {"_index": s.opensearch.index_name, "_id": _id}})
        actions.append(doc)
    if actions:
        client.bulk(body=actions, refresh=True)


def search_bm25(
    query: str,
    top_k: int = 20,
    hotel_name: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    s = get_settings()
    client = get_opensearch()
    must: List[Dict[str, Any]] = []
    if query:
        must.append({
            "multi_match": {
                "query": query,
                "fields": ["guest_name^3", "hotel_name^2", "hotel_address", "notes"],
                "type": "best_fields"
            }
        })
    if hotel_name:
        must.append({"match": {"hotel_name": {"query": hotel_name}}})
    range_filter: Dict[str, Any] = {}
    if date_from:
        range_filter.setdefault("check_in_date", {})["gte"] = date_from
    if date_to:
        range_filter.setdefault("check_out_date", {})["lte"] = date_to
    filters: List[Dict[str, Any]] = []
    if range_filter:
        filters.append({"range": range_filter})
    body = {
        "query": {
            "bool": {
                "must": must or [{"match_all": {}}],
                "filter": filters
            }
        },
        "size": top_k
    }
    res = client.search(index=s.opensearch.index_name, body=body)
    hits = res.get("hits", {}).get("hits", [])
    out: List[Tuple[str, float, Dict[str, Any]]] = []
    for h in hits:
        out.append((h["_id"], float(h.get("_score", 0.0)), h.get("_source", {})))
    return out


