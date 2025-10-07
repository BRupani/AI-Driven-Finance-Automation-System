from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field


class AwsSettings(BaseModel):
    region: str = Field(default=os.getenv("AWS_REGION", "us-east-1"))
    profile: Optional[str] = Field(default=os.getenv("AWS_PROFILE"))
    s3_bucket: str = Field(default=os.getenv("AWS_S3_BUCKET", ""))
    s3_prefix: str = Field(default=os.getenv("AWS_S3_PREFIX", "invoices/"))


class OpenSearchSettings(BaseModel):
    host: str = Field(default=os.getenv("OPENSEARCH_HOST", "http://localhost"))
    port: int = Field(default=int(os.getenv("OPENSEARCH_PORT", "9200")))
    username: str = Field(default=os.getenv("OPENSEARCH_USER", ""))
    password: str = Field(default=os.getenv("OPENSEARCH_PASSWORD", ""))
    index_name: str = Field(default=os.getenv("OPENSEARCH_INDEX", "invoices"))


class QdrantSettings(BaseModel):
    host: str = Field(default=os.getenv("QDRANT_HOST", "localhost"))
    port: int = Field(default=int(os.getenv("QDRANT_PORT", "6333")))
    collection_name: str = Field(default=os.getenv("QDRANT_COLLECTION", "invoices_vectors"))
    use_https: bool = Field(default=os.getenv("QDRANT_USE_HTTPS", "false").lower() == "true")


class EmbeddingSettings(BaseModel):
    model_name: str = Field(default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    dimension: int = Field(default=int(os.getenv("EMBEDDING_DIM", "384")))


class PipelineSettings(BaseModel):
    top_k_bm25: int = Field(default=int(os.getenv("TOP_K_BM25", "50")))
    top_k_vector: int = Field(default=int(os.getenv("TOP_K_VECTOR", "50")))
    blend_alpha: float = Field(default=float(os.getenv("BLEND_ALPHA", "0.6")))
    date_window_days: int = Field(default=int(os.getenv("DATE_WINDOW_DAYS", "7")))


class LoggingSettings(BaseModel):
    level: str = Field(default=os.getenv("LOG_LEVEL", "INFO"))
    json: bool = Field(default=os.getenv("LOG_JSON", "false").lower() == "true")


class Settings(BaseModel):
    aws: AwsSettings = Field(default_factory=AwsSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embed: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


