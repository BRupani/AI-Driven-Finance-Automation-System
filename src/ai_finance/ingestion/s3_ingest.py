from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, List, Optional

import boto3
import pandas as pd

from ai_finance.config import get_settings


@dataclass
class InvoiceRecord:
    invoice_id: str
    guest_name: Optional[str]
    hotel_name: Optional[str]
    hotel_address: Optional[str]
    check_in_date: Optional[str]
    check_out_date: Optional[str]
    notes: Optional[str]


def _read_object_to_dataframe(s3_client, bucket: str, key: str) -> pd.DataFrame:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    if key.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(body))
    if key.lower().endswith(".json"):
        return pd.read_json(io.BytesIO(body), lines=False)
    if key.lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(body))
    raise ValueError(f"Unsupported file type for key: {key}")


def list_s3_keys(prefix: Optional[str] = None) -> List[str]:
    settings = get_settings()
    session = boto3.Session(profile_name=settings.aws.profile) if settings.aws.profile else boto3.Session()
    s3 = session.client("s3", region_name=settings.aws.region)
    paginator = s3.get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=settings.aws.s3_bucket, Prefix=prefix or settings.aws.s3_prefix):
        for content in page.get("Contents", []):
            key = content["Key"]
            if key.endswith((".csv", ".json", ".parquet")):
                keys.append(key)
    return keys


def load_invoices_from_s3(prefix: Optional[str] = None) -> pd.DataFrame:
    settings = get_settings()
    session = boto3.Session(profile_name=settings.aws.profile) if settings.aws.profile else boto3.Session()
    s3 = session.client("s3", region_name=settings.aws.region)
    keys = list_s3_keys(prefix)
    frames: List[pd.DataFrame] = []
    for key in keys:
        df = _read_object_to_dataframe(s3, settings.aws.s3_bucket, key)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    # Normalize column names
    df_all.columns = [c.strip().lower() for c in df_all.columns]
    # Ensure required columns exist
    for col in [
        "invoice_id",
        "guest_name",
        "hotel_name",
        "hotel_address",
        "check_in_date",
        "check_out_date",
        "notes",
    ]:
        if col not in df_all.columns:
            df_all[col] = None
    return df_all


def iter_invoice_records(df: pd.DataFrame) -> Iterable[InvoiceRecord]:
    for _, row in df.iterrows():
        yield InvoiceRecord(
            invoice_id=str(row.get("invoice_id", "")),
            guest_name=row.get("guest_name"),
            hotel_name=row.get("hotel_name"),
            hotel_address=row.get("hotel_address"),
            check_in_date=row.get("check_in_date"),
            check_out_date=row.get("check_out_date"),
            notes=row.get("notes"),
        )


