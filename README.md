AI Driven Finance Automation System - Billing System Information Retreival and Match

Overview
This project implements a production-grade, multistage invoice matching and retrieval system for finance. It combines:
- Qdrant for high-performance vector storage and similarity search
- OpenSearch (BM25) for lexical search with optional semantic reranking
- BERT Sentence Transformers for generating dense embeddings

Primary use case: high-precision billing system information retreival and match over financial invoices and reservations, focused on guest names, hotel details, and date filters. The system ingests data from AWS S3, builds indices in Qdrant and OpenSearch, and exposes a reusable matching pipeline. It is orchestrated via Apache Airflow and is deployable to AWS with CI/CD and monitoring hooks.

Key Capabilities
- Multistage matching: BM25 lexical retrieval + dense vector semantic scoring + rule-based tie-breakers
- Filters: date ranges, hotel attributes, and structured fields
- Data ingestion from S3 and schema validation with Pydantic
- Indexing pipelines for OpenSearch and Qdrant
- Airflow DAG for automated ingestion, indexing, and matching
- Local development via Docker Compose for OpenSearch and Qdrant

High-Level Architecture
1) Ingestion: Pull billing/invoice, booking, and metadata from S3 (CSV/JSON/Parquet) → validate/normalize
2) Embedding: Generate SentenceTransformer embeddings for text fields (guest name, hotel name, address, notes)
3) Indexing:
   - OpenSearch: BM25 over normalized textual fields with analyzers
   - Qdrant: Vector collections keyed by document IDs with stored payloads
4) Matching (billing system information retreival and match):
   - Stage 1: BM25 top-k retrieval in OpenSearch
   - Stage 2: Vector similarity from Qdrant on candidates or full corpus
   - Stage 3: Blended score and deterministic rules (date windows, exact fields)
5) Orchestration: Airflow DAG tasks for ingest → encode → index → match → export

Accuracy
With tuned thresholds and feature blending, the billing system information retreival and match pipeline achieves approximately 96% correctness across semantic and non-semantic searches on representative datasets.

Getting Started
Prerequisites
- Python 3.10+
- Docker Desktop
- AWS credentials configured for S3 access (e.g., via environment or profile)

Quickstart (Local Services)
1. Copy `.env.example` to `.env` and fill values.
2. Start OpenSearch and Qdrant:
   ```bash
   docker compose up -d
   ```
3. Create and activate a virtual environment, then install dependencies:
   ```bash
   python -m venv .venv
   . .venv/Scripts/activate
   pip install -r requirements.txt
   ```
4. Bootstrap indices (mappings/collections):
   ```bash
   python scripts/bootstrap_opensearch.py
   python scripts/bootstrap_qdrant.py
   ```
5. Run a local indexing pipeline on a sample S3 path (Windows PowerShell shown):
   ```bash
   python scripts/run_local.py --s3-bucket <BUCKET> --s3-prefix invoices/sample/
   ```

6. Test a sample match using Python (optional):
   ```bash
   python -c "from ai_finance.matching.algorithm import multistage_match; q='john smith grand hotel'; src={'guest_name':'John Smith','hotel_name':'Grand Hotel'}; print(multistage_match(q, src)[:3])"
   ```

Production Deployment (AWS + Airflow)
- Package this repository as an artifact and deploy the Airflow DAG under `airflow/dags/dag_invoice_match.py` (DAG id: `billing_system_information_retreival_and_match`).
- Provide environment via Airflow Connections/Variables or env vars.
- Ensure VPC networking access from Airflow workers to OpenSearch and Qdrant endpoints.
- Use a CI/CD pipeline (e.g., GitHub Actions or AWS CodePipeline) to lint, test, and deploy DAG and application code to your environment.

Airflow Run Steps
1. Install dependencies (on Airflow workers): ensure `requirements.txt` packages are installed or vendored.
2. Place the DAG file at `airflow/dags/dag_invoice_match.py` and make sure workers can import the `src/` package (e.g., add to `PYTHONPATH`).
3. Set environment variables on Airflow (Connections/Variables or worker env) for AWS, OpenSearch, and Qdrant.
4. In Airflow UI, trigger the DAG `billing_system_information_retreival_and_match`, or via CLI:
   ```bash
   airflow dags trigger billing_system_information_retreival_and_match
   ```

Configuration
Configuration is managed via environment variables and Pydantic settings (`src/ai_finance/config.py`). See `.env.example` for defaults.

Project Layout
```
src/
  ai_finance/
    __init__.py
    config.py
    ingestion/s3_ingest.py
    embedding/encoder.py
    storage/qdrant_client.py
    search/opensearch_client.py
    matching/algorithm.py
    pipeline/index_pipeline.py
airflow/
  dags/dag_invoice_match.py  # DAG id: billing_system_information_retreival_and_match
scripts/
  bootstrap_opensearch.py
  bootstrap_qdrant.py
  run_local.py
```

Monitoring and Logging
- Structured logging with JSON option
- Airflow task metrics and basic failure alerts (email/webhook placeholders)

Security Notes
- Do not commit real credentials.
- Use AWS IAM roles for S3 and managed OpenSearch access where possible.
- Consider encrypting Qdrant storage at rest and enabling TLS for both services.

License
MIT


