import argparse
from ai_finance.pipeline.index_pipeline import run_index_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket", help="Override S3 bucket", default=None)
    parser.add_argument("--s3-prefix", help="S3 prefix with data", default=None)
    args = parser.parse_args()

    # Optionally override bucket via env to keep config centralized
    if args.s3_bucket:
        import os
        os.environ["AWS_S3_BUCKET"] = args.s3_bucket

    count = run_index_pipeline(s3_prefix=args.s3_prefix)
    print(f"Indexed {count} documents")


if __name__ == "__main__":
    main()


