from ai_finance.search.opensearch_client import ensure_index


def main() -> None:
    ensure_index()
    print("OpenSearch index ensured.")


if __name__ == "__main__":
    main()


