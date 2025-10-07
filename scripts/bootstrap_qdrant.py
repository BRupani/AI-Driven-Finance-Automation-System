from ai_finance.config import get_settings
from ai_finance.embedding.encoder import EmbeddingEncoder
from ai_finance.storage.qdrant_client import ensure_collection


def main() -> None:
    enc = EmbeddingEncoder()
    ensure_collection(enc.dimension)
    print("Qdrant collection ensured.")


if __name__ == "__main__":
    main()


