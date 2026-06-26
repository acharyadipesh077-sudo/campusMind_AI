import argparse
from pathlib import Path

from campusmind.config import get_settings
from campusmind.retrieval import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CampusMind FAISS + BM25 index from Excel dataset.")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to .xlsx dataset.")
    parser.add_argument("--index-dir", type=Path, default=None, help="Output index directory.")
    args = parser.parse_args()
    settings = get_settings()
    dataset = args.dataset or settings.dataset_path
    count = build_index(dataset, args.index_dir)
    print(f"Indexed {count} documents into {(args.index_dir or settings.index_dir)}")


if __name__ == "__main__":
    main()

