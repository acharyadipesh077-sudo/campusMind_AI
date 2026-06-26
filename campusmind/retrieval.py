import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from campusmind.config import get_settings
from campusmind.text import normalize_text


@dataclass
class Document:
    doc_id: int
    question: str
    context: str
    answer: str
    domain: str
    score: float = 0.0

    @property
    def combined(self) -> str:
        return f"Question: {self.question}\nContext: {self.context}\nAnswer: {self.answer}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "question": self.question,
            "context": self.context,
            "answer": self.answer,
            "domain": self.domain,
            "score": self.score,
        }


def tokenize(text: str) -> list[str]:
    return normalize_text(text).lower().split()


class HybridRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_dir = self.settings.index_dir
        self.embedding_model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.documents: list[Document] = []
        self.bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        index_path = self.index_dir / "faiss.index"
        docs_path = self.index_dir / "documents.jsonl"
        bm25_path = self.index_dir / "bm25.pkl"
        if not index_path.exists() or not docs_path.exists() or not bm25_path.exists():
            return
        self.index = faiss.read_index(str(index_path))
        self.documents = []
        with docs_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                self.documents.append(
                    Document(
                        doc_id=int(item["doc_id"]),
                        question=item["question"],
                        context=item["context"],
                        answer=item["answer"],
                        domain=item["domain"],
                    )
                )
        with bm25_path.open("rb") as handle:
            self.bm25 = pickle.load(handle)

    @property
    def ready(self) -> bool:
        return self.index is not None and self.bm25 is not None and bool(self.documents)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.settings.embedding_model)
        vectors = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=16,
        )
        return np.asarray(vectors, dtype="float32")

    def search(self, query: str, domain: str = "Unknown", domain_confidence: float = 0.0) -> list[Document]:
        if not self.ready:
            return []
        assert self.index is not None
        assert self.bm25 is not None
        query_vector = self.embed([query])
        semantic_scores, semantic_ids = self.index.search(query_vector, self.settings.top_k_semantic)
        semantic_map = {
            int(doc_id): float(score)
            for doc_id, score in zip(semantic_ids[0], semantic_scores[0])
            if int(doc_id) >= 0
        }
        bm25_scores = self.bm25.get_scores(tokenize(query))
        keyword_ids = np.argsort(bm25_scores)[::-1][: self.settings.top_k_keyword]
        max_bm25 = float(np.max(bm25_scores)) if len(bm25_scores) else 0.0
        keyword_map = {
            int(doc_id): float(bm25_scores[doc_id] / max_bm25) if max_bm25 > 0 else 0.0
            for doc_id in keyword_ids
        }
        candidate_ids = set(semantic_map) | set(keyword_map)
        results: list[Document] = []
        for doc_id in candidate_ids:
            doc = self.documents[doc_id]
            if domain_confidence > 0.7 and domain in {"Academic", "Mental Health"} and doc.domain != domain:
                continue
            score = (0.65 * semantic_map.get(doc_id, 0.0)) + (0.35 * keyword_map.get(doc_id, 0.0))
            results.append(
                Document(
                    doc_id=doc.doc_id,
                    question=doc.question,
                    context=doc.context,
                    answer=doc.answer,
                    domain=doc.domain,
                    score=score,
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[: self.settings.hybrid_top_k]


def build_index(dataset_path: Path, index_dir: Path | None = None) -> int:
    import pandas as pd

    settings = get_settings()
    target_dir = index_dir or settings.index_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_excel(dataset_path)
    required = {"question", "context", "answer", "domain"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    frame = frame.fillna("")
    documents = [
        Document(
            doc_id=index,
            question=str(row["question"]),
            context=str(row["context"]),
            answer=str(row["answer"]),
            domain=str(row["domain"]),
        )
        for index, row in frame.iterrows()
    ]
    model = SentenceTransformer(settings.embedding_model)
    vectors = model.encode(
        [doc.combined for doc in documents],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=16,
    )
    matrix = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(target_dir / "faiss.index"))
    with (target_dir / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
    bm25 = BM25Okapi([tokenize(doc.combined) for doc in documents])
    with (target_dir / "bm25.pkl").open("wb") as handle:
        pickle.dump(bm25, handle)
    return len(documents)
