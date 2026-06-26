import hashlib
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from campusmind.database import append_chat, get_recent_history, get_user_by_id
from campusmind.llm import OllamaClient
from campusmind.retrieval import Document, HybridRetriever
from campusmind.text import (
    emergency_response,
    is_emergency,
    normalize_text,
    simple_roman_hint_rewrite,
)


FALLBACK_ANSWER = "मसँग पर्याप्त जानकारी छैन"
ANSWER_SYSTEM_PROMPT = "You are CampusMind AI, a helpful Nepali student assistant."


def fast_domain(text: str) -> tuple[str, float]:
    t = text.lower()

    if any(w in t for w in ["तनाव", "डिप्रेस", "चिन्ता", "मानसिक", "depression", "anxiety", "stress"]):
        return "Mental Health", 0.7

    if any(w in t for w in ["परीक्षा", "कलेज", "इन्जिनियरिङ", "पढाइ", "exam", "college", "study"]):
        return "Academic", 0.7

    return "Unknown", 0.4


def fast_preprocess(text: str) -> str:
    normalized = normalize_text(text)
    return simple_roman_hint_rewrite(normalized)


def build_answer_prompt(context: str, question: str) -> str:
    return f"""You are CampusMind AI, a helpful Nepali student assistant.

Rules:
- Answer ONLY in Nepali
- Be natural, conversational, and helpful
- Do NOT sound robotic
- Use retrieved context if helpful
- If context is weak, say you don't have enough information
- Keep answer under 5 sentences

Context:
{context}

Question:
{question}
"""


def docs_to_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{index + 1}] Domain: {doc.domain}\nQuestion: {doc.question}\nContext: {doc.context}\nAnswer: {doc.answer}"
        for index, doc in enumerate(documents)
    )


def cache_key(question: str, documents: list[Document]) -> str:
    doc_part = "|".join(f"{doc.doc_id}:{doc.score:.4f}" for doc in documents)
    raw = f"{normalize_text(question).lower()}::{doc_part}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class FastChatResult:
    answer: str
    domain: str
    confidence: float
    rewritten_query: str
    retrieved_documents: list[Document]
    workflow: list[str]
    emergency: bool
    cached: bool
    latency_ms: int


class FastCampusMindEngine:
    def __init__(self) -> None:
        self.llm = OllamaClient()
        self.retriever = HybridRetriever()
        self.cache: dict[str, str] = {}

    def prepare(self, user_id: str, session_id: str, message: str) -> dict[str, Any]:
        started = time.perf_counter()
        workflow = ["Fast preprocessing"]
        query = fast_preprocess(message)
        user = get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        history = get_recent_history(user_id, session_id, limit_messages=10)

        if is_emergency(message):
            workflow.append("Emergency bypass")
            return {
                "started": started,
                "user": user,
                "history": history,
                "message": message,
                "query": query,
                "domain": "Emergency",
                "confidence": 1.0,
                "documents": [],
                "workflow": workflow,
                "emergency": True,
                "cache_key": "",
                "prompt": "",
            }

        domain, confidence = fast_domain(query)
        workflow.append("Rule-based domain classification")
        documents = self.retriever.search(query=query, domain=domain, domain_confidence=confidence)
        workflow.append("FAISS/BM25 retrieval")
        top_documents = documents[:3]
        context = docs_to_context(top_documents)
        key = cache_key(query, top_documents)
        prompt = build_answer_prompt(context or FALLBACK_ANSWER, query)
        workflow.append("Single streaming LLM answer")
        return {
            "started": started,
            "user": user,
            "history": history,
            "message": message,
            "query": query,
            "domain": domain,
            "confidence": confidence,
            "documents": top_documents,
            "workflow": workflow,
            "emergency": False,
            "cache_key": key,
            "prompt": prompt,
        }

    def _save(
        self,
        prepared: dict[str, Any],
        answer: str,
        latency_ms: int,
    ) -> None:
        append_chat(
            prepared["user"]["user_id"],
            prepared["session_id"],
            "user",
            prepared["message"],
            prepared["domain"],
            prepared["confidence"],
            latency_ms,
        )
        append_chat(
            prepared["user"]["user_id"],
            prepared["session_id"],
            "assistant",
            answer,
            prepared["domain"],
            prepared["confidence"],
            latency_ms,
        )

    def chat(self, user_id: str, session_id: str, message: str) -> FastChatResult:
        prepared = self.prepare(user_id, session_id, message)
        prepared["session_id"] = session_id
        cached = False

        if prepared["emergency"]:
            answer = emergency_response()
        elif not prepared["documents"]:
            answer = FALLBACK_ANSWER
        elif prepared["cache_key"] in self.cache:
            answer = self.cache[prepared["cache_key"]]
            cached = True
        else:
            answer = self.llm.generate(prepared["prompt"], system=ANSWER_SYSTEM_PROMPT, temperature=0.2)
            answer = answer.strip() or FALLBACK_ANSWER
            self.cache[prepared["cache_key"]] = answer

        latency_ms = int((time.perf_counter() - prepared["started"]) * 1000)
        self._save(prepared, answer, latency_ms)
        return FastChatResult(
            answer=answer,
            domain=prepared["domain"],
            confidence=prepared["confidence"],
            rewritten_query=prepared["query"],
            retrieved_documents=prepared["documents"],
            workflow=prepared["workflow"] + (["Response cache hit"] if cached else []),
            emergency=prepared["emergency"],
            cached=cached,
            latency_ms=latency_ms,
        )

    def stream_chat(self, user_id: str, session_id: str, message: str) -> Generator[str, None, None]:
        prepared = self.prepare(user_id, session_id, message)
        prepared["session_id"] = session_id
        chunks: list[str] = []

        if prepared["emergency"]:
            answer = emergency_response()
            yield answer
        elif not prepared["documents"]:
            answer = FALLBACK_ANSWER
            yield answer
        elif prepared["cache_key"] in self.cache:
            answer = self.cache[prepared["cache_key"]]
            yield answer
        else:
            for token in self.llm.generate_stream(prepared["prompt"], system=ANSWER_SYSTEM_PROMPT, temperature=0.2):
                chunks.append(token)
                yield token
            answer = "".join(chunks).strip() or FALLBACK_ANSWER
            self.cache[prepared["cache_key"]] = answer

        latency_ms = int((time.perf_counter() - prepared["started"]) * 1000)
        self._save(prepared, answer, latency_ms)


def result_to_jsonable(result: FastChatResult) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "domain": result.domain,
        "confidence": result.confidence,
        "rewritten_query": result.rewritten_query,
        "retrieved_documents": [doc.to_dict() for doc in result.retrieved_documents],
        "workflow": result.workflow,
        "emergency": result.emergency,
        "cached": result.cached,
        "latency_ms": result.latency_ms,
    }

