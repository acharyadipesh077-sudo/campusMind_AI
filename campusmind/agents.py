import json
import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from campusmind.config import get_settings
from campusmind.database import (
    append_chat,
    get_recent_history,
    get_user_by_id,
    get_user_memory,
)
from campusmind.llm import OllamaClient
from campusmind.retrieval import Document, HybridRetriever
from campusmind.text import (
    contains_devanagari,
    emergency_response,
    is_emergency,
    looks_romanized_nepali,
    normalize_text,
    simple_roman_hint_rewrite,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    user_id: str
    session_id: str
    message: str
    user: dict[str, Any]
    user_memory: list[dict[str, Any]]
    short_history: list[dict[str, Any]]
    follow_up: bool
    personalized_instruction: str
    rewritten_query: str
    domain: str
    domain_confidence: float
    retrieved_documents: list[Document]
    reranked_documents: list[Document]
    answer: str
    workflow: list[str]
    emergency: bool
    latency_ms: int
    started_at: float


class CampusMindGraph:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = OllamaClient()
        self.retriever = HybridRetriever()
        self.graph = self._compile()

    def _step(self, state: AgentState, name: str) -> None:
        state.setdefault("workflow", []).append(name)

    def memory_agent(self, state: AgentState) -> AgentState:
        self._step(state, "Memory Agent")
        history = get_recent_history(state["user_id"], state["session_id"], limit_messages=10)
        state["short_history"] = history
        follow_up_markers = {"à¤¯à¥‹", "à¤¤à¥à¤¯à¥‹", "à¤…à¤¨à¤¿", "à¤•à¤¿à¤¨", "à¤•à¤¸à¤°à¥€", "what about", "ani", "teso bhaye"}
        lowered = state["message"].lower()
        state["follow_up"] = bool(history) and any(marker in lowered for marker in follow_up_markers)
        return state

    def user_loader_agent(self, state: AgentState) -> AgentState:
        self._step(state, "User Loader Agent")
        user = get_user_by_id(state["user_id"])
        if not user:
            raise ValueError("User not found")
        user.pop("password_hash", None)
        state["user"] = user
        state["user_memory"] = get_user_memory(state["user_id"])
        return state

    def personalization_agent(self, state: AgentState) -> AgentState:
        self._step(state, "Personalization Agent")
        user = state.get("user", {})
        sensitivity = "à¤¸à¤‚à¤µà¥‡à¤¦à¤¨à¤¶à¥€à¤² à¤° à¤¸à¤¹à¤¯à¥‹à¤—à¥€ à¤¶à¥ˆà¤²à¥€ à¤…à¤ªà¤¨à¤¾à¤‰à¤¨à¥à¤¹à¥‹à¤¸à¥à¥¤" if user.get("mental_health_sensitive") else ""
        state["personalized_instruction"] = (
            f"à¤ªà¥à¤°à¤¯à¥‹à¤—à¤•à¤°à¥à¤¤à¤¾à¤•à¥‹ à¤¨à¤¾à¤® {user.get('name', 'à¤µà¤¿à¤¦à¥à¤¯à¤¾à¤°à¥à¤¥à¥€')} à¤¹à¥‹à¥¤ "
            f"à¤¶à¤¿à¤•à¥à¤·à¤¾ à¤¸à¥à¤¤à¤°: {user.get('education_level') or 'à¤…à¤œà¥à¤žà¤¾à¤¤'}à¥¤ "
            f"à¤°à¥à¤šà¤¿: {user.get('interests') or 'à¤…à¤œà¥à¤žà¤¾à¤¤'}à¥¤ {sensitivity}"
        )
        return state

    def emergency_router(self, state: AgentState) -> str:
        if is_emergency(state["message"]):
            return "emergency"
        return "continue"

    def emergency_agent(self, state: AgentState) -> AgentState:
        self._step(state, "Emergency Agent")
        state["domain"] = "Emergency"
        state["domain_confidence"] = 1.0
        state["answer"] = emergency_response()
        state["retrieved_documents"] = []
        state["reranked_documents"] = []
        state["rewritten_query"] = state["message"]
        state["emergency"] = True
        return state

    def query_rewriter_agent(self, state: AgentState) -> AgentState:
        self._step(state, "Query Rewriter Agent")
        message = normalize_text(state["message"])
        if contains_devanagari(message):
            state["rewritten_query"] = message
            return state
        if looks_romanized_nepali(message):
            hint = simple_roman_hint_rewrite(message)
        else:
            hint = message
        history_text = "\n".join(f"{h['role']}: {h['message']}" for h in state.get("short_history", []))
        prompt = f"""
Convert or rewrite the user query into clear Nepali Devanagari.
Preserve the meaning. If it is already clear English academic text, translate it to Nepali.
Use chat history only to resolve follow-up references.

Chat history:
{history_text}

User query:
{hint}

Return only the rewritten query.
"""
        try:
            state["rewritten_query"] = self.llm.generate(prompt, temperature=0.0)
        except Exception:
            logger.exception("Query rewriting failed; using deterministic fallback")
            state["rewritten_query"] = hint
        return state

    def domain_agent(self, state: AgentState) -> AgentState:
        self._step(state, "Domain Classification Agent")
        prompt = f"""
Classify this query into exactly one domain: Academic, Mental Health, Emergency, Unknown.
Return JSON only with keys domain and confidence.
Confidence must be 0 to 1.

Query: {state['rewritten_query']}
"""
        try:
            data = self.llm.generate_json(prompt, system="You are a strict JSON classifier.")
            domain = data.get("domain", "Unknown")
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            logger.exception("Domain classification failed; using fallback")
            text = state["rewritten_query"].lower()
            if any(word in text for word in ["à¤¤à¤¨à¤¾à¤µ", "à¤šà¤¿à¤¨à¥à¤¤à¤¾", "à¤¡à¤¿à¤ªà¥à¤°à¥‡à¤¸", "à¤®à¤¨", "à¤¨à¤¿à¤¨à¥à¤¦à¥à¤°à¤¾"]):
                domain, confidence = "Mental Health", 0.65
            elif any(word in text for word in ["à¤ªà¤°à¥€à¤•à¥à¤·à¤¾", "à¤ªà¤¢à¤¾à¤‡", "à¤•à¤²à¥‡à¤œ", "à¤…à¤¸à¤¾à¤‡à¤¨à¤®à¥‡à¤¨à¥à¤Ÿ", "à¤¶à¤¿à¤•à¥à¤·à¤•"]):
                domain, confidence = "Academic", 0.65
            else:
                domain, confidence = "Unknown", 0.4
        if domain not in {"Academic", "Mental Health", "Emergency", "Unknown"}:
            domain = "Unknown"
        state["domain"] = domain
        state["domain_confidence"] = max(0.0, min(confidence, 1.0))
        return state

    def retrieval_agent(self, state: AgentState) -> AgentState:
        self._step(state, "Hybrid Retrieval Agent")
        state["retrieved_documents"] = self.retriever.search(
            query=state["rewritten_query"],
            domain=state["domain"],
            domain_confidence=state["domain_confidence"],
        )
        return state

    def reranker_agent(self, state: AgentState) -> AgentState:
        self._step(state, "Reranker Agent")
        documents = state.get("retrieved_documents", [])
        if len(documents) <= self.settings.final_context_k:
            state["reranked_documents"] = documents
            return state
        candidates = "\n\n".join(
            f"ID: {doc.doc_id}\nDomain: {doc.domain}\nQuestion: {doc.question}\nContext: {doc.context}\nAnswer: {doc.answer}"
            for doc in documents
        )
        prompt = f"""
Select the 3 most useful document IDs for answering the query.
Return JSON only: {{"ids": [1, 2, 3]}}

Query: {state['rewritten_query']}

Candidates:
{candidates}
"""
        try:
            data = self.llm.generate_json(prompt, system="You are a precise reranker.")
            selected = [int(value) for value in data.get("ids", [])]
            selected_docs = [doc for doc_id in selected for doc in documents if doc.doc_id == doc_id]
            state["reranked_documents"] = selected_docs[: self.settings.final_context_k] or documents[: self.settings.final_context_k]
        except Exception:
            logger.exception("Reranking failed; using score order")
            state["reranked_documents"] = documents[: self.settings.final_context_k]
        return state

    def answer_agent(self, state: AgentState) -> AgentState:
        self._step(state, "Answer Agent")
        documents = state.get("reranked_documents", [])
        if not documents:
            state["answer"] = "à¤®à¤¸à¤à¤— à¤ªà¤°à¥à¤¯à¤¾à¤ªà¥à¤¤ à¤œà¤¾à¤¨à¤•à¤¾à¤°à¥€ à¤›à¥ˆà¤¨"
            return state
        context = "\n\n".join(
            f"[{index + 1}] {doc.context}\nà¤‰à¤¤à¥à¤¤à¤° à¤†à¤§à¤¾à¤°: {doc.answer}" for index, doc in enumerate(documents)
        )
        history_text = "\n".join(f"{h['role']}: {h['message']}" for h in state.get("short_history", []))
        prompt = f"""
You are CampusMind AI. Answer in Nepali only.
Use ONLY the retrieved context. Do not add facts outside the context.
Maximum 3 sentences.
If the context is insufficient, answer exactly: à¤®à¤¸à¤à¤— à¤ªà¤°à¥à¤¯à¤¾à¤ªà¥à¤¤ à¤œà¤¾à¤¨à¤•à¤¾à¤°à¥€ à¤›à¥ˆà¤¨

Personalization:
{state.get('personalized_instruction', '')}

Last 5 conversation pairs:
{history_text}

Retrieved context:
{context}

Question:
{state['rewritten_query']}
"""
        try:
            answer = self.llm.generate(prompt, temperature=0.1)
        except Exception:
            logger.exception("Answer generation failed")
            answer = documents[0].answer if documents else "à¤®à¤¸à¤à¤— à¤ªà¤°à¥à¤¯à¤¾à¤ªà¥à¤¤ à¤œà¤¾à¤¨à¤•à¤¾à¤°à¥€ à¤›à¥ˆà¤¨"
        sentences = [part.strip() for part in answer.replace("à¥¤", "à¥¤\n").splitlines() if part.strip()]
        state["answer"] = " ".join(sentences[:3]) or "à¤®à¤¸à¤à¤— à¤ªà¤°à¥à¤¯à¤¾à¤ªà¥à¤¤ à¤œà¤¾à¤¨à¤•à¤¾à¤°à¥€ à¤›à¥ˆà¤¨"
        return state

    def save_agent(self, state: AgentState) -> AgentState:
        self._step(state, "Persistence Agent")
        confidence = float(state.get("domain_confidence", 0.0))
        latency_ms = int((time.perf_counter() - float(state.get("started_at", time.perf_counter()))) * 1000)
        state["latency_ms"] = latency_ms
        append_chat(state["user_id"], state["session_id"], "user", state["message"], state.get("domain", "Unknown"), confidence)
        append_chat(
            state["user_id"],
            state["session_id"],
            "assistant",
            state["answer"],
            state.get("domain", "Unknown"),
            confidence,
            latency_ms,
        )
        return state

    def _compile(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("memory", self.memory_agent)
        workflow.add_node("user_loader", self.user_loader_agent)
        workflow.add_node("personalization", self.personalization_agent)
        workflow.add_node("emergency_handler", self.emergency_agent)
        workflow.add_node("rewrite", self.query_rewriter_agent)
        workflow.add_node("domain_classifier", self.domain_agent)
        workflow.add_node("retrieve", self.retrieval_agent)
        workflow.add_node("rerank", self.reranker_agent)
        workflow.add_node("answer_generator", self.answer_agent)
        workflow.add_node("save", self.save_agent)
        workflow.set_entry_point("memory")
        workflow.add_edge("memory", "user_loader")
        workflow.add_edge("user_loader", "personalization")
        workflow.add_conditional_edges("personalization", self.emergency_router, {"emergency": "emergency_handler", "continue": "rewrite"})
        workflow.add_edge("emergency_handler", "save")
        workflow.add_edge("rewrite", "domain_classifier")
        workflow.add_edge("domain_classifier", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "answer_generator")
        workflow.add_edge("answer_generator", "save")
        workflow.add_edge("save", END)
        return workflow.compile()

    def chat(self, user_id: str, session_id: str, message: str) -> AgentState:
        started = time.perf_counter()
        state: AgentState = {
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "workflow": [],
            "emergency": False,
            "started_at": started,
        }
        result = self.graph.invoke(state)
        result["latency_ms"] = int((time.perf_counter() - started) * 1000)
        if "Persistence Agent" not in result.get("workflow", []):
            self.save_agent(result)
        return result


def state_to_jsonable(state: AgentState) -> dict[str, Any]:
    docs = state.get("reranked_documents") or state.get("retrieved_documents") or []
    return {
        "answer": state.get("answer", "à¤®à¤¸à¤à¤— à¤ªà¤°à¥à¤¯à¤¾à¤ªà¥à¤¤ à¤œà¤¾à¤¨à¤•à¤¾à¤°à¥€ à¤›à¥ˆà¤¨"),
        "domain": state.get("domain", "Unknown"),
        "confidence": float(state.get("domain_confidence", 0.0)),
        "rewritten_query": state.get("rewritten_query", state.get("message", "")),
        "retrieved_documents": [doc.to_dict() for doc in docs],
        "workflow": state.get("workflow", []),
        "emergency": bool(state.get("emergency", False)),
    }
