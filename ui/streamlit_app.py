import uuid

import plotly.express as px
import requests
import streamlit as st


st.set_page_config(page_title="CampusMind AI", page_icon="CM", layout="wide")

API_URL = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans Devanagari', sans-serif; }
    .hero {
        padding: 1rem 1.25rem;
        border-radius: 8px;
        background: linear-gradient(120deg, #185a9d, #43cea2);
        color: white;
        margin-bottom: 1rem;
    }
    .metric-card {
        border: 1px solid rgba(255,255,255,.18);
        background: rgba(255,255,255,.08);
        border-radius: 8px;
        padding: .75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.get('token', '')}"}


def post(path: str, payload: dict) -> dict:
    response = requests.post(f"{API_URL}{path}", json=payload, headers=api_headers(), timeout=180)
    if not response.ok:
        try:
            detail = response.json().get("detail", response.json())
            if isinstance(detail, dict) and "reason" in detail:
                raise RuntimeError(detail["reason"])
            if isinstance(detail, str):
                raise RuntimeError(detail)
        except ValueError:
            pass
        response.raise_for_status()
    return response.json()


def get(path: str) -> dict:
    response = requests.get(f"{API_URL}{path}", headers=api_headers(), timeout=30)
    response.raise_for_status()
    return response.json()


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown('<div class="hero"><h2>CampusMind AI</h2><p>नेपाली Academic + Mental Health Agentic RAG Assistant</p></div>', unsafe_allow_html=True)

pages = ["Chat", "Signup/Login", "Analytics", "About"]
page = st.sidebar.radio("Page", pages)

if page == "Signup/Login":
    tab_login, tab_signup = st.tabs(["Login", "Signup"])
    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary"):
            try:
                data = post("/auth/login", {"email": email, "password": password})
                st.session_state.token = data["access_token"]
                st.session_state.user = data["user"]
                st.success("Logged in")
            except Exception as exc:
                st.error(f"Login failed: {exc}")
    with tab_signup:
        name = st.text_input("Name")
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
        gender = st.selectbox("Gender", ["male", "female", "other"])
        education_level = st.text_input("Education level")
        interests = st.text_area("Interests")
        sensitive = st.checkbox("Mental health sensitive support")
        if st.button("Create account", type="primary"):
            try:
                data = post(
                    "/auth/signup",
                    {
                        "name": name,
                        "email": signup_email,
                        "password": signup_password,
                        "age": age,
                        "gender": gender,
                        "education_level": education_level,
                        "interests": interests,
                        "mental_health_sensitive": sensitive,
                    },
                )
                st.session_state.token = data["access_token"]
                st.session_state.user = data["user"]
                st.success("Account created")
            except Exception as exc:
                st.error(f"Signup failed: {exc}")

elif page == "Chat":
    if not st.session_state.get("token"):
        st.info("Please login or create an account first.")
        st.stop()
    left, right = st.columns([0.68, 0.32])
    with left:
        for item in st.session_state.messages:
            with st.chat_message(item["role"]):
                st.write(item["content"])
        message = st.chat_input("नेपाली वा Roman Nepali मा प्रश्न लेख्नुहोस्")
        if message:
            st.session_state.messages.append({"role": "user", "content": message})
            with st.chat_message("user"):
                st.write(message)
            with st.chat_message("assistant"):
                with st.spinner("CampusMind agents are thinking..."):
                    try:
                        data = post("/chat", {"message": message, "session_id": st.session_state.session_id})
                        st.write(data["answer"])
                        st.session_state.last_response = data
                        st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
                    except Exception as exc:
                        st.error(f"Chat failed: {exc}")
    with right:
        data = st.session_state.get("last_response")
        st.subheader("Agent Trace")
        if data:
            st.caption(f"Domain: {data['domain']} | Confidence: {data['confidence']:.2f}")
            st.progress(min(float(data["confidence"]), 1.0))
            st.write(" -> ".join(data["workflow"]))
            st.subheader("Retrieved Documents")
            for doc in data["retrieved_documents"]:
                with st.expander(f"{doc['domain']} #{doc['doc_id']} | {doc['score']:.3f}"):
                    st.write(doc["question"])
                    st.write(doc["context"])
                    st.success(doc["answer"])
        else:
            st.caption("Ask a question to see domain, confidence, retrieved documents, and workflow.")

elif page == "Analytics":
    if not st.session_state.get("token"):
        st.info("Please login first.")
        st.stop()
    try:
        data = get("/analytics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Queries", data["queries"])
        c2.metric("Avg Confidence", f"{data['avg_confidence']:.2f}")
        c3.metric("Avg Latency ms", f"{data['avg_latency_ms']:.0f}")
        if data["domain_distribution"]:
            fig = px.bar(data["domain_distribution"], x="domain", y="count", color="domain")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Analytics failed: {exc}")

else:
    st.write(
        "CampusMind AI is a local Nepali agentic RAG system for academic and mental health domains. "
        "It uses short-term memory limited to the last five conversation pairs, long-term user profiles, "
        "hybrid FAISS/BM25 retrieval, reranking, grounded Nepali answers, and emergency self-harm handling."
    )
