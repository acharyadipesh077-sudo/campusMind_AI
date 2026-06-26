import sys
import uuid
from html import escape

import requests
import streamlit as st


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


st.set_page_config(page_title="CampusMind AI", page_icon="CM", layout="wide")


def init_state() -> None:
    defaults = {
        "api_url": "http://127.0.0.1:8000",
        "token": "",
        "user": None,
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "theme": "Dark",
        "last_prompt": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


def css(theme: str) -> str:
    dark = theme == "Dark"
    bg = "#0f1117" if dark else "#f7f8fb"
    panel = "#171a22" if dark else "#ffffff"
    text = "#f4f7fb" if dark else "#111827"
    muted = "#a8b3c7" if dark else "#667085"
    border = "#2a3040" if dark else "#e5e7eb"
    user = "#2563eb"
    ai = "#1f2937" if dark else "#eef2f7"
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Noto Sans Devanagari', system-ui, sans-serif;
        background: {bg};
        color: {text};
    }}
    .main .block-container {{
        max-width: 980px;
        padding-top: 1.2rem;
        padding-bottom: 6rem;
    }}
    [data-testid="stSidebar"] {{
        background: {panel};
        border-right: 1px solid {border};
    }}
    .brand {{
        display: flex;
        align-items: center;
        gap: .7rem;
        padding: .75rem .25rem 1rem;
        color: {text};
    }}
    .brand-mark {{
        width: 34px;
        height: 34px;
        display: grid;
        place-items: center;
        border-radius: 8px;
        background: linear-gradient(135deg, #dc2626, #2563eb);
        color: white;
        font-weight: 800;
    }}
    .brand-title {{
        font-size: 1.08rem;
        font-weight: 700;
        line-height: 1.1;
    }}
    .brand-subtitle {{
        color: {muted};
        font-size: .78rem;
    }}
    .chat-row {{
        display: flex;
        width: 100%;
        margin: .75rem 0;
    }}
    .chat-row.user {{
        justify-content: flex-end;
    }}
    .chat-row.assistant {{
        justify-content: flex-start;
    }}
    .bubble {{
        max-width: min(720px, 84%);
        padding: .85rem 1rem;
        border-radius: 14px;
        border: 1px solid {border};
        line-height: 1.65;
        white-space: pre-wrap;
        box-shadow: 0 8px 22px rgba(0, 0, 0, .08);
    }}
    .bubble.user {{
        background: {user};
        color: white;
        border-color: {user};
        border-bottom-right-radius: 4px;
    }}
    .bubble.assistant {{
        background: {ai};
        color: {text};
        border-bottom-left-radius: 4px;
    }}
    .meta {{
        color: {muted};
        font-size: .78rem;
        margin-top: .35rem;
    }}
    .toolbar {{
        display: flex;
        gap: .5rem;
        margin: .25rem 0 1rem;
    }}
    .empty {{
        text-align: center;
        margin: 12vh auto 2rem;
        color: {muted};
    }}
    .empty h1 {{
        color: {text};
        font-size: 2.1rem;
        margin-bottom: .35rem;
    }}
    </style>
    """


st.markdown(css(st.session_state.theme), unsafe_allow_html=True)


def headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def post_json(path: str, payload: dict, timeout: int = 60) -> dict:
    response = requests.post(f"{st.session_state.api_url}{path}", json=payload, headers=headers(), timeout=timeout)
    if not response.ok:
        try:
            detail = response.json().get("detail", response.json())
            if isinstance(detail, dict):
                raise RuntimeError(detail.get("reason", str(detail)))
            raise RuntimeError(str(detail))
        except ValueError:
            response.raise_for_status()
    return response.json()


def stream_answer(message: str):
    response = requests.post(
        f"{st.session_state.api_url}/chat/stream",
        json={"message": message, "session_id": st.session_state.session_id},
        headers=headers(),
        stream=True,
        timeout=(5, 180),
    )
    response.raise_for_status()
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            yield chunk


with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark">CM</div>
          <div>
            <div class="brand-title">CampusMind AI</div>
            <div class="brand-subtitle">Nepali Academic + Mental Health</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.api_url = st.text_input("API URL", value=st.session_state.api_url)
    st.session_state.theme = st.radio("Theme", ["Dark", "Light"], horizontal=True)
    if st.button("New chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.last_prompt = ""
        st.rerun()

    st.divider()
    st.caption("Account")
    tab_login, tab_signup = st.tabs(["Login", "Signup"])
    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary", use_container_width=True):
            try:
                data = post_json("/auth/login", {"email": email, "password": password})
                st.session_state.token = data["access_token"]
                st.session_state.user = data["user"]
                st.success("Logged in")
            except Exception as exc:
                st.error(f"Login failed: {exc}")
    with tab_signup:
        name = st.text_input("Name")
        signup_email = st.text_input("Signup email")
        signup_password = st.text_input("Signup password", type="password")
        age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
        gender = st.selectbox("Gender", ["male", "female", "other"])
        education_level = st.text_input("Education level")
        interests = st.text_area("Interests", height=80)
        sensitive = st.checkbox("Mental health sensitive support")
        if st.button("Create account", use_container_width=True):
            try:
                data = post_json(
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

    st.divider()
    if st.session_state.user:
        st.caption(f"Signed in as {st.session_state.user['name']}")
    else:
        st.caption("Sign in to start chatting.")


if not st.session_state.token:
    st.markdown(
        """
        <div class="empty">
          <h1>CampusMind AI</h1>
          <p>Login or create an account from the sidebar to begin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty">
          <h1>आज के सहयोग चाहिन्छ?</h1>
          <p>Academic प्रश्न, exam stress, study planning, वा mental health support सोध्नुहोस्।</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


for message in st.session_state.messages:
    role = message["role"]
    content = escape(message["content"])
    st.markdown(
        f"""
        <div class="chat-row {role}">
          <div>
            <div class="bubble {role}">{content}</div>
            <div class="meta">{'You' if role == 'user' else 'CampusMind AI'}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


col_copy, col_regen = st.columns([1, 5])
with col_copy:
    last_ai = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "assistant"), "")
    if last_ai:
        st.download_button("Copy", data=last_ai, file_name="campusmind-answer.txt", use_container_width=True)
with col_regen:
    if st.session_state.last_prompt and st.button("Regenerate", use_container_width=False):
        st.session_state.messages = [m for m in st.session_state.messages if m["role"] != "assistant"]
        prompt = st.session_state.last_prompt
        with st.chat_message("assistant"):
            placeholder = st.empty()
            text = ""
            for token in stream_answer(prompt):
                text += token
                placeholder.markdown(text + "▌")
            placeholder.markdown(text)
        st.session_state.messages.append({"role": "assistant", "content": text})
        st.rerun()


prompt = st.chat_input("नेपाली वा Roman Nepali मा सन्देश लेख्नुहोस्")
if prompt:
    st.session_state.last_prompt = prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(
        f"""
        <div class="chat-row user">
          <div>
            <div class="bubble user">{escape(prompt)}</div>
            <div class="meta">You</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    placeholder = st.empty()
    answer = ""
    try:
        for token in stream_answer(prompt):
            answer += token
            placeholder.markdown(
                f"""
                <div class="chat-row assistant">
                  <div>
                    <div class="bubble assistant">{escape(answer)}▌</div>
                    <div class="meta">CampusMind AI is typing...</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        placeholder.markdown(
            f"""
            <div class="chat-row assistant">
              <div>
                <div class="bubble assistant">{escape(answer)}</div>
                <div class="meta">CampusMind AI</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as exc:
        st.error(f"Chat failed: {exc}")
