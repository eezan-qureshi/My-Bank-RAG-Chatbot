"""
app.py - My Bank virtual assistant (customer-facing chat UI).

Run:  streamlit run app.py

Knowledge comes from the PDF(s) in the data/ folder (loaded automatically).
Every chat is stored in SQLite (memory.py), so history survives restarts.
Evaluation lives in a separate dashboard: evaluation_app.py
"""
import streamlit as st

import memory
import rag

st.set_page_config(page_title="My Bank | Virtual Assistant", page_icon="🏦", layout="centered")
memory.init_db()

st.markdown(
    """<style>
    [data-testid="stSidebar"] .stButton button {width: 100%; text-align: left;}
    [data-testid="stToolbar"], [data-testid="stDecoration"], footer {visibility: hidden; height: 0;}
    .block-container {padding-top: 2rem; max-width: 820px;}
    </style>""",
    unsafe_allow_html=True,
)

if not rag.has_api_key():
    st.error("GROQ_API_KEY is missing. Create a `.env` file (copy of `.env.example`), add your key "
             "as `GROQ_API_KEY=...`, and restart the app.")
    st.stop()


@st.cache_resource(show_spinner="Preparing the knowledge base (the first run can take a minute)...")
def load_kb():
    return rag.ensure_index()


kb = load_kb()

if "session_id" not in st.session_state:
    st.session_state.session_id = memory.new_session()


# ------------------------------------------------------------------ callbacks
def new_chat():
    st.session_state.session_id = memory.new_session()


def open_chat(session_id):
    st.session_state.session_id = session_id


def remove_chat(session_id):
    memory.delete_session(session_id)
    if st.session_state.session_id == session_id:
        st.session_state.session_id = memory.new_session()


def rate(message_id, session_id, value):
    memory.save_feedback(message_id, session_id, value)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🏦 My Bank")
    st.caption("Virtual Assistant")
    st.button("＋ New chat", on_click=new_chat)

    st.markdown("**Chat history**")
    sessions = memory.list_sessions()
    if not sessions:
        st.caption("No previous chats yet.")
    for s in sessions:
        c1, c2 = st.columns([5, 1])
        is_current = s["id"] == st.session_state.session_id
        c1.button(s["title"], key=f"open_{s['id']}", on_click=open_chat, args=(s["id"],),
                  type="primary" if is_current else "secondary", help=s["created_at"])
        c2.button("🗑", key=f"del_{s['id']}", on_click=remove_chat, args=(s["id"],), help="Delete chat")

    st.divider()
    st.caption("Answers are based on the official My Bank handbook. The assistant cannot access your "
               "account. Never share your PIN, OTP, CVV, password or CNIC in this chat.")

# ------------------------------------------------------------------ chat
sid = st.session_state.session_id
messages = memory.get_messages(sid)
feedback_map = memory.get_feedback_map(sid)

st.title("My Bank Assistant")
st.caption("Your virtual banking assistant. Ask about accounts, cards, transfers, loans and more.")

if kb["chunks"] == 0:
    st.warning("No documents were found. Add a PDF to the `data` folder and restart the app.")

for m in messages:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🏦"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m["sources"]:
                with st.expander(f"Sources ({len(m['sources'])})"):
                    for src in m["sources"]:
                        st.markdown(f"**{src['source']}**, page {src['page']}")
                        st.caption(src["text"])
            current = feedback_map.get(m["id"])
            b1, b2, b3 = st.columns([1, 1, 10])
            b1.button("👍", key=f"up_{m['id']}", on_click=rate, args=(m["id"], sid, 1),
                      type="primary" if current == 1 else "secondary", help="Helpful")
            b2.button("👎", key=f"down_{m['id']}", on_click=rate, args=(m["id"], sid, 0),
                      type="primary" if current == 0 else "secondary", help="Not helpful")
            if current is not None:
                b3.caption("Thanks for your feedback.")

if not messages:
    st.markdown("**Try asking:**")
    examples = ["What is the minimum deposit for a Savings Account?",
                "What should I do if my card is lost?",
                "How do I register for the mobile app?"]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            st.session_state.pending_q = ex
            st.rerun()

prompt = st.chat_input("Type your question here...") or st.session_state.pop("pending_q", None)

if prompt:
    blocked = bool(rag.find_sensitive(prompt))
    shown = rag.mask_sensitive(prompt) if blocked else prompt
    history = memory.get_history(sid)                      # earlier messages (without the new question)

    with st.chat_message("user", avatar="🧑"):
        st.markdown(shown)
    with st.chat_message("assistant", avatar="🏦"):
        try:
            with st.spinner("Searching the handbook..."):
                result = rag.answer(prompt, history)
        except Exception as e:
            st.error("Sorry, I could not get a response. Please check your internet connection and "
                     "API key, then try again.")
            st.caption(f"{type(e).__name__}: {str(e)[:200]}")
            st.stop()

    if not messages:
        memory.set_title(sid, shown)
    memory.add_message(sid, "user", shown)                  # sensitive input is stored masked
    memory.add_message(sid, "assistant", result["answer"], sources=result["sources"],
                       latency=result["latency"])
    st.rerun()

st.caption("The assistant can make mistakes. For urgent matters, please contact the bank helpline.")
