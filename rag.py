"""
rag.py - My Bank RAG engine.

Pipeline
  1. LOAD      data/ folder ki PDFs (aur .txt) padho, source + page yaad rakho
  2. CHUNK     heading-aware chunks (har chunk ke sath uski heading)
  3. STORE     Chroma (disk par, to restart par dobara embed nahi karna parta)
  4. RETRIEVE  question ke closest chunks (+ distance cut-off)
  5. ANSWER    Groq model sirf un chunks + chat history se jawab deta hai

Extra:
  - Safety guard: PIN/OTP/CVV/CNIC/card number detect karke mask kar deta hai
  - Query rewrite: follow-up ("aur uski fee?") aur Roman Urdu sawal ko
    standalone English search query bana deta hai (PDF English mein ho to bhi kaam kare)
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------ settings
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
DATA_DIR = Path(os.getenv("MY_BANK_DATA", "data"))
CHROMA_DIR = Path(os.getenv("MY_BANK_CHROMA", "chroma_db"))
COLLECTION = "my_bank_kb"

MAX_CHARS = 500          # chunk ka maximum size
N_RESULTS = 4            # model ko kitne chunks dene hain
MAX_DISTANCE = float(os.getenv("MY_BANK_MAX_DISTANCE", "0.8"))  # cosine distance cut-off
USE_REWRITE = os.getenv("MY_BANK_REWRITE", "1") == "1"
INDEX_VERSION = f"v2-{MAX_CHARS}"   # chunking badle to index khud dobara banega

SYSTEM_PROMPT = """You are the friendly virtual assistant of My Bank (a fictional bank).

- For banking facts (fees, limits, steps, timings, products) use ONLY the CONTEXT passages given with the question.
- If the CONTEXT does not contain the answer, say you are not sure and suggest contacting the bank helpline or a branch (quote the helpline number only if it appears in the CONTEXT). Never invent numbers, fees or rates.
- You may use the conversation history (for example, remember the customer's name or earlier questions).
- Never ask for or accept a PIN, OTP, CVV, password, full card number or CNIC. Bank staff never ask for these.
- You cannot see real accounts, balances or transactions.
- No legal or investment advice. If the question is not about banking, politely say you can only help with banking topics.
- Language: reply in English if the customer writes English. If the customer writes Roman Urdu (Urdu typed in English letters), reply in Roman Urdu using English letters, never in Urdu script. Use Urdu script only if the customer writes in Urdu script.
- Tone: polite, clear and professional.
- Keep answers short: at most 5 sentences, or a short numbered list for steps."""

REWRITE_PROMPT = """You turn a customer's last message into ONE standalone English search query for a bank FAQ handbook.
- Use the chat history to resolve words like "it", "that", "uski", "second one".
- Translate Urdu / Roman Urdu into English.
- If the message is just a greeting or personal chit-chat, return it unchanged.
- Output ONLY the query, nothing else."""

SAFETY_MESSAGE = (
    "For your security, please do not share your PIN, OTP, CVV, password, CNIC or full card number here. "
    "Bank staff will never ask you for these details. "
    "Your message has been masked and its sensitive part was not stored."
)

# ------------------------------------------------------------------ safety guard
_SENSITIVE = [
    ("CNIC", re.compile(r"\b\d{5}-\d{7}-\d\b")),
    ("card / account number", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("OTP / PIN / CVV / password",
     re.compile(r"\b(otp|pin|cvv|cvc|password|passcode)\b[^\d\n]{0,15}\d{3,8}\b", re.I)),
    ("OTP / PIN / CVV / password",
     re.compile(r"\b\d{4,8}\b[^\d\n]{0,15}\b(otp|pin|cvv|cvc)\b", re.I)),
]


def find_sensitive(text):
    """Return list of what looks sensitive in `text` (empty list = safe)."""
    found = []
    for label, pattern in _SENSITIVE:
        if pattern.search(text) and label not in found:
            found.append(label)
    return found


def mask_sensitive(text):
    """Sensitive hisson ko **** se badal do (database mein masked text hi jata hai)."""
    for _, pattern in _SENSITIVE:
        text = pattern.sub("[hidden]", text)
    return text


# ------------------------------------------------------------------ 1. LOAD
def _read_pdf(path):
    from pypdf import PdfReader

    pages = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        pages.append({"text": page.extract_text() or "", "source": path.name, "page": number})
    return pages


def _read_txt(path):
    return [{"text": path.read_text(encoding="utf-8", errors="ignore"), "source": path.name, "page": 1}]


def list_files(folder=None):
    folder = Path(folder or DATA_DIR)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in (".pdf", ".txt"))


def load_pages(folder=None):
    pages = []
    for path in list_files(folder):
        pages += _read_pdf(path) if path.suffix.lower() == ".pdf" else _read_txt(path)
    return pages


# ------------------------------------------------------------------ 2. CHUNK
_BULLETS = "•●▪-–*"


def _is_heading(line):
    s = line.strip()
    return (
        0 < len(s) <= 60
        and len(s.split()) <= 8
        and s[-1] not in ".!?:,;"
        and not s[0].islower()
        and s[0] not in _BULLETS
    )


def _sections(text, heading=""):
    """Page text ko [(heading, body)] sections mein todo. Aakhri heading wapas bhi do."""
    sections, buf = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            buf.append("\x00")                       # khali line = sentence boundary
        elif _is_heading(line):
            body = " ".join(buf).strip("\x00 ")
            if body:
                sections.append((heading, body))
            buf, heading = [], line
        elif line[0] in _BULLETS:
            buf.append("\x00" + line.lstrip(_BULLETS + " "))   # bullet = nayi sentence
        else:
            buf.append(line)
    body = " ".join(buf).strip("\x00 ")
    if body:
        sections.append((heading, body))
    return sections, heading


def chunk_section(heading, body, max_chars=MAX_CHARS):
    """Poore sentences ko pack karo (1 sentence overlap) aur har chunk par heading lagao."""
    sentences = [" ".join(s.split()) for s in re.split(r"\x00|(?<=[.!?])\s+", body)]
    sentences = [s for s in sentences if s]
    prefix = f"{heading}: " if heading else ""
    budget = max(120, max_chars - len(prefix))
    chunks, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > budget:
            chunks.append(prefix + " ".join(current))
            last = current[-1]
            current = [last] if len(last) < budget // 3 else []     # overlap
        current.append(sentence)
    if current and (not chunks or " ".join(current) != chunks[-1][len(prefix):]):
        chunks.append(prefix + " ".join(current))
    return chunks


def build_chunks(folder=None, max_chars=MAX_CHARS):
    chunks, skipped, heading, last_source = [], [], "", None
    for page in load_pages(folder):
        if page["source"] != last_source:
            heading, last_source = "", page["source"]
        if not page["text"].strip():
            skipped.append(f'{page["source"]} page {page["page"]}')
            continue
        sections, heading = _sections(page["text"], heading)
        for head, body in sections:
            for piece in chunk_section(head, body, max_chars):
                chunks.append({"text": piece, "source": page["source"], "page": page["page"]})
    return chunks, skipped


# ------------------------------------------------------------------ 3. STORE
_cache = {}


def _embedder():
    if "embed" not in _cache:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        _cache["embed"] = DefaultEmbeddingFunction()
    return _cache["embed"]


def _client():
    if "client" not in _cache:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _cache["client"] = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _cache["client"]


def _create_collection():
    client = _client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    try:
        return client.create_collection(
            COLLECTION, embedding_function=_embedder(), configuration={"hnsw": {"space": "cosine"}}
        )
    except TypeError:                                   # purana chromadb
        return client.create_collection(
            COLLECTION, embedding_function=_embedder(), metadata={"hnsw:space": "cosine"}
        )


def _fingerprint(files):
    parts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {"version": INDEX_VERSION, "files": parts}


def _manifest_path():
    return CHROMA_DIR / "manifest.json"


def ensure_index(force=False):
    """PDFs badli hon (ya nayi add hon) tabhi dobara index banao, warna purana use karo."""
    files = list_files()
    fp = _fingerprint(files)
    old = None
    if _manifest_path().exists():
        try:
            old = json.loads(_manifest_path().read_text())
        except Exception:
            old = None

    if not force and old and old.get("fingerprint") == fp:
        try:
            col = _client().get_collection(COLLECTION, embedding_function=_embedder())
            if col.count() > 0:
                return {"rebuilt": False, "chunks": col.count(), "files": [p.name for p in files],
                        "skipped": old.get("skipped", [])}
        except Exception:
            pass                                         # collection ghayab - dobara banao

    chunks, skipped = build_chunks()
    col = _create_collection()
    if chunks:
        col.upsert(
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
            ids=[f'{Path(c["source"]).stem}-p{c["page"]}-{i}' for i, c in enumerate(chunks)],
        )
    _manifest_path().write_text(json.dumps({"fingerprint": fp, "skipped": skipped}))
    return {"rebuilt": True, "chunks": len(chunks), "files": [p.name for p in files], "skipped": skipped}


# ------------------------------------------------------------------ 4. RETRIEVE
def retrieve(query, n=N_RESULTS, max_distance=None):
    """n sab se qareeb chunks. max_distance se door wale chunks nikal do."""
    col = _client().get_collection(COLLECTION, embedding_function=_embedder())
    if col.count() == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(n, col.count()))
    found = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        found.append({"text": text, "source": meta["source"], "page": meta["page"], "distance": round(dist, 3)})
    if max_distance is not None:
        found = [f for f in found if f["distance"] <= max_distance]
    return found


# ------------------------------------------------------------------ 5. ANSWER
def has_api_key():
    return bool(os.getenv("GROQ_API_KEY"))


def _chat(messages, max_tokens=700):
    if "groq" not in _cache:
        from groq import Groq

        _cache["groq"] = Groq(api_key=os.environ["GROQ_API_KEY"])
    kwargs = {}
    if MODEL.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"               # chhote jawab, kam token
    resp = _cache["groq"].chat.completions.create(
        model=MODEL, messages=messages, temperature=0, max_tokens=max_tokens, **kwargs
    )
    return (resp.choices[0].message.content or "").strip()


def rewrite_query(question, history):
    """Follow-up / Roman Urdu sawal -> standalone English search query. Fail ho to original."""
    if not USE_REWRITE:
        return question
    recent = "\n".join(f'{m["role"]}: {m["content"][:300]}' for m in history[-4:])
    try:
        out = _chat(
            [
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": f"Chat history:\n{recent or '(none)'}\n\nLast message: {question}"},
            ],
            max_tokens=300,
        )
        return out.splitlines()[0].strip() if out else question
    except Exception:
        return question


def answer(question, history=None, n=N_RESULTS):
    """
    Poora RAG step. Return:
      {"answer", "sources", "query", "latency", "blocked"}
    `history` = [{"role","content"}, ...] (pichli baatein, naya sawal shamil nahi)
    """
    history = history or []
    start = time.perf_counter()

    if find_sensitive(question):
        return {"answer": SAFETY_MESSAGE, "sources": [], "query": question,
                "latency": round(time.perf_counter() - start, 2), "blocked": True}

    query = rewrite_query(question, history)
    sources = retrieve(query, n=n, max_distance=MAX_DISTANCE)
    context = "\n\n".join(f'[{s["source"]}, page {s["page"]}]\n{s["text"]}' for s in sources) \
        or "(no relevant passage found in the handbook)"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-8:] + [
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nCUSTOMER QUESTION: {question}"}
    ]
    reply = _chat(messages) or "Maaf kijiye, abhi jawab nahi ban saka. Dobara koshish karein."
    return {"answer": reply, "sources": sources, "query": query,
            "latency": round(time.perf_counter() - start, 2), "blocked": False}
