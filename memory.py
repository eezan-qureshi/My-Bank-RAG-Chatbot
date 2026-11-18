"""
memory.py - My Bank ki permanent memory (SQLite).

Yeh file chat history, user feedback (👍/👎) aur survey ko ek local database
file (my_bank_memory.db) mein save karti hai. App band karke dobara kholo to
purani chats wapas mil jati hain.
"""
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.getenv("MY_BANK_DB", "my_bank_memory.db")


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _db() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                latency REAL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                message_id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                helpful INTEGER NOT NULL,          -- 1 = 👍, 0 = 👎
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS survey (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ease INTEGER, accuracy INTEGER, speed INTEGER, overall INTEGER,
                comment TEXT,
                created_at TEXT NOT NULL
            );
            """
        )


# ----------------------------------------------------------------- sessions
def new_session():
    sid = uuid.uuid4().hex[:12]
    with _db() as c:
        c.execute("INSERT INTO sessions (id, created_at) VALUES (?, ?)", (sid, _now()))
    return sid


def list_sessions(limit=30):
    """Sirf woh chats jin mein kam az kam ek message ho, naye pehle."""
    with _db() as c:
        rows = c.execute(
            """SELECT s.id, s.title, s.created_at FROM sessions s
               WHERE EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id)
               ORDER BY s.created_at DESC, s.rowid DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def set_title(session_id, title):
    with _db() as c:
        c.execute("UPDATE sessions SET title = ? WHERE id = ?", (title[:45], session_id))


def delete_session(session_id):
    with _db() as c:
        c.execute("DELETE FROM feedback WHERE session_id = ?", (session_id,))
        c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


# ----------------------------------------------------------------- messages
def add_message(session_id, role, content, sources=None, latency=None):
    with _db() as c:
        cur = c.execute(
            "INSERT INTO messages (session_id, role, content, sources, latency, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, json.dumps(sources) if sources else None, latency, _now()),
        )
        return cur.lastrowid


def get_messages(session_id):
    with _db() as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d["sources"]) if d["sources"] else []
        out.append(d)
    return out


def get_history(session_id, last_n=8):
    """Model ko dene ke liye: sirf role + content, aakhri last_n messages."""
    msgs = get_messages(session_id)[-last_n:]
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


# ----------------------------------------------------------------- feedback
def save_feedback(message_id, session_id, helpful):
    with _db() as c:
        c.execute(
            "INSERT OR REPLACE INTO feedback (message_id, session_id, helpful, created_at) "
            "VALUES (?, ?, ?, ?)",
            (message_id, session_id, int(helpful), _now()),
        )


def get_feedback_map(session_id):
    with _db() as c:
        rows = c.execute(
            "SELECT message_id, helpful FROM feedback WHERE session_id = ?", (session_id,)
        ).fetchall()
    return {r["message_id"]: r["helpful"] for r in rows}


def feedback_stats():
    with _db() as c:
        row = c.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(helpful), 0) AS up FROM feedback"
        ).fetchone()
        answers = c.execute("SELECT COUNT(*) FROM messages WHERE role='assistant'").fetchone()[0]
    total, up = row["total"], row["up"]
    return {
        "answers": answers,
        "rated": total,
        "up": up,
        "down": total - up,
        "helpful_pct": round(100 * up / total, 1) if total else None,
    }


def disliked_questions(limit=10):
    """Woh sawal jin ke jawab par 👎 mila (improvement ke liye)."""
    with _db() as c:
        rows = c.execute(
            """SELECT q.content AS question, a.content AS answer
               FROM feedback f
               JOIN messages a ON a.id = f.message_id
               JOIN messages q ON q.id = (SELECT MAX(id) FROM messages
                                          WHERE session_id = a.session_id AND role='user' AND id < a.id)
               WHERE f.helpful = 0 ORDER BY f.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def avg_latency():
    with _db() as c:
        v = c.execute("SELECT AVG(latency) FROM messages WHERE role='assistant' AND latency IS NOT NULL").fetchone()[0]
    return round(v, 2) if v else None


# ----------------------------------------------------------------- survey
def add_survey(ease, accuracy, speed, overall, comment=""):
    with _db() as c:
        c.execute(
            "INSERT INTO survey (ease, accuracy, speed, overall, comment, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ease, accuracy, speed, overall, comment.strip(), _now()),
        )


def survey_stats():
    with _db() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n, AVG(ease) AS ease, AVG(accuracy) AS accuracy, "
            "AVG(speed) AS speed, AVG(overall) AS overall FROM survey"
        ).fetchone()
        comments = c.execute(
            "SELECT comment, overall, created_at FROM survey WHERE comment != '' "
            "ORDER BY id DESC LIMIT 10"
        ).fetchall()
    r = lambda v: round(v, 2) if v is not None else None
    return {
        "n": row["n"],
        "ease": r(row["ease"]),
        "accuracy": r(row["accuracy"]),
        "speed": r(row["speed"]),
        "overall": r(row["overall"]),
        "comments": [dict(c) for c in comments],
    }
