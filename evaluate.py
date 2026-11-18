import argparse
import json
import re
import time
from datetime import datetime

import memory
import rag

# Yeh test set data/my_bank_handbook.pdf (sample PDF) ke liye hai.
# Apni PDF ho to --testset se apne sawal dein (format README mein hai).
TEST_SET = [
    # ---- in_scope: jawab PDF mein hai. expect = in mein se koi ek lafz jawab/chunk mein hona chahiye
    {"type": "in_scope", "q": "What is the minimum deposit to open a Savings Account?", "expect": ["5000"]},
    {"type": "in_scope", "q": "Mera card kho gaya hai, ab kya karun?", "expect": ["block", "0800-00000"]},
    {"type": "in_scope", "q": "What is the daily ATM withdrawal limit of the Classic card?", "expect": ["50000"]},
    {"type": "in_scope", "q": "Dusre bank ke ATM se paise nikalne ki fee kitni hai?", "expect": ["25"]},
    {"type": "in_scope", "q": "What is the daily limit for transfers to other banks?", "expect": ["200000"]},
    {"type": "in_scope", "q": "Account kholne ke liye kaun se documents chahiye?", "expect": ["cnic"]},
    {"type": "in_scope", "q": "Is the branch open on Saturday?", "expect": ["1 pm", "1pm", "1 baje", "13:00"]},
    {"type": "in_scope", "q": "What is the maximum personal loan amount?", "expect": ["3000000", "3 million", "30 lakh"]},
    {"type": "in_scope", "q": "Account dormant kab hota hai?", "expect": ["12 months", "12 mahine", "12 maah", "one year", "1 year", "aik saal"]},
    {"type": "in_scope", "q": "How long does it take to resolve a complaint?", "expect": ["10 working", "10 days", "10 din", "ten"]},
    {"type": "in_scope", "q": "Replacement card ki fee kya hai?", "expect": ["500"]},
    {"type": "in_scope", "q": "How much does a paper statement cost?", "expect": ["200"]},
    {"type": "in_scope", "q": "App ka passcode kitne digit ka hota hai?", "expect": ["6-digit", "6 digit", "6 hindsay", "six", "6 digits"]},
    {"type": "in_scope", "q": "What is the annual fee of the Platinum card?", "expect": ["3000"]},
    # ---- out_of_scope: PDF mein nahi / banking nahi -> chatbot ko inkar karna chahiye
    {"type": "out_of_scope", "q": "Who is the CEO of My Bank?"},
    {"type": "out_of_scope", "q": "Aaj Lahore ka mausam kaisa hai?"},
    {"type": "out_of_scope", "q": "Kya mujhe Bitcoin mein paisa lagana chahiye?"},
    {"type": "out_of_scope", "q": "What is the capital of France?"},
    # ---- sensitive: guard ko roknaa chahiye (expect_block True) ya nahi roknaa (False = control)
    {"type": "sensitive", "q": "Mera OTP 483920 hai, check kar do", "expect_block": True},
    {"type": "sensitive", "q": "My CNIC is 42101-1234567-1, open an account", "expect_block": True},
    {"type": "sensitive", "q": "Card number 4111 1111 1111 1111 hai, block kar do", "expect_block": True},
    {"type": "sensitive", "q": "Mujhe apna card block karna hai", "expect_block": False},
    # ---- memory: do turns, aakhri jawab mein naam aana chahiye
    {"type": "memory", "turns": ["Mera naam Ahmed hai", "Mera naam kya hai?"], "expect": ["ahmed"]},
]

REFUSAL_MARKERS = [
    "not sure", "don't know", "do not know", "could not find", "couldn't find", "cannot find",
    "not mentioned", "no information", "nahi mila", "nahi mili", "maloom nahi", "maloomat nahi",
    "pakka nahi", "pakki", "helpline", "branch", "only help", "can only", "sirf banking",
    "banking topics", "banking se", "not related", "not able to",
]


# ------------------------------------------------------------------ helpers
def norm(text):
    """Lowercase + numbers se comma hatao ('5,000' -> '5000') taake match asaan ho."""
    text = text.lower()
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def contains_any(text, options):
    t = norm(text)
    return any(norm(o) in t for o in options)


def _row(item, **kw):
    base = {"type": item["type"], "question": item.get("q") or " -> ".join(item.get("turns", [])),
            "passed": False, "retrieval_hit": None, "answer_ok": None, "top_distance": None,
            "latency": None, "answer": "", "note": ""}
    base.update(kw)
    return base


# ------------------------------------------------------------------ A) automatic test
def run_auto_eval(testset=None, use_llm=True, progress=None):
    testset = testset or TEST_SET
    rows = []
    for i, item in enumerate(testset, start=1):
        if progress:
            progress(i, len(testset), item.get("q") or item["turns"][-1])
        kind = item["type"]
        try:
            if kind == "sensitive":
                blocked = bool(rag.find_sensitive(item["q"]))
                rows.append(_row(item, passed=blocked == item["expect_block"],
                                 answer="(blocked by guard)" if blocked else "(allowed)",
                                 note=f"expected block={item['expect_block']}, got {blocked}"))

            elif kind == "in_scope":
                if use_llm:
                    res = rag.answer(item["q"], [])
                    sources, answer, latency = res["sources"], res["answer"], res["latency"]
                else:
                    sources = rag.retrieve(item["q"], max_distance=rag.MAX_DISTANCE)
                    answer, latency = "", None
                hit = contains_any(" ".join(s["text"] for s in sources), item["expect"])
                ok = contains_any(answer, item["expect"]) if use_llm else None
                rows.append(_row(item, retrieval_hit=hit, answer_ok=ok,
                                 passed=hit and (ok if use_llm else True),
                                 top_distance=sources[0]["distance"] if sources else None,
                                 latency=latency, answer=answer,
                                 note=f"expected one of {item['expect']}"))

            elif kind == "out_of_scope":
                if use_llm:
                    res = rag.answer(item["q"], [])
                    refused = contains_any(res["answer"], REFUSAL_MARKERS)
                    rows.append(_row(item, passed=refused, answer=res["answer"], latency=res["latency"],
                                     top_distance=res["sources"][0]["distance"] if res["sources"] else None,
                                     note=f"relevant chunks passed to model: {len(res['sources'])}"))
                else:
                    sources = rag.retrieve(item["q"], max_distance=rag.MAX_DISTANCE)
                    rows.append(_row(item, passed=len(sources) == 0,
                                     top_distance=sources[0]["distance"] if sources else None,
                                     note=f"{len(sources)} chunk(s) wrongly considered relevant"))

            elif kind == "memory":
                if not use_llm:
                    rows.append(_row(item, passed=False, note="skipped (needs LLM)"))
                    continue
                history, res = [], None
                for turn in item["turns"]:
                    res = rag.answer(turn, history)
                    history += [{"role": "user", "content": turn},
                                {"role": "assistant", "content": res["answer"]}]
                rows.append(_row(item, passed=contains_any(res["answer"], item["expect"]),
                                 answer=res["answer"], latency=res["latency"],
                                 note=f"expected one of {item['expect']}"))
        except Exception as e:                                   # network / API error
            rows.append(_row(item, note=f"ERROR: {type(e).__name__}: {str(e)[:120]}"))

    return {"rows": rows, "summary": summarize(rows, use_llm), "use_llm": use_llm}


def _pct(num, den):
    return round(100 * num / den, 1) if den else None


def summarize(rows, use_llm):
    by = lambda t: [r for r in rows if r["type"] == t and not r["note"].startswith("skipped")]
    ins, oos, sens, mem = by("in_scope"), by("out_of_scope"), by("sensitive"), by("memory")
    lat = [r["latency"] for r in rows if r["latency"]]
    counted = [r for r in rows if not r["note"].startswith("skipped")]
    return {
        "total": len(counted),
        "passed": sum(r["passed"] for r in counted),
        "overall_pct": _pct(sum(r["passed"] for r in counted), len(counted)),
        "retrieval_hit_pct": _pct(sum(bool(r["retrieval_hit"]) for r in ins), len(ins)),
        "answer_accuracy_pct": _pct(sum(bool(r["answer_ok"]) for r in ins), len(ins)) if use_llm else None,
        "out_of_scope_pct": _pct(sum(r["passed"] for r in oos), len(oos)),
        "safety_pct": _pct(sum(r["passed"] for r in sens), len(sens)),
        "memory_pct": _pct(sum(r["passed"] for r in mem), len(mem)) if use_llm else None,
        "avg_latency": round(sum(lat) / len(lat), 2) if lat else None,
        "errors": sum(r["note"].startswith("ERROR") for r in rows),
    }


# ------------------------------------------------------------------ B) user opinion
def user_opinion():
    return {
        "feedback": memory.feedback_stats(),
        "survey": memory.survey_stats(),
        "disliked": memory.disliked_questions(),
        "avg_latency": memory.avg_latency(),
    }


# ------------------------------------------------------------------ report
def _fmt(v, suffix=""):
    return "n/a" if v is None else f"{v}{suffix}"


def build_report(auto, opinion, index_info=None):
    s = auto["summary"]
    fb, sv = opinion["feedback"], opinion["survey"]
    L = [f"# My Bank RAG Chatbot - Evaluation Report",
         f"_Generated: {datetime.now():%Y-%m-%d %H:%M}_  |  Model: `{rag.MODEL}`  |  "
         f"Distance cut-off: {rag.MAX_DISTANCE}  |  Mode: {'full (LLM)' if auto['use_llm'] else 'retrieval-only'}"]
    if index_info:
        L.append(f"Knowledge base: {index_info['chunks']} chunks from {', '.join(index_info['files']) or 'no files'}")

    L += ["", "## A. Automatic test results", "",
          "| Metric | Result |", "|---|---|",
          f"| Overall pass rate | {_fmt(s['overall_pct'], '%')} ({s['passed']}/{s['total']}) |",
          f"| Retrieval hit rate (right chunk found) | {_fmt(s['retrieval_hit_pct'], '%')} |",
          f"| Answer accuracy (right fact in answer) | {_fmt(s['answer_accuracy_pct'], '%')} |",
          f"| Out-of-scope handling (says \"not sure\") | {_fmt(s['out_of_scope_pct'], '%')} |",
          f"| Safety guard (PIN/OTP/CNIC blocked) | {_fmt(s['safety_pct'], '%')} |",
          f"| Memory (remembers earlier message) | {_fmt(s['memory_pct'], '%')} |",
          f"| Average response time | {_fmt(s['avg_latency'], ' s')} |",
          f"| Errors during test | {s['errors']} |", "",
          "### Per-question details", "",
          "| # | Type | Question | Result | Top distance | Note |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(auto["rows"], 1):
        q = r["question"].replace("|", "/")[:70]
        L.append(f"| {i} | {r['type']} | {q} | {'PASS' if r['passed'] else 'FAIL'} | "
                 f"{_fmt(r['top_distance'])} | {r['note'][:60].replace('|', '/')} |")

    L += ["", "## B. User opinion", "",
          "### Per-answer feedback (👍 / 👎)", "",
          f"- Answers given: {fb['answers']}  |  rated by users: {fb['rated']}",
          f"- Helpful (👍): {fb['up']}  |  Not helpful (👎): {fb['down']}  |  "
          f"**Helpful rate: {_fmt(fb['helpful_pct'], '%')}**", "",
          "### Survey (1-5)", ""]
    if sv["n"]:
        L += [f"- Responses: {sv['n']}", f"- Ease of use: {sv['ease']}", f"- Answer accuracy: {sv['accuracy']}",
              f"- Speed: {sv['speed']}", f"- **Overall satisfaction: {sv['overall']} / 5**"]
        if sv["comments"]:
            L += ["", "User comments:"] + [f"- \"{c['comment']}\" ({c['overall']}/5)" for c in sv["comments"]]
    else:
        L.append("- No survey responses yet.")
    if opinion["disliked"]:
        L += ["", "### Questions that got 👎", ""] + [f"- {d['question']}" for d in opinion["disliked"]]

    L += ["", "## C. Conclusion", ""] + conclusion(s, opinion)
    return "\n".join(L) + "\n"


def conclusion(s, opinion):
    pts = []
    if s["retrieval_hit_pct"] is not None:
        if s["retrieval_hit_pct"] >= 85:
            pts.append(f"Retrieval is strong ({s['retrieval_hit_pct']}%): the right chunk is usually found.")
        else:
            pts.append(f"Retrieval needs work ({s['retrieval_hit_pct']}%): try smaller/larger chunks, "
                       f"higher N_RESULTS or a looser distance cut-off.")
    if s["answer_accuracy_pct"] is not None:
        pts.append(f"Answer accuracy is {s['answer_accuracy_pct']}% on in-scope questions.")
    if s["out_of_scope_pct"] is not None:
        pts.append(f"Out-of-scope questions were handled correctly {s['out_of_scope_pct']}% of the time"
                   + (" (hallucination risk is low)." if s["out_of_scope_pct"] >= 75 else " - the bot may be guessing; tighten the distance cut-off."))
    if s["safety_pct"] is not None:
        pts.append(f"Safety guard passed {s['safety_pct']}% of the sensitive-data tests.")
    fb = opinion["feedback"]
    if fb["helpful_pct"] is not None:
        pts.append(f"Real users found {fb['helpful_pct']}% of rated answers helpful.")
    if opinion["survey"]["n"]:
        pts.append(f"Average overall satisfaction from the survey: {opinion['survey']['overall']}/5.")
    if not pts:
        pts.append("Not enough data yet - run the tests and collect some user feedback.")
    return [f"- {p}" for p in pts]


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description="Evaluate the My Bank RAG chatbot")
    ap.add_argument("--retrieval-only", action="store_true", help="LLM/API key ke baghair sirf search test")
    ap.add_argument("--testset", help="JSON file with your own test questions")
    ap.add_argument("--out", default="evaluation_report.md")
    args = ap.parse_args()

    use_llm = not args.retrieval_only
    if use_llm and not rag.has_api_key():
        raise SystemExit("GROQ_API_KEY nahi mili (.env file dekhein) - ya --retrieval-only chalayein.")

    testset = json.load(open(args.testset, encoding="utf-8")) if args.testset else None
    memory.init_db()
    info = rag.ensure_index()
    print(f"Knowledge base ready: {info['chunks']} chunks. Running tests...\n")

    auto = run_auto_eval(testset, use_llm, progress=lambda i, n, q: print(f"[{i}/{n}] {q[:70]}"))
    report = build_report(auto, user_opinion(), info)
    open(args.out, "w", encoding="utf-8").write(report)

    print("\n" + json.dumps(auto["summary"], indent=2))
    print(f"\nReport saved to {args.out}")


if __name__ == "__main__":
    main()
