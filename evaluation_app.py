import pandas as pd
import streamlit as st

import evaluate
import memory
import rag

st.set_page_config(page_title="My Bank | Evaluation Dashboard", page_icon="📊", layout="wide")
memory.init_db()


@st.cache_resource(show_spinner="Loading the knowledge base...")
def load_kb():
    return rag.ensure_index()


kb = load_kb()

st.title("My Bank Chatbot — Evaluation Dashboard")
st.caption(f"Model: {rag.MODEL}  ·  Knowledge base: {kb['chunks']} chunks from "
           f"{', '.join(kb['files']) or 'no files'}  ·  Distance cut-off: {rag.MAX_DISTANCE}")

tab_opinion, tab_auto, tab_report = st.tabs(["User opinion", "Automatic test", "Report"])

# ================================================================== USER OPINION
with tab_opinion:
    form_col, stats_col = st.columns([2, 3], gap="large")

    with form_col:
        st.subheader("Tester survey")
        st.caption("Ask each tester to try the chatbot, then rate their experience here.")
        with st.form("survey", clear_on_submit=True):
            ease = st.select_slider("Ease of use", options=[1, 2, 3, 4, 5], value=4)
            acc = st.select_slider("Accuracy of answers", options=[1, 2, 3, 4, 5], value=4)
            speed = st.select_slider("Response speed", options=[1, 2, 3, 4, 5], value=4)
            overall = st.select_slider("Overall satisfaction", options=[1, 2, 3, 4, 5], value=4)
            comment = st.text_area("Comments or suggestions (optional)")
            if st.form_submit_button("Submit"):
                memory.add_survey(ease, acc, speed, overall, comment)
                st.success("Thank you. The response was saved.")

    opinion = evaluate.user_opinion()          # read after the form so numbers are fresh
    fb, sv = opinion["feedback"], opinion["survey"]

    with stats_col:
        st.subheader("Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Answers rated", f"{fb['rated']} / {fb['answers']}")
        c2.metric("Helpful (👍)", f"{fb['helpful_pct']}%" if fb["helpful_pct"] is not None else "—")
        c3.metric("Survey rating", f"{sv['overall']} / 5" if sv["n"] else "—")
        c4.metric("Avg. response", f"{opinion['avg_latency']} s" if opinion["avg_latency"] else "—")

        if sv["n"]:
            st.caption(f"Survey averages from {sv['n']} response(s), scale 1–5")
            st.bar_chart(pd.DataFrame(
                {"Average": [sv["ease"], sv["accuracy"], sv["speed"], sv["overall"]]},
                index=["Ease of use", "Accuracy", "Speed", "Overall"]), y_label="Rating (1–5)")
        else:
            st.info("No survey responses yet.")

        if opinion["disliked"]:
            with st.expander(f"Questions that received 👎 ({len(opinion['disliked'])})"):
                for d in opinion["disliked"]:
                    st.write("• " + d["question"])
        if sv["comments"]:
            with st.expander("Tester comments"):
                for c in sv["comments"]:
                    st.write(f"“{c['comment']}” — {c['overall']}/5")

# ================================================================== AUTOMATIC TEST
with tab_auto:
    st.subheader("Automatic test")
    st.caption("Runs a fixed set of questions and checks retrieval, answer accuracy, out-of-scope handling, "
               "the safety guard and conversation memory. The built-in test set matches the sample handbook; "
               "for another PDF use `python evaluate.py --testset my_tests.json`.")
    mode = st.radio("Mode", ["Full (uses the LLM)", "Retrieval only (fast, no LLM)"], horizontal=True)

    if st.button("Run automatic test", type="primary"):
        full = mode.startswith("Full")
        if full and not rag.has_api_key():
            st.error("GROQ_API_KEY is missing. Add it to `.env`, or choose retrieval-only mode.")
        else:
            bar = st.progress(0.0, text="Starting...")
            st.session_state.auto = evaluate.run_auto_eval(
                use_llm=full,
                progress=lambda i, n, q: bar.progress(i / n, text=f"[{i}/{n}] {q[:70]}"),
            )
            bar.empty()

    auto = st.session_state.get("auto")
    if auto:
        s = auto["summary"]
        pct = lambda v: f"{v}%" if v is not None else "—"
        m = st.columns(6)
        m[0].metric("Overall", pct(s["overall_pct"]), f"{s['passed']}/{s['total']} passed", delta_color="off")
        m[1].metric("Retrieval hit", pct(s["retrieval_hit_pct"]))
        m[2].metric("Answer accuracy", pct(s["answer_accuracy_pct"]))
        m[3].metric("Out-of-scope", pct(s["out_of_scope_pct"]))
        m[4].metric("Safety guard", pct(s["safety_pct"]))
        m[5].metric("Memory", pct(s["memory_pct"]))
        if s["errors"]:
            st.warning(f"{s['errors']} test(s) ended with an error (see the Note column).")

        df = pd.DataFrame(auto["rows"])
        df["Result"] = df["passed"].map({True: "✅ Pass", False: "❌ Fail"})
        st.dataframe(
            df.rename(columns={"type": "Type", "question": "Question", "top_distance": "Top distance",
                               "latency": "Time (s)", "note": "Note"})[
                ["Result", "Type", "Question", "Top distance", "Time (s)", "Note"]],
            hide_index=True)

        with st.expander("Model answers"):
            for r in auto["rows"]:
                if r["answer"]:
                    st.markdown(f"**{r['question']}**")
                    st.caption(r["answer"])
    else:
        st.info("Press **Run automatic test** to start.")

# ================================================================== REPORT
with tab_report:
    st.subheader("Evaluation report")
    auto = st.session_state.get("auto")
    if not auto:
        st.info("Run the automatic test first. The report combines its results with the user opinion data.")
    else:
        report = evaluate.build_report(auto, evaluate.user_opinion(), kb)
        st.download_button("Download report (.md)", report, file_name="evaluation_report.md",
                           mime="text/markdown", type="primary")
        st.markdown(report)
