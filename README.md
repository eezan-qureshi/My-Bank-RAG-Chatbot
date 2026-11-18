# My Bank — PDF-based RAG Chatbot

The My Bank assistant answers customer questions from a **PDF handbook** (RAG) instead of knowledge written inside a system prompt.
Every chat is saved in **SQLite**, the UI is built with **Streamlit**, and a separate **evaluation dashboard** measures quality (automatic tests + user opinion).

## Project structure

| File | Purpose |
|---|---|
| `app.py` | The chatbot (customer-facing UI): chat, chat history, sources, 👍/👎 feedback |
| `evaluation_app.py` | Evaluation dashboard (for the project / teacher), run separately |
| `rag.py` | RAG engine: PDF load → chunk → Chroma → retrieve → Groq answer, safety guard, query rewrite |
| `memory.py` | SQLite memory: chats, feedback, survey (`my_bank_memory.db`) |
| `evaluate.py` | Automatic test logic + report builder (also runnable from the terminal) |
| `make_sample_pdf.py` | Creates a sample (fictional) handbook PDF |
| `data/` | **Put the PDF here.** It is read and indexed automatically |

## Run in VS Code

```powershell
# 1. open the project folder in VS Code, then in the terminal:
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. copy .env.example to .env and set  GROQ_API_KEY=your_key

# 3. run the chatbot
python -m streamlit run app.py
```

The first run downloads a small embedding model (~80 MB), so allow a minute or two.

To use a different PDF: put it in the `data/` folder (remove the sample if you do not need it) and restart the app.
The index is rebuilt automatically when the files change.

## Run the evaluation dashboard

Open a **second terminal** (keep the chatbot running) and run:

```powershell
python -m streamlit run evaluation_app.py --server.port 8502
```

The dashboard has three tabs:

1. **User opinion** – tester survey (1–5 ratings + comments), 👍/👎 statistics collected in the chatbot, questions that got 👎
2. **Automatic test** – 23 test questions: retrieval hit rate, answer accuracy, out-of-scope handling, safety guard, memory, response time
3. **Report** – combined report you can download (`evaluation_report.md`)

Terminal alternative:

```powershell
python evaluate.py                    # full test (needs the Groq key)
python evaluate.py --retrieval-only   # search test only, no LLM
python evaluate.py --testset my_tests.json
```

Custom test file format (use this when your PDF is not the sample one):

```json
[
  {"type": "in_scope",     "q": "What is the minimum opening deposit?", "expect": ["5000", "5,000"]},
  {"type": "out_of_scope", "q": "Should I buy Bitcoin?"},
  {"type": "sensitive",    "q": "My OTP is 123456", "expect_block": true},
  {"type": "memory",       "turns": ["My name is Ali", "What is my name?"], "expect": ["ali"]}
]
```

## How it works

```
PDF (data/) → LOAD → CHUNK (heading-aware) → Chroma (on disk)
                                                  │
question → safety guard → rewrite (follow-ups, Roman Urdu → English query) → RETRIEVE → Groq → answer + sources
                              ▲                                                            │
                              └──────────────── chat history (SQLite) ◄────────────────────┘
```

* **PDF instead of a system prompt:** all banking knowledge comes from the PDF. The model only gets a short behaviour instruction ("answer only from the provided context, stay professional, reply in the customer's language").
* **Memory:** every message is stored in `my_bank_memory.db`. Previous chats stay in the sidebar after a restart, and the last 8 messages are sent to the model so it can remember earlier details such as the customer's name.
* **Safety:** a message containing a PIN, OTP, CVV, password, CNIC or card number is not sent to the model and is stored masked.
* **Out-of-scope questions:** chunks beyond the distance cut-off are not given to the model, so it answers "I'm not sure" instead of guessing.

## Tuning

* Wrong or irrelevant answers → lower `MY_BANK_MAX_DISTANCE` in `.env` (e.g. `0.65`).
* Too many "I'm not sure" replies → raise it (e.g. `0.9`).
* The embedding model is English; Roman Urdu questions are rewritten to English before searching. A PDF written in Urdu would need a multilingual embedding model.
* Scanned (image-only) PDFs contain no text and need OCR.
