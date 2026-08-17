# NER Anonymization Agent — Locations & Dates

Individual project for **ML Summer Camp 2026** (UNIDATA LAB), mentor: Anastasiia.

Fine-tunes a BERT-based model to detect and anonymize **locations** and **dates** in text, wraps it in a LangChain agent alongside a RAG tool for answering questions about the student, and exposes both through MCP, Docker, and a Telegram bot.

## Architecture

```
Kaggle (GPU, training)          Hugging Face Hub           Local (PyCharm / Docker)
─────────────────────           ─────────────────           ─────────────────────────
fine-tune bert-base-cased  ───▶  hosted model weights  ───▶  Tool 1: anonymize_text
on tner/ontonotes5                                           Tool 2: answer_about_student (RAG)
                                                                        │
                                                               MCP server (mcp_server.py)
                                                                        │
                                                        ┌───────────────┴───────────────┐
                                                   agent.py (CLI)              telegram_bot.py
```

## Step 1 — Fine-tuning

- **Dataset:** [`tner/ontonotes5`](https://huggingface.co/datasets/tner/ontonotes5) — a flat version of OntoNotes 5.0 with ready-made train/validation/test splits (59,924 / 8,528 / 8,262 examples).
- **Labels:** collapsed from the original 37 (18 entity types × B-/I- + O) down to 5: `O`, `B-LOC`, `I-LOC`, `B-DATE`, `I-DATE`. Both `GPE` (cities/countries) and `LOC` (mountains, rivers, etc.) map to `LOC`.
- **Split:** used the dataset's **official** train/val/test splits rather than a custom one — the flat format has no document ID, so a self-made split couldn't guarantee no leakage between splits (sentences from the same source document ending up in both train and test).
- **Base model:** `bert-base-cased` (cased specifically — capitalization is a strong NER signal).
- **Metric:** entity-level F1 (via `seqeval`), not raw token accuracy. Accuracy is misleadingly high on NER because the `O` class dominates (see [Findings](#findings--design-decisions) below).
- **Hyperparameter experiments:** compared `learning_rate` 2e-5 vs 1e-5 over 3 epochs. Both converge to the same F1 ceiling (~0.895–0.896); 2e-5 reaches it faster (epoch 1 vs epoch 3). Also observed mild overfitting past epoch 1 (validation loss creeping up while training loss kept falling) — kept the epoch-1 checkpoint as final.

**Final test set results:**

| Metric    | Value |
|-----------|-------|
| Precision | 0.897 |
| Recall    | 0.909 |
| F1        | 0.903 |
| Accuracy  | 0.993 |

Well above the 80% threshold from the spec (agreed with mentor to use entity-F1 instead of raw accuracy as the target metric).

Model published publicly: [`marianaY/ner-loc-date-anonymizer`](https://huggingface.co/marianaY/ner-loc-date-anonymizer).

Training notebooks (require a GPU — run on Kaggle): `notebooks/01-dataset-exploration.ipynb`, `notebooks/02-training.ipynb`.

## Step 2 — Agent & Tools

Built with LangChain (`create_tool_calling_agent` + `AgentExecutor`).

- **Tool 1 — `anonymize_text`** (`src/anonymize.py`): wraps the fine-tuned model via a `transformers` pipeline (`aggregation_strategy="simple"` merges subword-level predictions back into whole entities). Replaces found entities right-to-left to avoid shifting character offsets of entities not yet processed.
- **Tool 2 — `answer_about_student`** (`src/rag.py`): RAG over a short personal facts file (`src/student_facts.txt`). Chunked by paragraph, embedded with OpenAI (`text-embedding-3-small`), stored in a local FAISS index, answered with `gpt-4o-mini` (temperature=0). Prompt explicitly forbids inferring facts not present in the retrieved context (e.g. calculating age from a birth date) — falls back to a polite "I don't have access to that information."

## Step 3 — Deployment

- **MCP:** `src/mcp_server.py` exposes both tools over the Model Context Protocol (`FastMCP`). `agent.py` and `telegram_bot.py` connect to it as MCP clients over stdio instead of importing the tools directly.
- **Docker:** `Dockerfile` containerizes the agent (`ENV PYTHONUTF8=1` works around a locale-related `UnicodeEncodeError` that only shows up inside the minimal `python:3.9-slim` image, not locally on macOS).
- **Telegram bot:** `src/telegram_bot.py`, built on `python-telegram-bot`. Opens a fresh MCP connection per incoming message (simpler and more robust than trying to keep one long-lived connection alive across the bot's per-message async tasks).

## Testing

`tests/` — `pytest` tests for both tools (correct redaction, no false positives, correct RAG answers on known facts, hallucination guardrail on unknown ones).

```
pytest tests/ -v
```

## Findings & Design Decisions

- **Entity-level F1 vs. accuracy:** token accuracy on the test set is 0.993 — deceptively high, since most tokens are simply `O`. F1 (0.903) is the metric that actually reflects how well the model finds entities.
- **Dataset annotation noise:** found a real mislabeled example in OntoNotes 5.0 itself ("Salvador" tagged `I-DATE`) — a reminder that even standard NER benchmarks contain some label noise.
- **Capitalization sensitivity:** the cased model relies heavily on capitalization. Lowercase "poland" in a sentence gets only partially anonymized ("[LOC]oland") because the tokenizer splits it into subwords the model doesn't fully recognize as a continuing entity — a direct, documented tradeoff of choosing a cased model.
- **RAG chunk quality:** a short, single-sentence fact ("...enjoys singing...") was consistently outranked by longer, richer chunks in similarity search, even when it was the correct answer. Fixed by enriching the chunk's wording rather than just increasing `k`.
- **Domain shift:** the model reliably detects locations in formal, news-style phrasing ("the president of Poland", "the United States of America" — confidence >0.99), but consistently misses the same countries (Poland, Germany, Canada) in casual sentences like "X traveled to Y" — regardless of a nearby date. Likely cause: OntoNotes 5.0 is a news corpus, where this everyday travel-narrative phrasing is underrepresented relative to formal/political phrasing.

## Setup & Usage

```bash
git clone https://github.com/YuvchenkoMariana/ner-anonymization.git
cd ner-anonymization
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-key-here
TELEGRAM_BOT_TOKEN=your-token-here
```

**Run the CLI agent:**
```bash
python src/agent.py
```

**Run via Docker:**
```bash
docker build -t ner-anonymization-agent .
docker run -it --env-file .env ner-anonymization-agent
```

**Run the Telegram bot:**
```bash
python src/telegram_bot.py
```

**Run tests:**
```bash
pytest tests/ -v
```

## Project structure

```
ner-anonymization/
├── notebooks/
│   ├── 01-dataset-exploration.ipynb   # dataset selection, preprocessing (Kaggle)
│   └── 02-training.ipynb              # model training, hyperparameter experiments (Kaggle)
├── src/
│   ├── anonymize.py       # Tool 1: NER-based anonymization
│   ├── rag.py              # Tool 2: RAG over student facts
│   ├── student_facts.txt   # source facts for Tool 2
│   ├── agent.py             # LangChain agent, CLI entry point, MCP client
│   ├── mcp_server.py        # MCP server exposing both tools
│   └── telegram_bot.py      # Telegram bot on top of the MCP-connected agent
├── tests/
│   ├── conftest.py
│   ├── test_anonymize.py
│   └── test_rag.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── pytest.ini
```
