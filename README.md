# NER Anonymization Agent — Locations & Dates

Individual project for **ML Summer Camp 2026** (UNIDATA LAB), mentor: Anastasiia.

Fine-tunes BERT-based models (English and Ukrainian) to detect and anonymize **locations** and **dates** in text, wraps them in a LangChain agent alongside a RAG tool for answering questions about the student, and exposes all three tools through MCP — backed by an independent FastAPI microservice architecture, Docker, and a Telegram bot.

## Architecture

```
Kaggle (GPU, training)                 Hugging Face Hub                    Docker (docker-compose)
─────────────────────                  ─────────────────                    ─────────────────────────
fine-tune bert-base-cased        ───▶   marianaY/ner-loc-date-       ───▶   ner-service (FastAPI)
on tner/ontonotes5 (English)            anonymizer                          POST /anonymize/en, /anonymize/uk
                                                                                       │
fine-tune xlm-roberta-base       ───▶   marianaY/ner-loc-date-       ───▶            │
on Goader/ner-uk-2.0 (Ukrainian)        anonymizer-uk                                │
                                                                              rag-service (FastAPI)
                                                                              POST /answer
                                                                                       │
                                                                  both called over HTTP from
                                                                                       │
                                                                       mcp_server.py (MCP server)
                                                                                       │
                                                                    ┌──────────────────┴──────────────────┐
                                                               agent.py (CLI)                    telegram_bot.py
```

MCP is the **required**, outward-facing protocol — `agent.py` and `telegram_bot.py` only ever talk to `mcp_server.py` over MCP, unchanged regardless of what's running underneath. The FastAPI microservices are a **bonus** layer sitting *under* `mcp_server.py`: each tool call is now an HTTP request to an independent, separately-deployable service instead of in-process inference.

## Step 1 — Fine-tuning

### English

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

### Ukrainian (bonus)

- **Dataset:** [`Goader/ner-uk-2.0`](https://huggingface.co/datasets/Goader/ner-uk-2.0) (NER-UK 2.0) — chosen over the simpler `benjamin/ner-uk` because it includes a `DATE` type, not just PER/ORG/LOC/MISC.
- **Labels:** same 5-class scheme as English. `DATE` also absorbs `PERIOD` (time ranges, e.g. "2014-2015 роки") — per the official NER-UK 2.0 annotation guidelines, both are calendar-time information sensitive for anonymization. (Caught and fixed a real bug here: the label-remapping function originally mapped `PERIOD` onto `DATE` but never handled the dataset's *native* `DATE` tag, silently dropping real dates from training — fixed before training.)
- **Base model:** `xlm-roberta-base` — multilingual, with far more pretraining data covering Ukrainian than narrow Ukrainian-only models trained on smaller corpora.
- **Experiments:** 3 epochs → F1 0.798; 6 epochs → F1 0.788 (plateaued, no clear overfitting yet — unlike English, likely because the training set is ~5x smaller: 11k vs 60k examples). A higher-learning-rate run hit a GPU `OutOfMemoryError`; decided not to keep chasing it since this is a bonus feature and the result was already close to the 80% target.

**Final validation results (6-epoch run):**

| Metric    | Value |
|-----------|-------|
| F1        | 0.788 |

Below the 80% target, but reasoned and documented as acceptable for a bonus, given the training set size relative to English.

Model published publicly: [`marianaY/ner-loc-date-anonymizer-uk`](https://huggingface.co/marianaY/ner-loc-date-anonymizer-uk).

Training notebook: `notebooks/03-ukrainian-dataset-exploration.ipynb`.

## Step 2 — Agent & Tools

Built with LangChain (`create_tool_calling_agent` + `AgentExecutor`).

- **Tool 1 — `anonymize_text`** (`src/anonymize.py`): wraps the fine-tuned English model via a `transformers` pipeline (`aggregation_strategy="simple"` merges subword-level predictions back into whole entities). Replaces found entities right-to-left to avoid shifting character offsets of entities not yet processed.
- **Tool 2 — `answer_about_student`** (`src/rag.py`): RAG over a short personal facts file (`src/student_facts.txt`). Chunked by paragraph, embedded with OpenAI (`text-embedding-3-small`), stored in a local FAISS index, answered with `gpt-4o-mini` (temperature=0). Prompt explicitly forbids inferring facts not present in the retrieved context (e.g. calculating age from a birth date) — falls back to a polite "I don't have access to that information."
- **Tool 3 — `anonymize_text_uk`** (`src/anonymize_uk.py`, bonus): same pattern as Tool 1, using the Ukrainian model. Adds a `merge_overlapping_entities` step before replacement to fix a real bug found in inflected Ukrainian word forms (see [Findings](#findings--design-decisions)). Added with **zero changes** to `agent.py` or `telegram_bot.py` — MCP tools are auto-discovered from the server, so a new tool is genuinely "just another arrow."

## Step 3 — Deployment

- **MCP (required):** `src/mcp_server.py` exposes all three tools over the Model Context Protocol (`FastMCP`). `agent.py` and `telegram_bot.py` connect to it as MCP clients over stdio.
- **Microservices (bonus):** `mcp_server.py`'s tools no longer run inference in-process — each one makes an HTTP request to an independent FastAPI service instead:
  - `src/ner_service.py` — `POST /anonymize/en`, `POST /anonymize/uk` (both language models)
  - `src/rag_service.py` — `POST /answer`
  
  Each service has its own `Dockerfile` (`Dockerfile.ner`, `Dockerfile.rag`), and `docker-compose.yml` networks them together with the agent's own container (`Dockerfile`) so they can reach each other by service name (`http://ner-service:8001`) instead of `localhost`. Service URLs are read from environment variables (`NER_SERVICE_URL`, `RAG_SERVICE_URL`) with `localhost` defaults, so the same code works standalone or via compose.
- **Docker:** all three Dockerfiles use `python:3.11-slim` (upgraded from `3.9-slim` — the `mcp` package requires Python ≥3.10). `ENV PYTHONUTF8=1` works around a locale-related `UnicodeEncodeError` that only shows up inside the minimal image, not locally on macOS — the same fix was needed again for Cyrillic text passing through the MCP subprocess.
- **Telegram bot:** `src/telegram_bot.py`, built on `python-telegram-bot`. Opens a fresh MCP connection per incoming message (simpler and more robust than trying to keep one long-lived connection alive across the bot's per-message async tasks).
- **Load testing (bonus):** `locustfile.py` simulates concurrent users against the NER and RAG HTTP endpoints — see [Findings](#findings--design-decisions) for what it revealed about relative latency.
- **Logging:** `mcp_server.py` and `telegram_bot.py` log every tool call, its duration, and any errors to `logs/*.log` (file only, kept out of the console so the interactive CLI and LangChain's `verbose=True` trace stay readable).

## Testing

`tests/` — `pytest` tests for all three tools (correct redaction in both languages, no false positives, correct RAG answers on known facts, hallucination guardrail on unknown ones). The Ukrainian test suite includes a characterization test that documents a known model limitation rather than hiding it, so a future retraining shows up as a visible test change instead of going unnoticed.

```
pytest tests/ -v
```

## Findings & Design Decisions

- **Entity-level F1 vs. accuracy:** token accuracy on the English test set is 0.993 — deceptively high, since most tokens are simply `O`. F1 (0.903) is the metric that actually reflects how well the model finds entities.
- **Dataset annotation noise:** found a real mislabeled example in OntoNotes 5.0 itself ("Salvador" tagged `I-DATE`) — a reminder that even standard NER benchmarks contain some label noise.
- **Capitalization sensitivity:** the cased English model relies heavily on capitalization. Lowercase "poland" in a sentence gets only partially anonymized ("[LOC]oland") because the tokenizer splits it into subwords the model doesn't fully recognize as a continuing entity — a direct, documented tradeoff of choosing a cased model.
- **RAG chunk quality:** a short, single-sentence fact ("...enjoys singing...") was consistently outranked by longer, richer chunks in similarity search, even when it was the correct answer. Fixed by enriching the chunk's wording rather than just increasing `k`.
- **Domain shift (English):** the model reliably detects locations in formal, news-style phrasing ("the president of Poland", "the United States of America" — confidence >0.99), but consistently misses the same countries (Poland, Germany, Canada) in casual sentences like "X traveled to Y" — regardless of a nearby date. Likely cause: OntoNotes 5.0 is a news corpus, where this everyday travel-narrative phrasing is underrepresented relative to formal/political phrasing.
- **Overlapping entity spans (Ukrainian):** inflected word forms broke the naive right-to-left replacement — e.g. "Варшаві" (locative case of Варшава) was split by SentencePiece into overlapping/touching subword predictions, producing a garbled `[LOC]ршав[LOC]` instead of one clean `[LOC]`. Fixed with a `merge_overlapping_entities` helper that merges touching/overlapping same-type spans before replacing.
- **Multi-word date phrases (Ukrainian):** "минулого тижня" ("last week") is only partially redacted — "минулого" is tagged `DATE`, but "тижня" isn't recognized as a continuation. Likely due to the ~5x smaller Ukrainian training set (11k vs 60k examples) underrepresenting this specific phrasing, the same underlying issue as the English domain-shift finding, just attributable to dataset size rather than style.
- **NER vs. RAG latency (load testing):** Locust load tests show the NER endpoints responding in ~60–130ms (local model inference) versus the RAG endpoint at ~950–3400ms (depends on external OpenAI API round-trips). Splitting the backend into microservices made this bottleneck directly visible and measurable — harder to isolate in a monolithic design.

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

**Run the CLI agent (talks to mcp_server.py, which calls services at localhost):**
```bash
python src/agent.py
```

**Run the full microservices stack via Docker Compose:**
```bash
docker-compose up -d ner-service rag-service   # backend services, detached
docker-compose run --rm agent                  # interactive CLI agent
```

**Run a single container (agent only, in-process — no separate services):**
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

**Run load tests** (make sure `ner-service`/`rag-service` are running first, e.g. via `docker-compose up -d ner-service rag-service`):
```bash
locust -f locustfile.py
```
Opens a web UI at `http://localhost:8089` to configure simulated users and watch live stats.

## Project structure

```
ner-anonymization/
├── notebooks/
│   ├── 01-dataset-exploration.ipynb           # English dataset selection, preprocessing (Kaggle)
│   ├── 02-training.ipynb                      # English model training, hyperparameter experiments (Kaggle)
│   └── 03-ukrainian-dataset-exploration.ipynb # Ukrainian dataset, preprocessing, training (Kaggle)
├── src/
│   ├── anonymize.py         # Tool 1: English NER-based anonymization
│   ├── anonymize_uk.py      # Tool 3: Ukrainian NER-based anonymization (bonus)
│   ├── rag.py                # Tool 2: RAG over student facts
│   ├── student_facts.txt     # source facts for Tool 2
│   ├── agent.py               # LangChain agent, CLI entry point, MCP client
│   ├── mcp_server.py          # MCP server — tools delegate to microservices over HTTP
│   ├── telegram_bot.py        # Telegram bot on top of the MCP-connected agent
│   ├── ner_service.py         # FastAPI microservice wrapping Tools 1 & 3 (bonus)
│   └── rag_service.py         # FastAPI microservice wrapping Tool 2 (bonus)
├── tests/
│   ├── conftest.py
│   ├── test_anonymize.py
│   ├── test_anonymize_uk.py
│   └── test_rag.py
├── logs/                    # runtime tool-call/error logs (gitignored)
├── locustfile.py            # load tests for the microservices (bonus)
├── Dockerfile                # agent container
├── Dockerfile.ner            # NER microservice container (bonus)
├── Dockerfile.rag            # RAG microservice container (bonus)
├── docker-compose.yml        # networks agent + both microservices (bonus)
├── .dockerignore
├── requirements.txt
└── pytest.ini
```
