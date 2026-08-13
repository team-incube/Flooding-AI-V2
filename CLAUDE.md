# Flooding-AI-V2

## Overview

AI chatbot + music-recommendation backend for the school website **"Flooding"**.
FastAPI service; the entrypoint is `app.main:app`.

The chatbot currently answers questions about how to use the site via a RAG pipeline.
The music feature turns a user's recent song requests into YouTube recommendations.

**Tech stack:** Python ≥3.12, FastAPI, Uvicorn · LangChain 1.x (Core / Community) + LangGraph · LangChain-Chroma (vector DB) · OpenAI (`langchain-openai`) for the chat LLM and embeddings · `langchain-tavily` for the graph's `web_search` branch (a top-level route destination, not a RAG-spine fallback — see `docs/v2-architecture.md`) · Spotipy (Spotify API) + google-api-python-client (YouTube API) for music · `ragas` / `datasets` for RAG evaluation (present in deps, eval skill in progress).
Package manager: **`uv`** (`.venv`, `uv.lock` are committed).

> **Status — the LangGraph rewrite is live.**
> `POST /ai/chat` (`app/api/chat_API.py`) now calls `app.langgraph_services.graph.ask`,
> **not** `app/services/chatbot.py` — the old `create_agent` version is kept only for
> the RAGAS before/after baseline (`run-ragas-eval --target chatbot`), it is no
> longer in the serving path. The live graph is `app/langgraph_services/graph.py`
> (nodes in `nodes.py`, routing in `routing.py`, prompts in `prompts.py`); its full
> node/edge spec, including the booking sub-graph (apply-only — study room /
> massage chair / wake-up music; **no cancellation, no confirmation step**), lives
> in **`docs/v2-architecture.md`** — keep that doc in sync when nodes/edges change.
>
> Note: the RAG spine's retry logic has **two separate, independently-capped retry
> counters** (retrieval-side vs. answer-side) — see `docs/v2-architecture.md`
> for the exact loop shape before touching `grade_documents` / `grade_answer`.
>
> Booking calls the dev API at `https://dev.flooding.kr` with the `Authorization`
> header `chat_API.py` receives on `/ai/chat`, forwarded through `GraphState.auth_token`
> — never logged. See `docs/v2-architecture.md`'s "Booking sub-graph" section.

## Development Commands

```bash
# Local dev (inside the uv environment)
uv run uvicorn app.main:app --reload

# Docker (serves on port 8000)
docker build -t flooding-ai .
docker run -p 8000:8000 --env-file app/services/.env flooding-ai
```

- There is **no project test suite yet** (only third-party tests inside `.venv`). Do not invent a `pytest` command that passes green — there is nothing to run.
- After changing RAG source docs you must rebuild the vector DB — use the `rebuild-vectordb` skill (see harness) or see "Cautions".

## Architecture

Layered around FastAPI: **API layer → Service layer**. Keep business logic out of the API modules.

```
app/
├── main.py            # FastAPI app, router registration
├── schemas.py         # Pydantic models (UserInput, ChatResponse, SongRequest, ...)
├── api/
│   ├── chat_API.py    # POST /ai/chat   (thin: parse/validate -> service)
│   └── music_API.py   # POST /ai/song   (thin: parse/validate -> service)
└── services/
    ├── chatbot.py     # RAG chatbot (LangChain create_agent + search_document tool)
    ├── embedding.py   # Chroma build/load, RAG chunk shaping
    └── music_chain.py # recent songs -> LLM analysis -> Spotify search -> 3 YouTube links
data/
├── flooding_rag.json         # RAG source (Q&A, procedures, principles)
└── Flooding_RAG용_문서.pdf   # original reference doc
```

- `api/*` modules stay thin: validate input, delegate to a service, shape the response. No LLM/DB logic here.
- `services/*` owns all business logic (LLM calls, retrieval, external APIs).
- Pydantic models in `schemas.py` are the layer boundary — pass typed models across it, not raw dicts.

### API endpoints

- `POST /ai/chat` — `{ user_input: str }` → `{ response: str }`. Empty string → **400**.
- `POST /ai/song` — `{ recent_songs: [{ title, artist }] }` (max 5) → `{ youtube_links: [str] }` (**exactly 3**). If 3 links cannot be produced → **502**.

### RAG pipeline (`embedding.py`)

- `data/flooding_rag.json` → per-chunk shaping (branches by type: Q&A / procedure / principle / ...) → OpenAI embeddings (`text-embedding-3-small`, local cache in `./cache/`) → stored in Chroma (`./chroma_db/`).
- Retriever: **MMR** with `k=7, fetch_k=20, lambda_mult=0.8`.

### How the LangGraph chatbot is structured

`route_question` classifies every turn into one of **four** top-level destinations —
`vectorstore` (RAG spine), `booking` (apply-only sub-graph), `general_chat`, or
`web_search` (Tavily, for non-school factual questions) — stored in `state["mode"]`.
The RAG spine (`retrieve` → `grade_documents` → `generate`) is self-corrective with
its own bounded re-search loop; `web_search` is a sibling top-level branch, **not**
a fallback reached from inside the RAG spine (a vectorstore miss answers "don't
know" rather than escaping to the open web). All four destinations converge on a
shared `grade_answer` gate before `END`. Retrieval failures and answer failures are
corrected by **two distinct, capped retry loops** (`retrieve_retry_count`,
`answer_retry_count`) that never share a counter — see `docs/v2-architecture.md`
for the exact node/edge spec, routing rules, and retry caps; that is the single
source of truth for the graph. Keep the existing RAGAS metrics comparable so
before/after quality can be measured on the same question set.

## Core Principles

- **Single responsibility:** one reason to change per class/module; keep API modules thin and services focused.
- **Boundaries via types:** cross the API↔service boundary with Pydantic models from `schemas.py`.
- **Interface-based seams** where a component is likely to be swapped (e.g. retriever, LLM client) so the LangGraph re-architecture stays localized.
- **Tests (aspirational):** there is no suite today; when adding one, prioritize covering service-layer behavior before large refactors.

## Coding Conventions

- Follow the harness rules and skills under `.claude/` (see below).
- **Git branches:** `main` (release) ← `develop` (integration) ← `feature/*` / `feat/*` (work). PRs created via the `write-pr` skill auto-detect the base branch from this rule.

## `.claude/` harness

- **Hooks** (`.claude/hooks/*.py`, Python): block dangerous commands, prevent secret exposure, log commands — run automatically on every action.
- **Skills** (`.claude/skills/*/SKILL.md`), invoked when relevant:
  - `write-pr`, `git-commit`, `systematic-debugging` — general workflow.
  - `rebuild-vectordb` — safely rebuild the Chroma DB after editing `flooding_rag.json`. Runs `scripts/`: validate JSON → (confirm) delete `./chroma_db` → rebuild via `get_retriever()` (keeps `./cache/` so unchanged chunks skip re-embedding) → verify with sample queries. Exists because Chroma is **not** auto-rebuilt (see Cautions).
  - `run-ragas-eval` — *(in progress)* RAGAS quality scoring for before/after comparison across the re-architecture.
- **Language convention:** hooks are Python (they parse JSON from stdin); skill scripts like `create-pr.sh` are Bash (git/gh shell calls only). Rule of thumb is "right tool for the job," not one-language-for-everything.

## Cautions

- **`.env` lives at `app/services/.env`, not the repo root.** All three services (`chatbot.py`, `embedding.py`, `music_chain.py`) load it via `os.path.dirname(__file__)/.env`.
- **Chroma is not auto-rebuilt.** If `./chroma_db` already exists, changes to `flooding_rag.json` are ignored. After editing RAG docs, use the `rebuild-vectordb` skill (or delete `./chroma_db/` and re-run) to reflect them.
- **Spotify creds are required for music.** Without `SPOTIPY_CLIENT_ID` / `SPOTIPY_CLIENT_SECRET`, `/ai/song` returns empty results.
- **Tavily key is optional but recommended for `web_search`.** Without `TAVILY_API_KEY`, the `web_search` node safely falls back to an empty context (no crash) and `generate` answers "don't know" instead of fabricating.
- **Booking has no confirmation step.** `extract_booking_slot` runs `execute_booking` immediately once an apply intent (and slot, if any) is resolved — there is no "are you sure?" turn. This is deliberate (a text-only chat UI can't do a button confirmation); the trade-off is that a misclassified apply intent can only be undone through the site's own cancel UI, not through the chatbot. See `docs/v2-architecture.md`'s "Booking sub-graph" section.
- **Booking is excluded from the answer-retry loop.** `grade_answer` still runs for `mode == "booking"` (for observability), but `route_after_grade_answer` sends it straight to `END` regardless of pass/fail — never back to `execute_booking`. `execute_booking` calls a real, side-effecting apply API; retrying it on a failed grade would risk a duplicate application.
- **Dockerfile ↔ pyproject mismatch (resolved 2026-08-13):** `httpx` was missing from the Dockerfile's `pip install` list and has been added; `langchain-tavily` was already present. All other deps in `pyproject.toml` now have a matching entry **except `ragas` and `datasets`**, which remain intentionally excluded (not used in the serving path — keep in mind if evaluation code ever moves into the runtime image). Also fixed: every `pkg>=version` spec in that `RUN pip install` line is now double-quoted — unquoted, the shell parses `>` as a redirect operator (e.g. `fastapi>=0.135.1` → runs `fastapi` with stdout redirected to a file named `=0.135.1`), silently dropping version pins and littering `/app` with junk files. Added `.dockerignore` (excludes `.venv/`, `.git/`, `cache/`, `.claude/`, `docs/`, `.env*`) — without it the build context included the local `.venv`, making `docker build` take 15+ minutes just to transfer context; **`.dockerignore` intentionally keeps `chroma_db/`** since `app/langgraph_services/nodes.py` builds the retriever at import time and `get_retriever()` calls OpenAI embeddings immediately if `./chroma_db` is missing — omitting it would make the server fail to start without a real `OPENAI_API_KEY`. **Verified 2026-08-13**: `docker build -t flooding-ai .` completes clean, and `docker run -p 8000:8000 --env-file app/services/.env flooding-ai` boots Uvicorn with no import errors (`/docs` → 200) even with a placeholder `.env`. Note: the Docker Hub image `baeyongbin/flooding-ai:latest` had been stale since 2026-04-27 (predated the LangGraph rewrite entirely) — re-pushed 2026-08-13 with the fixes above; a `2026-08-13` dated tag was also pushed alongside `latest` as a rollback point.

## Environment Variables (`app/services/.env`)

- `OPENAI_API_KEY` — chat LLM, embeddings, and music LLM all need it.
- `YOUTUBE_API_KEY` — YouTube link search.
- `SPOTIPY_CLIENT_ID`, `SPOTIPY_CLIENT_SECRET` — Spotify search.
- `TAVILY_API_KEY` — `web_search` node in the LangGraph chatbot; optional (safe fallback if unset).
- `ALLOWED_ORIGINS` — comma-separated CORS allowlist for `app/main.py`'s `CORSMiddleware`. Optional; defaults to `https://flooding.kr,http://localhost:3000` if unset. Must include every frontend origin that calls `/ai/chat` / `/ai/song` with credentials (e.g. add `https://prod.flooding.kr` if the frontend itself is served from there instead of `https://flooding.kr`).

Booking does **not** read a `BOOKING_API_BASE_URL` env var — the dev API base
(`https://dev.flooding.kr`) is a hardcoded constant in
`app/langgraph_services/nodes.py`. Booking actions also need a per-request user
**accessToken**: `chat_API.py` forwards the `Authorization` header it receives on
`/ai/chat` into `GraphState.auth_token`, and `execute_booking` sends it as a
`Bearer` token to the dev API; without it, booking replies "로그인이 필요합니다."
instead of attempting the call.