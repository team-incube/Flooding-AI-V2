# Flooding-AI-V2

## Overview

AI chatbot + music-recommendation backend for the school website **"Flooding"**.
FastAPI service; the entrypoint is `app.main:app`.

The chatbot currently answers questions about how to use the site via a RAG pipeline.
The music feature turns a user's recent song requests into YouTube recommendations.

**Tech stack:** Python ≥3.12, FastAPI, Uvicorn · LangChain 1.x (Core / Community) + LangGraph · LangChain-Chroma (vector DB) · OpenAI (`langchain-openai`) for the chat LLM and embeddings · `langchain-tavily` for the RAG spine's web-search fallback · Spotipy (Spotify API) + google-api-python-client (YouTube API) for music · `ragas` / `datasets` for RAG evaluation (present in deps, eval skill in progress).
Package manager: **`uv`** (`.venv`, `uv.lock` are committed).

<!-- ─────────────────────────────────────────────────────────────
     CURRENT-WORK STATUS  ·  swap this whole block when the graph lands
     ───────────────────────────────────────────────────────────── -->
> **Status — read before editing `services/chatbot.py`.**
> The chatbot is being **re-architected onto LangGraph** — this is a re-architecture, not a like-for-like port: it adds a web-search fallback, booking actions, and an Adaptive / self-corrective RAG loop. Target design: **`docs/v2-architecture.md`**.
>
> As of this writing the graph **does not exist yet** — `services/chatbot.py` is still the LangChain `create_agent` version (implicit agent loop + `search_document` tool). This work is starting now, so treat `docs/v2-architecture.md` as the spec to build against. Once the graph is in place, delete this block and describe the live graph structure here instead.
>
> Note: the RAG spine's retry logic has **two separate, independently-capped retry
> counters** (retrieval-side vs. answer-side) — see `docs/v2-architecture.md`
> for the exact loop shape before touching `grade_documents` / `grade_answer`.

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

### Where the LangGraph re-architecture is headed

The target is an Adaptive / self-corrective RAG graph with a separate booking sub-graph and a fallback-only web search. Retrieval failures and answer failures are corrected by **two distinct, capped retry loops** rather than one shared loop — see `docs/v2-architecture.md` for the exact node/edge spec, routing rules, and retry caps; that is the single source of truth for the rewrite. Keep the existing RAGAS metrics comparable so before/after quality can be measured on the same question set.

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
- **Tavily key is optional but recommended for `web_search`.** Without `TAVILY_API_KEY`, the RAG spine's `web_search` node safely falls back to an empty context (no crash) and `generate` answers "don't know" instead of fabricating.
- **Dockerfile ↔ pyproject mismatch:** the Docker `pip install` list omits `ragas` and `datasets` (present in `pyproject.toml`). Harmless today (not used in the serving path) but keep in mind if evaluation code moves into the runtime image.

## Environment Variables (`app/services/.env`)

- `OPENAI_API_KEY` — chat LLM, embeddings, and music LLM all need it.
- `YOUTUBE_API_KEY` — YouTube link search.
- `SPOTIPY_CLIENT_ID`, `SPOTIPY_CLIENT_SECRET` — Spotify search.
- `TAVILY_API_KEY` — `web_search` fallback node in the LangGraph chatbot; optional (safe fallback if unset).