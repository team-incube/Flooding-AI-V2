# Flooding-AI-V2 — LangGraph Re-architecture Spec

Target design for rewriting the chatbot from LangChain `create_agent` onto a
LangGraph graph. This is the **single source of truth for the rewrite**; when a
node/edge is built, it should match what's here. This is a re-architecture, not a
port — it adds a web-search fallback, booking actions, and an Adaptive /
self-corrective RAG loop that the current agent doesn't have.

> Status is tracked in `CLAUDE.md`. While that file's status block says the graph
> doesn't exist yet, this document is aspirational (the plan). Once nodes land,
> keep this doc in sync with the actual graph.

## Why re-architect

The current `create_agent` chatbot decides tool use inside an opaque agent loop,
and hallucination is held back only by prompt constraints (`[절대 금지]` block).
A graph makes control flow explicit — usage questions can be *forced* through
retrieval and document grading, and the "answer only from context" rule becomes a
structural branch rather than a prompt promise. This raises controllability and
debuggability, and is measurable against the existing RAGAS baseline.

## Top-level routing

`route_question` classifies each turn into one of **four** destinations:

- **vectorstore** → RAG spine (site usage + dorm rules). Everything reachable
  from here is treated as school-related, so answers here **must** stay
  document-grounded — see the "why `web_search` isn't in the RAG spine" note
  below.
- **booking** → a separate sub-graph for actions (study-room / massage-chair /
  song requests). See "Booking sub-graph". Currently a stub (`booking_stub`).
- **general_chat** → small talk / no factual content needed (e.g. "안녕", "너
  누구야"). One LLM call, no retrieval, no search.
- **web_search** → the question needs current/factual information but is
  **not** about the school (e.g. "오늘 날씨 어때", "2025년 노벨 물리학상 누가
  받았어"). Runs a real web search (Tavily) and answers from those results.

All four destinations converge on the same answer-gate — see "Answer gate"
below. `route_question` is a single LLM call classifying into these four
labels; it does not need a second pass to decide "does this need a search."

## RAG spine (self-corrective)

```
retrieve → grade_documents ──(relevant)───────────────────────→ generate
              │  (not relevant)
              ▼
        retrieve_retry_count < 1 ?
              │ yes                    │ no
              ▼                        ▼
        transform_query ─────► generate (context stays empty → "don't know")
        (rewrite query,
         back to retrieve,
         retrieve_retry_count+=1)
```

`generate` here feeds into the shared **answer gate** (`generate → grade_answer`),
described below — this diagram only covers retrieval + document grading.

This graph has **two separate, independently-capped retry loops** — they must not
share a counter, and neither should leak into the other:

- **Retrieval-side loop** (`retrieve_retry_count`, cap = **1**): when
  `grade_documents` finds nothing relevant, the *first* time this happens for a
  turn it goes to `transform_query` and back to `retrieve` (re-search the vector
  store with a rewritten query). If it's *still* not relevant after that one
  reformulated re-search, the graph gives up — straight to `generate` with an
  empty `context`, which answers "don't know" (see below, and "why `web_search`
  isn't in the RAG spine").
- **Answer-side loop** (`answer_retry_count`, cap = **1**): when `grade_answer`
  fails (hallucination or off-topic), the graph does **not** re-retrieve or
  re-rewrite the query — it goes straight back to whichever node produced the
  answer, with the **same** context, i.e. it's a "try producing a better answer
  from what we already have" retry, not a search retry. This loop is shared by
  every top-level destination — see "Answer gate" below.

Add both counters to graph `State` (e.g. `retrieve_retry_count: int`,
`answer_retry_count: int`), reset per user turn.

- **retrieve** — MMR retrieval from the usage/rules vector store
  (`k=7, fetch_k=20, lambda_mult=0.8`, matching the current retriever).
- **grade_documents** — relevance check on retrieved docs.
  - relevant → `generate`
  - not relevant, `retrieve_retry_count == 0` → `transform_query` → back to
    `retrieve` (increments `retrieve_retry_count`)
  - not relevant, `retrieve_retry_count >= 1` (already retried once) →
    `generate` (empty context, "don't know")
- **transform_query** — rewrites the question for a better vector-store re-search.
  Feeds back into `retrieve` — the rewritten query gets one shot at the vector
  store before the graph gives up on it for this turn.
- **generate** — answer from context only. If context is empty, answer "don't
  know" rather than inventing (this replaces the prompt-only `[절대 금지]` rule
  with a structural branch). Feeds into the shared answer gate.

### Why `web_search` isn't in the RAG spine

Reaching the RAG spine (`vectorstore` mode) means the question is about the
school by construction — `route_question` only sends usage-shaped questions
here. A live web search **cannot reliably answer internal-only school
procedures** (dorm laundry, dorm rules, etc.) and will confidently cite
unrelated sources if allowed to try (observed in testing: a laundry-machine
question with no matching RAG doc got answered from an unrelated blog about a
different building's laundry app). So for `vectorstore`, "not found after one
retry" means **"don't know"**, never "let the open web guess" — no fallback to
`web_search` from inside the RAG spine.

`web_search` is real (Tavily, see "Answer gate"), but only reachable as its own
top-level `route_question` destination, for questions that are explicitly
**not** about the school. Dorm-rules documents are also still **not** loaded
into the vector store (only usage/how-to documents are indexed) — belt-and-
suspenders with the same reasoning.

## Answer gate (shared by every path)

Every top-level destination produces its answer through a **different** node,
but they all converge on the same two nodes before `END`:

```
generate ─────────┐
general_chat ──────┼──→ grade_answer ──(pass)──→ END
booking_stub ──────┘         │ (fail, answer_retry_count < 1)
                              ▼
                    back to whichever node produced the answer
                    (generate / general_chat / booking_stub — same context,
                     no re-retrieval, no re-routing)
                              │ (fail, answer_retry_count >= 1)
                              ▼
                             END  (best-effort answer, no infinite loop)
```

- **generate** — used by `vectorstore` (RAG context) and `web_search` (Tavily
  results). Answers from `context` only; empty `context` → "don't know".
- **general_chat** — free-form reply, no `context` at all.
- **booking_stub** — fixed placeholder reply (booking sub-graph isn't built yet).
- **grade_answer** — runs after *all three*, uniformly. There's no special-case
  skip for `general_chat`/`booking_stub` just because they have no `context` —
  in practice `grade_answer`'s `grounded_in_context` check doesn't penalize an
  answer that isn't claiming anything from a document (e.g. a greeting), so
  this doesn't manufacture spurious failures on every chat turn.
- Retry destination on `fail` is **mode-aware** (`route_after_grade_answer`
  looks at `state["mode"]`): `general_chat` retries `general_chat`, `booking`
  retries `booking_stub`, everything else (`vectorstore`, `web_search`) retries
  `generate`. This is the same `answer_retry_count` loop described above — one
  shared counter and cap, regardless of which node it's retrying.

## `web_search` (Tavily, real search)

- Reachable **only** from `route_question` classifying a turn as `web_search`
  (not about the school, needs current/factual info) — never from inside the
  RAG spine (see "Why `web_search` isn't in the RAG spine").
- Fetches a handful of results (max 5, snippet-only — no full-page crawl) and
  turns them into `Document`s with `metadata["source"]` set to the result URL,
  same shape `retrieve` produces, so `generate` handles both identically.
- `generate` cites `metadata["source"]` in the reply when present (RAG-store
  docs don't carry `source`, so citation only shows up for real web results).
- If `TAVILY_API_KEY` is missing or the request fails, falls back to an empty
  `context` — `generate` answers "don't know" instead of crashing or inventing.
- Results are **not** re-graded by `grade_documents` — they go straight into
  `generate`, then through the same `grade_answer` gate as everything else.

## Booking sub-graph (actions, outside the RAG loop)

Booking is not an information-retrieval problem, so it lives outside the grading/
rewrite loop.

```
extract_booking_slot → confirm_with_user → execute_booking
```

- Each action needs different slots, e.g.:
  - study room: date, time slot, seat number
  - massage chair: time slot
  - song request: title, artist
- `extract_booking_slot` identifies the action type and whether required slots are
  filled; if not, it asks back (multi-turn).
- `confirm_with_user` — actions that write/reserve a real resource **must be
  confirmed before execution** (prevents double-booking / typos).
- `execute_booking` — writes to the DB and returns a completion message.

## Cost / latency guidance

- Don't run every grader as a full LLM call. Prefer a **retrieval-score threshold**
  (e.g. cosine < 0.75 → "not relevant") for the first pass in `grade_documents`,
  and only send ambiguous cases to an LLM grader.
- Keep both retry loops bounded: `retrieve_retry_count` cap **1**,
  `answer_retry_count` cap **1**. This matters under concurrency (the async
  handling that already cut response time ~2×).

## Measurement

Keep RAGAS metrics comparable to the current baseline (Faithfulness, Answer
Relevancy, Context Relevance) on the **same question set**, so the re-architecture
can be reported as a before/after delta. Capture the baseline on the current
`create_agent` version *before* the rewrite — see the `run-ragas-eval` skill.

## Suggested build order

1. Minimal skeleton: `route_question` + `retrieve` + `generate` (hand-written).
   **Done.**
2. Add grading + correction: `grade_documents`, `transform_query`, `grade_answer`,
   with the **two separate retry caps** described above (`retrieve_retry_count`
   cap 1, `answer_retry_count` cap 1). **Done.**
3. Add real `web_search` (Tavily) as its own `route_question` destination, and
   the shared answer gate (`generate`/`general_chat`/`booking_stub` → `grade_answer`).
   **Done.**
4. Add the real booking sub-graph (replace `booking_stub`).
5. Re-run `run-ragas-eval` and compare against baseline.