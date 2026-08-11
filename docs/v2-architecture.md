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

`route_question` classifies each turn into one of three destinations:

- **vectorstore** → RAG spine (site usage + dorm rules).
- **booking** → a separate sub-graph for actions (study-room / massage-chair /
  song requests). See "Booking sub-graph".
- **general_chat** → plain LLM reply (small talk / out-of-scope).

## RAG spine (self-corrective)

```
retrieve → grade_documents ──(relevant)──────────────→ generate → grade_answer ──(pass)──→ END
              │  (not relevant)                                       │ (fail, same docs,
              ▼                                                       │  capped 1–2 retries)
        retrieve_retry_count < 1 ?                                    │
              │ yes                    │ no                           │
              ▼                        ▼                               │
        transform_query ─────────► web_search ──► generate ◄──────────┘
        (rewrite query,           (fallback,      (loops back into
         back to retrieve,         not re-graded)   grade_answer above)
         retrieve_retry_count+=1)
```

This graph has **two separate, independently-capped retry loops** — they must not
share a counter, and neither should leak into the other:

- **Retrieval-side loop** (`retrieve_retry_count`, cap = **1**): when
  `grade_documents` finds nothing relevant, the *first* time this happens for a
  turn it goes to `transform_query` and back to `retrieve` (re-search the vector
  store with a rewritten query). If it's *still* not relevant after that one
  reformulated re-search, the graph gives up on the vector store and falls
  through to `web_search` instead of trying `transform_query` again.
- **Answer-side loop** (`answer_retry_count`, cap = **1–2**): when `grade_answer`
  fails (hallucination or off-topic), the graph does **not** re-retrieve or
  re-rewrite the query — it goes straight back to `generate` with the **same**
  retrieved/fallback documents, i.e. it's a "try producing a better answer from
  what we already have" retry, not a search retry.

Add both counters to graph `State` (e.g. `retrieve_retry_count: int`,
`answer_retry_count: int`), reset per user turn.

- **retrieve** — MMR retrieval from the usage/rules vector store
  (`k=7, fetch_k=20, lambda_mult=0.8`, matching the current retriever).
- **grade_documents** — relevance check on retrieved docs.
  - relevant → `generate`
  - not relevant, `retrieve_retry_count == 0` → `transform_query` → back to
    `retrieve` (increments `retrieve_retry_count`)
  - not relevant, `retrieve_retry_count >= 1` (already retried once) →
    `web_search`
- **transform_query** — rewrites the question for a better vector-store re-search.
  In this design it feeds back into `retrieve`, **not** directly into
  `web_search` — the rewritten query gets one shot at the vector store before
  the graph falls back to the web.
- **web_search** — fallback only, fires after the one allowed retrieval retry is
  exhausted. Results go **straight to `generate`**; they are **not** re-graded
  by `grade_documents` (avoids a re-grading loop).
- **generate** — answer from context only. If context is empty, answer "don't
  know" rather than inventing (this replaces the prompt-only `[절대 금지]` rule
  with a structural branch).
- **grade_answer** — hallucination + question-relevance check.
  - pass → `END`
  - fail → back to **`generate`** (regenerate from the same context), **capped
    at 1–2 retries** via `answer_retry_count` so latency can't blow up. This
    loop does not touch retrieval or the query at all.

### `web_search` is fallback-only

It fires in exactly two cases, and never on a normal turn (token/latency cost):

1. `route_question` decides the answer isn't in our docs (routes straight to web).
2. `grade_documents` rejects **all** retrieved docs on the retry attempt (i.e.
   after the one `transform_query → retrieve` reformulation has already been
   used up for this turn).

Note: dorm rules are internal docs, so web search generally can't answer
rules questions. **Decision:** dorm-rules documents are **not** loaded into the
vector store — only usage/how-to documents are indexed, so this edge case does
not arise for now. Revisit if rules content is added later.

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
  `answer_retry_count` cap **1–2**. This matters under concurrency (the async
  handling that already cut response time ~2×).

## Measurement

Keep RAGAS metrics comparable to the current baseline (Faithfulness, Answer
Relevancy, Context Relevance) on the **same question set**, so the re-architecture
can be reported as a before/after delta. Capture the baseline on the current
`create_agent` version *before* the rewrite — see the `run-ragas-eval` skill.

## Suggested build order

1. Minimal skeleton: `route_question` + `retrieve` + `generate` (hand-written).
   **Done.**
2. Add grading + correction: `grade_documents`, `transform_query`, `web_search`,
   `grade_answer`, with the **two separate retry caps** described above
   (`retrieve_retry_count` cap 1, `answer_retry_count` cap 1–2). ← current step.
3. Add the booking sub-graph.
4. Re-run `run-ragas-eval` and compare against baseline.