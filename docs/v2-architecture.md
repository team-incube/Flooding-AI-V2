# Flooding-AI-V2 — LangGraph Architecture Spec

Spec for the chatbot graph that replaced the old LangChain `create_agent`
implementation. This is the **single source of truth for the graph**; when a
node/edge changes in code, update this doc in the same session. This was a
re-architecture, not a port — it added a dedicated `web_search` branch, booking
actions, and an Adaptive / self-corrective RAG loop that the old agent didn't have.

> **Status: live and built.** `CLAUDE.md`'s status block confirms `POST /ai/chat`
> calls `app.langgraph_services.graph.ask` — this is the serving path, not a plan.
> `app/services/chatbot.py` (the old `create_agent` version) is kept only for the
> RAGAS before/after baseline. Everything below describes the graph as it exists
> in `app/langgraph_services/{graph,nodes,routing,prompts}.py` today; keep this
> doc in sync whenever those files change.

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
  wake-up-music apply only — cancellation out of scope). See "Booking
  sub-graph".
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
but they all converge on `grade_answer` before `END`:

```
generate ─────────┐
general_chat ──────┼──→ grade_answer ──(pass, or mode="booking")──→ END
execute_booking ───┘         │ (fail, mode≠"booking", answer_retry_count < 1)
                              ▼
                    back to whichever node produced the answer
                    (generate / general_chat — same context,
                     no re-retrieval, no re-routing)
                              │ (fail, answer_retry_count >= 1)
                              ▼
                             END  (best-effort answer, no infinite loop)
```

- **generate** — used by `vectorstore` (RAG context) and `web_search` (Tavily
  results). Answers from `context` only; empty `context` → "don't know".
- **general_chat** — free-form reply, no `context` at all.
- **execute_booking** — calls the real apply API and reports the result (see
  "Booking sub-graph"). `extract_booking_slot`'s own direct replies (slot
  re-ask, cancel guidance) bypass this gate entirely — see below.
- **grade_answer** — runs after *all three*, uniformly (kept for observability/
  measurement even for booking), but **booking is excluded from the retry
  loop**: `route_after_grade_answer` sends `mode == "booking"` straight to
  `END` regardless of the grade. `execute_booking` calls a real, side-effecting
  apply API — unlike `generate`/`general_chat`, "retry" here can't mean
  "regenerate text from the same context," it can only mean "call the apply
  API again," which would risk a duplicate application. So the answer-side
  retry cap effectively doesn't apply to booking; `execute_booking` never
  bumps `answer_retry_count`.
- Retry destination on `fail` for the paths that *do* retry is **mode-aware**
  (`route_after_grade_answer` looks at `state["mode"]`): `general_chat`
  retries `general_chat`, everything else non-booking (`vectorstore`,
  `web_search`) retries `generate`. This is the same `answer_retry_count`
  loop described above — one shared counter and cap, regardless of which node
  it's retrying.
- `extract_booking_slot`'s own replies (missing `musicUrl`, cancel guidance)
  never reach `grade_answer` either — there's no claim to fact-check, so they
  go straight to `END` (see "Booking sub-graph").

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

**Scope:** only the three apply actions below. **Cancellation is explicitly out of
scope** — `extract_booking_slot` detects cancel intent and tells the user to cancel
on the site directly, without calling any API. There is **no confirmation step**:
once an apply intent (and its slot, if any) is resolved, `execute_booking` runs
immediately.

The original design in this doc had a `confirm_with_user` step between slot
extraction and execution; it was removed because this is a text-only chat UI with
no button/card affordance to confirm against — a confirmation step would just be
another free-text turn the router has to reinterpret, which doesn't reliably
reduce misclassification risk and adds a round-trip to every booking. The
trade-off: if `extract_booking_slot` misclassifies intent (e.g. applies for the
massage chair when the user meant the study room), the **only** way to undo it is
the site's own cancel UI — the chatbot itself has no cancel path (see above).

```
                    ┌─(apply_study / apply_massage, or apply_music with a link)──→ execute_booking → grade_answer
route_question ──→ extract_booking_slot
 (mode=booking)     └─(apply_music w/o link, or cancel)──→ reply directly → END
                        (no grade_answer — nothing to fact-check)
```

- Actions and slots:
  - `apply_study` (자습실 신청) — `POST /domitory/study`, no slot, body `{}`.
  - `apply_massage` (안마의자 신청) — `POST /domitory/massage`, no slot, body `{}`.
  - `apply_music` (기상음악 신청) — `POST /domitory/music`, requires a
    `musicUrl` slot (a `youtube.com`/`youtu.be` link extracted from the
    message); if missing, `extract_booking_slot` asks the user for the link
    and ends the turn instead of calling `execute_booking`.
- `extract_booking_slot` classifies the message into
  `apply_study | apply_massage | apply_music | cancel | unclear` (LLM call),
  extracts `musicUrl` when relevant, and decides whether the turn is ready to
  execute (`booking_ready`). `cancel` and `unclear` reply directly and skip
  `execute_booking`/`grade_answer` entirely.
- `execute_booking` calls the matching endpoint on `BOOKING_API_BASE_URL`
  (`https://dev.flooding.kr`, the dev server) with
  `Authorization: Bearer {auth_token}` (`auth_token` comes from `GraphState`,
  populated by `chat_API.py` from the incoming `Authorization` header — never
  logged). No `auth_token` → "로그인이 필요합니다." without attempting the call.
  Response handling collapses to four buckets: 2xx → per-action success
  message; 401 → session-expired message; 400/409 → the server's own error
  message when present, else a generic "can't apply right now" message;
  anything else (timeout, network error, unexpected status) → a generic retry
  message. Every branch is wrapped so a request exception never propagates out
  of the node.
- `grade_answer` still runs for `mode == "booking"` (observability/measurement),
  but **booking never retries** on `fail` — `route_after_grade_answer` sends it
  straight to `END` regardless of the grade. `execute_booking` is a
  side-effecting node (it calls a real apply API); re-entering it on a failed
  grade would re-issue the API call and risk a duplicate application, so it's
  deliberately excluded from the answer-retry loop. See "Answer gate" below.

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
4. Add the real booking sub-graph (replace `booking_stub` with
   `extract_booking_slot` / `execute_booking`, apply-only, no confirmation
   step). **Done.**
5. Re-run `run-ragas-eval` and compare against baseline.