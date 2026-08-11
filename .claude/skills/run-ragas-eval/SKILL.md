---
name: run-ragas-eval
description: Run a RAGAS quality evaluation (Faithfulness, AnswerRelevancy, ContextRelevance) against the chatbot and save a timestamped score snapshot for before/after comparison. Use this whenever the user says things like "ragas 평가", "ragas로 평가", "RAG 성능 측정", "챗봇 성능 before/after 비교", "baseline 측정", "LangGraph 재설계 전에 점수 남겨놔", or asks whether a chatbot/RAG change made things better or worse. Especially relevant right now: this measures the baseline BEFORE the LangGraph rewrite so post-rewrite scores have something to compare against. Always use this skill instead of writing a one-off eval script from scratch — it already knows how to pull retrieved contexts out of the LangChain agent's tool-call messages, which is the part that's easy to get wrong.
---

# Run RAGAS Eval

## Why this exists

Comparing chatbot quality "before" and "after" a change only means something if
both runs use the same questions, the same metrics, and a result that's saved
somewhere instead of just scrolled past in the terminal. This skill fixes the
question set (`data/eval_questions.json`), fixes the metrics (Faithfulness,
AnswerRelevancy, ContextRelevance), and writes every run to its own timestamped
file in `data/eval_results/` so nothing gets overwritten.

The tricky part this skill handles for you: `app/services/chatbot.py`'s `ask()`
only returns the final text response — it doesn't expose what the
`search_document` tool actually retrieved. Faithfulness and ContextRelevance
need those retrieved contexts to score anything meaningful. This skill invokes
the same `agent` object directly and pulls the contexts back out of the
`ToolMessage`s in `result["messages"]`, instead of modifying `chatbot.py`'s
public API just for evaluation purposes.

The script supports **two targets** via `--target`:
- `chatbot` (default) — `app.services.chatbot.agent`, the old `create_agent` version.
- `langgraph` — `app.langgraph_services.graph.app_graph`, the LangGraph rewrite.
  Contexts here come directly from the graph's `state["context"]` — no
  tool-message parsing needed, since the graph already exposes it.

Run one of each with the same question set to get a real before/after
comparison instead of two baselines of the same system.

## Read this before interpreting scores

**Low AnswerRelevancy is not automatically a regression.** The chatbot is
deliberately designed to defer — "학교 담당 부서에 문의해 주세요." — instead of
inventing an answer when `flooding_rag.json` has no relevant content for the
question. `data/eval_questions.json` includes `out_of_scope` category
questions specifically to exercise this path. If a run scores low
AnswerRelevancy on those, check the actual response text first: if it's
correctly deferring, that's the system working as intended, not a bug to fix.
Only worry about AnswerRelevancy dropping on the categories that ARE supposed
to be answered from the docs (login, dormitory_study, club_lookup, etc).

## Steps

### 1. Make sure the question set has real questions

Check `data/eval_questions.json`. If most `question` fields are still empty
strings (the shipped template), tell the user which categories still need
filling in before running — don't try to invent questions yourself, the user
asked to fill these in themselves. Empty questions are skipped automatically,
so a partially-filled file still runs, just with fewer samples.

### 2. Run the evaluation

```bash
# old create_agent baseline
uv run python .claude/skills/run-ragas-eval/scripts/run_eval.py --target chatbot

# new LangGraph rewrite
uv run python .claude/skills/run-ragas-eval/scripts/run_eval.py --target langgraph
```

(`--target` defaults to `chatbot` if omitted.) This will, per question:
1. Invoke the real system for that target (`app.services.chatbot.agent` for
   `chatbot`, `app.langgraph_services.graph.app_graph` for `langgraph`)
2. Collect the response and retrieved contexts (from tool-call messages for
   `chatbot`, directly from graph state for `langgraph`)
3. Score the whole batch with RAGAS (Faithfulness, AnswerRelevancy, ContextRelevance)
4. Print a summary + per-question breakdown to the console (for `langgraph`,
   each line also shows which `mode` route_question picked)
5. Save the full result to `data/eval_results/<timestamp>_<target>.json`

This calls OpenAI once per question for the chatbot itself, plus more calls
per question per metric for RAGAS's judge model — expect it to take a while
and to cost real API usage for anything beyond a handful of questions.

### 3. Compare against a previous run

To compare before/after, diff the `aggregate` blocks of two files in
`data/eval_results/`:

```bash
ls data/eval_results/
```

Pick a `_chatbot` run and a `_langgraph` run (or two runs of the same target
taken at different times) and read both `aggregate` sections. Also skim
`per_question` for any single question whose score moved a lot — an aggregate
average can hide a question that got much worse while others improved. For
`langgraph` runs, also check the `mode` field on questions whose contexts are
empty — an empty context from `general_chat`/`booking`/route_question sending
it straight to `web_search` is a routing decision, not the same thing as the
RAG spine failing to find anything.

## Metrics reference

| Metric | Needs | What it measures |
|---|---|---|
| `faithfulness` | response + contexts | Does the response only claim things the retrieved context actually supports? |
| `answer_relevancy` | question + response | Does the response actually address the question asked? (see the caveat above about deferred answers) |
| `nv_context_relevance` | question + contexts | Are the retrieved documents actually relevant to the question, independent of how the chatbot answered? |

## Known project quirks (don't "fix" these while using this skill)

- `ragas` and `datasets` are in `pyproject.toml` but intentionally left out of
  `Dockerfile`'s `pip install` list — this eval only runs locally via `uv run`,
  never in the deployed container, so that's not a bug.
- The retriever config being evaluated is MMR with `k=7, fetch_k=20,
  lambda_mult=0.8` (from `app/services/embedding.py`). If that config changes,
  scores from old `eval_results/` files are still comparable in principle, but
  the comparison is weaker — note the config change when reporting results.
