#!/usr/bin/env python3
"""Run a RAGAS quality evaluation against the chatbot.

Usage:
  uv run python .claude/skills/run-ragas-eval/scripts/run_eval.py

Reads data/eval_questions.json, runs each question through the real chatbot
agent (app.services.chatbot.agent), collects the retrieved contexts from the
search_document tool calls, then scores the (question, response, contexts)
triples with RAGAS: Faithfulness, AnswerRelevancy, ContextRelevance.

IMPORTANT — expected tradeoff, not a bug:
Low AnswerRelevancy on out-of-scope questions is intentional. The chatbot is
designed to defer ("학교 담당 부서에 문의해 주세요.") instead of hallucinating an
answer when flooding_rag.json has no relevant content for the question. If a
future rebuild scores lower on AnswerRelevancy specifically for the
"out_of_scope" category in eval_questions.json, that's not necessarily a
regression — check whether it's still deferring correctly before treating it
as one.
"""
import asyncio
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas.*")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval_questions.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval_results"

METRIC_NAMES = ["faithfulness", "answer_relevancy", "nv_context_relevance"]


async def collect_samples() -> list[dict]:
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [q for q in data.get("questions", []) if q.get("question", "").strip()]
    if not questions:
        print(f"No filled-in questions found in {QUESTIONS_PATH}.")
        print("Fill in the 'question' field for at least a few entries first.")
        sys.exit(1)

    # Imported here, not at module scope: this triggers chatbot.py's module-level
    # get_retriever() (builds/loads the Chroma DB) and needs OPENAI_API_KEY, so we
    # don't want to pay that cost just to discover the questions file is still empty.
    from langchain_core.messages import ToolMessage

    from app.services.chatbot import agent

    loop = asyncio.get_running_loop()
    samples = []
    for q in questions:
        question_text = q["question"].strip()
        result = await loop.run_in_executor(
            None,
            lambda qt=question_text: agent.invoke({"messages": [("human", qt)]}),
        )
        messages = result["messages"]
        response = messages[-1].content
        contexts = [m.content for m in messages if isinstance(m, ToolMessage) and m.content]

        samples.append(
            {
                "id": q.get("id", "?"),
                "category": q.get("category", ""),
                "question": question_text,
                "response": response,
                "contexts": contexts,
            }
        )
        print(f"  collected [{q.get('id', '?')}] {question_text[:50]}")

    return samples


def score_samples(samples: list[dict]):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextRelevance, Faithfulness

    # LangchainLLMWrapper/LangchainEmbeddingsWrapper are marked deprecated in
    # favor of ragas' native provider factories, but this project already
    # builds ChatOpenAI/OpenAIEmbeddings everywhere else (chatbot.py,
    # embedding.py) — reusing that pattern here keeps one consistent way of
    # talking to OpenAI instead of introducing a second one.
    judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    ragas_samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["response"],
            retrieved_contexts=s["contexts"] or [""],
        )
        for s in samples
    ]
    dataset = EvaluationDataset(samples=ragas_samples)

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextRelevance(llm=judge_llm),
    ]

    return evaluate(dataset, metrics=metrics)


def build_report(samples: list[dict], result) -> dict:
    per_question = []
    for i, s in enumerate(samples):
        row = {
            "id": s["id"],
            "category": s["category"],
            "question": s["question"],
            "response": s["response"],
            "num_contexts": len(s["contexts"]),
        }
        for metric in METRIC_NAMES:
            row[metric] = result[metric][i]
        per_question.append(row)

    aggregate = {
        metric: sum(result[metric]) / len(result[metric]) for metric in METRIC_NAMES
    }

    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "num_questions": len(samples),
        "questions_file": str(QUESTIONS_PATH.relative_to(PROJECT_ROOT)),
        "retriever": {"search_type": "mmr", "k": 7, "fetch_k": 20, "lambda_mult": 0.8},
        "aggregate": aggregate,
        "per_question": per_question,
    }


def print_report(report: dict) -> None:
    print("\n=== RAGAS Evaluation Summary ===")
    print(f"Questions: {report['num_questions']}")
    for metric, value in report["aggregate"].items():
        print(f"  {metric:>20}: {value:.3f}")

    print("\n--- Per-question scores ---")
    for row in report["per_question"]:
        print(f"[{row['id']}] ({row['category'] or 'uncategorized'}) {row['question'][:40]}")
        print(
            f"    faithfulness={row['faithfulness']:.2f}  "
            f"answer_relevancy={row['answer_relevancy']:.2f}  "
            f"context_relevance={row['nv_context_relevance']:.2f}"
        )


def save_report(report: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{report['timestamp']}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return out_path


def main() -> None:
    print(f"Loading questions from {QUESTIONS_PATH.relative_to(PROJECT_ROOT)}...")
    samples = asyncio.run(collect_samples())

    print(f"\nScoring {len(samples)} samples with RAGAS (this calls OpenAI for each metric)...")
    result = score_samples(samples)

    report = build_report(samples, result)
    print_report(report)

    out_path = save_report(report)
    print(f"\nSaved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
