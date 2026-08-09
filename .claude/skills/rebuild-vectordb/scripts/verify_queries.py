#!/usr/bin/env python3
"""Sanity-check the rebuilt retriever with a few sample queries.

Usage:
  python verify_queries.py                  # runs the default queries below
  python verify_queries.py "커스텀 질문" "질문2"  # runs your own queries instead
"""
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_QUERIES = [
    "기숙사 자습 신청은 어떻게 하나요?",
    "로그인은 어떻게 하나요?",
    "동아리 개설 절차가 뭐예요?",
]


def main() -> None:
    from app.services.embedding import get_retriever

    retriever = get_retriever()
    queries = sys.argv[1:] or DEFAULT_QUERIES

    for query in queries:
        docs = retriever.invoke(query)
        print(f"\n=== Query: {query!r} ===")
        if not docs:
            print("  (no documents returned — retriever may be broken or DB is empty)")
            continue
        for i, doc in enumerate(docs[:3], 1):
            snippet = doc.page_content[:120].replace("\n", " ")
            print(f"  {i}. [{doc.metadata.get('id', '?')}] {snippet}...")


if __name__ == "__main__":
    main()
