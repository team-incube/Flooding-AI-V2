#!/usr/bin/env python3
"""Validate data/flooding_rag.json before touching the vector DB.

Run this first. If it fails, stop — do not delete chroma_db yet.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAG_JSON = PROJECT_ROOT / "data" / "flooding_rag.json"


def main() -> None:
    if not RAG_JSON.exists():
        print(f"NOT FOUND: {RAG_JSON}")
        sys.exit(1)

    try:
        with RAG_JSON.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"INVALID JSON: {e}")
        sys.exit(1)

    docs = data.get("documents", [])
    if not docs:
        print("WARNING: valid JSON but 'documents' is empty")
        sys.exit(1)

    print(f"OK: valid JSON, {len(docs)} documents in {RAG_JSON.relative_to(PROJECT_ROOT)}")
    for d in docs:
        print(f"  - {d.get('id', '?')}: {d.get('title', '(no title)')}")


if __name__ == "__main__":
    main()
