#!/usr/bin/env python3
"""Delete the persisted Chroma DB and rebuild it from data/flooding_rag.json.

get_retriever() in app/services/embedding.py only builds the DB when
./chroma_db doesn't exist yet — if it's already there, edits to
flooding_rag.json are silently ignored. Deleting chroma_db first is what
forces a real rebuild.

Only run this after the user has confirmed it's OK to delete chroma_db.
The embedding cache in ./cache/ is left untouched (safe to keep, saves API calls).
"""
import os
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

CHROMA_PATH = PROJECT_ROOT / "chroma_db"


def main() -> None:
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
        print(f"Deleted {CHROMA_PATH}")
    else:
        print(f"{CHROMA_PATH} does not exist yet — building fresh")

    from app.services.embedding import get_retriever

    get_retriever()
    print("Rebuild complete: chroma_db regenerated from data/flooding_rag.json")


if __name__ == "__main__":
    main()
