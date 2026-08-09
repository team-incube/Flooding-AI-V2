---
name: rebuild-vectordb
description: Rebuild the Chroma vector DB after editing data/flooding_rag.json. Use this whenever the user edits, adds to, or fixes flooding_rag.json, or says anything like "벡터 DB 재빌드", "임베딩 다시", "rebuild", "챗봇이 새 문서를 못 찾아", or "RAG 문서 수정했는데 반영이 안 돼". Critical context: get_retriever() in app/services/embedding.py only builds the DB when ./chroma_db doesn't already exist — if it's already there, edits to flooding_rag.json are silently ignored and the chatbot keeps answering from stale data with no error. Always use this skill instead of manually deleting chroma_db or re-running the app, since skipping the confirmation step or the deletion step is exactly how stale-data bugs happen.
---

# Rebuild Vector DB

## Why this exists

`get_retriever()` checks `if os.path.exists(CHROMA_PATH)` and, if true, just loads
the existing Chroma DB — it never re-reads `flooding_rag.json` in that case. So
editing the RAG source file and restarting the app does *nothing* unless
`chroma_db` is deleted first. There's no error or warning when this happens;
the chatbot just keeps giving old answers. This skill exists to make the
delete-then-rebuild step impossible to forget.

The embedding cache (`./cache/`, `CacheBackedEmbeddings`) is separate from
`chroma_db` and should NOT be deleted — it just avoids re-paying for OpenAI
embedding calls on chunks that haven't changed, and rebuilding reads from it
automatically when the text is unchanged.

## Steps

### 1. Validate the JSON first

```bash
uv run python .claude/skills/rebuild-vectordb/scripts/validate_json.py
```

If this reports invalid JSON or an empty `documents` list, stop here and fix
`data/flooding_rag.json` before doing anything else — don't delete `chroma_db`
against a broken source file.

### 2. Confirm with the user before deleting

Show the user that `./chroma_db` is about to be deleted and rebuilt from
`data/flooding_rag.json`, and wait for their go-ahead before running step 3.
This is a destructive, irreversible-ish step (the DB has to be fully
re-embedded from scratch) — don't skip the confirmation just because step 1
passed.

### 3. Delete and rebuild

```bash
uv run python .claude/skills/rebuild-vectordb/scripts/rebuild_db.py
```

This deletes `./chroma_db` if it exists, then calls `get_retriever()`, which
rebuilds it from `data/flooding_rag.json` (reusing `./cache/` for any
unchanged chunks). This step can take a while the first time a chunk's text
changes, since that chunk needs a fresh OpenAI embedding call.

### 4. Verify the rebuild actually picked up the changes

```bash
uv run python .claude/skills/rebuild-vectordb/scripts/verify_queries.py
```

This runs a few sample queries against the fresh retriever and prints the
top matches. Check that:
- The documents returned actually look relevant to each query
- If you just edited a specific section of `flooding_rag.json`, run a query
  that should hit that section specifically (pass your own queries as
  arguments instead of the defaults) and confirm the new content shows up

If a query that should obviously match returns nothing relevant, the rebuild
likely didn't pick up the change — re-check that `chroma_db` was actually
deleted in step 3 before concluding anything else is wrong.
