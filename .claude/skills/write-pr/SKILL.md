---
name: write-pr
description: Generate PR title, body from commits since the base branch, then create the PR on GitHub. Handles base branch detection and PR creation end-to-end.
allowed-tools: Bash(git *:*), Bash(bash *create-pr.sh:*), Bash(cat *:*), Read, Write
---

## Step 1 — Gather Context

```bash
git branch --show-current
git log origin/main..HEAD --oneline 2>/dev/null || git log --oneline -15
git diff origin/main...HEAD --stat 2>/dev/null || git diff HEAD~5...HEAD --stat
git diff origin/main...HEAD 2>/dev/null || git diff HEAD~5...HEAD
```

If `.github/PULL_REQUEST_TEMPLATE.md` exists, read it and follow its structure for the body. Otherwise use the default structure in Step 3.

## Step 2 — Generate PR Content

**Title** — Generate 3 options in the format `[scope] description`:
- Scope: infer from changed file paths (e.g. `app/api` → `[api]`, `app/services` → `[chatbot]`/`[rag]`). Use `[global]` for cross-cutting changes only.
- Description: Korean, concise, no emojis, max 50 characters total
- Wrap class/function names, file names, and technical terms in backticks (e.g. `` `search_document` ``, `` `chat_API.py` ``)

**Body** — If no PR template exists, use:
```
## 변경 사항
- ...

## 변경 이유
- ...

## 테스트
- [ ] ...
```
- Korean 합쇼체: `~하였습니다`, `~되었습니다`, `~추가하였습니다`
- No emojis, max 2500 characters
- Wrap all proper nouns and technical identifiers in backticks

## Step 3 — Write Body & Show Preview

Write the body to `PR_BODY.md`, then display:

```
## PR 제목 후보
1. [title1]
2. [title2]
3. [title3]

## PR 본문 미리보기
[body content]
```

Use AskUserQuestion to ask the user which title to use (present options 1/2/3). Wait for the answer before proceeding.

## Step 4 — Create PR

```bash
bash "${CLAUDE_SKILL_DIR}/scripts/create-pr.sh" "<confirmed-title>" "PR_BODY.md"
```

Base branch is auto-detected by the script: `feature/*`/`feat/*` → `develop`, `develop` → `main`, otherwise falls back to the current PR's base or `main`.

After creation, display the PR URL. Cleanup: remove `PR_BODY.md`.
