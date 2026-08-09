---
name: git-commit
description: Create Git commits by splitting changes into logical units following project conventions.
allowed-tools: Bash(git *:*)
---

## Commit Message Rules

Format: `type(scope): description`

- **Types**: `add` / `update` / `fix` / `refactor` / `ci/cd` / `docs` / `test` / `merge`
- **Scope**: domain name by default — for the full selection table, read `${CLAUDE_SKILL_DIR}/references/scope-guide.md`; for type/scope conventions, read `${CLAUDE_SKILL_DIR}/references/commit-conventions.md`
- **Description**: Korean, no period, avoid endings: `~한다/~된다`, `~하기`, `~합니다/~됩니다`, `~했습니다`
  - Good examples: `검색 문서 없을 때 응답 처리 수정`, `프롬프트 마무리 문구 제거`, `RAG 응답 로직 개선`
- Subject line only (no body)

## Commit Flow

1. Inspect changes: `git status`, `git diff`
2. Categorize into logical units (feature / bug fix / refactoring / etc.) — don't bundle unrelated changes into one commit
3. Group files per unit
4. For each group:
   - Stage only relevant files with `git add`
   - Write a commit message following the rules above
   - `git commit -m "message"`
5. Verify with `git log --oneline -n <count>`
