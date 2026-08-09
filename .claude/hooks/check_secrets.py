#!/usr/bin/env python3
import json
import os
import re
import sys

PATTERNS = [
    r"AKIA[0-9A-Z]{16}",
    r"ghp_[A-Za-z0-9]{36}",
    r"ghs_[A-Za-z0-9]{36}",
    r"github_pat_[A-Za-z0-9_]{82}",
    r"sk-[A-Za-z0-9]{48}",
    r"sk-proj-[A-Za-z0-9_-]{50,}",
    r"-----BEGIN\s*(RSA\s*|EC\s*|OPENSSH\s*)?PRIVATE KEY-----",
    r"xox[baprs]-[A-Za-z0-9-]+",
]


def main() -> int:
    sys.stdin.reconfigure(encoding="utf-8")
    data = json.load(sys.stdin)
    tool_name = data.get("tool_name")
    if tool_name not in ("Write", "Edit"):
        return 0

    tool_input = data.get("tool_input", {})
    content = tool_input.get("content") or tool_input.get("new_string") or ""
    file_path = tool_input.get("file_path", "")

    basename = os.path.basename(file_path)
    if basename == ".env" or basename.startswith(".env."):
        return 0

    for pattern in PATTERNS:
        if re.search(pattern, content):
            name = os.path.basename(file_path)
            print(f"[Hook] Potential secret detected in {name}. Pattern: {pattern}", file=sys.stderr)
            print("Possible secret or credential detected in the file content. Review before writing.")
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
