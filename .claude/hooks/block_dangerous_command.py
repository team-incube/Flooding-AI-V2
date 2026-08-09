#!/usr/bin/env python3
import json
import re
import sys

BLOCKED_PATTERNS = [
    r"rm\s+-rf\s*/\s*$",
    r"sudo\s+rm",
    r">\s*/dev/(?!null|zero|stdout|stderr)\w*",
    r"dd\s+if=",
    r"mkfs",
    r"curl.*\|\s*sh",
    r"wget.*\|\s*sh",
    r"git\s+push\b.*--force",
    r"git\s+push\b.*(^|\s)-f(\s|$)",
    r"git\s+reset\s+--hard",
    r"git\s+clean\b(?=[^\n]*-\w*f)(?=[^\n]*-\w*d)",
]


def main() -> int:
    sys.stdin.reconfigure(encoding="utf-8")
    data = json.load(sys.stdin)
    if data.get("tool_name") != "Bash":
        return 0

    command = data.get("tool_input", {}).get("command", "")

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command):
            print(f"[Hook] Blocked dangerous command: {command}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
