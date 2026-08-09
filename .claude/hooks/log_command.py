#!/usr/bin/env python3
import json
import sys
from datetime import datetime
from pathlib import Path


def main() -> int:
    sys.stdin.reconfigure(encoding="utf-8")
    data = json.load(sys.stdin)
    if data.get("tool_name") != "Bash":
        return 0

    command = data.get("tool_input", {}).get("command", "")
    cwd = data.get("cwd", ".")

    log_file = Path(cwd) / ".claude" / "logs" / "command.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {command}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
