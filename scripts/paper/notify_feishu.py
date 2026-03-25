#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a Feishu bot notification.")
    parser.add_argument("--webhook", required=True, help="Feishu bot webhook URL")
    parser.add_argument("--title", required=True, help="Short message title")
    parser.add_argument("--body", required=True, help="Message body")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "msg_type": "text",
        "content": {"text": f"{args.title}\n{args.body}"},
    }
    request = urllib.request.Request(
        args.webhook,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        print(
            f"failed to send Feishu notification: HTTP {exc.code}: {error_text}",
            file=sys.stderr,
        )
        return 1
    except urllib.error.URLError as exc:
        print(f"failed to send Feishu notification: {exc}", file=sys.stderr)
        return 1

    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError:
        print(
            f"unexpected Feishu response (non-JSON): {response_text}",
            file=sys.stderr,
        )
        return 1

    if response_json.get("code", 0) != 0:
        print(f"Feishu rejected notification: {response_json}", file=sys.stderr)
        return 1

    print("feishu_notification=sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
