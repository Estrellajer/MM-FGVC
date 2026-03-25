"""MCP server for sending Feishu bot notifications over stdio.

The server exposes a single tool, ``feishu-bot-message``, which accepts a
plain text message and forwards it to a Feishu bot webhook.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any
from urllib import error, request

from mcp.server.fastmcp import FastMCP

DEFAULT_WEBHOOK_URL = (
    "https://open.feishu.cn/open-apis/bot/v2/hook/"
    "c3394338-4009-48c8-9c97-7fd4d7be73bc"
)
WEBHOOK_ENV_VAR = "FEISHU_BOT_WEBHOOK_URL"

server = FastMCP("feishu-bot-message")


def _get_webhook_url() -> str:
    webhook_url = os.environ.get(WEBHOOK_ENV_VAR, DEFAULT_WEBHOOK_URL).strip()
    if not webhook_url:
        raise RuntimeError(
            f"Missing Feishu webhook URL. Set {WEBHOOK_ENV_VAR} in the environment."
        )
    return webhook_url


def _post_feishu_text(content: str) -> dict[str, Any]:
    payload = {
        "msg_type": "text",
        "content": {"text": content},
    }
    encoded_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        _get_webhook_url(),
        data=encoded_payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=15) as response:
            raw_body = response.read().decode("utf-8")
            body = json.loads(raw_body) if raw_body else {}
            status_code = response.getcode()
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Feishu webhook returned HTTP {exc.code}: {error_body or exc.reason}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach Feishu webhook: {exc.reason}") from exc

    if status_code != 200:
        raise RuntimeError(f"Feishu webhook returned unexpected status {status_code}")

    if isinstance(body, dict) and body.get("code") not in (None, 0):
        raise RuntimeError(f"Feishu webhook rejected the message: {body}")

    return {
        "status_code": status_code,
        "response": body,
    }


@server.tool(
    name="feishu-bot-message",
    description="Send a concise text notification to the Feishu bot webhook.",
)
def feishu_bot_message(content: str) -> dict[str, Any]:
    """Send a text message to Feishu.

    Args:
        content: The final message text to send to Feishu.
    """

    normalized_content = content.strip()
    if not normalized_content:
        raise ValueError("content must not be empty")

    result = _post_feishu_text(normalized_content)
    return {
        "ok": True,
        "message": "Feishu message sent",
        **result,
    }


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    sys.exit(main())