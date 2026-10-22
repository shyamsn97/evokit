"""A minimal coding agent: ask the model, run the tools it calls, repeat until it stops calling tools.

    python agent.py --dir WORKDIR --task "..."

The last line printed is JSON: {"summary": ..., "cost_usd": ..., "tokens": ...}.
"""

import argparse
import json
import os
import sys
import urllib.request

from prompts import SYSTEM
from tools import TOOLS, run_tool

MODEL = os.environ.get("DGM_MODEL", "claude-sonnet-4-5")
PRICE = (3.0, 15.0)  # dollars per million input and output tokens
MAX_STEPS = 30


def call_model(messages):
    body = {"model": MODEL, "max_tokens": 8000, "system": SYSTEM, "tools": TOOLS, "messages": messages}
    request = urllib.request.Request(
        os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com") + "/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    messages = [{"role": "user", "content": args.task}]
    cost, tokens, summary, failed = 0.0, 0, "", False
    try:
        for _ in range(MAX_STEPS):
            reply = call_model(messages)
            usage = reply["usage"]
            tokens += usage["input_tokens"] + usage["output_tokens"]
            cost += (usage["input_tokens"] * PRICE[0] + usage["output_tokens"] * PRICE[1]) / 1e6
            messages.append({"role": "assistant", "content": reply["content"]})
            summary = "\n".join(b["text"] for b in reply["content"] if b["type"] == "text") or summary
            calls = [b for b in reply["content"] if b["type"] == "tool_use"]
            if not calls:
                break
            results = [
                {"type": "tool_result", "tool_use_id": c["id"], "content": run_tool(args.dir, c["name"], c["input"])}
                for c in calls
            ]
            messages.append({"role": "user", "content": results})
    except Exception as exc:
        summary, failed = f"agent error: {type(exc).__name__}: {exc}", True
    print(json.dumps({"summary": summary, "cost_usd": cost, "tokens": tokens}))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
