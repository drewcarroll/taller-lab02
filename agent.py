"""The agent loop.

Each turn, the model either calls tools or returns the final report:

    model takes a turn
      -> did it ask for tools?
           yes -> run them, append the results, loop again
           no  -> that text is the final report, stop

The model is stateless, so the conversation lives in the `messages` list, which
grows each turn and is re-sent in full on every call.
"""

import json
import os

from dotenv import load_dotenv
from groq import BadRequestError, Groq

from executors import execute_tool
from tools import TOOLS

# Load GROQ_API_KEY (and TAVILY_API_KEY) from .env, then create the client.
load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are a thorough research agent. Follow this process and do
NOT skip steps:

1. Call web_search to find relevant sources for the topic.
2. Pick the 2 most promising results and call fetch_url on EACH of them to read
   the full page. Search snippets alone are never sufficient — you must read the
   actual pages before writing.
3. For long or dense pages, call analyze_data with a clear focus to distill the
   key points before you synthesize.
4. Only after you have read multiple sources, write the final report.

The report must be comprehensive, well-structured, grounded strictly in what the
tools returned (not prior assumptions), and cite every source inline with its URL.
Do not write the report until you have fetched at least two full pages.
"""


def _complete(messages: list, max_tries: int = 3):
    """One model turn: let it decide whether to call tools or answer.

    Retries when the model emits a malformed tool call, which the API rejects as
    `tool_use_failed`. The generation is stochastic, so a re-roll usually clears it.
    """
    for attempt in range(max_tries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except BadRequestError as exc:
            if "tool_use_failed" in str(exc) and attempt < max_tries - 1:
                continue
            raise


def run_agent(topic: str, max_iterations: int = 15) -> dict:
    """Research `topic` and return {report, tool_calls, iterations}."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Research the following topic and produce a comprehensive report: {topic}"},
    ]
    tool_calls: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        msg = _complete(messages).choices[0].message

        # (b) No tool calls -> the model is done. Its text is the final report.
        if not msg.tool_calls:
            report = (msg.content or "").strip()
            if report:
                return {
                    "report": report,
                    "tool_calls": tool_calls,
                    "iterations": iteration,
                }
            # Empty turn: nudge it to actually write the report.
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user",
                             "content": "Please write the final report now, "
                                        "based on what you gathered, citing sources with URLs."})
            continue

        # (c) Record the assistant's tool-call turn. This message (with its
        #     tool_calls) must precede the matching tool result messages.
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name,
                                 "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        # (d) Run each tool and append its result as a `tool` message keyed by
        #     the tool_call_id, so the model knows which call it answers.
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(tc.function.name, args)
            tool_calls.append({"tool": tc.function.name, "input": args})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": result,
            })

    # (f) Ran out of iterations without a final text answer.
    return {
        "report": (
            "Research did not converge within the iteration limit. "
            "Try a narrower topic or a higher max_iterations."
        ),
        "tool_calls": tool_calls,
        "iterations": max_iterations,
    }


if __name__ == "__main__":
    out = run_agent("Best sunscreen?")
    print(json.dumps({"iterations": out["iterations"],
                       "tool_calls": out["tool_calls"]}, indent=2))
    print("\n=== REPORT ===\n")
    print(out["report"])
