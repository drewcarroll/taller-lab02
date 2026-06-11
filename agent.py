"""Step 3: The agent loop.

We hand Gemini a topic plus the tool declarations, then loop:

    model takes a turn
      -> did it ask for tools?
           yes -> run them, send the results back, loop again
           no  -> that text IS the final report, stop

The model decides *what* to call and *when*. Our job is only to run whatever it
asks for and feed the results back until it stops asking.
"""

import os

import google.generativeai as genai
from dotenv import load_dotenv

from executors import execute_tool
from tools import TOOLS

# Load GOOGLE_API_KEY (and TAVILY_API_KEY) from .env, then configure the SDK once.
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

SYSTEM_PROMPT = """You are a thorough research agent. Follow this process and do
NOT skip steps:

1. Call web_search to find relevant sources for the topic.
2. Pick the 4-5 most promising results and call fetch_url on EACH of them to read
   the full page. Search snippets alone are never sufficient — you must read the
   actual pages before writing.
3. For long or dense pages, call analyze_data with a clear focus to distill the
   key points before you synthesize.
4. Only after you have read multiple sources, write the final report.

The report must be comprehensive, well-structured, grounded strictly in what the
tools returned (not prior assumptions), and cite every source inline with its URL.
Do not write the report until you have fetched at least two full pages.
"""


def _args_to_dict(function_call) -> dict:
    """Convert a Gemini FunctionCall's args (a proto map) into a plain dict."""
    return {key: value for key, value in function_call.args.items()}


def _extract_text(response) -> str:
    """Pull text out of a response's parts.

    We avoid `response.text` (the SDK shortcut) because it raises when a turn
    contains no text part. Joining parts ourselves returns "" instead.
    """
    parts = response.candidates[0].content.parts
    return "".join(getattr(p, "text", "") or "" for p in parts).strip()


def run_agent(topic: str, max_iterations: int = 15) -> dict:
    """Research `topic` and return {report, tool_calls, iterations}."""
    model = genai.GenerativeModel(
        MODEL,
        tools=[TOOLS],
        system_instruction=SYSTEM_PROMPT,
    )
    # start_chat manages the running conversation history for us, so we don't
    # have to hand-assemble the message list each turn.
    chat = model.start_chat()

    tool_calls: list[dict] = []

    # The opening user message kicks off the research.
    response = chat.send_message(
        f"Research the following topic and produce a comprehensive report: {topic}"
    )

    for iteration in range(1, max_iterations + 1):
        # (a) Pull any function_call parts out of the model's latest turn.
        parts = response.candidates[0].content.parts
        function_calls = [p.function_call for p in parts if p.function_call.name]

        # (b) No tool calls -> the model is done. Its text is the final report.
        if not function_calls:
            report = _extract_text(response)
            if report:
                return {
                    "report": report,
                    "tool_calls": tool_calls,
                    "iterations": iteration,
                }
            # Empty closing turn (a flash-lite quirk): nudge it to write the report.
            response = chat.send_message(
                "Please write the final report now, based on what you gathered, "
                "citing sources with their URLs."
            )
            continue

        # (c) Otherwise, run every tool the model asked for this turn...
        function_responses = []
        for fc in function_calls:
            args = _args_to_dict(fc)
            result = execute_tool(fc.name, args)          # runs the real executor
            tool_calls.append({"tool": fc.name, "input": args})  # track for metadata

            # (d) Wrap each result as a function_response part keyed by tool name,
            #     so the model knows which call this answers.
            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fc.name,
                        response={"result": result},
                    )
                )
            )

        # (e) Send all the results back; the model takes another turn -> loop.
        response = chat.send_message(function_responses)

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
    # Quick manual smoke test: `python agent.py`
    import json

    out = run_agent("Best sunscreen?")
    print(json.dumps({"iterations": out["iterations"],
                       "tool_calls": out["tool_calls"]}, indent=2))
    print("\n=== REPORT ===\n")
    print(out["report"])
