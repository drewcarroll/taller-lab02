"""Tool executors and the router.

Each ``execute_*`` function performs the real work behind a tool. ``execute_tool``
dispatches by name and always returns a string (never raises) so a single failing
tool call cannot crash the agent loop.
"""

import os

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_FETCH_TIMEOUT = 15.0
_FETCH_MAX_CHARS = 2500
_STRIP_TAGS = ("script", "style", "nav", "footer", "header")

# Model used for the analyze_data sub-call.
_ANALYZE_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

_client = None


def _groq() -> Groq:
    """Return a lazily created Groq client (so import doesn't require the key)."""
    global _client
    if _client is None:
        _client = Groq()
    return _client


def execute_web_search(args: dict) -> str:
    """Search the web via the Tavily API and return formatted results."""
    query = args["query"]
    max_results = int(args.get("max_results") or 5)
    max_results = max(1, min(max_results, 10))

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is not set; cannot perform web search."

    resp = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
        },
        timeout=_FETCH_TIMEOUT,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return f"No search results found for query: {query!r}"

    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"{i}. {r.get('title', '(no title)')}\n"
            f"   URL: {r.get('url', '')}\n"
            f"   Snippet: {r.get('content', '').strip()}"
        )
    return "\n\n".join(lines)


def execute_fetch_url(args: dict) -> str:
    """Fetch a URL and return its main text content with HTML stripped."""
    url = args["url"]
    resp = httpx.get(
        url,
        timeout=_FETCH_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": "research-report-agent/0.1 (+https://github.com)"},
    )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(list(_STRIP_TAGS)):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse blank lines / whitespace runs into something readable.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)

    if len(cleaned) > _FETCH_MAX_CHARS:
        cleaned = cleaned[:_FETCH_MAX_CHARS] + "\n\n[...truncated...]"
    return cleaned or "(page contained no extractable text)"


def _frame_content(content: str, focus: str) -> str:
    """Fallback: frame raw text under a focus heading (no LLM call)."""
    snippet = content.strip()
    if len(snippet) > _FETCH_MAX_CHARS:
        snippet = snippet[:_FETCH_MAX_CHARS] + "\n\n[...truncated...]"
    return (
        f"Analysis focus: {focus}\n"
        f"Content length: {len(content)} chars\n"
        f"--- content under review ---\n"
        f"{snippet}"
    )


def execute_analyze_data(args: dict) -> str:
    """Distill raw text down to the key points for a given focus.

    A single stateless model call: text in, focused bullet points out. If the
    call fails, it falls back to framing the raw content so a failure here never
    breaks the agent loop.
    """
    content = args["content"]
    focus = args["focus"]

    # Cap input so a huge page can't blow the model's token budget.
    trimmed = content.strip()
    if len(trimmed) > _FETCH_MAX_CHARS:
        trimmed = trimmed[:_FETCH_MAX_CHARS]

    prompt = (
        "You are a research assistant. Extract the key points from the text "
        f"below that are relevant to this focus: {focus}\n\n"
        "Return concise bullet points. Preserve any specific facts, numbers, "
        "names, and dates. Ignore navigation text, ads, and boilerplate.\n\n"
        f"--- TEXT ---\n{trimmed}"
    )

    try:
        resp = _groq().chat.completions.create(
            model=_ANALYZE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        distilled = (resp.choices[0].message.content or "").strip()
        if not distilled:
            return _frame_content(content, focus)
        return f"Analysis focus: {focus}\n\n{distilled}"
    except Exception:  # noqa: BLE001 - degrade gracefully, never break the loop
        return _frame_content(content, focus)


_ROUTER = {
    "web_search": execute_web_search,
    "fetch_url": execute_fetch_url,
    "analyze_data": execute_analyze_data,
}


def execute_tool(name: str, args: dict) -> str:
    """Dispatch to the named executor. Always returns a string, never raises."""
    fn = _ROUTER.get(name)
    if fn is None:
        return f"Error: unknown tool {name!r}."
    try:
        return fn(args)
    except Exception as exc:  # noqa: BLE001 - surface failures to the model as text
        return f"Error while running tool {name!r}: {type(exc).__name__}: {exc}"
