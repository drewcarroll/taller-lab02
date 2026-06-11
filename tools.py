"""Step 1: Gemini function declarations.

These schemas are sent to the Gemini API so the model knows which tools it
can call and what arguments each one expects. The actual implementations live
in ``executors.py``.
"""

import google.generativeai as genai

_web_search = genai.protos.FunctionDeclaration(
    name="web_search",
    description=(
        "Search the web for current information on a topic. "
        "Returns search results with titles, URLs, and snippets."
    ),
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "query": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The search query.",
            ),
            "max_results": genai.protos.Schema(
                type=genai.protos.Type.INTEGER,
                description="Number of results to return, between 1 and 10. Defaults to 5.",
            ),
        },
        required=["query"],
    ),
)

_fetch_url = genai.protos.FunctionDeclaration(
    name="fetch_url",
    description=(
        "Fetch the text content of a web page. "
        "Returns the main text content stripped of HTML."
    ),
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "url": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The absolute URL of the page to fetch.",
            ),
        },
        required=["url"],
    ),
)

_analyze_data = genai.protos.FunctionDeclaration(
    name="analyze_data",
    description=(
        "Analyze and extract structured insights from raw text data, "
        "focused on a specific question or angle."
    ),
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "content": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The raw text to analyze.",
            ),
            "focus": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="What to focus the analysis on (a question or angle).",
            ),
        },
        required=["content", "focus"],
    ),
)

# A single Tool object grouping all three declarations, ready to pass to the model.
TOOLS = genai.protos.Tool(
    function_declarations=[_web_search, _fetch_url, _analyze_data],
)
