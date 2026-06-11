"""Tool declarations in the function-calling schema the model expects.

These schemas tell the model which tools exist and what arguments each takes.
The implementations live in ``executors.py``.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information on a topic. "
                "Returns search results with titles, URLs, and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results, 1-10. Defaults to 5.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch the text content of a web page. "
                "Returns the main text content stripped of HTML."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The absolute URL of the page to fetch.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": (
                "Analyze and extract structured insights from raw text data, "
                "focused on a specific question or angle."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The raw text to analyze.",
                    },
                    "focus": {
                        "type": "string",
                        "description": "What to focus the analysis on (a question or angle).",
                    },
                },
                "required": ["content", "focus"],
            },
        },
    },
]
