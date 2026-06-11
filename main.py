"""Step 4: HTTP API.

Wraps the agent in a FastAPI service so it can be called over the network and
deployed. Two endpoints:

    POST /research  -> run the agent on a topic, return the report
    GET  /health    -> liveness check for the deployment platform

Run locally:  uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent import run_agent

app = FastAPI(title="Research & Report Agent", version="0.1.0")


# --- Request / response shapes (Pydantic validates these automatically) ---

class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="The topic to research.")
    max_iterations: int = Field(
        15, ge=1, le=30, description="Max agent loop iterations."
    )


class ResearchResponse(BaseModel):
    report: str
    tool_calls_count: int
    iterations: int


# --- Endpoints ---

@app.get("/health")
def health() -> dict:
    """Liveness probe — deployment platforms ping this to confirm the app is up."""
    return {"status": "ok"}


# NOTE: a plain `def` (not `async def`) on purpose. run_agent() blocks for many
# seconds on network calls; FastAPI runs sync endpoints in a threadpool, so one
# slow research request doesn't freeze the whole server.
@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest) -> ResearchResponse:
    result = run_agent(req.topic, max_iterations=req.max_iterations)
    return ResearchResponse(
        report=result["report"],
        tool_calls_count=len(result["tool_calls"]),
        iterations=result["iterations"],
    )
