from fastapi import FastAPI
from pydantic import BaseModel

from app.orchestrator import route_query
from app.prompt_builder import build_prompt
from app.llm_client import generate_response

app = FastAPI()


class AskRequest(BaseModel):
    question: str
    user_id: str
    session_id: str


@app.post("/ask")
def ask(request: AskRequest):
    routed = route_query(request.question)

    if routed["route"] == "clarify":
        return {
            "message": routed["message"]
        }

    messages = build_prompt(
        query=routed["query"],
        route=routed["route"],
        context=routed["context"]
    )

    answer = generate_response(messages)

    return {
        "route": routed["route"],
        "answer": answer
    }