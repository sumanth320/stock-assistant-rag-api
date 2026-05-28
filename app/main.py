from fastapi import FastAPI
from pydantic import BaseModel

from app.orchestrator import route_query
from app.prompt_builder import build_prompt
from app.llm_client import generate_response

from app.logger import Logger

app = FastAPI()


class AskRequest(BaseModel):
    question: str
    user_id: str
    session_id: str


@app.post("/ask")
def ask(request: AskRequest):
    logger = Logger()
    logger.start()

    logger.log("query", request.question)
    logger.log("user_id", request.user_id)
    logger.log("session_id", request.session_id)

    routed = route_query(request.question)
    logger.log("route_decision", routed["route"])

    if routed["route"] == "clarify":
        logger.log("final_status", "clarify")
        trace = logger.end()
        logger.print()

        return {
            "message": routed["message"]
        }

    messages = build_prompt(
        query=routed["query"],
        route=routed["route"],
        context=routed["context"]
    )

    answer = generate_response(messages)

    logger.log("final_status", "success")
    trace = logger.end()
    logger.print()

    return {
        "route": routed["route"],
        "answer": answer
    }