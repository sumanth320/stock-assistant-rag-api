from app.prefilter import compute_intent_scores
from app.rag import retrieve
from app.tools import get_stock_price, extract_ticker
from app.config import RAG_CONFIG, HIGH_CONFIDENCE


def get_dynamic_k(top1, top2):
    gap = top1 - top2 if top2 is not None else 1.0

    if gap >= RAG_CONFIG["high_gap"]:
        return RAG_CONFIG["k_small"]

    if gap >= RAG_CONFIG["low_gap"]:
        return RAG_CONFIG["k_medium"]

    return RAG_CONFIG["k_large"]


def is_rag_reliable(top1_score):
    return top1_score >= RAG_CONFIG["min_top1_score"]


def route_query(query: str):
    scores = compute_intent_scores(query)

    best_route = max(scores, key=scores.get)
    best_score = scores[best_route]

    if best_route == "tool":
        if best_score < HIGH_CONFIDENCE:
            return {
                "route": "clarify",
                "message": "Could you specify the stock symbol or company name?"
            }

        ticker = extract_ticker(query)

        if not ticker:
            return {
                "route": "clarify",
                "message": "Could you specify the stock symbol (e.g. AAPL)?"
            }

        tool_result = get_stock_price(ticker)

        return {
            "route": "tool",
            "query": query,
            "context": tool_result
        }

    if best_route == "direct":
        return {
            "route": "direct",
            "query": query,
            "context": None
        }

    chunks = retrieve(query)

    if not chunks:
        return {
            "route": "direct",
            "query": query,
            "context": None
        }

    top1 = chunks[0]["score"]
    top2 = chunks[1]["score"] if len(chunks) > 1 else None

    if not is_rag_reliable(top1):
        return {
            "route": "direct",
            "query": query,
            "context": None
        }

    k = get_dynamic_k(top1, top2)

    return {
        "route": "rag",
        "query": query,
        "context": chunks[:k]
    }