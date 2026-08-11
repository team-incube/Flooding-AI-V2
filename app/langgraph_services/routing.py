from langgraph.graph import END

from app.langgraph_services.nodes import MAX_ANSWER_RETRY, MAX_RETRIEVE_RETRY, GraphState


def route_after_question(state: GraphState) -> str:
    if state["mode"] == "booking":
        return "booking_stub"
    if state["mode"] == "general_chat":
        return "general_chat_stub"
    return "retrieve"


def route_after_grade_documents(state: GraphState) -> str:
    if state.get("documents_grade") == "relevant":
        return "generate"
    if state.get("retrieve_retry_count", 0) < MAX_RETRIEVE_RETRY:
        return "transform_query"
    return "web_search"


def route_after_grade_answer(state: GraphState) -> str:
    if state.get("answer_grade") == "pass":
        return END
    if state.get("answer_retry_count", 0) < MAX_ANSWER_RETRY:
        return "generate"
    return END
