from langgraph.graph import END

from app.langgraph_services.nodes import MAX_ANSWER_RETRY, MAX_RETRIEVE_RETRY, GraphState


def route_after_question(state: GraphState) -> str:
    mode = state["mode"]
    if mode == "booking":
        return "booking_stub"
    if mode == "general_chat":
        return "general_chat"
    if mode == "web_search":
        return "web_search"
    return "retrieve"


def route_after_grade_documents(state: GraphState) -> str:
    """vectorstore로 온 질문은 학교 관련 내용이므로, 문서에 없으면 web_search로
    보내지 않고 그대로 generate로 보낸다 — 문서 기반이 아닌 답변은 허용하지 않는다."""
    if state.get("documents_grade") == "relevant":
        return "generate"
    if state.get("retrieve_retry_count", 0) < MAX_RETRIEVE_RETRY:
        return "transform_query"
    return "generate"


def route_after_grade_answer(state: GraphState) -> str:
    """모든 경로(vectorstore/web_search/general_chat/booking)가 여기로 모인다.
    재시도는 항상 답을 만들어낸 그 노드로 되돌아간다 — mode로 어느 노드였는지 구분한다."""
    if state.get("answer_grade") == "pass":
        return END
    if state.get("answer_retry_count", 0) >= MAX_ANSWER_RETRY:
        return END

    mode = state.get("mode")
    if mode == "general_chat":
        return "general_chat"
    if mode == "booking":
        return "booking_stub"
    return "generate"
