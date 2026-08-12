from langgraph.graph import END

from app.langgraph_services.nodes import MAX_ANSWER_RETRY, MAX_RETRIEVE_RETRY, GraphState


def route_after_question(state: GraphState) -> str:
    mode = state["mode"]
    if mode == "booking":
        return "extract_booking_slot"
    if mode == "general_chat":
        return "general_chat"
    if mode == "web_search":
        return "web_search"
    return "retrieve"


def route_after_booking_slot(state: GraphState) -> str:
    """슬롯이 확정돼 바로 실행 가능하면 execute_booking으로, 되묻거나(링크 없음)
    취소 안내로 끝난 경우엔 grade_answer 없이 바로 END로 보낸다."""
    if state.get("booking_ready"):
        return "execute_booking"
    return END


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
    재시도는 항상 답을 만들어낸 그 노드로 되돌아간다 — mode로 어느 노드였는지 구분한다.

    booking은 예외: execute_booking은 실제 신청 API를 호출하는 부작용이 있는 노드라서,
    fail로 재진입시키면 신청이 중복 실행된다. grade_answer는 다른 경로와 동일하게
    거치되(관측용 answer_grade 기록), 재시도 없이 판정과 무관하게 바로 END로 보낸다."""
    mode = state.get("mode")
    if mode == "booking":
        return END

    if state.get("answer_grade") == "pass":
        return END
    if state.get("answer_retry_count", 0) >= MAX_ANSWER_RETRY:
        return END

    if mode == "general_chat":
        return "general_chat"
    return "generate"
