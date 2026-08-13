import os
import re
from typing import Literal, NotRequired, TypedDict
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from app.langgraph_services.prompts import (
    ANSWER_GRADE_SYSTEM_PROMPT,
    BOOKING_INTENT_SYSTEM_PROMPT,
    DOCUMENT_GRADE_SYSTEM_PROMPT,
    GENERAL_CHAT_SYSTEM_PROMPT,
    GENERATE_SYSTEM_PROMPT,
    GENERATE_WEB_SEARCH_ADDENDUM,
    ROUTE_MODES,
    ROUTE_SYSTEM_PROMPT,
    TRANSFORM_QUERY_SYSTEM_PROMPT,
    WEB_SEARCH_QUERY_SYSTEM_PROMPT,
)
from app.services.embedding import get_retriever

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "services", ".env"))

retriever = get_retriever()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# text-embedding-3-small 기준 실측하며 튜닝할 것. 애매한 구간만 LLM 그레이더로 넘어간다.
DOCUMENT_RELEVANCE_THRESHOLD = 0.75
DOCUMENT_IRRELEVANCE_THRESHOLD = 0.3

# retrieve_retry_count와 answer_retry_count는 서로 다른 루프의 카운터이므로 절대 공유하지 않는다.
MAX_RETRIEVE_RETRY = 1
MAX_ANSWER_RETRY = 1

# 전체 페이지 크롤링이 아니라 요약 스니펫만 받도록 include_raw_content=False로 고정.
MAX_WEB_SEARCH_RESULTS = 5

# generate()가 실시간 정보를 못 찾았다고 정직하게 답할 때 쓰는 고정 문구. 이 마커가 답변에
# 있으면 출처를 강제로 붙이지 않는다 — 근거로 쓰지 않은 검색 결과를 출처처럼 붙이면 오해를 준다.
WEB_SEARCH_NO_INFO_MARKER = "정확한 실시간 정보를 찾지 못했"
WEB_SEARCH_NO_INFO_REPLY = f"{WEB_SEARCH_NO_INFO_MARKER}어요. 관련 웹사이트나 앱에서 직접 확인해보시는 게 정확할 것 같아요."

# 개발 서버. 취소는 이번 범위에서 제외 — 신청 3종만 지원.
BOOKING_API_BASE_URL = "https://dev.flooding.kr"
BOOKING_ENDPOINTS = {
    "apply_study": ("POST", "/domitory/study"),
    "apply_massage": ("POST", "/domitory/massage"),
    "apply_music": ("POST", "/domitory/music"),
}
BOOKING_SUCCESS_MESSAGES = {
    "apply_study": "자습실 신청이 완료되었습니다.",
    "apply_massage": "안마의자 신청이 완료되었습니다.",
    "apply_music": "기상음악 신청이 완료되었습니다.",
}

_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}
_URL_PATTERN = re.compile(r"https?://\S+")

try:
    web_search_tool = TavilySearch(
        max_results=MAX_WEB_SEARCH_RESULTS,
        search_depth="basic",
        include_raw_content=False,
    )
except Exception:
    web_search_tool = None  # TAVILY_API_KEY 없음 등 — web_search가 빈 context로 안전하게 폴백


class DocumentRelevanceGrade(BaseModel):
    binary_score: bool = Field(description="검색된 문서가 질문과 관련 있으면 true, 없으면 false")


class AnswerGrade(BaseModel):
    grounded_in_context: bool = Field(description="답변이 문서/컨텍스트에 근거하고 있으면 true, 환각이면 false")
    addresses_question: bool = Field(description="답변이 실제로 질문에 답하고 있으면 true, 아니면 false")


class BookingIntent(BaseModel):
    action: Literal["apply_study", "apply_massage", "apply_music", "cancel", "unclear"] = Field(
        description="사용자가 요청한 신청 액션. 취소 의도면 cancel, 셋 중 무엇인지 특정할 수 없으면 unclear"
    )


document_grader_llm = llm.with_structured_output(DocumentRelevanceGrade)
answer_grader_llm = llm.with_structured_output(AnswerGrade)
booking_intent_llm = llm.with_structured_output(BookingIntent)


class GraphState(TypedDict):
    messages: list[AnyMessage]
    mode: Literal["vectorstore", "booking", "general_chat", "web_search"]
    context: list[Document]
    documents_grade: NotRequired[Literal["relevant", "not_relevant"]]
    answer_grade: NotRequired[Literal["pass", "fail"]]
    query: NotRequired[str]
    original_query: NotRequired[str]
    retrieve_retry_count: NotRequired[int]
    answer_retry_count: NotRequired[int]
    auth_token: NotRequired[str | None]
    booking_action: NotRequired[Literal["apply_study", "apply_massage", "apply_music"] | None]
    booking_slots: NotRequired[dict]
    booking_ready: NotRequired[bool]


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _last_message_content(messages: list[AnyMessage], message_type: type) -> str:
    for message in reversed(messages):
        if isinstance(message, message_type):
            return message.content
    return ""


async def route_question(state: GraphState) -> dict:
    """LLM으로 질문을 vectorstore / booking / general_chat / web_search 로 분류한다.

    booking은 extract_booking_slot으로 이어져 자습/안마의자/기상음악 신청을 처리한다.
    """
    question = state["messages"][-1].content
    response = await llm.ainvoke([
        SystemMessage(content=ROUTE_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ])

    mode = response.content.strip().lower()
    if mode not in ROUTE_MODES:
        mode = "general_chat"

    return {"mode": mode}


def _bump_answer_retry_if_reentering(state: GraphState) -> dict:
    """grade_answer가 "fail"을 낸 뒤 같은 노드로 재진입한 경우에만 answer_retry_count를 올린다."""
    if state.get("answer_grade") == "fail":
        return {"answer_retry_count": state.get("answer_retry_count", 0) + 1}
    return {}


def _extract_youtube_url(text: str) -> str | None:
    """메시지에서 youtube.com/youtu.be 도메인의 URL만 뽑아낸다. 서브도메인 스푸핑
    (예: youtube.com.evil.com) 방지를 위해 netloc을 정확히 검사한다."""
    for match in _URL_PATTERN.finditer(text):
        url = match.group(0).rstrip(").,'\"")
        if urlparse(url).netloc.lower() in _YOUTUBE_HOSTS:
            return url
    return None


def _extract_booking_error_message(response: httpx.Response) -> str | None:
    try:
        data = response.json()
    except ValueError:
        return None
    if isinstance(data, dict):
        message = data.get("message") or data.get("error")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return None


async def extract_booking_slot(state: GraphState) -> dict:
    """3가지 신청 액션(자습/안마의자/기상음악) 중 무엇인지 분류한다. 확인 단계 없이,
    슬롯이 다 채워지면 바로 execute_booking으로 보낸다(booking_ready=True).
    기상음악은 유튜브 링크가 없으면 되묻고, 취소 의도는 안내만 하고 종료한다
    (booking_ready=False — route_after_booking_slot이 이 경우 END로 보낸다)."""
    question = state["messages"][-1].content

    try:
        intent = await booking_intent_llm.ainvoke([
            SystemMessage(content=BOOKING_INTENT_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ])
        action = intent.action
    except Exception:
        action = "unclear"

    if action == "cancel":
        reply = "취소는 사이트에서 직접 진행해주세요."
        return {
            "booking_ready": False,
            "messages": state["messages"] + [AIMessage(content=reply)],
        }

    if action == "unclear":
        reply = "자습실/안마의자/기상음악 중 어떤 것을 신청하시겠어요?"
        return {
            "booking_ready": False,
            "messages": state["messages"] + [AIMessage(content=reply)],
        }

    if action == "apply_music":
        music_url = _extract_youtube_url(question)
        if music_url is None:
            reply = "기상음악으로 등록할 유튜브 링크를 알려주세요."
            return {
                "booking_action": action,
                "booking_ready": False,
                "messages": state["messages"] + [AIMessage(content=reply)],
            }
        return {"booking_action": action, "booking_slots": {"musicUrl": music_url}, "booking_ready": True}

    # apply_study / apply_massage: 별도 슬롯 없이 바로 실행 대상.
    return {"booking_action": action, "booking_slots": {}, "booking_ready": True}


async def execute_booking(state: GraphState) -> dict:
    """auth_token으로 실제 신청 API를 호출한다. 성공/인증만료/신청불가/서버·네트워크오류
    4갈래로 안내하며, 어떤 경우에도 예외로 죽지 않는다. auth_token 값은 로그에 남기지 않는다.

    grade_answer를 거치긴 하지만(route_after_grade_answer 참고) booking은 재시도 대상에서
    제외되므로, 다른 노드들과 달리 answer_retry_count를 올리지 않는다 — 실제 신청을
    중복 호출하지 않기 위함이다."""
    auth_token = state.get("auth_token")

    if not auth_token:
        reply = "로그인이 필요합니다."
        return {"messages": state["messages"] + [AIMessage(content=reply)]}

    action = state["booking_action"]

    try:
        method, path = BOOKING_ENDPOINTS[action]
        body = state.get("booking_slots") or {}
        bearer_token = auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"

        async with httpx.AsyncClient(base_url=BOOKING_API_BASE_URL, timeout=10.0) as client:
            response = await client.request(
                method, path, json=body, headers={"Authorization": bearer_token}
            )

        if response.is_success:
            reply = BOOKING_SUCCESS_MESSAGES[action]
        elif response.status_code == 401:
            reply = "로그인이 만료됐어요. 다시 로그인 후 시도해주세요."
        elif response.status_code in (400, 409):
            reply = _extract_booking_error_message(response) or "지금은 신청할 수 없어요."
        else:
            reply = "처리 중 문제가 발생했어요. 잠시 후 다시 시도해주세요."
    except Exception:
        reply = "처리 중 문제가 발생했어요. 잠시 후 다시 시도해주세요."

    return {"messages": state["messages"] + [AIMessage(content=reply)]}


async def general_chat(state: GraphState) -> dict:
    """RAG 검색/채점은 거치지 않지만, generate와 동일하게 grade_answer 검증 대상이다."""
    update = _bump_answer_retry_if_reentering(state)
    question = state["messages"][-1].content
    response = await llm.ainvoke([
        SystemMessage(content=GENERAL_CHAT_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ])
    return {**update, "messages": state["messages"] + [AIMessage(content=response.content)]}


async def retrieve(state: GraphState) -> dict:
    """MMR 검색(k=7, fetch_k=20, lambda_mult=0.8). transform_query가 재작성한
    query가 있으면 그걸 쓰고, 없으면(첫 retrieve) 원본 질문으로 초기화한다."""
    original_query = state.get("original_query") or state["messages"][-1].content
    query = state.get("query") or original_query
    docs = await retriever.ainvoke(query)
    return {"context": docs, "query": query, "original_query": original_query}


async def grade_documents(state: GraphState) -> dict:
    """문서별 cosine 유사도로 1차 필터링하고, 애매한 구간만 LLM 그레이더에게 넘긴다."""
    question = state["messages"][-1].content
    context = state.get("context") or []

    if not context:
        return {"context": [], "documents_grade": "not_relevant"}

    embeddings = retriever.vectorstore.embeddings
    query_embedding = await embeddings.aembed_query(question)
    doc_embeddings = await embeddings.aembed_documents([doc.page_content for doc in context])

    relevant_docs = []
    for doc, doc_embedding in zip(context, doc_embeddings):
        score = _cosine_similarity(query_embedding, doc_embedding)

        if score >= DOCUMENT_RELEVANCE_THRESHOLD:
            is_relevant = True
        elif score < DOCUMENT_IRRELEVANCE_THRESHOLD:
            is_relevant = False
        else:
            try:
                grade = await document_grader_llm.ainvoke([
                    SystemMessage(content=DOCUMENT_GRADE_SYSTEM_PROMPT),
                    HumanMessage(content=f"[질문]\n{question}\n\n[문서]\n{doc.page_content}"),
                ])
                is_relevant = grade.binary_score
            except Exception:
                is_relevant = False  # 그레이딩 실패 시 관련없음으로 안전하게 폴백

        if is_relevant:
            relevant_docs.append(doc)

    documents_grade = "relevant" if relevant_docs else "not_relevant"
    return {"context": relevant_docs, "documents_grade": documents_grade}


async def transform_query(state: GraphState) -> dict:
    """query만 재작성하고 original_query는 그대로 보존한다."""
    original_query = state.get("original_query") or state["messages"][-1].content
    response = await llm.ainvoke([
        SystemMessage(content=TRANSFORM_QUERY_SYSTEM_PROMPT),
        HumanMessage(content=original_query),
    ])
    rewritten_query = response.content.strip()

    return {
        "query": rewritten_query,
        "original_query": original_query,
        "retrieve_retry_count": state.get("retrieve_retry_count", 0) + 1,
    }


async def web_search(state: GraphState) -> dict:
    """Tavily로 폴백 검색한다. 도구가 없거나(TAVILY_API_KEY 미설정) 호출이 실패하거나
    결과가 없으면 빈 context를 반환해 generate가 "모른다" 답변으로 안전하게 폴백한다.

    Tavily에 넘기기 전에 인사말/추임새를 걷어낸 검색어로 정제한다 — "안녕! 오늘 날씨
    어때?"처럼 인사말이 섞인 원문 그대로 검색하면 날씨 자체가 아니라 "날씨 표현/동요"
    같은 무관한 콘텐츠가 걸리는 게 실측으로 확인됐다. 정제 실패 시 원문 질문으로 폴백한다."""
    if web_search_tool is None:
        return {"context": []}

    raw_query = state.get("query") or state["messages"][-1].content

    try:
        query_rewrite = await llm.ainvoke([
            SystemMessage(content=WEB_SEARCH_QUERY_SYSTEM_PROMPT),
            HumanMessage(content=raw_query),
        ])
        query = query_rewrite.content.strip() or raw_query
    except Exception:
        query = raw_query

    try:
        raw_results = await web_search_tool.ainvoke({"query": query})
    except Exception:
        return {"context": []}

    results = raw_results.get("results") if isinstance(raw_results, dict) else None
    if not results:
        return {"context": []}

    docs = [
        Document(
            page_content=result["content"],
            metadata={"source": result.get("url", ""), "title": result.get("title", "")},
        )
        for result in results
        if result.get("content")
    ]
    return {"context": docs}


async def generate(state: GraphState) -> dict:
    """검색/웹검색 context로만 답변 생성. context 가 비면 '모른다' 답변으로 분기.

    grade_answer 재시도 시(answer_grade="fail") 같은 context로만 재생성하며,
    재검색이나 query 재작성은 하지 않는다.
    """
    update = _bump_answer_retry_if_reentering(state)

    context = state.get("context") or []

    is_web_search = state.get("mode") == "web_search"

    if not context:
        if is_web_search:
            reply = WEB_SEARCH_NO_INFO_REPLY
        else:
            reply = "관련 문서를 찾을 수 없어 답변드릴 수 없습니다. 학교 담당 부서에 문의해 주세요."
        return {**update, "messages": state["messages"] + [AIMessage(content=reply)]}

    question = state["messages"][-1].content
    context_text = "\n\n".join(
        f"{doc.page_content}\n(출처: {doc.metadata['source']})" if doc.metadata.get("source") else doc.page_content
        for doc in context
    )

    system_prompt = GENERATE_SYSTEM_PROMPT + GENERATE_WEB_SEARCH_ADDENDUM if is_web_search else GENERATE_SYSTEM_PROMPT

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"[문서]\n{context_text}\n\n[질문]\n{question}"),
    ])
    answer = response.content

    # 출처가 있는 문서(web_search 결과)인데 LLM이 인용을 빠뜨리면, 프롬프트 준수에만
    # 맡기지 않고 코드에서 확정적으로 붙여준다. source가 없는 RAG 문서는 애초에
    # 이 목록이 비어 있어서 아무것도 덧붙지 않는다. 단, "정보를 못 찾았다"고 정직하게
    # 물러난 답변에는 그 근거가 되지 않은 출처를 붙이면 오히려 오해를 주므로 붙이지 않는다.
    if "(출처:" not in answer and WEB_SEARCH_NO_INFO_MARKER not in answer:
        sources = [doc.metadata["source"] for doc in context if doc.metadata.get("source")]
        if sources:
            answer = f"{answer}\n\n(출처: {sources[0]})"

    return {**update, "messages": state["messages"] + [AIMessage(content=answer)]}


async def grade_answer(state: GraphState) -> dict:
    """grounded_in_context와 addresses_question이 둘 다 true여야 "pass"."""
    question = _last_message_content(state["messages"], HumanMessage)
    answer = _last_message_content(state["messages"], AIMessage)
    context = state.get("context") or []
    context_text = "\n\n".join(doc.page_content for doc in context)

    try:
        grade = await answer_grader_llm.ainvoke([
            SystemMessage(content=ANSWER_GRADE_SYSTEM_PROMPT),
            HumanMessage(content=f"[문서]\n{context_text}\n\n[질문]\n{question}\n\n[답변]\n{answer}"),
        ])
        passed = grade.grounded_in_context and grade.addresses_question
    except Exception:
        passed = False

    return {"answer_grade": "pass" if passed else "fail"}
