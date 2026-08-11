import os
from typing import Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from app.langgraph_services.prompts import (
    ANSWER_GRADE_SYSTEM_PROMPT,
    DOCUMENT_GRADE_SYSTEM_PROMPT,
    GENERAL_CHAT_SYSTEM_PROMPT,
    ROUTE_MODES,
    ROUTE_SYSTEM_PROMPT,
    TRANSFORM_QUERY_SYSTEM_PROMPT,
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


document_grader_llm = llm.with_structured_output(DocumentRelevanceGrade)
answer_grader_llm = llm.with_structured_output(AnswerGrade)


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

    booking은 아직 실제 예약 서브그래프가 없어 자리표시(booking_stub)로 응답한다.
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


async def booking_stub(state: GraphState) -> dict:
    """다음 단계(3단계)에서 실제 예약 서브그래프로 교체될 자리표시. grade_answer도 다른 경로와
    동일하게 거치므로, 재시도 진입 시 answer_retry_count도 똑같이 올려준다."""
    update = _bump_answer_retry_if_reentering(state)
    reply = "예약 기능은 아직 준비 중입니다. (booking 서브그래프는 다음 단계에서 구현)"
    return {**update, "messages": state["messages"] + [AIMessage(content=reply)]}


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
    결과가 없으면 빈 context를 반환해 generate가 "모른다" 답변으로 안전하게 폴백한다."""
    if web_search_tool is None:
        return {"context": []}

    query = state.get("query") or state["messages"][-1].content

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

    if not context:
        if state.get("mode") == "web_search":
            reply = "죄송해요, 관련 정보를 찾지 못했어요."
        else:
            reply = "관련 문서를 찾을 수 없어 답변드릴 수 없습니다. 학교 담당 부서에 문의해 주세요."
        return {**update, "messages": state["messages"] + [AIMessage(content=reply)]}

    question = state["messages"][-1].content
    context_text = "\n\n".join(
        f"{doc.page_content}\n(출처: {doc.metadata['source']})" if doc.metadata.get("source") else doc.page_content
        for doc in context
    )

    prompt = (
        "당신은 학교 웹사이트(Flooding)의 AI 챗봇입니다. "
        "아래 문서 내용만 사용해서 질문에 답변하세요. 존댓말을 사용하세요. "
        "문서에 (출처: ...)가 붙어 있으면 답변 마지막에 참고한 출처를 밝히세요.\n\n"
        f"[문서]\n{context_text}\n\n"
        f"[질문]\n{question}"
    )
    response = await llm.ainvoke(prompt)
    return {**update, "messages": state["messages"] + [AIMessage(content=response.content)]}


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
