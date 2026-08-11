import asyncio
import os
from typing import Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.services.embedding import get_retriever

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "services", ".env"))

retriever = get_retriever()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

ROUTE_MODES = ("vectorstore", "booking", "general_chat")

ROUTE_SYSTEM_PROMPT = """당신은 학교 웹사이트(Flooding)의 챗봇 라우터입니다.
사용자 질문을 아래 세 카테고리 중 정확히 하나로 분류하세요.

- vectorstore: 웹사이트 사용법 문의 (예: "다크모드 어떻게 켜요", "세탁기 사용법", "동아리 신청 방법이 뭐예요")
- booking: 자습실/안마의자/노래신청처럼 실제로 무언가를 예약·신청하려는 의도 (예: "스터디룸 예약할래요", "안마의자 신청하고 싶어요")
- general_chat: 위 둘 다 아닌 잡담이나 사이트 범위 밖의 질문 (예: "화성에 사람이 살 수 있나요")

반드시 vectorstore, booking, general_chat 중 하나의 단어만 정확히 출력하세요. 다른 설명은 절대 덧붙이지 마세요."""

# grade_documents 1차 필터용 cosine 임계값. text-embedding-3-small 기준 실측하며 튜닝할 것.
# score >= DOCUMENT_RELEVANCE_THRESHOLD 면 LLM 호출 없이 즉시 관련있음으로 판단.
# score < DOCUMENT_IRRELEVANCE_THRESHOLD 면 LLM 호출 없이 즉시 관련없음으로 판단.
# 그 사이(애매한 구간)만 LLM 그레이더에게 넘긴다.
DOCUMENT_RELEVANCE_THRESHOLD = 0.75
DOCUMENT_IRRELEVANCE_THRESHOLD = 0.3

# RAG spine의 두 재시도 루프는 서로 다른 카운터를 쓰고, 절대 서로의 카운터를 건드리지 않는다.
# retrieve_retry_count: grade_documents가 "관련없음"일 때 transform_query -> retrieve 재검색 허용 횟수.
# answer_retry_count: grade_answer가 "실패"일 때 같은 context로 generate 재생성 허용 횟수.
MAX_RETRIEVE_RETRY = 1
MAX_ANSWER_RETRY = 1

TRANSFORM_QUERY_SYSTEM_PROMPT = """당신은 벡터스토어 재검색을 돕는 질문 재작성기입니다.
사용자의 원래 의도는 유지한 채, 벡터 검색에서 더 잘 걸리도록 핵심 키워드를 보강하거나
다른 표현으로 바꿔 한 문장으로 다시 작성하세요. 재작성한 질문 문장만 출력하고
다른 설명은 절대 덧붙이지 마세요."""

DOCUMENT_GRADE_SYSTEM_PROMPT = """당신은 검색된 문서가 사용자 질문과 관련이 있는지 평가하는 채점자입니다.
문서에 질문에 답하는 데 도움이 되는 키워드나 의미적 내용이 포함되어 있으면 관련 있다고(true) 판단하세요.
완벽히 일치하는 답이 아니어도, 질문과 주제가 통하면 관련 있다고 판단합니다."""

ANSWER_GRADE_SYSTEM_PROMPT = """당신은 챗봇 답변의 품질을 평가하는 채점자입니다.
아래 문서, 질문, 답변을 보고 두 가지를 평가하세요.
1. grounded_in_context: 답변의 내용이 주어진 문서에서 실제로 확인되면 true, 문서에 없는 내용을 지어냈다면(환각) false.
2. addresses_question: 답변이 실제로 질문이 요구하는 내용에 답하고 있으면 true, 동문서답이면 false."""


class DocumentRelevanceGrade(BaseModel):
    binary_score: bool = Field(description="검색된 문서가 질문과 관련 있으면 true, 없으면 false")


class AnswerGrade(BaseModel):
    grounded_in_context: bool = Field(description="답변이 문서/컨텍스트에 근거하고 있으면 true, 환각이면 false")
    addresses_question: bool = Field(description="답변이 실제로 질문에 답하고 있으면 true, 아니면 false")


document_grader_llm = llm.with_structured_output(DocumentRelevanceGrade)
answer_grader_llm = llm.with_structured_output(AnswerGrade)


class GraphState(TypedDict):
    messages: list[AnyMessage]
    mode: Literal["vectorstore", "booking", "general_chat"]
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
    """LLM으로 질문을 vectorstore / booking / general_chat 로 분류한다.

    지금 단계에선 vectorstore 경로만 실제로 동작한다.
    booking / general_chat 은 다음 단계까지의 자리표시용 스텁이다.
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


def route_after_question(state: GraphState) -> str:
    if state["mode"] == "booking":
        return "booking_stub"
    if state["mode"] == "general_chat":
        return "general_chat_stub"
    return "retrieve"


async def booking_stub(state: GraphState) -> dict:
    """다음 단계(3단계)에서 실제 예약 서브그래프로 교체될 자리표시."""
    reply = "예약 기능은 아직 준비 중입니다. (booking 서브그래프는 다음 단계에서 구현)"
    return {"messages": state["messages"] + [AIMessage(content=reply)]}


async def general_chat_stub(state: GraphState) -> dict:
    """다음 단계에서 실제 일반 대화 응답으로 교체될 자리표시."""
    reply = "일반 대화 응답은 아직 준비 중입니다. (general_chat 분기는 다음 단계에서 구현)"
    return {"messages": state["messages"] + [AIMessage(content=reply)]}


async def retrieve(state: GraphState) -> dict:
    """embedding.py 의 get_retriever() (MMR k=7, fetch_k=20, lambda_mult=0.8) 로 검색.

    state["query"] 가 있으면(transform_query가 재작성한 검색어) 그걸로 검색하고,
    턴의 첫 retrieve 라면 원본 질문을 query/original_query 로 채워둔다.
    original_query는 이후 transform_query가 재작성을 반복해도 절대 덮어쓰지 않는다.
    """
    original_query = state.get("original_query") or state["messages"][-1].content
    query = state.get("query") or original_query
    docs = await retriever.ainvoke(query)
    return {"context": docs, "query": query, "original_query": original_query}


async def grade_documents(state: GraphState) -> dict:
    """RAG spine 의 문서 관련성 채점 (아직 그래프에 연결되지 않은 독립 노드).

    각 문서를 cosine 유사도로 1차 필터링한다.
    - score >= DOCUMENT_RELEVANCE_THRESHOLD: 즉시 관련있음
    - score < DOCUMENT_IRRELEVANCE_THRESHOLD: 즉시 관련없음
    - 그 사이(애매한 경우)만 LLM에게 True/False 그레이딩을 요청한다.
    관련있다고 판단된 문서만 남겨 context 를 갱신하고, 하나라도 남으면 "relevant".
    """
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
                is_relevant = False  # 파싱/호출 실패 시 안전하게 관련없음으로 폴백

        if is_relevant:
            relevant_docs.append(doc)

    documents_grade = "relevant" if relevant_docs else "not_relevant"
    return {"context": relevant_docs, "documents_grade": documents_grade}


async def transform_query(state: GraphState) -> dict:
    """재검색을 위해 질문을 재작성한다. original_query는 보존하고 query만 갱신하며,
    retrieve_retry_count를 1 증가시켜 retrieve로 되돌아간다."""
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
    """재검색까지 소진된 뒤의 폴백. 결과는 재채점 없이 바로 generate로 간다.

    booking_stub / general_chat_stub과 같은 자리표시 노드다. 이 프로젝트에는 아직
    실제 웹 검색 도구(Tavily/DDG 등)가 연동되어 있지 않아서, 지금은 빈 context를
    반환해 generate가 "모른다" 답변으로 안전하게 폴백하게 한다. 실제 검색 API 연동은
    별도 작업으로 진행할 것.
    """
    return {"context": []}


async def generate(state: GraphState) -> dict:
    """검색된 context 로만 답변 생성. context 가 비면 '모른다' 답변으로 분기.

    state["answer_grade"] 가 이미 "fail" 인 채로 다시 들어오면(=grade_answer 재시도
    경로) answer_retry_count를 1 증가시킨다. 이 루프는 같은 context로만 재생성하며
    재검색이나 query 재작성은 하지 않는다.
    """
    update: dict = {}
    if state.get("answer_grade") == "fail":
        update["answer_retry_count"] = state.get("answer_retry_count", 0) + 1

    context = state.get("context") or []

    if not context:
        reply = "관련 문서를 찾을 수 없어 답변드릴 수 없습니다. 학교 담당 부서에 문의해 주세요."
        return {**update, "messages": state["messages"] + [AIMessage(content=reply)]}

    question = state["messages"][-1].content
    context_text = "\n\n".join(doc.page_content for doc in context)

    prompt = (
        "당신은 학교 웹사이트(Flooding)의 AI 챗봇입니다. "
        "아래 문서 내용만 사용해서 질문에 답변하세요. 존댓말을 사용하세요.\n\n"
        f"[문서]\n{context_text}\n\n"
        f"[질문]\n{question}"
    )
    response = await llm.ainvoke(prompt)
    return {**update, "messages": state["messages"] + [AIMessage(content=response.content)]}


async def grade_answer(state: GraphState) -> dict:
    """generate 답변의 환각 여부와 질문 적합성을 채점 (아직 그래프에 연결되지 않은 독립 노드).

    grounded_in_context(환각 아님)와 addresses_question(질문에 답함)이 둘 다 true 여야 "pass".
    하나라도 false 이거나 그레이딩 자체가 실패하면 "fail".
    """
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
        passed = False  # 파싱/호출 실패 시 안전하게 실패로 폴백 (재시도 경로를 타도록)

    return {"answer_grade": "pass" if passed else "fail"}


def route_after_grade_documents(state: GraphState) -> str:
    """retrieve_retry_count < MAX_RETRIEVE_RETRY(1) 이면 transform_query로 재검색,
    이미 소진됐으면 web_search로 폴백. 두 경우 다 재채점 루프로 다시 들어오지 않는다."""
    if state.get("documents_grade") == "relevant":
        return "generate"
    if state.get("retrieve_retry_count", 0) < MAX_RETRIEVE_RETRY:
        return "transform_query"
    return "web_search"


def route_after_grade_answer(state: GraphState) -> str:
    """answer_retry_count < MAX_ANSWER_RETRY(1) 이면 같은 context로 generate 재시도,
    소진됐으면 최선의 답변으로 그냥 END (무한루프 방지)."""
    if state.get("answer_grade") == "pass":
        return END
    if state.get("answer_retry_count", 0) < MAX_ANSWER_RETRY:
        return "generate"
    return END


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("route_question", route_question)
    graph.add_node("booking_stub", booking_stub)
    graph.add_node("general_chat_stub", general_chat_stub)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("transform_query", transform_query)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)
    graph.add_node("grade_answer", grade_answer)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges(
        "route_question",
        route_after_question,
        {
            "booking_stub": "booking_stub",
            "general_chat_stub": "general_chat_stub",
            "retrieve": "retrieve",
        },
    )
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grade_documents,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "web_search": "web_search",
        },
    )
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", "grade_answer")
    graph.add_conditional_edges(
        "grade_answer",
        route_after_grade_answer,
        {
            "generate": "generate",
            END: END,
        },
    )
    graph.add_edge("booking_stub", END)
    graph.add_edge("general_chat_stub", END)

    return graph.compile()


app_graph = build_graph()


async def ask(user_input: str) -> str:
    result = await app_graph.ainvoke({
        "messages": [HumanMessage(content=user_input)],
        "mode": "vectorstore",
        "context": [],
    })
    return result["messages"][-1].content


async def _test_graders():
    """grade_documents / grade_answer 단독(함수 단위) 테스트. 그래프를 통하지 않고 직접 호출한다."""
    relevant_doc = Document(
        page_content="다크모드는 헤더 우측에 있는 달 아이콘을 클릭하면 켜집니다. 다시 클릭하면 라이트모드로 전환됩니다.",
        metadata={"id": "settings"},
    )
    irrelevant_doc = Document(
        page_content="동아리 개설 신청은 학기 초 2주간 가능하며, 지도교사 승인이 필요합니다.",
        metadata={"id": "club"},
    )
    question = "다크모드는 어떻게 켜나요?"

    print("\n=== grade_documents 단독 테스트 ===")

    case1 = await grade_documents({
        "messages": [HumanMessage(content=question)],
        "mode": "vectorstore",
        "context": [relevant_doc],
    })
    print(f"1) 관련 있는 문서 -> documents_grade={case1['documents_grade']}")

    case2 = await grade_documents({
        "messages": [HumanMessage(content=question)],
        "mode": "vectorstore",
        "context": [irrelevant_doc],
    })
    print(f"2) 관련 없는 문서 -> documents_grade={case2['documents_grade']}")

    print("\n=== grade_answer 단독 테스트 ===")

    grounded_answer = "다크모드는 헤더 우측에 있는 달 아이콘을 클릭하면 켜집니다."
    case3 = await grade_answer({
        "messages": [HumanMessage(content=question), AIMessage(content=grounded_answer)],
        "mode": "vectorstore",
        "context": [relevant_doc],
    })
    print(f"3) 문서에 근거한 답변 -> answer_grade={case3['answer_grade']}")

    hallucinated_answer = "다크모드를 켜려면 설정 메뉴에서 '테마 변경 코드'를 입력하고 관리자에게 이메일로 인증 요청을 보내야 합니다."
    case4 = await grade_answer({
        "messages": [HumanMessage(content=question), AIMessage(content=hallucinated_answer)],
        "mode": "vectorstore",
        "context": [relevant_doc],
    })
    print(f"4) 지어낸(환각) 답변 -> answer_grade={case4['answer_grade']}")


async def _test_rag_spine():
    """RAG spine 조건부 엣지 전체 배선 확인용 통합 테스트. app_graph를 통해 실행하고
    두 재시도 카운터가 spec대로 끝나는지 확인한다."""
    print("\n=== RAG spine 통합 테스트 ===")

    print("\n1) 문서에 바로 있는 질문 -> relevant -> generate -> pass -> END")
    result1 = await app_graph.ainvoke({
        "messages": [HumanMessage(content="다크모드는 어떻게 켜나요?")],
        "mode": "vectorstore",
        "context": [],
    })
    print(
        f"   documents_grade={result1.get('documents_grade')} "
        f"answer_grade={result1.get('answer_grade')} "
        f"retrieve_retry_count={result1.get('retrieve_retry_count', 0)} "
        f"answer_retry_count={result1.get('answer_retry_count', 0)}"
    )
    print(f"   AI: {result1['messages'][-1].content}")
    assert result1.get("retrieve_retry_count", 0) == 0
    assert result1.get("answer_retry_count", 0) == 0

    print("\n2) 문서에 없는 질문 -> not_relevant -> transform_query -> retrieve"
          " -> 그래도 not_relevant -> web_search -> generate")
    result2 = await app_graph.ainvoke({
        "messages": [HumanMessage(content="웹사이트 폰트 크기는 어떻게 조절하나요?")],
        "mode": "vectorstore",
        "context": [],
    })
    print(
        f"   documents_grade={result2.get('documents_grade')} "
        f"retrieve_retry_count={result2.get('retrieve_retry_count', 0)} "
        f"query={result2.get('query')!r}"
    )
    print(f"   AI: {result2['messages'][-1].content}")
    assert result2.get("retrieve_retry_count", 0) == 1

    print("\n3) grade_answer를 강제로 실패시켜 generate 재진입 -> 상한 후 END")

    class _AlwaysFailGrader:
        """answer_grader_llm은 pydantic RunnableSequence라 속성 몽키패치가 막혀 있어,
        grade_answer 로직은 그대로 두고 모듈 전역 이름만 통째로 바꿔치기한다."""

        async def ainvoke(self, *_args, **_kwargs):
            return AnswerGrade(grounded_in_context=False, addresses_question=False)

    global answer_grader_llm
    original_grader = answer_grader_llm
    answer_grader_llm = _AlwaysFailGrader()
    try:
        result3 = await app_graph.ainvoke({
            "messages": [HumanMessage(content="다크모드는 어떻게 켜나요?")],
            "mode": "vectorstore",
            "context": [],
        })
    finally:
        answer_grader_llm = original_grader

    print(
        f"   answer_grade={result3.get('answer_grade')} "
        f"answer_retry_count={result3.get('answer_retry_count', 0)}"
    )
    print(f"   AI: {result3['messages'][-1].content}")
    assert result3.get("answer_grade") == "fail"
    assert result3.get("answer_retry_count", 0) == MAX_ANSWER_RETRY

    print("\n=== RAG spine 통합 테스트 통과 ===")


if __name__ == "__main__":
    sample_questions = [
        "기숙사 세탁기는 어떻게 신청하나요?",
        "다크모드는 어떻게 켜나요?",
        "화성에 사람이 살 수 있나요?",
        "안마의자 예약하고 싶어요",
        "기숙사 세탁기는 어떻게 신청하나요",
    ]

    async def main():
        for q in sample_questions:
            print(f"\n나: {q}")
            result = await app_graph.ainvoke({
                "messages": [HumanMessage(content=q)],
                "mode": "vectorstore",
                "context": [],
            })
            print(f"[mode: {result['mode']}]")
            print(f"AI: {result['messages'][-1].content}")

        await _test_graders()
        await _test_rag_spine()

    asyncio.run(main())
