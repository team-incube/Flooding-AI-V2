import asyncio

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

import app.langgraph_services.nodes as nodes
from app.langgraph_services.graph import app_graph
from app.langgraph_services.nodes import MAX_ANSWER_RETRY, AnswerGrade, grade_answer, grade_documents


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
        grade_answer 로직은 그대로 두고 nodes 모듈 전역 이름만 통째로 바꿔치기한다."""

        async def ainvoke(self, *_args, **_kwargs):
            return AnswerGrade(grounded_in_context=False, addresses_question=False)

    original_grader = nodes.answer_grader_llm
    nodes.answer_grader_llm = _AlwaysFailGrader()
    try:
        result3 = await app_graph.ainvoke({
            "messages": [HumanMessage(content="다크모드는 어떻게 켜나요?")],
            "mode": "vectorstore",
            "context": [],
        })
    finally:
        nodes.answer_grader_llm = original_grader

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
