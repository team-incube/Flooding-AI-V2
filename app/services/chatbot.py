from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from app.services.embedding import get_retriever
from dotenv import load_dotenv
import asyncio
import os


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
os.getenv("OPENAI_API_KEY")

retriever = get_retriever()


SYSTEM_PROMPT = """당신은 학교 웹사이트(Flooding)의 AI 챗봇입니다. 아래 두 가지 방식으로 질문에 답변합니다.

[절대 금지] 답변 마지막에 "이상입니다.", "감사합니다.", "추가로 궁금하신 점이 있으면 말씀해 주세요." 같은 마무리 문장을 쓰지 않습니다. 답변 내용이 끝나면 바로 종료합니다.

1. Flooding 사용법 질문 (신청 방법, 절차, 기능 안내 등):
   - 반드시 search_document 도구를 사용하여 관련 문서를 검색합니다.
   - 검색된 문서 내용만 사용합니다. 문서에 없는 내용은 절대 추가하지 않습니다.
   - 검색된 문서에 Q&A 형식의 답변이 있으면 그 내용을 기반으로 답변합니다.
   - 문서에 관련 내용이 있으면 추가 질문 없이 바로 답변합니다. 절대 되묻지 않습니다.
   - 문서에 관련 내용이 전혀 없을 때만 "학교 담당 부서에 문의해 주세요."라고 합니다.
   - 단계는 번호를 붙여 설명합니다.

2. 그 외 모든 질문 (인사, 추천, 아이디어, 일반 대화 등):
   - 자유롭게 답변합니다.

공통 규칙:
- 존댓말을 사용합니다."""


@tool
def search_document(query: str) -> str:
    """동아리 및 기숙사 관리 웹사이트 사용법 등 학교 관련 정보를 검색합니다."""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs) if docs else "관련 문서를 찾을 수 없습니다."


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_agent(
    model=llm,
    tools=[search_document],
    system_prompt=SYSTEM_PROMPT,
)


async def ask(user_input: str) -> str:
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: agent.invoke({"messages": [("human", user_input)]})
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    user_input = input("\n나: ").strip()
    print("AI:", asyncio.run(ask(user_input)))