from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from app.services.embedding import get_retriever
from dotenv import load_dotenv
import asyncio
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

retriever = get_retriever()

import logging
logging.basicConfig(level=logging.DEBUG)

SYSTEM_PROMPT = """
당신은 학교 웹사이트에서 동아리 및 기숙사 정보를 안내하는 AI 챗봇입니다.
사용자의 질문에 친절하고 정확하게 답변하며, 반드시 제공된 문서 정보만을 기반으로 답변합니다.

[담당 기능]
- 회원가입 / 로그인 방법 안내
- 동아리 신청 및 조회 방법 안내
- 기숙사 신청 및 조회 방법 안내
- 기타 웹사이트 주요 기능 사용법 안내

[답변 규칙]
- 항상 존댓말을 사용합니다.
- 사용법은 번호를 붙여 단계별로 설명합니다.
- 사용자가 직접 따라할 수 있도록 구체적으로 설명합니다.
- 간결하고 명확하게 설명하되, 필요한 정보는 빠짐없이 포함합니다.
- 질문의 의도가 불명확한 경우, 되묻지 않고 가장 가능성 높은 의도로 답변합니다.

[도구 사용 규칙]
- 모든 답변 전에 반드시 search_document 도구를 사용하여 관련 문서를 검색합니다.
- 검색 결과가 없거나 관련 정보를 찾지 못한 경우 "해당 내용은 확인이 필요합니다. 학교 담당 부서에 문의해 주세요."라고 안내합니다.

[금지 사항]
- 문서에 없는 내용을 추측하거나 임의로 답변하지 않습니다.
- 존재하지 않는 기능을 설명하지 않습니다.
- 개인정보(학번, 비밀번호 등) 입력을 직접 요구하지 않습니다.
"""


@tool
def search_document(query: str) -> str:
    """동아리 및 기숙사 관리 웹사이트 사용법 등 학교 관련 정보를 검색합니다."""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs) if docs else "관련 문서를 찾을 수 없습니다."


llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

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