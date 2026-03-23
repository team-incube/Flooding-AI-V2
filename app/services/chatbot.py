from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from embedding import get_retriever
from dotenv import load_dotenv
import os

load_dotenv()

retriever = get_retriever()

@tool
def search_document(query: str) -> str:
    """동아리 및 기숙사 관리 웹사이트 사용법 등 학교 관련 정보를 검색합니다."""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs) if docs else "관련 문서를 찾을 수 없습니다."

llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[search_document],
    prompt="""
당신은 학교 웹사이트에서 동아리 및 기숙사 정보를 안내하는 AI 챗봇입니다.

[기능]

- 웹사이트 사용 방법 안내:
  - 회원가입 방법
  - 로그인 방법
  - 동아리 신청 방법
  - 기숙사 신청 방법
  - 주요 기능 사용법

[사용법 안내 규칙]
- 사용법은 단계별로 설명합니다.
- 각 단계는 번호를 사용합니다.
- 사용자가 따라할 수 있도록 구체적으로 설명합니다.

[응답 규칙]
- 항상 존댓말 사용
- 간결하고 명확하게 설명
- 필요한 경우 단계별 설명

[데이터 규칙]
- 제공된 정보 기반으로만 답변
- 모르면 "확인이 필요합니다"라고 답변

[금지]
- 추측 금지
- 존재하지 않는 기능 설명 금지
"""
)

def ask(user_input: str) -> str:
    result = agent.invoke({"messages": [("human", user_input)]})
    return result["messages"][-1].content

if __name__ == "__main__":
    user_input = input("\n나: ").strip()
    print("AI:", ask(user_input))