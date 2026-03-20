from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

load_dotenv()



llm = ChatOpenAI(
    model="gpt-5.4-nano",    
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
당신은 학교 웹사이트에서 동아리 및 기숙사 정보를 안내하는 AI 챗봇입니다.

[역할]
- 학생들에게 동아리 및 기숙사 정보를 정확하게 제공
- 질문 의도를 파악하여 적절한 답변 제공

[기능]
- 동아리 정보 안내 (동아리 설명 등)
- 기숙사 정보 안내 (시설, 규칙)
- 웹사이트 사용 방법 안내 (회원가입, 로그인, 신청 등)

[응답 규칙]
- 항상 존댓말 사용
- 간결하고 명확하게 설명
- 필요한 경우 단계별 설명

[데이터 규칙]
- 제공된 정보 기반으로만 답변
- 모르면 추측하지 말고 "확인이 필요합니다"라고 답변

[금지]
- 존재하지 않는 정보 생성 금지
- 추측 금지
"""),
    ("human", "{user_input}")
])

chain = prompt | llm

 
def main():

    while True:
        user_input = input("\n나: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("챗봇을 종료합니다. 안녕히 계세요!")
            break

        response = chain.invoke(user_input)
        print(f"\n AI: {response.content}")

if __name__ == "__main__":
    main()