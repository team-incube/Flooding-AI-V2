from fastapi import APIRouter, Header, HTTPException
from app.schemas import UserInput, ChatResponse
from app.langgraph_services.graph import ask

router = APIRouter(prefix="/ai", tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat(request: UserInput, authorization: str | None = Header(default=None)):
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="메시지를 입력해주세요.")
    reply = await ask(request.user_input, auth_token=authorization)
    return ChatResponse(response=reply)