from fastapi import APIRouter, HTTPException
from app.schemas import UserInput, ChatResponse
from app.services.chatbot import ask

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def chat(request: UserInput):
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="메시지를 입력해주세요.")
    reply = await ask(request.user_input)
    return ChatResponse(response=reply)