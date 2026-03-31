from fastapi import FastAPI
from app.api.chat_API import router as chat_router

app = FastAPI(title="학교 챗봇 API")

app.include_router(chat_router)