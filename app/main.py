import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat_API import router as chat_router
from app.api.music_API import router as music_router

load_dotenv(os.path.join(os.path.dirname(__file__), "services", ".env"))

DEFAULT_ALLOWED_ORIGINS = "https://flooding.kr,http://localhost:3000"
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",")
    if origin.strip()
]

app = FastAPI(title="학교 챗봇 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(chat_router)

app.include_router(music_router)