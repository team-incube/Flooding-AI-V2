from pydantic import BaseModel, Field

class UserInput(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    response: str

class SongRequest(BaseModel):
    title: str = Field(..., min_length=1)
    artist: str = Field(..., min_length=1)


class MusicLinksRequest(BaseModel):
    recent_songs: list[SongRequest] = Field(default_factory=list, max_length=5)


class MusicLinksResponse(BaseModel):
    youtube_links: list[str] = Field(..., min_length=3, max_length=3)