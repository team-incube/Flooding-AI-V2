from pydantic import BaseModel

class UserInput(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    response: str