from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# Serve your frontend directly from Python so you only need 1 deployment
@app.get("/")
def read_index():
    return FileResponse("index.html")

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    # TODO: Plug your PDF reading logic / LLM agent here later
    reply = f"Agent received: {request.message}"
    return {"response": reply}
