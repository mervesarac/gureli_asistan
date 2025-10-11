import os
import uuid
import json
import logging
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from db_agent import bot_answer
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Gureli Denetim Asistanı API")
security = HTTPBasic()

# Simple auth: username == password
def same_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != credentials.password:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s",
    filename=os.path.join(os.path.dirname(__file__), "data", "analiz-bot.log"),
    filemode="a",
    encoding="utf-8"
)

# Serve static files (for chatbot.html and assets)
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.post("/chat")
async def chat_endpoint(request: Request, username: str = Depends(same_auth)):
    data = await request.json()
    message = data.get("text") or data.get("message")
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    if not message:
        return JSONResponse(status_code=400, content={"error": "Missing 'text' in request body."})
    logging.info(">" * 80)
    logging.info(f"[FROM {username}]:[{conversation_id}] {message}")
    response = await bot_answer(message, conversation_id, username)
    logging.info(f"[TO   {username}]:[{conversation_id}] {str(response)}")
    return {"response": response, "conversation_id": conversation_id}


# Auth-protected health check endpoint for login validation
@app.get("/healthz")
def health_check(username: str = Depends(same_auth)):
    return {"status": "ok"}


# Serve chatbot UI at root
@app.get("/")
def serve_chatbot_html():
    html_path = os.path.join(static_dir, "chatbot.html")
    return FileResponse(html_path, media_type="text/html")

# Example curl:
# curl -u user:user -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{"text": "Paperzero nasıl çalışır?"}'
