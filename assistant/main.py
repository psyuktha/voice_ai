from fastapi import FastAPI, Form, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import json
import re
import os
import requests
import httpx  
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

load_dotenv()

vapi_key = os.getenv("VAPI_API_KEY")
assistant_id = os.getenv("VAPI_ASSISTANT_ID")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

active_connection = None
refined = []
chat_log = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_connection
    await websocket.accept()
    active_connection = websocket

    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        print(f"⚠️ WebSocket disconnected: {e}")

@app.get("/get-call-summary")
async def get_call_summary():
    try:
        print(vapi_key[:5]+"..."+vapi_key[-5:])  # Print first and last 5 characters of vapi_key
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.vapi.ai/call",
                headers={"Authorization": f"Bearer {vapi_key}"}
            )
            response.raise_for_status()
        summary_str = response.json()[0].get("summary")
        print("📄 Raw Summary:", response.json()[0])
        summary = json.loads(summary_str)
        stringified = json.dumps(summary, indent=4)
        print("Call summary:", stringified)

        if active_connection:
            await active_connection.send_text(stringified)
            print("✅ Sent summary to client")

        return JSONResponse(content={"summary": stringified})

    except Exception as e:
        print(f"❌ Error fetching summary: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
@app.post("/start-call")
async def start_call():
    if active_connection:
        await active_connection.send_text("start-call")
        return {"status": "trigger sent"}
    return {"error": "no client connected"}

GEMINI_PROMPT_TEMPLATE = """
You are a prompt refinement assistant for a voice AI agent. Given a raw user command, do the following:
1. Extract all possible entities from the user input. Use these keys:
   - target_name
   - location
   - time
   - item
   - intent_category (choose from: Inquiry, Appointment, Rescheduling, Finding Lost Items, Follow-up)
   - quantity (if applicable)
   - number of people (if applicable for appointments or scheduled events)
2. follow_up_questions 
   Add all possible follow-up questions that might be needed to clarify the user's intent. if any entities are missing, include them in the follow-up questions. Do not ask users name.
If it is an impossible event like a visit to another planet, return a message saying "This is not possible" and do not generate any entities or follow-up questions.
Respond a JSON object without any '```json ```' backticks with the following:
- entities (dictionary)
- follow_up_questions (array of strings)
"""

@app.post("/process")
async def process(request: Request, phone_number: str = Form(...), query: str = Form(...)):
    try:
        prompt = GEMINI_PROMPT_TEMPLATE + f"\nInput: {query}"
        model = genai.GenerativeModel("gemini-2.0-flash")

        response = model.generate_content(prompt)
        cleaned = re.sub(r'```json|```', '', response.text).strip()
        parsed = json.loads(cleaned)

        print(parsed)
        refined.clear()  
        refined.append(parsed.get("entities"))
        followups = parsed.get("follow_up_questions")

        print(f"Follow-up questions: {followups}")
        return JSONResponse(content={"follow_up_questions": followups})
    except Exception as e:
        return JSONResponse(content={"follow_up_questions": [], "error": str(e)}, status_code=500)


class QAItem(BaseModel):
    question: str
    answer: str

class ChatHistory(BaseModel):
    phone_number: str
    query: str
    follow_up_questions: List[QAItem]

@app.post("/save")
async def save_chat(data: ChatHistory):
    final = data.dict()
    final.update({"entities": refined})
    chat_log.append(final)
    print(f"📝 Chat log updated")
    return json.dumps(final, indent=4)

@app.get("/context")
async def get_context():
    if not chat_log:
        return JSONResponse(content={"error": "No context available yet."}, status_code=404)

    context = json.dumps(chat_log[-1], indent=4)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""You are an assistant that summarizes phone call conversations.

    Find out the final task the assistant needs to perform based on the conversation history.

    Conversation history:
    {context}
    """

    response = model.generate_content(prompt)
    return response.text


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/call-ui", response_class=HTMLResponse)
async def call_ui(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "vapi_key": vapi_key,
        "assistant_id": assistant_id
    })
if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
