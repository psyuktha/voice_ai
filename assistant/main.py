from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import json
import re
import os

# Initialize the app
app = FastAPI()
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Serve frontend HTML/JS/CSS from the "frontend" folder
app = FastAPI()

# Enable templates from /templates folder
templates = Jinja2Templates(directory="templates")

# (Optional CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve HTML via Jinja2
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
# Your Gemini prompt template

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
refined=[]
@app.post("/process")
async def process(request: Request, phone_number: str = Form(...), query: str = Form(...)):
    try:
        # Prepare prompt
        prompt = GEMINI_PROMPT_TEMPLATE + f"\nInput: {query}"
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Generate response
        response = model.generate_content(prompt)
        cleaned = re.sub(r'```json|```', '', response.text).strip()
        parsed = json.loads(cleaned)
        print(parsed)
        refined.append(parsed.get("entities"))
        followups = parsed.get("follow_up_questions")
        print(f"Follow-up questions: {followups}")
        return JSONResponse(content={"follow_up_questions": followups})
    except Exception as e:
        return JSONResponse(content={"follow_up_questions": [], "error": str(e)}, status_code=500)

from pydantic import BaseModel
from typing import List

class QAItem(BaseModel):
    question: str
    answer: str

class ChatHistory(BaseModel):
    phone_number: str
    query: str
    follow_up_questions: List[QAItem]

chat_log = []
import json

@app.post("/save")
async def save_chat(data: ChatHistory):
    final=data.dict()
    print(final)
    final.update({"entities": refined})
    chat_log.append(final)
    print(f"Chat log updated: {chat_log}")
    return json.dumps(final, indent=4)

@app.get("/context")
async def get_context():
    print(f"Chat log: {chat_log}")
    if not chat_log:
        return JSONResponse(content={"error": "No context available yet."}, status_code=404)
    model = genai.GenerativeModel("gemini-2.0-flash")
    context= json.dumps(chat_log[-1], indent=4)
    print(context)
    prompt = f"""You are an assistant that summarizes phone call conversations.

    Find out the final task the assistant needs to perform based on the conversation history.

    Conversation history:
    {context}
    """
    # return JSONResponse(content=chat_log[-1])
    response = model.generate_content(prompt)
    return response.text