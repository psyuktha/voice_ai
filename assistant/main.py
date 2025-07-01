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
1. Convert it into a natural voice agent prompt.
2. Extract all possible entities from the user input. Use these keys:
   - user_name
   - target_name
   - location
   - time
   - item
   - intent_category (choose from: Inquiry, Appointment, Rescheduling, Finding Lost Items, Follow-up)
   - follow_up_questions (if any)
   Add possible follow-up questions that might be needed to clarify the user's intent. if any entities are missing, include them in the follow-up questions.
   add atleast one follow up question.
If it is an impossible event like a visit to another planet, return a message saying "This is not possible" and do not generate any entities or follow-up questions.
Respond a JSON object without any '```json ```' backticks with the following:
- refined_prompt
- entities (dictionary)
- follow_up_questions (array of strings)
"""
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
        followups = parsed.get("follow_up_questions", [])

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
    with open("chat_log.json", "a") as f:
        f.write(json.dumps(data.dict()) + "\n")
    return json.load(open("chat_log.json"))
