from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
import google.generativeai as genai  # Assuming you're using Google Gemini
from dotenv import load_dotenv
import os   

load_dotenv()  # Load environment variables from .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Get your Gemini API key from .env
# Initialize your Gemini API key

genai.configure(api_key=GEMINI_API_KEY)  # Set your Gemini API key from .env

# Define your Gemini prompt template
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
SUMMARY_PROMPT_TEMPLATE = """
You are an assistant that summarizes phone call conversations.

Given the full conversation transcript below, extract and return a structured JSON object without any '```json ```' backticks containing:
- intent: choose from: Inquiry, Appointment, Rescheduling, Finding Lost Items, Follow-up,
- entities: a dictionary of all entities extracted from the conversation
- status: Overall result of the conversation (e.g., "Item found", "Appointment scheduled", "Information unavailable")
- action_taken: What was done during the call
- notes: Any important details
- natural_language_summary: A concise 2-3 sentence summary of the call in plain English

Conversation:
"""

# FastAPI app
app = FastAPI()

# Input schema
class IntentInput(BaseModel):
    raw_intent: str
    phone_number: str = None  # Optional, can be used if needed
class ConversationInput(BaseModel):
    conversation: str
@app.post("/summarize")
def summarize_conversation(input: ConversationInput):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        print(input.conversation.strip())
        prompt = SUMMARY_PROMPT_TEMPLATE + input.conversation.strip()
        response = model.generate_content(prompt)

        raw_text = response.parts[0].text.strip()
        print(raw_text)

        # Extract JSON from response
        with open ('response.json', 'a') as f:
            f.write(raw_text)
     
        return raw_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Function to call Gemini and refine prompt
def refine_prompt(raw_intent: str):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            contents=GEMINI_PROMPT_TEMPLATE + "\nInput: " + raw_intent,
        )
        print(response.text)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI route
@app.post("/get_intent_details")
async def get_intent_details(input: IntentInput):
    return refine_prompt(input.raw_intent)
