SUMMARY_PROMPT_TEMPLATE = """
You are an assistant that summarizes phone call conversations.

Given the full conversation transcript below, extract and return a structured JSON containing:

- status: Overall result of the conversation (e.g., "Item found", "Appointment scheduled", "Information unavailable")
- action_taken: What was done during the call
- follow_up_required: true/false
- notes: Any important details
- natural_language_summary: A concise 2-3 sentence summary of the call in plain English

Conversation:
"""
import google.generativeai as genai
import json
import re

genai.configure(api_key="YOUR_GEMINI_API_KEY")

def summarize_conversation(conversation_text: str):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = SUMMARY_PROMPT_TEMPLATE + conversation_text.strip()

        response = model.generate_content(prompt)
        raw_text = response.parts[0].text.strip()

        # Extract JSON
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not match:
            raise ValueError("Gemini output did not contain valid JSON.")
        
        return json.loads(match.group())
    except Exception as e:
        raise RuntimeError(f"Failed to summarize: {str(e)}")
