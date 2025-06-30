# Voice AI Coordination Tool

This project is a Voice AI Chatbot that can make phone calls and coordinate tasks on behalf of users using two different approaches:

- Vapi Voice Agent Workflow
- Gemini Vertex AI Voice Agent

## 🔁 Project Structure

```
.
├── vapi_workflow.json          # Workflow definition for Vapi voice agent
├── main.py                     # FastAPI backend for handling Vapi webhooks
├── response.json               # Sample post-call summaries
├── gemini_voice_agent/         # Gemini-based voice AI implementation
│   └── ...                     # Gemini Vertex AI logic for call handling
└── README.md                   # You're here!
```

## 🔧 1. Vapi Voice AI Agent (Using Vapi API)

### ✅ Features
- Handles backend logic with FastAPI.
- Generates post-call summaries stored in `response.json`.

### 📂 Files Involved
- `vapi_workflow.json`: Defines the conversation flow and logic.
- `main.py`: FastAPI server for handling webhook callbacks.
- `response.json`: Sample summaries of previously tested conversations.
![image](https://github.com/user-attachments/assets/dbe2312a-e9f8-4f06-8100-d00eb3d1be3b)


### ☎️ Test Number
+1 (956) 528-4589 (0019565284589 for international format)

### ⚠️ Deployment Note
The backend server is not yet deployed. To test:

1. Use `ngrok` or `cloudflared` to expose `main.py` to the internet:
   ```
   ngrok http 8000
   ```
2. Update the public URL in `vapi_workflow.json`:
   - Line 44 and Line 114 should point to the public webhook URL returned by `ngrok`/`cloudflared`.

## 🌐 2. Gemini Vertex Voice Agent (Using Google Cloud Gemini)

### ✅ Features
- Uses Gemini from Vertex AI for intent parsing and natural language interaction.
- Ideal for more complex or dynamic voice tasks.

### 📂 Directory
All Gemini-related code is contained in:

```
gemini_voice_agent/
```

This includes:

- Prompt engineering
- Intent classification
- Entity extraction
- Call logic orchestration

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Vapi account (for Approach 1)
- Google Cloud account with Gemini access (for Approach 2)

### Install Dependencies
```
pip install fastapi uvicorn requests
```

### Run FastAPI Server
```
uvicorn main:app --reload
```
Then expose it using `ngrok` or `cloudflared`.

## 📘 Sample Summary (`response.json`)

```json
[
 {
  "intent": "Inquiry",
  "entities": {
    "service": "Darshan",
    "location": "Balaji Temple, Trimal",
    "date": "second week of next month"
  },
  "status": "Inquiry received",
  "action_taken": "None",
  "notes": "Customer is looking to book a pilgrimage for Darshan at Balaji Temple in Trimal during the second week of next month.",
  "natural_language_summary": "The caller is inquiring about pilgrimage booking for Darshan at Balaji Temple in Trimal. They are interested in booking for the second week of next month. No action was taken during the call."
}
]
```

## 📞 Demo Call

To test the full experience, place a call to:
+1 (956) 528-4589 (or 0019565284589 internationally)

## 📌 TODO

- Deploy FastAPI server to a stable cloud endpoint (Render, Railway, or GCP).
