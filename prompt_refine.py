from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import requests, os, json, io
from fastapi.responses import StreamingResponse
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import atexit
import signal

load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Vapi Configuration
VAPI_API_KEY = os.getenv("VAPI_API_KEY")
VAPI_PHONE_NUMBER_ID = os.getenv("VAPI_PHONE_NUMBER_ID")
VAPI_BASE_URL = "https://api.vapi.ai"

app = FastAPI()

# Register cleanup function to save conversations on exit
def cleanup_conversations():
    """Save all conversations before server shutdown"""
    print("\n🔄 Server shutting down, saving all conversations...")
    saved_files = save_all_conversations()
    print(f"✅ Saved {len(saved_files)} conversations before shutdown")

atexit.register(cleanup_conversations)

# Handle SIGTERM and SIGINT
def signal_handler(signum, frame):
    print(f"\n📡 Received signal {signum}")
    cleanup_conversations()
    exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class VapiCallRequest(BaseModel):
    phone_number: str
    name: Optional[str] = None
    system_message: Optional[str] = None

class VapiWebhookData(BaseModel):
    type: str
    call: Optional[Dict] = None
    message: Optional[Dict] = None

# In-memory conversation storage (in production, use a database)
conversations: Dict[str, List[Dict]] = {}

# Serve the HTML file at the root
@app.get("/")
async def read_index():
    return FileResponse('/Users/yuktha/Desktop/voice_ai/voice_ai/index_with_history.html')

# Alternative endpoint for the original interface
@app.get("/simple")
async def read_simple_index():
    return FileResponse('/Users/yuktha/Desktop/voice_ai/voice_ai/index.html')

# --- CONFIG ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID") or "EXAVITQu4vr4xnSDxMaL"  # Replace with actual ID

# --- Gemini Prompt Template ---
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

Respond a JSON object without any '```json ```' backticks with the following:
- refined_prompt
- entities (dictionary)
- follow_up_questions (array of strings)
"""

# --- Gemini Refiner ---
# @app.post("/refine_prompt")
def refine_prompt(raw_intent):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            contents=GEMINI_PROMPT_TEMPLATE + "\nInput: " + raw_intent,
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation")
async def conversation(audio: UploadFile = File(...), session_id: str = Form(None)):
    try:
        # Get or create conversation session - maintain existing session if provided
        if session_id and session_id in conversations:
            print(f"🔗 Continuing existing session: {session_id}")
        elif session_id:
            print(f"🔗 Session {session_id} not found, creating new one")
            session_id = get_or_create_conversation()
        else:
            print("🔗 No session provided, creating new one")
            session_id = get_or_create_conversation()
        
        print(f"🔗 Using session: {session_id}")
        
        # Step 1: Speech → Text (STT)
        
        print("📤 Sending audio to ElevenLabs STT...")
        stt_resp = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
             data={"model_id": "scribe_v1"},
          headers={"xi-api-key": ELEVENLABS_API_KEY},
            files={"file": (audio.filename, await audio.read(), audio.content_type)},
        )
        print("📥 STT Response Status:", stt_resp.status_code)
        print("📥 STT Response Content:", stt_resp.text)

        if stt_resp.status_code != 200:
            raise HTTPException(status_code=500, detail="STT failed")

        user_text = stt_resp.json()["text"]
        print("🎧 User said:", user_text)
        
        # Check if user wants to end the conversation
        end_keywords = ["bye", "goodbye", "end conversation", "stop", "quit", "exit", "see you later", "talk to you later"]
        is_ending = any(keyword in user_text.lower() for keyword in end_keywords)
        
        # Add user message to conversation history
        add_to_conversation(session_id, "user", user_text)
        
        # Print current conversation history
        print(f"📚 Current conversation history for {session_id}:")
        for i, msg in enumerate(conversations[session_id], 1):
            role = "🧑 User" if msg['role'] == 'user' else "🤖 Assistant"
            print(f"   {i}. {role}: {msg['content']}")

        # Step 2: Get conversation context and generate response
        context = get_conversation_context(session_id)
        
        if is_ending:
            # Generate goodbye response
            model = genai.GenerativeModel('gemini-1.5-flash')
            chat_prompt = f"""You are a helpful voice assistant. The user is ending the conversation. 
            
Previous conversation context:
{context if context else 'This was the start of our conversation.'}

Current user message: {user_text}

Give a friendly goodbye response and thank them for the conversation. Keep it brief and warm."""
            
            gemini_response = model.generate_content(chat_prompt)
            bot_reply = gemini_response.text
        else:
            # Continue normal conversation with history context
            refined = refine_prompt(raw_intent=user_text)
            print("🔧 Refined prompt:", refined)
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Create enhanced prompt with conversation context
            if context:
                chat_prompt = f"""You are a helpful voice assistant. Here's our conversation history:

{context}

Current refined user request: {refined['refined_prompt']}
Extracted entities: {json.dumps(refined['entities'])}
Follow-up questions: {json.dumps(refined['follow_up_questions'])}

Respond naturally and clearly, taking into account the conversation history. You can ask follow-up questions if needed or provide the answer if you know it."""
            else:
                chat_prompt = f"""You are a helpful voice assistant.

User request: {refined['refined_prompt']}
Entities: {json.dumps(refined['entities'])}
Follow-up questions: {json.dumps(refined['follow_up_questions'])}

Generate a natural response. You can ask follow-up questions if needed or provide the answer if you know it."""
            
            gemini_response = model.generate_content(chat_prompt)
            bot_reply = gemini_response.text
        
        print("🧠 Gemini replied:", bot_reply)
        
        # Add assistant response to conversation history
        add_to_conversation(session_id, "assistant", bot_reply)
        
        # If user is ending conversation, mark session as ended
        if is_ending:
            print(f"🔚 User ended conversation for session: {session_id}")
            add_to_conversation(session_id, "system", "Conversation ended by user")

        # Step 3: TTS
        tts_resp = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/9BWtsMINqrJLrRacOk9x",
            headers={
                "Accept": "audio/mpeg",
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            },
            json={"text": bot_reply}
        )
        print("🔊 TTS Response Status:", tts_resp.status_code)
        if tts_resp.status_code != 200:
            print("🔴 TTS failed:", tts_resp.text)
            raise HTTPException(status_code=500, detail="TTS failed")

        # Return audio with session info in headers
        response = StreamingResponse(io.BytesIO(tts_resp.content), media_type="audio/mpeg")
        response.headers["X-Session-ID"] = session_id
        response.headers["X-Conversation-Ended"] = str(is_ending).lower()
        return response

    except Exception as e:
        print("❌ Internal error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Conversation history management functions
def get_or_create_conversation(session_id: str = None) -> str:
    """Get existing conversation or create a new one"""
    if session_id and session_id in conversations:
        return session_id
    
    # Create new conversation
    new_session_id = str(uuid.uuid4())
    conversations[new_session_id] = []
    print(f"✨ Created new conversation session: {new_session_id}")
    return new_session_id

def add_to_conversation(session_id: str, role: str, content: str, auto_save_enabled: bool = True):
    """Add a message to conversation history"""
    if session_id not in conversations:
        conversations[session_id] = []
    
    conversations[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep only last 15 messages to prevent context from getting too long
    if len(conversations[session_id]) > 15:
        conversations[session_id] = conversations[session_id][-15:]
    
    # Auto-save conversation periodically
    if auto_save_enabled:
        auto_save_conversation(session_id)

def get_conversation_context(session_id: str) -> str:
    """Get formatted conversation history for context"""
    if session_id not in conversations or not conversations[session_id]:
        return ""
    
    context = "Previous conversation:\n"
    for msg in conversations[session_id]:
        if msg['role'] != 'system':  # Skip system messages in context
            context += f"{msg['role']}: {msg['content']}\n"
    
    return context

def save_conversation_to_file(session_id: str, auto_save: bool = False):
    """Save conversation history to a file"""
    if session_id not in conversations:
        return
    
    # Create conversations directory if it doesn't exist
    conversations_dir = os.path.join(os.getcwd(), "conversations")
    os.makedirs(conversations_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if auto_save:
        filename = f"auto_save_{session_id}.json"
    else:
        filename = f"conversation_{session_id}.json"
    
    filepath = os.path.join(conversations_dir, filename)
    
    try:
        # Prepare conversation data with metadata
        conversation_data = {
            "session_id": session_id,
            "total_messages": len(conversations[session_id]),
            "started_at": conversations[session_id][0]['timestamp'] if conversations[session_id] else None,
            "ended_at": conversations[session_id][-1]['timestamp'] if conversations[session_id] else None,
            "saved_at": datetime.now().isoformat(),
            "auto_saved": auto_save,
            "messages": conversations[session_id],
            "summary": {
                "user_messages": len([msg for msg in conversations[session_id] if msg['role'] == 'user']),
                "assistant_messages": len([msg for msg in conversations[session_id] if msg['role'] == 'assistant']),
                "system_messages": len([msg for msg in conversations[session_id] if msg['role'] == 'system']),
                "conversation_ended": any(msg.get('role') == 'system' and 'ended' in msg.get('content', '').lower() for msg in conversations[session_id])
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        if auto_save:
            print(f"💾 Auto-saved conversation to: {filepath}")
        else:
            print(f"💾 Conversation saved to: {filepath}")
        
        return filepath
    except Exception as e:
        print(f"❌ Error saving conversation: {e}")
        return None

def auto_save_conversation(session_id: str):
    """Automatically save conversation after every few messages"""
    if session_id in conversations and len(conversations[session_id]) > 0:
        # Auto-save every 5 messages or when conversation ends
        message_count = len(conversations[session_id])
        if message_count % 5 == 0 or any(msg.get('role') == 'system' and 'ended' in msg.get('content', '').lower() for msg in conversations[session_id]):
            save_conversation_to_file(session_id, auto_save=True)

def save_all_conversations():
    """Save all active conversations to files"""
    saved_files = []
    for session_id in conversations.keys():
        if conversations[session_id]:  # Only save if there are messages
            filepath = save_conversation_to_file(session_id)
            if filepath:
                saved_files.append(filepath)
    
    print(f"💾 Saved {len(saved_files)} conversations")
    return saved_files

def load_conversation_from_file(filepath: str):
    """Load a conversation from a saved file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session_id = data['session_id']
        messages = data['messages']
        
        # Restore conversation to memory
        conversations[session_id] = messages
        
        print(f"📂 Loaded conversation {session_id} from {filepath}")
        return session_id
    except Exception as e:
        print(f"❌ Error loading conversation: {e}")
        return None

# API endpoints for conversation management
@app.get("/conversation/new")
async def new_conversation():
    """Create a new conversation session"""
    session_id = get_or_create_conversation()
    return {"session_id": session_id}

@app.get("/conversation/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "session_id": session_id,
        "messages": conversations[session_id]
    }

@app.get("/conversation/{session_id}/save")
async def save_conversation(session_id: str):
    """Save conversation history to a file"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    filepath = save_conversation_to_file(session_id)
    if filepath:
        return {"message": "Conversation saved successfully", "filepath": filepath}
    else:
        raise HTTPException(status_code=500, detail="Failed to save conversation")

@app.post("/conversations/save-all")
async def save_all_conversations_endpoint():
    """Save all active conversations to files"""
    saved_files = save_all_conversations()
    return {
        "message": f"Saved {len(saved_files)} conversations",
        "saved_files": saved_files
    }

@app.get("/conversations/files")
async def list_conversation_files():
    """List all saved conversation files"""
    conversations_dir = os.path.join(os.getcwd(), "conversations")
    if not os.path.exists(conversations_dir):
        return {"files": []}
    
    files = []
    for filename in os.listdir(conversations_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(conversations_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                files.append({
                    "filename": filename,
                    "filepath": filepath,
                    "session_id": data.get('session_id'),
                    "total_messages": data.get('total_messages', 0),
                    "saved_at": data.get('saved_at'),
                    "auto_saved": data.get('auto_saved', False),
                    "conversation_ended": data.get('summary', {}).get('conversation_ended', False)
                })
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    
    return {
        "total_files": len(files),
        "files": sorted(files, key=lambda x: x['saved_at'], reverse=True)
    }

@app.post("/conversations/load/{filename}")
async def load_conversation_file(filename: str):
    """Load a conversation from a saved file"""
    conversations_dir = os.path.join(os.getcwd(), "conversations")
    filepath = os.path.join(conversations_dir, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    session_id = load_conversation_from_file(filepath)
    if session_id:
        return {"message": "Conversation loaded successfully", "session_id": session_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to load conversation")

@app.get("/conversations/all")
async def get_all_conversations():
    """Get all conversation sessions and their message counts"""
    result = {}
    for session_id, messages in conversations.items():
        result[session_id] = {
            "message_count": len(messages),
            "last_message_time": messages[-1]['timestamp'] if messages else None,
            "first_message_time": messages[0]['timestamp'] if messages else None,
            "is_ended": any(msg.get('role') == 'system' and 'ended' in msg.get('content', '').lower() for msg in messages)
        }
    
    return {
        "total_conversations": len(conversations),
        "conversations": result
    }

@app.post("/conversation/{session_id}/end")
async def end_conversation(session_id: str):
    """Manually end a conversation and save it"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Add system message to mark as ended
    add_to_conversation(session_id, "system", "Conversation ended manually")
    
    # Save to file
    filepath = save_conversation_to_file(session_id)
    
    return {
        "message": "Conversation ended and saved",
        "session_id": session_id,
        "filepath": filepath
    }

# Pydantic models
class ConversationRequest(BaseModel):
    session_id: str = None

# Command-line continuous conversation function
def continuous_conversation_cli():
    """Run a continuous conversation loop in the command line until user says bye"""
    import speech_recognition as sr
    import pyttsx3
    
    # Initialize speech recognition and text-to-speech
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    tts_engine = pyttsx3.init()
    
    # Create a new conversation session
    session_id = get_or_create_conversation()
    print(f"🎙️ Started new conversation session: {session_id}")
    print("🗣️ Say something to start the conversation, or say 'bye' to end...")
    
    while True:
        try:
            # Listen for audio
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                print("👂 Listening...")
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
            
            # Convert speech to text
            try:
                user_text = recognizer.recognize_google(audio)
                print(f"🧑 You said: {user_text}")
            except sr.UnknownValueError:
                print("🤔 Sorry, I didn't understand that. Please try again.")
                continue
            except sr.RequestError as e:
                print(f"❌ Speech recognition error: {e}")
                continue
            
            # Check if user wants to end conversation
            end_keywords = ["bye", "goodbye", "end conversation", "stop", "quit", "exit", "see you later", "talk to you later"]
            is_ending = any(keyword in user_text.lower() for keyword in end_keywords)
            
            # Add user message to conversation history
            add_to_conversation(session_id, "user", user_text)
            
            # Print conversation history
            print(f"\n📚 Conversation History:")
            for i, msg in enumerate(conversations[session_id], 1):
                role = "🧑 You" if msg['role'] == 'user' else "🤖 Assistant"
                print(f"   {i}. {role}: {msg['content']}")
            print()
            
            # Generate response
            context = get_conversation_context(session_id)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            if is_ending:
                chat_prompt = f"""You are a helpful voice assistant. The user is ending the conversation.
                
Previous conversation context:
{context if context else 'This was the start of our conversation.'}

Current user message: {user_text}

Give a friendly goodbye response and thank them for the conversation. Keep it brief and warm."""
                
                gemini_response = model.generate_content(chat_prompt)
                bot_reply = gemini_response.text
            else:
                # Use the refine_prompt function for better context
                refined = refine_prompt(raw_intent=user_text)
                
                if context:
                    chat_prompt = f"""You are a helpful voice assistant. Here's our conversation history:

{context}

Current refined user request: {refined['refined_prompt']}
Extracted entities: {json.dumps(refined['entities'])}
Follow-up questions: {json.dumps(refined['follow_up_questions'])}

Respond naturally and clearly, taking into account the conversation history."""
                else:
                    chat_prompt = f"""You are a helpful voice assistant.

User request: {refined['refined_prompt']}
Entities: {json.dumps(refined['entities'])}
Follow-up questions: {json.dumps(refined['follow_up_questions'])}

Generate a natural response."""
                
                gemini_response = model.generate_content(chat_prompt)
                bot_reply = gemini_response.text
            
            print(f"🤖 Assistant: {bot_reply}")
            
            # Add assistant response to conversation history
            add_to_conversation(session_id, "assistant", bot_reply)
            
            # Speak the response
            tts_engine.say(bot_reply)
            tts_engine.runAndWait()
            
            # If ending conversation, save and break
            if is_ending:
                add_to_conversation(session_id, "system", "Conversation ended by user")
                filepath = save_conversation_to_file(session_id)
                print(f"\n🔚 Conversation ended and saved to: {filepath}")
                print("👋 Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n⏹️ Conversation interrupted by user")
            add_to_conversation(session_id, "system", "Conversation interrupted")
            filepath = save_conversation_to_file(session_id)
            print(f"💾 Conversation saved to: {filepath}")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

# Add CLI command support
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Run continuous conversation in CLI mode
        continuous_conversation_cli()
    else:
        # Run FastAPI server
        import uvicorn
        print("🚀 Starting FastAPI server...")
        print("🌐 Web interface: http://localhost:8000")
        print("💬 For CLI mode: python prompt_refine.py --cli")
        uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/conversations/{session_id}/summary")
async def generate_conversation_summary(session_id: str):
    """Generate a structured post-call summary for a conversation"""
    messages = None
    print(conversations)
    # First, try to find in memory
    if session_id in conversations:
        messages = conversations[session_id]
    else:
        # If not in memory, search in saved files
        print(f"🔍 Session {session_id} not in memory, searching saved files...")
        conversations_dir = os.path.join(os.getcwd(), "conversations")
        
        if os.path.exists(conversations_dir):
            for filename in os.listdir(conversations_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(conversations_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if data.get('session_id') == session_id:
                            messages = data.get('messages', [])
                            print(f"📂 Found conversation in file: {filename}")
                            break
                    except Exception as e:
                        continue
        
        if not messages:
            raise HTTPException(status_code=404, detail=f"Conversation {session_id} not found in memory or saved files")
    
    if not messages:
        raise HTTPException(status_code=400, detail="No messages in conversation")
    
    try:
        # Prepare conversation text for analysis
        conversation_text = ""
        for msg in messages:
            if msg['role'] in ['user', 'assistant']:
                role_name = "User" if msg['role'] == 'user' else "Assistant"
                conversation_text += f"{role_name}: {msg['content']}\n"
        
        # Generate summary using Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        summary_prompt = f"""
Analyze this conversation and provide a structured post-call summary in JSON format.

Conversation:
{conversation_text}

Provide a JSON response with the following structure:
{{
  "outcome": "Brief description of what was accomplished (e.g., 'item found', 'appointment booked', 'information provided', 'issue resolved', 'no resolution')",
  "status": "completed|pending|failed|partially_completed",
  "action_taken": "Specific actions that were taken during the conversation",
  "follow_up_required": "yes|no",
  "follow_up_details": "What follow-up actions are needed (if any)",
  "key_entities": {{
    "people": ["names mentioned"],
    "locations": ["places mentioned"],
    "dates_times": ["dates/times mentioned"],
    "items": ["items/objects mentioned"],
    "organizations": ["companies/organizations mentioned"]
  }},
  "intent_category": "inquiry|appointment|rescheduling|lost_items|customer_service|other",
  "resolution_level": "fully_resolved|partially_resolved|unresolved",
  "notes": "Additional important details or context",
  "natural_summary": "A short, natural-language summary for user display (2-3 sentences max)"
}}

Respond with valid JSON only, no markdown formatting.
"""
        
        response = model.generate_content(summary_prompt)
        
        try:
            summary_data = json.loads(response.text)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            summary_data = {
                "outcome": "conversation completed",
                "status": "completed",
                "action_taken": "Had a conversation with the AI assistant",
                "follow_up_required": "no",
                "follow_up_details": "",
                "key_entities": {
                    "people": [],
                    "locations": [],
                    "dates_times": [],
                    "items": [],
                    "organizations": []
                },
                "intent_category": "other",
                "resolution_level": "completed",
                "notes": "Summary generation failed, using fallback",
                "natural_summary": f"Had a conversation with {len([m for m in messages if m['role'] == 'user'])} user messages and {len([m for m in messages if m['role'] == 'assistant'])} assistant responses."
            }
        
        # Add metadata
        conversation_metadata = {
            "session_id": session_id,
            "generated_at": datetime.now().isoformat(),
            "total_messages": len(messages),
            "duration_minutes": calculate_conversation_duration(messages),
            "user_messages": len([m for m in messages if m['role'] == 'user']),
            "assistant_messages": len([m for m in messages if m['role'] == 'assistant']),
            "conversation_start": messages[0]['timestamp'] if messages else None,
            "conversation_end": messages[-1]['timestamp'] if messages else None
        }
        
        return {
            "metadata": conversation_metadata,
            "summary": summary_data
        }
        
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@app.get("/conversations/file/{filename}/summary")
async def generate_file_summary(filename: str):
    """Generate a structured summary from a saved conversation file"""
    conversations_dir = os.path.join(os.getcwd(), "conversations")
    filepath = os.path.join(conversations_dir, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Load conversation from file
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session_id = data['session_id']
        messages = data['messages']
        
        # Temporarily load into memory for processing
        conversations[session_id] = messages
        
        # Generate summary
        summary_response = await generate_conversation_summary(session_id)
        
        # Clean up temporary data
        if session_id in conversations:
            del conversations[session_id]
        
        return summary_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/conversations/batch-summary")
async def generate_batch_summaries():
    """Generate summaries for all saved conversation files"""
    conversations_dir = os.path.join(os.getcwd(), "conversations")
    if not os.path.exists(conversations_dir):
        return {"summaries": []}
    
    summaries = []
    files = [f for f in os.listdir(conversations_dir) if f.endswith('.json')]
    
    for filename in files:
        try:
            summary = await generate_file_summary(filename)
            summaries.append({
                "filename": filename,
                "summary": summary
            })
        except Exception as e:
            summaries.append({
                "filename": filename,
                "error": str(e)
            })
    
    return {
        "total_files": len(files),
        "processed": len(summaries),
        "summaries": summaries
    }

def calculate_conversation_duration(messages):
    """Calculate conversation duration in minutes"""
    if len(messages) < 2:
        return 0
    
    try:
        start_time = datetime.fromisoformat(messages[0]['timestamp'])
        end_time = datetime.fromisoformat(messages[-1]['timestamp'])
        duration = (end_time - start_time).total_seconds() / 60
        return round(duration, 2)
    except:
        return 0

# --- Vapi Integration ---
VAPI_API_KEY = os.getenv("VAPI_API_KEY")
VAPI_BASE_URL = "https://api.vapi.ai"

# Vapi Assistant Configuration
VAPI_ASSISTANT_CONFIG = {
    "model": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "systemMessage": "You are a helpful voice AI assistant that can help with inquiries, appointments, lost items, and general customer service. Be friendly, concise, and professional."
    },
    "voice": {
        "provider": "11labs",
        "voiceId": "EXAVITQu4vr4xnSDxMaL"
    },
    "firstMessage": "Hello! This is your AI assistant. How can I help you today?",
    "recordingEnabled": True,
    "endCallOnSilence": True,
    "silenceTimeoutSeconds": 30,
    "maxDurationSeconds": 600,  # 10 minutes max
    "backgroundSound": "office"
}

@app.post("/vapi/create-call")
async def create_vapi_call(call_request: VapiCallRequest):
    """Create a new phone call using Vapi"""
    print(f"🔍 Received call request: {call_request}")
    
    if not VAPI_API_KEY or not VAPI_PHONE_NUMBER_ID:
        print(f"❌ Missing config - API Key: {'Set' if VAPI_API_KEY else 'Missing'}, Phone ID: {'Set' if VAPI_PHONE_NUMBER_ID else 'Missing'}")
        raise HTTPException(status_code=500, detail="Vapi API key or phone number ID not configured")
    
    try:
        # Create a new session for this call
        session_id = get_or_create_conversation()
        print(f"✨ Created session: {session_id}")
        
        # Default system message if not provided
        system_message = call_request.system_message or "You are a helpful AI assistant. Be polite, professional, and helpful."
        
        # Prepare the call payload with corrected structure
        call_payload = {
            "phoneNumberId": VAPI_PHONE_NUMBER_ID,
            "customer": {
                "number": call_request.phone_number
            },
            "assistant": {
                "model": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "systemMessage": system_message
                },
                "voice": {
                    "provider": "11labs", 
                    "voiceId": ELEVENLABS_VOICE_ID
                },
                "firstMessage": "Hello! This is your AI assistant. How can I help you today?"
            }
        }
        
        # Add metadata if name is provided
        if call_request.name:
            call_payload["metadata"] = {
                "session_id": session_id,
                "customer_name": call_request.name
            }
        else:
            call_payload["metadata"] = {
                "session_id": session_id
            }
        
        print(f"🚀 Sending payload to Vapi: {json.dumps(call_payload, indent=2)}")
        
        # Make the API call to Vapi
        response = requests.post(
            f"{VAPI_BASE_URL}/call",
            headers={
                "Authorization": f"Bearer {VAPI_API_KEY}",
                "Content-Type": "application/json"
            },
            json=call_payload,
            timeout=10
        )
        
        print(f"📡 Vapi response status: {response.status_code}")
        print(f"📡 Vapi response headers: {dict(response.headers)}")
        print(f"📡 Vapi response body: {response.text}")
        
        if response.status_code == 201:
            call_data = response.json()
            print(f"📞 Created Vapi call: {call_data.get('id')} for session: {session_id}")
            
            # Add initial log to conversation
            add_to_conversation(session_id, "system", f"Outbound call initiated to {call_request.phone_number}")
            if call_request.name:
                add_to_conversation(session_id, "system", f"Customer name: {call_request.name}")
            
            return {
                "success": True,
                "call_id": call_data.get("id"),
                "session_id": session_id,
                "status": call_data.get("status"),
                "message": "Call initiated successfully"
            }
        else:
            error_detail = f"Status: {response.status_code}, Body: {response.text}"
            print(f"❌ Vapi call creation failed: {error_detail}")
            
            # Try to parse error response
            try:
                error_data = response.json()
                error_message = error_data.get("message", error_data.get("error", "Unknown error"))
            except:
                error_message = response.text or "Unknown error"
            
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Vapi call creation failed: {error_message}"
            )
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout")
        raise HTTPException(status_code=408, detail="Request timeout - Vapi API is not responding")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/vapi/webhook/{session_id}")
async def vapi_webhook(session_id: str, request: Request):
    """Handle Vapi webhooks for call events"""
    try:
        body = await request.json()
        event_type = body.get("message", {}).get("type")
        
        print(f"🔔 Vapi webhook for session {session_id}: {event_type}")
        
        if event_type == "conversation-update":
            # Handle conversation updates
            conversation_data = body.get("message", {}).get("conversation", [])
            
            for message in conversation_data:
                if message.get("role") == "user":
                    add_to_conversation(session_id, "user", message.get("content", ""))
                elif message.get("role") == "assistant":
                    add_to_conversation(session_id, "assistant", message.get("content", ""))
        
        elif event_type == "call-ended":
            # Handle call end
            add_to_conversation(session_id, "system", "Call ended")
            
            # Auto-save conversation
            filename = save_conversation(session_id)
            print(f"💾 Auto-saved conversation to {filename}")
        
        elif event_type == "function-call":
            # Handle function calls
            function_call = body.get("message", {}).get("functionCall", {})
            function_name = function_call.get("name")
            parameters = function_call.get("parameters", {})
            
            print(f"🔧 Function call: {function_name} with params: {parameters}")
            
            # Execute the function
            result = await process_function_call(function_name, parameters, session_id)
            
            # Return the result to Vapi
            return {"result": result}
        
        return {"status": "received"}
        
    except Exception as e:
        print(f"❌ Webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")

@app.get("/vapi/call/{call_id}/status")
async def get_call_status(call_id: str):
    """Get the status of a Vapi call"""
    try:
        response = requests.get(
            f"{VAPI_BASE_URL}/call/{call_id}",
            headers={
                "Authorization": f"Bearer {VAPI_API_KEY}"
            }
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to get call status")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting call status: {str(e)}")

@app.post("/vapi/call/{call_id}/end")
async def end_call(call_id: str):
    """End a Vapi call"""
    try:
        response = requests.post(
            f"{VAPI_BASE_URL}/call/{call_id}/end",
            headers={
                "Authorization": f"Bearer {VAPI_API_KEY}"
            }
        )
        
        if response.status_code == 200:
            return {"success": True, "message": "Call ended successfully"}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to end call")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending call: {str(e)}")

async def process_function_call(function_name: str, parameters: dict, session_id: str):
    """Process function calls from Vapi (e.g., booking appointments, checking availability)"""
    
    if function_name == "book_appointment":
        # Example function for booking appointments
        date = parameters.get("date")
        time = parameters.get("time")
        service = parameters.get("service")
        
        # Add your booking logic here
        result = f"Appointment booked for {service} on {date} at {time}"
        add_to_conversation(session_id, "system", f"Function call: {function_name} - {result}")
        
        return {
            "success": True,
            "message": result,
            "appointment_id": f"APT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
    
    elif function_name == "check_availability":
        # Example function for checking availability
        date = parameters.get("date")
        
        # Add your availability checking logic here
        result = f"Checking availability for {date}"
        add_to_conversation(session_id, "system", f"Function call: {function_name} - {result}")
        
        return {
            "available_slots": ["9:00 AM", "2:00 PM", "4:00 PM"],
            "date": date
        }
    
    elif function_name == "transfer_to_human":
        # Example function for transferring to human agent
        reason = parameters.get("reason", "Customer request")
        
        add_to_conversation(session_id, "system", f"Transfer to human requested: {reason}")
        
        return {
            "transfer_initiated": True,
            "reason": reason,
            "message": "Transferring you to a human agent now"
        }
    
    else:
        return {"error": f"Unknown function: {function_name}"}

# --- VAPI FRONTEND ---
@app.get("/vapi")
async def vapi_interface():
    """Serve the Vapi phone call interface"""
    return FileResponse('/Users/yuktha/Desktop/voice_ai/voice_ai/vapi_calls.html')

# --- TEST ENDPOINTS ---
@app.get("/vapi/test-config")
async def test_vapi_config():
    """Test Vapi configuration"""
    return {
        "vapi_api_key": "Set" if VAPI_API_KEY else "Missing",
        "vapi_phone_number_id": "Set" if VAPI_PHONE_NUMBER_ID else "Missing",
        "elevenlabs_voice_id": ELEVENLABS_VOICE_ID,
        "base_url": VAPI_BASE_URL
    }

@app.post("/vapi/test-payload")
async def test_vapi_payload(call_request: VapiCallRequest):
    """Test what payload would be sent to Vapi without actually making the call"""
    session_id = "test-session"
    system_message = call_request.system_message or "You are a helpful AI assistant. Be polite, professional, and helpful."
    
    call_payload = {
        "phoneNumberId": VAPI_PHONE_NUMBER_ID,
        "customer": {
            "number": call_request.phone_number
        },
        "assistant": {
            "model": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "systemMessage": system_message
            },
            "voice": {
                "provider": "11labs", 
                "voiceId": ELEVENLABS_VOICE_ID
            },
            "firstMessage": "Hello! This is your AI assistant. How can I help you today?"
        }
    }
    
    if call_request.name:
        call_payload["metadata"] = {
            "session_id": session_id,
            "customer_name": call_request.name
        }
    else:
        call_payload["metadata"] = {
            "session_id": session_id
        }
    
    return {
        "would_send_payload": call_payload,
        "request_received": {
            "phone_number": call_request.phone_number,
            "name": call_request.name,
            "system_message": call_request.system_message
        }
    }

# --- VAPI INTEGRATION ENDPOINTS ---
