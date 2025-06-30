from fastapi import FastAPI, Request
import os
from dotenv import load_dotenv
import requests
from retell import Retell

load_dotenv()
app = FastAPI()

RETELL_API_KEY = os.getenv("RETELL_API_KEY")

@app.post("/start_call")
async def start_call():
    
    payload = {
        "phone_number": "+916361472456",   # <-- Your test number
        "agent_id": "agent_ce88f0f9942d8ca72c99e8a594"  # <-- From Retell dashboard
    }
    headers = {
            "Authorization": f"Bearer {RETELL_API_KEY}"
        }

    

    client = Retell(
        api_key=os.getenv("RETELL_API_KEY"),
    )
    phone_call_response = client.call.create_phone_call(
        from_number="+14157774444",
        to_number="+916361472456",  # <-- Your test number
        agent_id="agent_ce88f0f9942d8ca72c99e8a594",  # <-- From Retell dashboard
    )
    print(phone_call_response.agent_id)

            
    
 

@app.post("/webhook")
async def transcript_webhook(request: Request):
    data = await request.json()
    print("🎤 Transcript received from Retell:")
    print(data)
    return {"status": "ok"}
