import requests
import os
from dotenv import load_dotenv

#1. Load the real password from the .env file
load_dotenv()
REAL_BOT_TOKEN = os.getenv("DISCORD_TOKEN") 

#CONFIGURATION
APPLICATION_ID = "1457475133135524116"

url = f"https://discord.com/api/v10/applications/{APPLICATION_ID}/commands"

json_body = {
    "name": "watcher", 
    "type": 1,
    "description": "Summon The Watcher to observe your timeline",
    "options": [
        {
            "name": "query",
            "description": "What nexus event troubles you?",
            "type": 3, 
            "required": True
        }
    ]
}

headers = {"Authorization": f"Bot {REAL_BOT_TOKEN}"}

response = requests.post(url, headers=headers, json=json_body)

if response.status_code in [200, 201]:
    print("Success! Command '/watcher' is registered.")
else:
    print(f"Error: {response.status_code}")
    print(response.json())