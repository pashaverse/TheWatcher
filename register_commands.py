import requests

#CONFIGURATION
APPLICATION_ID = "1457475133135524116"
BOT_TOKEN = "BOT_TOKEN" 

url = f"https://discord.com/api/v10/applications/1457475133135524116/commands"

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

headers = {"Authorization": f"Bot {BOT_TOKEN}"}

response = requests.post(url, headers=headers, json=json_body)

if response.status_code in [200, 201]:
    print("✅ Success! Command '/watcher' is registered.")
else:
    print(f"Error: {response.status_code}")
    print(response.json())