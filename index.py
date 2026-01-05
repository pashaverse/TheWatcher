from fastapi import FastAPI, Request, HTTPException
from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError
from groq import Groq
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import os

app = FastAPI()

# --- THE WATCHER'S PERSONA (TUNED FOR NATURAL CONVERSATION) ---
SYSTEM_PROMPT = (
    "You are The Watcher, an observer across timelines and policies at ITU Punjab. "
    "You have access to the 'Sacred Texts' (Student Handbook). "
    "Your goal is to be helpful, mystical, and accurate.\n\n"
    
    "--- RULES FOR ANSWERING ---\n"
    "1. **Analyze the Input:** Is the user asking for specific INFORMATION (rules, dates, policies) or just CHATTING (greetings, identity, jokes)?\n"
    "2. **IF CHATTING:** Ignore the handbook context. Speak naturally as The Watcher. Do NOT say 'This is not in the handbook'. Just be yourself.\n"
    "3. **IF ASKING FOR INFO:** Inspect the provided 'Context'.\n"
    "   - **Found it:** Answer strictly based on the context. Start with: 'According to the archives…'\n"
    "   - **Not Found:** If the context is irrelevant to the question, state clearly: 'The archives are silent on this specific matter.' Then, offer general advice if possible, but warn that it is your own observation."
)

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DISCORD_PUBLIC_KEY = os.getenv("DISCORD_PUBLIC_KEY")

# --- CLIENTS ---
groq_client = Groq(api_key=GROQ_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
verify_key = VerifyKey(bytes.fromhex(DISCORD_PUBLIC_KEY))
embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

@app.post("/interactions")
async def interaction(request: Request):
    # 1. Verify Signature
    try:
        signature = request.headers.get("X-Signature-Ed25519")
        timestamp = request.headers.get("X-Signature-Timestamp")
        body = await request.body()
        if not signature or not timestamp: raise HTTPException(401)
        verify_key.verify(timestamp.encode() + body, bytes.fromhex(signature))
    except BadSignatureError:
        raise HTTPException(status_code=401, detail="Invalid request signature")

    data = await request.json()

    # 2. PING
    if data["type"] == 1: return {"type": 1}

    # 3. SLASH COMMAND
    if data["type"] == 2:
        try:
            # Extract user query
            if "options" in data["data"] and len(data["data"]["options"]) > 0:
                user_query = data["data"]["options"][0]["value"]
            else:
                user_query = "What do you see?"

            # --- RAG: RETRIEVE CONTEXT ---
            query_vector = list(embed_model.embed([user_query]))[0].tolist()

            # SEARCH: INCREASED LIMIT TO 10 (Finds more relevant chunks)
            search_results = qdrant_client.search(
                collection_name="knowledge_base",
                query_vector=query_vector,
                limit=10  
            )

            # FILTER: Only keep results that are actually relevant (score > 0.35 is a safe bet for MiniLM)
            # If everything is low score, we assume the handbook has no answer.
            relevant_hits = [hit for hit in search_results if hit.score > 0.35]

            if not relevant_hits:
                context_text = "No relevant archives found."
            else:
                context_text = "\n\n".join([hit.payload['text'] for hit in relevant_hits])

            # --- GENERATE ANSWER ---
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context from Handbook:\n{context_text}\n\nStudent Question: {user_query}"}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.6 # Added a little creativity
            )
            
            response_content = chat_completion.choices[0].message.content

            return {"type": 4, "data": {"content": response_content}}

        except Exception as e:
            print(f"ERROR: {e}")
            return {"type": 4, "data": {"content": "A disturbance in the timeline... (Internal Error)"}}

    return {"type": 4, "data": {"content": "Unknown nexus event."}}