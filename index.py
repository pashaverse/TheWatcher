from fastapi import FastAPI, Request, HTTPException
from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError
from groq import Groq
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import os

app = FastAPI()

# --- THE WATCHER'S PERSONA ---
SYSTEM_PROMPT = (
    "You are The Watcher, an observer across timelines and policies. "
    "You behold the 'Sacred Texts' (the ITU Punjab Student Handbook) and speak with cosmic clarity. "
    "When responding:\n"
    "1. FIRST, inspect the provided Context. If it contains the answer (rules, policies, procedures), "
    "   answer strictly according to the handbook and preface with 'According to the archives…'.\n"
    "2. IF the Context lacks relevant handbook data (e.g., queries about academics, coding, or general advice), "
    "   answer based on your observed logic and wisdom but refrain from hallucinating specifics. "
    "   In this case preface with 'This is not in the handbook, but in my observation…'.\n"
    "3. IF the question cannot be answered from Handbook nor from sound reasoning, "
    "   gently acknowledge the unknown: 'Even I, who have seen many paths, cannot find certainty here.'\n"
    "Tone: detached, thoughtful, benevolent — cosmic, yet concise (no more than 4 sentences). "
    "In your voice, let every answer feel like a narration from beyond, guiding without interference."
)

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DISCORD_PUBLIC_KEY = os.getenv("DISCORD_PUBLIC_KEY")

# --- INITIALIZE CLIENTS ---
groq_client = Groq(api_key=GROQ_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
verify_key = VerifyKey(bytes.fromhex(DISCORD_PUBLIC_KEY))

# --- FIX: MODEL MATCHING ---
# We switched this to match your ingest script exactly
embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

@app.post("/interactions")
async def interaction(request: Request):
    # 1. Verify Signature
    try:
        signature = request.headers.get("X-Signature-Ed25519")
        timestamp = request.headers.get("X-Signature-Timestamp")
        body = await request.body()
        
        if not signature or not timestamp:
            raise HTTPException(401)
        
        verify_key.verify(timestamp.encode() + body, bytes.fromhex(signature))
    except BadSignatureError:
        raise HTTPException(status_code=401, detail="Invalid request signature")

    data = await request.json()

    # 2. Handle PING
    if data["type"] == 1:
        return {"type": 1}

    # 3. Handle Slash Command
    if data["type"] == 2:
        try:
            if "options" in data["data"] and len(data["data"]["options"]) > 0:
                user_query = data["data"]["options"][0]["value"]
            else:
                user_query = "What do you see?"

            # --- RAG: RETRIEVE CONTEXT ---
            
            # A. Vectorize Question
            query_vector = list(embed_model.embed([user_query]))[0].tolist()

            # B. Search Qdrant (Updated for older versions compatibility)
            # If .search() fails, this method is the most reliable fallback
            search_results = qdrant_client.search(
                collection_name="knowledge_base",
                query_vector=query_vector,
                limit=3
            )

            # C. Combine Context
            if not search_results:
                context_text = "The archives are silent on this matter."
            else:
                context_text = "\n".join([hit.payload['text'] for hit in search_results])

            # --- GENERATE ANSWER ---
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context from Handbook:\n{context_text}\n\nStudent Question: {user_query}"}
                ],
                model="llama3-8b-8192",
            )
            
            response_content = chat_completion.choices[0].message.content

            return {
                "type": 4, 
                "data": {
                    "content": response_content
                }
            }

        except Exception as e:
            print(f"Error: {e}")
            return {
                "type": 4,
                "data": {
                    "content": "A disturbance in the timeline prevented me from answering. (Internal Error)"
                }
            }

    return {"type": 4, "data": {"content": "Unknown nexus event."}}