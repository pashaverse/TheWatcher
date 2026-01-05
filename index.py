from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError
from groq import Groq
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import os
import httpx 
import logging

# Initialize Logging (Helps debug issues in Render logs)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TheWatcher")

app = FastAPI()

# --- 1. CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DISCORD_PUBLIC_KEY = os.getenv("DISCORD_PUBLIC_KEY")

# --- 2. INITIALIZE CLIENTS ---
# We load these once when the bot starts to save memory
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    verify_key = VerifyKey(bytes.fromhex(DISCORD_PUBLIC_KEY))
    # Using the same model as your ingest script
    embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Clients initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize clients: {e}")

# --- 3. THE PERSONA ---
SYSTEM_PROMPT = (
    "You are The Watcher, an observer across timelines and policies at ITU Punjab. "
    "You have access to the 'Sacred Texts' (Student Handbook). "
    "Your goal is to be helpful, mystical, and accurate.\n\n"
    "--- RULES ---\n"
    "1. **Chatting?** Be mystical and friendly. Ignore the handbook. Do NOT mention the archives.\n"
    "2. **Asking Info?** Use the provided CONTEXT.\n"
    "   - If context has the answer: Start with 'According to the archives...'\n"
    "   - If context is empty or irrelevant: 'The archives are silent on this specific matter.' (Do not make up rules)."
)

# --- 4. KEEP-ALIVE ENDPOINT (For Cron-job.org) ---
@app.get("/")
async def home():
    """
    This route exists solely to keep the bot awake.
    Point cron-job.org here to ping it every 10 minutes.
    """
    return {"status": "The Watcher is active.", "system": "online"}


# --- 5. BACKGROUND AI PROCESSING (Prevents Timeouts) ---
async def process_and_respond(interaction_token: str, application_id: str, user_query: str):
    """
    Runs in the background. Performs RAG + AI generation + Webhook response.
    """
    try:
        # A. SEARCH DATABASE (RAG)
        # Convert query to vector
        query_vector = list(embed_model.embed([user_query]))[0].tolist()
        
        # Search Qdrant
        search_results = qdrant_client.search(
            collection_name="knowledge_base",
            query_vector=query_vector,
            limit=5  # Get top 5 matches
        )

        # Filter for quality (score > 0.35 implies relevance)
        relevant_text = [hit.payload['text'] for hit in search_results if hit.score > 0.35]
        
        if relevant_text:
            context_text = "\n\n".join(relevant_text)
        else:
            context_text = "No relevant archives found."

        # B. GENERATE AI ANSWER
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.6,
        )
        
        ai_response = chat_completion.choices[0].message.content

    except Exception as e:
        logger.error(f"Error in background task: {e}")
        ai_response = "A temporal disturbance interrupted my thought process. (Internal Error)"

    # C. SEND BACK TO DISCORD (Via Webhook)
    # We use the interaction token to update the "Thinking..." message
    webhook_url = f"https://discord.com/api/v10/webhooks/{application_id}/{interaction_token}/messages/@original"
    
    async with httpx.AsyncClient() as client:
        await client.patch(webhook_url, json={"content": ai_response})


# --- 6. MAIN INTERACTION ENDPOINT ---
@app.post("/interactions")
async def interactions(request: Request, background_tasks: BackgroundTasks):
    # A. VERIFY SIGNATURE (Security Requirement)
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

    # B. HANDLE PING (Discord Health Check)
    if data["type"] == 1:
        return {"type": 1}

    # C. HANDLE SLASH COMMANDS
    if data["type"] == 2:
        # Extract user question
        options = data["data"].get("options", [])
        if options:
            user_query = options[0]["value"]
        else:
            user_query = "Hello Watcher"

        # Get IDs for the callback
        interaction_token = data["token"]
        application_id = data["application_id"]

        # Add the heavy work to the background queue
        background_tasks.add_task(
            process_and_respond, 
            interaction_token, 
            application_id, 
            user_query
        )

        # IMMEDIATELY return Type 5 ("Deferred Channel Message")
        # This makes the bot say "The Watcher is thinking..." while it works.
        return {"type": 5}

    return {"error": "Unknown type"}