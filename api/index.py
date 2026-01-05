from fastapi import FastAPI, Request, HTTPException
from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError
from groq import Groq
from pinecone import Pinecone
import cohere
import os

app = FastAPI()

# --- THE WATCHER'S PERSONA (HYBRID MODE) ---
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

@app.post("/api/interactions")
async def interaction(request: Request):
    # 1. Verify Signature
    try:
        signature = request.headers.get("X-Signature-Ed25519")
        timestamp = request.headers.get("X-Signature-Timestamp")
        body = await request.body()
        if not signature or not timestamp: raise HTTPException(401)
        VerifyKey(bytes.fromhex(os.environ["DISCORD_PUBLIC_KEY"])).verify(timestamp.encode() + body, bytes.fromhex(signature))
    except: raise HTTPException(401, "Invalid signature")

    data = await request.json()
    if data["type"] == 1: return {"type": 1}

    # 2. Handle Slash Command
    if data["type"] == 2 and data["data"]["name"] == "watcher":
        user_query = data["data"]["options"][0]["value"]
        
        try:
            # --- RAG: RETRIEVE CONTEXT ---
            # A. Convert Question to Numbers (Cohere)
            # FIXED: Pointing to the variable name, not the key value
            co = cohere.Client(os.environ["COHERE_API_KEY"])
            embed_response = co.embed(
                texts=[user_query], 
                model="embed-english-v3.0", 
                input_type="search_query"
            )
            query_embedding = embed_response.embeddings[0]

            # B. Search Pinecone (The Archive)
            # FIXED: Pointing to the variable name, not the key value
            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            index = pc.Index("watcher-memory") 
            search_results = index.query(
                vector=query_embedding, 
                top_k=3, 
                include_metadata=True
            )

            # C. Combine Context
            matches = search_results['matches']
            if not matches:
                context_text = "No relevant knowledge found in the Sacred Texts."
            else:
                context_text = "\n".join([match['metadata']['text'] for match in matches])

            # --- GENERATE ANSWER (GROQ) ---
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context from Handbook:\n{context_text}\n\nStudent Question: {user_query}"}
                ],
                model="llama3-8b-8192",
            )
            
            return {"type": 4, "data": {"content": chat_completion.choices[0].message.content}}

        except Exception as e:
            return {"type": 4, "data": {"content": f"The Archive is silent... (Error: {str(e)})"}}

    return {"type": 4, "data": {"content": "Unknown nexus event."}}