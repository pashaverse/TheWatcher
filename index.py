from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding
import os
import httpx 
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import uuid
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TheWatcher")

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DISCORD_PUBLIC_KEY = os.getenv("DISCORD_PUBLIC_KEY")
ITU_LINKS_STR = os.getenv("ITU_LINKS", "")
UPDATE_SECRET = os.getenv("UPDATE_SECRET", "change_this_to_a_random_password") 

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    verify_key = VerifyKey(bytes.fromhex(DISCORD_PUBLIC_KEY))
    embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Clients initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize clients: {e}")

SYSTEM_PROMPT = (
    "You are The Watcher, an ancient and mystical observer of ITU created by faby0001. "
    "You speak with a slightly cryptic, magical tone, but your information is precise. "
    "You have access to the 'Archives' (Student Handbook & Website Data).\n\n"
    "--- CORE DIRECTIVES ---\n"
    "1. **Synthesize, Don't Just Quote:** Users may ask subjective questions (e.g., 'What is the best option?', 'How strict is the policy?'). "
    "The archives won't contain these exact opinions. You must analyze the **facts** in the Context (dates, numbers, lists, qualifications) "
    "to construct a logical, evidence-based answer.\n"
    "2. **Bridge the Gap:** If a user asks a fragmented question (e.g., 'fee?', 'dates?', 'dean?'), "
    "interpret their intent broadly. Assume they want the most relevant schedule, structure, or person related to that keyword.\n"
    "3. **Maximize Utility:** Never say 'Archives have no related data' if you have even partial information. "
    "If the specific answer is missing, provide the closest relevant facts that might help the user."
)

def get_precision_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200: 
            return None 

        soup = BeautifulSoup(response.content, 'html.parser')

        #Avada Pricing Tables
        for pricing in soup.select(".fusion-pricing-table"):
            headers = [h.get_text(strip=True) for h in pricing.select(".panel-heading")]
            prices = [p.get_text(strip=True) for p in pricing.select(".panel-body")]
            features = [f.get_text(strip=True) for f in pricing.select(".list-group-item")]
            pricing_text = f"\n=== FEE/PRICING DATA ===\nPlans: {', '.join(headers)}\nPrices: {', '.join(prices)}\nDetails: {', '.join(features)}\n========================\n"
            pricing.replace_with(pricing_text)

        #Standard Tables
        for table in soup.find_all("table"):
            table_str = "\n--- TABLE DATA ---\n"
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all(["td", "th"])
                row_text = " | ".join(ele.get_text(strip=True) for ele in cols)
                table_str += row_text + "\n"
            table_str += "------------------\n"
            table.replace_with(table_str)

        noise_selectors = [".fusion-footer", ".fusion-header-wrapper", "#sliders-container", ".fusion-sliding-bar", ".fusion-page-title-bar", "#side-header", ".fusion-sharing-box", "script", "style", "iframe", "form", "nav"]
        for selector in noise_selectors:
            for element in soup.select(selector):
                element.decompose()

        main_content = soup.find(id="main")
        text = main_content.get_text(separator="\n") if main_content else soup.get_text(separator="\n")

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        chunks = []
        chunk_size = 12   
        overlap = 4       
        for i in range(0, len(lines), chunk_size - overlap):
            window = lines[i : i + chunk_size]
            if len(window) < 3: continue 
            full_record = f"Source: {url}\nContent: " + "\n".join(window)
            chunks.append(full_record)
        return chunks
    except Exception as e:
        logger.error(f"Scrape error on {url}: {e}")
        return None

def discover_internal_links(seed_url):
    try:
        response = requests.get(seed_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        found_links = set()
        base_domain = urlparse(seed_url).netloc

        for link in soup.find_all('a', href=True):
            full_url = urljoin(seed_url, link['href'])
            if urlparse(full_url).netloc != base_domain: continue
            if any(x in full_url for x in [".pdf", ".jpg", "#", "wp-content", "login", "feed"]): continue
            if any(k in full_url for k in ["academics", "faculty", "program", "department", "admissions", "fee", "examinations", "research", "administration"]):
                found_links.add(full_url)
        return list(found_links)
    except:
        return []

def run_smart_update():
    logger.info("Starting scheduled web ingestion...")
    seed_urls = [url.strip() for url in ITU_LINKS_STR.split(",") if url.strip()]
    final_url_list = set(seed_urls)
    
    for seed in seed_urls:
        discovered = discover_internal_links(seed)
        final_url_list.update(discovered)
    
    urls_to_process = list(final_url_list)[:300]
    logger.info(f"Targeting {len(urls_to_process)} pages.")

    #Create indexes if missing
    try:
        qdrant_client.create_payload_index(collection_name="knowledge_base", field_name="url", field_schema="keyword")
        qdrant_client.create_payload_index(collection_name="knowledge_base", field_name="source_type", field_schema="keyword")
    except: pass

    for url in urls_to_process:
        new_chunks = get_precision_content(url)
        if new_chunks is None:
            continue # Skip failed pages, keep old data

        try:
            # 1. Vectorize
            embeddings = list(embed_model.embed(new_chunks))
            points = []
            for doc, vector in zip(new_chunks, embeddings):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={"text": doc, "source_type": "website", "url": url} 
                ))

            # 2. Delete old data for this specific URL
            qdrant_client.delete(
                collection_name="knowledge_base",
                points_selector=Filter(must=[FieldCondition(key="url", match=MatchValue(value=url))])
            )

            # 3. Upload new data
            qdrant_client.upsert(collection_name="knowledge_base", points=points)
            time.sleep(1) # Polite delay
        except Exception as e:
            logger.error(f"DB Error on {url}: {e}")
    
    logger.info("Ingestion complete.")

def optimize_search_query(original_query: str) -> str:
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a database query optimizer. Convert the user's question into a string of factual keywords likely to appear in a university handbook or website. Output ONLY keywords."},
                {"role": "user", "content": f"User Input: '{original_query}'"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=60
        )
        return completion.choices[0].message.content
    except:
        return original_query 

async def process_and_respond(interaction_token: str, application_id: str, user_query: str):
    try:
        search_query = optimize_search_query(user_query)
        logger.info(f"Original: '{user_query}' -> Optimized: '{search_query}'")

        query_vector = list(embed_model.embed([search_query]))[0].tolist()
        
        search_results = qdrant_client.search(
            collection_name="knowledge_base",
            query_vector=query_vector,
            limit=12
        )

        relevant_text = [hit.payload['text'] for hit in search_results if hit.score > 0.30]
        context_text = "\n\n".join(relevant_text) if relevant_text else "The archives revealed no specific records matching this vibration."

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context from Archives:\n{context_text}\n\nUser Question: {user_query}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
        )
        
        ai_response = chat_completion.choices[0].message.content

    except Exception as e:
        logger.error(f"Error in background task: {e}")
        ai_response = "A temporal disturbance interrupted my thought process. (Internal Error)"

    webhook_url = f"https://discord.com/api/v10/webhooks/{application_id}/{interaction_token}/messages/@original"
    async with httpx.AsyncClient() as client:
        await client.patch(webhook_url, json={"content": ai_response})

@app.get("/")
async def home():
    return {"status": "Watcher is online"}

@app.post("/interactions")
async def interactions(request: Request, background_tasks: BackgroundTasks):
    try:
        signature = request.headers.get("X-Signature-Ed25519")
        timestamp = request.headers.get("X-Signature-Timestamp")
        body = await request.body()
        if not signature or not timestamp: raise HTTPException(401)
        verify_key.verify(timestamp.encode() + body, bytes.fromhex(signature))
    except BadSignatureError:
        raise HTTPException(status_code=401, detail="Invalid request signature")

    data = await request.json()

    if data["type"] == 1: return {"type": 1}

    if data["type"] == 2:
        options = data["data"].get("options", [])
        user_query = options[0]["value"] if options else "Hello Watcher"
        interaction_token = data["token"]
        application_id = data["application_id"]
        background_tasks.add_task(process_and_respond, interaction_token, application_id, user_query)
        return {"type": 5}

    return {"error": "Unknown type"}

@app.post("/trigger-update")
async def trigger_update(request: Request, background_tasks: BackgroundTasks):
    #Security check to prevent random people from triggering it
    secret = request.query_params.get("secret")
    if secret != UPDATE_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    background_tasks.add_task(run_smart_update)
    return {"status": "Update process started in background"}