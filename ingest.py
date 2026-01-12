import os
import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
import uuid
import time

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "knowledge_base"

ITU_LINKS_STR = os.getenv("ITU_LINKS", "")
SEED_URLS = [url.strip() for url in ITU_LINKS_STR.split(",") if url.strip()]

#Limits
SAFE_LIMIT = 300       #Max pages to scrape
SLEEP_TIMER = 1.5      #Seconds to wait between pages

#Initialize Clients
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

#CRAWLER FUNCTION 
def discover_internal_links(seed_url):
    """
    Visits a main page and finds all relevant sub-pages.
    """
    try:
        print(f"Crawling Hub: {seed_url} ...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(seed_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        found_links = set()
        base_domain = urlparse(seed_url).netloc

        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(seed_url, href)
            parsed_url = urlparse(full_url)

            if parsed_url.netloc != base_domain: continue
            if any(x in full_url for x in [".pdf", ".jpg", "#", "wp-content", "login", "feed"]): continue
            
            #Keywords to keep relevant pages only
            if any(k in full_url for k in ["academics", "faculty", "program", "department", "admissions", "fee", "examinations", "research", "administration"]):
                found_links.add(full_url)

        print(f"   > Found {len(found_links)} sub-pages.")
        return list(found_links)

    except Exception as e:
        print(f"   ! Error crawling {seed_url}: {e}")
        return []

#SCRAPER FUNCTION (With Table Support)
def get_precision_content(url):
    try:
        #Browser header to prevent getting blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code != 200: return []

        soup = BeautifulSoup(response.content, 'html.parser')

        #SPECIAL HANDLER: AVADA PRICING TABLES
        #The theme uses divs with class "fusion-pricing-table" for fees
        for pricing in soup.select(".fusion-pricing-table"):
            headers = [h.get_text(strip=True) for h in pricing.select(".panel-heading")]
            prices = [p.get_text(strip=True) for p in pricing.select(".panel-body")]
            features = [f.get_text(strip=True) for f in pricing.select(".list-group-item")]
            
            #Convert to text block
            pricing_text = f"\n=== FEE/PRICING DATA ===\nPlans: {', '.join(headers)}\nPrices: {', '.join(prices)}\nDetails: {', '.join(features)}\n========================\n"
            pricing.replace_with(pricing_text)

        #B. SPECIAL HANDLER: STANDARD TABLES 
        for table in soup.find_all("table"):
            table_str = "\n--- TABLE DATA ---\n"
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all(["td", "th"])
                #Join columns with a pipe | to keep structure
                row_text = " | ".join(ele.get_text(strip=True) for ele in cols)
                table_str += row_text + "\n"
            table_str += "------------------\n"
            table.replace_with(table_str)

        #C. STANDARD CLEANUP (Avada Specifics)
        noise_selectors = [
            ".fusion-footer", ".fusion-header-wrapper", "#sliders-container", 
            ".fusion-sliding-bar", ".fusion-page-title-bar", "#side-header", 
            ".fusion-sharing-box", "script", "style", "iframe", "form", "nav"
        ]
        for selector in noise_selectors:
            for element in soup.select(selector):
                element.decompose()

        #D. GET TEXT 
        #Try to find the main content box first
        main_content = soup.find(id="main")
        text = main_content.get_text(separator="\n") if main_content else soup.get_text(separator="\n")

        #E. CHUNKING 
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        chunks = []
        chunk_size = 12   
        overlap = 4       
        
        for i in range(0, len(lines), chunk_size - overlap):
            window = lines[i : i + chunk_size]
            if len(window) < 3: continue 
            
            chunk_text = "\n".join(window)
            full_record = f"Source: {url}\nContent: {chunk_text}"
            chunks.append(full_record)
            
        return chunks

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

#MAIN EXECUTION
if __name__ == "__main__":
    if not SEED_URLS:
        print("No links found in .env.")
        exit()

    #A. Setup Qdrant Collection
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    #B. Create Index (Fixes "Bad Request" Error)
    print("Checking indexes...")
    try:
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="source_type",
            field_schema="keyword"
        )
        print("   Index created/verified.")
    except Exception as e:
        print(f"   Index note: {e}")

    #C. Discover Pages
    print("--- Starting Discovery ---")
    final_url_list = set(SEED_URLS)
    for seed in SEED_URLS:
        discovered = discover_internal_links(seed)
        final_url_list.update(discovered)
    
    urls_to_process = list(final_url_list)[:SAFE_LIMIT]
    print(f"\nTargeted {len(urls_to_process)} pages for processing.")

    #D. Clean Old Data (Full Refresh)
    print("Cleaning old website memory...")
    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(must=[FieldCondition(key="source_type", match=MatchValue(value="website"))])
    )

    #E. Scrape & Process
    all_documents = []
    print("Starting Scraping (This may take a few minutes)...")
    
    for idx, url in enumerate(urls_to_process):
        print(f"[{idx+1}/{len(urls_to_process)}] Reading: {url}")
        site_chunks = get_precision_content(url)
        all_documents.extend(site_chunks)
        time.sleep(SLEEP_TIMER)

    if not all_documents:
        print("No text found.")
        exit()

    #F. Vectorize & Upload
    print(f"\n🧠 Vectorizing {len(all_documents)} text chunks...")
    embeddings = list(embed_model.embed(all_documents))

    points = []
    for doc, vector in zip(all_documents, embeddings):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={"text": doc, "source_type": "website"} 
        ))

    print(f"Uploading to Qdrant...")
    batch_size = 50
    for i in range(0, len(points), batch_size):
        try:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i : i + batch_size]
            )
            print(f"Uploaded batch {i} - {i + batch_size}")
            time.sleep(0.2)
        except Exception as e:
            print(f"Upload error: {e}")

    print("\nSuccess! The Watcher has fully re-learned the website.")