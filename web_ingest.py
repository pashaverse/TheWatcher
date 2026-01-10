import os
import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, MatchAny
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

SAFE_LIMIT = 300       
SLEEP_TIMER = 1.5      

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def discover_internal_links(seed_url):
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
            if any(k in full_url for k in ["academics", "faculty", "program", "department", "admissions", "fee", "examinations", "research", "administration"]):
                found_links.add(full_url)

        print(f"   > Found {len(found_links)} sub-pages.")
        return list(found_links)
    except Exception as e:
        print(f"   ! Error crawling {seed_url}: {e}")
        return []

def get_precision_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200: 
            print(f"Status Code {response.status_code}")
            return None 

        soup = BeautifulSoup(response.content, 'html.parser')

        for pricing in soup.select(".fusion-pricing-table"):
            headers = [h.get_text(strip=True) for h in pricing.select(".panel-heading")]
            prices = [p.get_text(strip=True) for p in pricing.select(".panel-body")]
            features = [f.get_text(strip=True) for f in pricing.select(".list-group-item")]
            pricing_text = f"\n=== FEE/PRICING DATA ===\nPlans: {', '.join(headers)}\nPrices: {', '.join(prices)}\nDetails: {', '.join(features)}\n========================\n"
            pricing.replace_with(pricing_text)

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
            chunk_text = "\n".join(window)
            full_record = f"Source: {url}\nContent: {chunk_text}"
            chunks.append(full_record)
            
        return chunks

    except Exception as e:
        print(f"Network Error: {e}")
        return None 

def update_page_safely(url):
    """
    1. Scrapes the new page.
    2. IF successful: Deletes OLD data for this URL -> Uploads NEW data.
    3. IF failed: Does nothing (Old data remains safe).
    """
    print(f"   Processing: {url}")
    new_chunks = get_precision_content(url)

    # FAILURE: Stop here if scrape failed
    if new_chunks is None:
        print(f"Scrape Failed. KEEPING OLD VERSION of {url}")
        return

    # SUCCESS: Continue to upload (This logic was previously unreachable!)
    try:
        embeddings = list(embed_model.embed(new_chunks))
        points = []
        for doc, vector in zip(new_chunks, embeddings):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                # IMPORTANT: We add 'url' to payload so we can target it later
                payload={"text": doc, "source_type": "website", "url": url} 
            ))

        # Delete ONLY this specific page's data
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(key="url", match=MatchValue(value=url))
                ]
            )
        )

        # Upload new data
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"Updated: {len(points)} chunks.")
        
    except Exception as e:
        print(f"DB Error on {url}: {e}")


if __name__ == "__main__":
    if not SEED_URLS:
        print("No links found in .env.")
        exit()

    # A. Setup Qdrant & Indexes
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    # Create Index on 'url' so we can delete specific pages fast
    print("Verifying URL index...")
    try:
        qdrant_client.create_payload_index(collection_name=COLLECTION_NAME, field_name="url", field_schema="keyword")
        qdrant_client.create_payload_index(collection_name=COLLECTION_NAME, field_name="source_type", field_schema="keyword")
    except: pass

    print("---Starting Discovery ---")
    final_url_list = set(SEED_URLS)
    for seed in SEED_URLS:
        discovered = discover_internal_links(seed)
        final_url_list.update(discovered)
    
    urls_to_process = list(final_url_list)[:SAFE_LIMIT]
    print(f"\nTargeted {len(urls_to_process)} pages for Safe Updates.")

    # Optional: Wipe only legacy data that doesn't have a 'url' tag yet
    print("Checking for legacy data...")
    try:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="source_type", match=MatchValue(value="website"))],
                must_not=[FieldCondition(key="url", match=MatchAny(any=urls_to_process))] 
            )
        )
    except: pass

    print("Starting Smart Updates...")
    for idx, url in enumerate(urls_to_process):
        print(f"[{idx+1}/{len(urls_to_process)}]", end=" ")
        update_page_safely(url)
        time.sleep(SLEEP_TIMER)

    print("\nThe Watcher has finished the update cycle.")