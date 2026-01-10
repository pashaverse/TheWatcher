from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from dotenv import load_dotenv
import os

load_dotenv()

# Setup
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
collection = "knowledge_base"

def peek_inside(search_term):
    print(f"\n--- 🔍 SEARCHING FOR: '{search_term}' ---")
    
    # Convert text to vector
    vector = list(model.embed([search_term]))[0].tolist()
    
    # Search DB
    results = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=3 # Show top 3 matches
    )

    for i, hit in enumerate(results):
        print(f"\n[Result {i+1}] (Score: {hit.score:.2f})")
        print(f"Source: {hit.payload.get('source_type', 'Unknown')}")
        print("-" * 30)
        # Print the actual text the bot reads
        print(hit.payload['text']) 
        print("-" * 30)

if __name__ == "__main__":
    #change this to whatever is missing
    peek_inside("Fee Structure")