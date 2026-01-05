import os
import time
from pinecone import Pinecone
import cohere
from pypdf import PdfReader
from cohere.errors import TooManyRequestsError

# --- 🔑 YOUR KEYS ---
PINECONE_KEY = "pcsk_4LSFaz_3AymY2oTqu7KdfJoptXzkb5JwU4um4mzgnyopmQTwAjG4jbVh5DWpp6La2iA6D8"
COHERE_KEY = "GIL16ifLJwOhalqYndZBYAdXc9H4SnH6NdkJKEOr"

# --- ⚙️ CONFIGURATION ---
INDEX_NAME = "watcher-memory" 
PDF_PATH = "handbook.pdf"

# Initialize
pc = Pinecone(api_key=PINECONE_KEY)
co = cohere.Client(COHERE_KEY)

# 1. Read PDF
print(f"📖 Reading {PDF_PATH}...")
if not os.path.exists(PDF_PATH):
    print("❌ PDF not found!")
    exit()

reader = PdfReader(PDF_PATH)
text = ""
for page in reader.pages:
    extract = page.extract_text()
    if extract:
        text += extract + "\n"

# 2. Chunking
CHUNK_SIZE = 1000
chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
total_chunks = len(chunks)
print(f"🧩 Split into {total_chunks} chunks.")

# 3. Embed & Upload (With Progress Bar)
print("🚀 Uploading to The Archive...")
index = pc.Index(INDEX_NAME)
batch_size = 20
total_batches = (total_chunks + batch_size - 1) // batch_size # Calculate total batches

for i in range(0, total_chunks, batch_size):
    batch_chunks = chunks[i:i+batch_size]
    current_batch = (i // batch_size) + 1
    
    # Progress Indicator
    percent = int((current_batch / total_batches) * 100)
    print(f"\n📦 Processing Batch {current_batch}/{total_batches} ({percent}%)")
    
    while True:
        try:
            response = co.embed(
                texts=batch_chunks, 
                model="embed-english-v3.0", 
                input_type="search_document"
            )
            
            vectors = []
            for j, embedding in enumerate(response.embeddings):
                vectors.append({
                    "id": f"chunk_{i+j}",
                    "values": embedding,
                    "metadata": {"text": batch_chunks[j]}
                })

            index.upsert(vectors=vectors)
            print(f"   ✅ Success! Waiting 12s to respect rate limit...")
            time.sleep(12) 
            break 

        except TooManyRequestsError:
            print("   ⏳ Rate limit hit. Cooling down for 60 seconds... (Don't close window)")
            time.sleep(60)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            break

print("\n🎉 MISSION ACCOMPLISHED. The Watcher is ready.")