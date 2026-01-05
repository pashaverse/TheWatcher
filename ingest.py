import os
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Load secrets
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "knowledge_base"

# --- CHUNKING CONFIGURATION ---
# 800 chars is roughly 150-200 words (a solid paragraph).
# 100 chars overlap prevents cutting sentences in half at the edge.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def get_text_chunks(text, chunk_size, overlap):
    """Splits text into overlapping chunks so context isn't lost."""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move forward, but step back by 'overlap' to capture the seam
        start += chunk_size - overlap
    
    return chunks

# Initialize Client
# We set a 60s timeout to prevent errors on slow connections
qdrant_client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
    timeout=60.0 
)

model = SentenceTransformer('all-MiniLM-L6-v2') 

# 1. CLEAN SLATE: Delete old bad data
# We must delete the old collection because it contains the broken "line-by-line" data.
if qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    print(f"Deleting old '{COLLECTION_NAME}' to remove broken fragments...")
    qdrant_client.delete_collection(collection_name=COLLECTION_NAME)

# 2. Re-create Collection
qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
print(f"Created fresh collection: '{COLLECTION_NAME}'")

# 3. Read & CLEAN PDF
print("Reading 'handbook.pdf'...")
try:
    reader = PdfReader("handbook.pdf")
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            # Append with a space, NOT a newline, to merge pages seamlessly
            full_text += text + " "
    
    # CRITICAL FIX: Merge broken lines into one giant block of text
    # This heals sentences that were split across lines in the PDF
    clean_text = full_text.replace('\n', ' ').replace('  ', ' ')
    
    print(f"Total raw text length: {len(clean_text)} characters.")

    # 4. Create Semantic Chunks
    documents = get_text_chunks(clean_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Split into {len(documents)} semantic chunks (approx {CHUNK_SIZE} chars each).")

except FileNotFoundError:
    print("ERROR: 'handbook.pdf' not found. Please make sure the file is in this folder.")
    exit()

# 5. Vectorize
print("Converting text to vectors (this may take a minute)...")
embeddings = model.encode(documents)

points = []
for idx, (doc, vector) in enumerate(zip(documents, embeddings)):
    points.append(PointStruct(
        id=idx,
        vector=vector.tolist(),
        payload={"text": doc}
    ))

# 6. Upload in Batches
print(f"Starting upload of {len(points)} points...")
batch_size = 50 

for i in range(0, len(points), batch_size):
    batch = points[i : i + batch_size]
    try:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        print(f"Uploaded batch {i} to {i + len(batch)}...")
    except Exception as e:
        print(f"Error uploading batch: {e}")
        time.sleep(2)
    time.sleep(0.2) 

print("Success! Your Watcher is now fully educated with high-quality data.")