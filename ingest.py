import os
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import time

# Load secrets
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "knowledge_base"

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY)

# FIX 1: Set a long timeout (60 seconds) so it doesn't give up easily
qdrant_client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
    timeout=60.0 
)

model = SentenceTransformer('all-MiniLM-L6-v2') 

# 1. Create Collection
if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"Created collection: '{COLLECTION_NAME}'")

# 2. Read PDF
print("Reading 'handbook.pdf'...")
documents = []
try:
    reader = PdfReader("handbook.pdf")
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    
    # Split text by newlines and remove empty lines
    documents = [line.strip() for line in full_text.split('\n') if line.strip()]
    print(f"I found {len(documents)} text chunks.")

except FileNotFoundError:
    print("ERROR: 'handbook.pdf' not found.")
    exit()

# 3. Vectorize
print("Converting text to vectors (this might take a moment)...")
embeddings = model.encode(documents)

points = []
for idx, (doc, vector) in enumerate(zip(documents, embeddings)):
    points.append(PointStruct(
        id=idx,
        vector=vector.tolist(),
        payload={"text": doc}
    ))

# 4. Upload in Smaller Batches
print(f"Starting upload of {len(points)} points...")

# FIX 2: Smaller batch size (50) to be safer
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
        print(f"Error uploading batch starting at {i}: {e}")
        # Wait a bit longer if there is an error
        time.sleep(2)
        
    time.sleep(0.2) 

print("Success! Process finished.")