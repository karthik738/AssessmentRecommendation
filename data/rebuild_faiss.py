import os
import json
import faiss
import numpy as np
from tqdm import tqdm
import google.generativeai as genai

# Load environment variables
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY is missing in your .env file")

# Configure Gemini SDK
genai.configure(api_key=GOOGLE_API_KEY)

# Load the SHL docstore
with open("shl/data/docstore.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
metadata = []

print(f"📄 Total items in docstore: {len(data)}")

for item in data:
    name = item.get("name", "")
    description = item.get("text", "")
    combined_text = f"{name}. {description}"
    texts.append(combined_text)
    metadata.append(item)

# Embed and build FAISS index
vectors = []
for text in tqdm(texts, desc="🔍 Embedding texts with Gemini"):
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        vector = response['embedding']
        vectors.append(vector)
    except Exception as e:
        print(f"❌ Error embedding {text[:40]}: {e}")

if not vectors:
    print("⚠️ No embeddings were generated. FAISS index not built.")
    exit()

# Create and save FAISS index
dimension = len(vectors[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors).astype("float32"))

faiss.write_index(index, "shl/data/faiss_index.idx")
with open("shl/data/index_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ FAISS index built and saved with {len(vectors)} vectors.")
