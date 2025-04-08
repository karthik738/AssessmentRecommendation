import faiss
import google.generativeai as genai
import numpy as np
import os

# ✅ STEP 1: Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Ensure this is set in env vars

# Sample query
query = "Software developer with experience in Python and backend systems."

# ✅ STEP 2: Get Gemini embedding
response = genai.embed_content(
    model="models/embedding-001",
    content=query,
    task_type="retrieval_query",
)

embedding = response['embedding']
embedding_np = np.array(embedding, dtype='float32')
print(f"Gemini embedding dimension: {embedding_np.shape[0]}")

# ✅ STEP 3: Load FAISS index
index = faiss.read_index("data\\faiss_index.idx")  # replace with your actual file path
print(f"FAISS index dimension: {index.d}")

# ✅ STEP 4: Compare
if embedding_np.shape[0] == index.d:
    print("✅ Dimensions match! You’re good to go.")
else:
    print("❌ Mismatch detected! Gemini vector != FAISS index vector size.")
