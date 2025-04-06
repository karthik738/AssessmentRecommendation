# 📁 File: embedding.py

import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

# === Config ===
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
INPUT_JSON_PATH = "data\shl_product_details_full.json"
OUTPUT_INDEX_PATH = "data\\faiss_index.bin"
OUTPUT_DOCSTORE_PATH = "data\docstore.json"

# === Step 1: Load Model ===
print("\n🔍 Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL_NAME)

# === Step 2: Load Product JSON ===
print("📖 Loading SHL products...")
with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# === Step 3: Convert to Flat List of Chunks ===
documents = []
doc_id = 0

def format_downloads(download_list):
    if not download_list:
        return "None"
    return "\n".join(
        f"- {item.get('title', '')}: {item.get('url', '')} ({item.get('language', '')})"
        for item in download_list
    )

print("\n🧱 Preparing documents for embedding...")
for category in ["pre_packaged_solutions", "individual_test_solutions"]:
    for item in raw_data[category]:
        text = f"""
        Name: {item.get('name', '')}
        Title: {item.get('title', '')}
        Description: {item.get('description', '')}
        Job Levels: {item.get('job_levels', '')}
        Languages: {item.get('languages', '')}
        Assessment Length: {item.get('assessment_length', '')}
        Remote Testing: {item.get('remote_testing', '')}
        Adaptive/IRT: {item.get('adaptive_irt', '')}
        Test Types: {', '.join(item.get('test_types', []))}
        Link: {item.get('link', '')}
        Downloads:
        {format_downloads(item.get('downloads', []))}
        """
        text = text.strip().replace("\n", " ")
        documents.append({
            "id": doc_id,
            "text": text,
            "source": item.get("link", ""),
            "name": item.get("name", "")
        })
        doc_id += 1

print(f"📚 Total documents to embed: {len(documents)}")

# === Step 4: Embed Texts ===
print("\n📌 Embedding documents...")
texts = [doc["text"] for doc in documents]
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# === Step 5: Build FAISS Index ===
print("\n📦 Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
faiss.write_index(index, OUTPUT_INDEX_PATH)
print(f"✅ Saved FAISS index to: {OUTPUT_INDEX_PATH}")

# === Step 6: Save Metadata ===
print(f"🧾 Saving docstore metadata to: {OUTPUT_DOCSTORE_PATH}")
with open(OUTPUT_DOCSTORE_PATH, "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=2, ensure_ascii=False)

print("\n✅ All Done! You can now use FAISS + docstore for RAG.")
