# 📁 recommender/core.py

import os
import json
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from difflib import get_close_matches
import re
from openai import OpenAI

class SHLRecommender:
    def __init__(self,
                 embed_model_name="BAAI/bge-small-en-v1.5",
                 index_path="data/faiss_index.bin",
                 docstore_path="data/docstore.json",
                 top_k=10,  # 🎯 Precision mode: fewer results
                 spell_threshold=0.7):

        self.embed_model_name = embed_model_name
        self.index_path = index_path
        self.docstore_path = docstore_path
        self.top_k = top_k
        self.spell_threshold = spell_threshold

        self.model = SentenceTransformer(embed_model_name)
        self.index = faiss.read_index(index_path)

        with open(docstore_path, "r", encoding="utf-8") as f:
            self.docstore = json.load(f)

        self.product_names = [doc["name"] for doc in self.docstore]

        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.llm_model = "anthropic/claude-3-haiku"

    def _fuzzy_match(self, query):
        match = get_close_matches(query, self.product_names, n=1, cutoff=self.spell_threshold)
        return match[0] if match else None

    def _search(self, query):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        _, indices = self.index.search(query_vec, self.top_k)
        return [self.docstore[i] for i in indices[0]]

    def recommend(self, query):
        match = self._fuzzy_match(query)
        if match and match.lower() != query.lower():
            query = match

        results = self._search(query)
        return [
            {
                "name": doc["name"],
                "link": doc["source"],
                "snippet": doc["text"][:300] + "...",
            }
            for doc in results
        ]

    def recommend_detailed(self, query):
        match = self._fuzzy_match(query)
        if match and match.lower() != query.lower():
            query = match

        results = self._search(query)

        simplified = []
        for doc in results:
            text_raw = doc["text"]
            text = text_raw.lower()
            downloads = self._extract_downloads(text_raw)

            structured = {
                "name": doc["name"],
                "url": doc["source"],
                "remote_testing": "yes" if "remote testing: yes" in text else "no",
                "adaptive_irt": "yes" if "adaptive/irt: yes" in text else "no",
                "duration": self._extract_duration(text),
                "test_types": self._extract_test_types(text),
                "downloads": downloads if downloads else []
            }
            simplified.append(structured)

        ranked = self._rerank_with_openrouter(query, simplified)
        return ranked

    def _extract_duration(self, text):
        match = re.search(r"completion time in minutes\s*=\s*(\d+)", text)
        return f"{match.group(1)} minutes" if match else "Unknown"

    def _extract_test_types(self, text):
        test_type_block = text.split("test types:")[-1].split("link:")[0].strip()
        return [t.strip() for t in test_type_block.split(",") if t.strip()]

    def _extract_downloads(self, text):
        match = re.search(r"(?i)downloads:(.*?)(link:|$)", text, re.DOTALL)
        if not match:
            return []

        downloads_block = match.group(1)
        matches = re.findall(r"([^:]+):\s*(https?://[^\s]+)\s*\(([^)]+)\)", downloads_block)

        downloads = []
        for title, url, language in matches:
            downloads.append({
                "title": title.strip(" -").strip(),
                "url": url.strip(),
                "language": language.strip()
            })
        return downloads

    def _rerank_with_openrouter(self, query, assessments):
        try:
            prompt = f"""
You are an assistant that reranks SHL assessments.

Query: {query}

Assessments:
{json.dumps(assessments, indent=2)}

Reorder and return only the **top 5 to 10 most relevant** assessments as valid JSON.
If fewer than 10 are highly relevant, return fewer.
Only include the sorted list as output.
""".strip()

            completion = self.llm_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "shl-recommender"
                },
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant trained on SHL assessments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )

            content = completion.choices[0].message.content.strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                else:
                    raise ValueError("Could not extract valid JSON from LLM response.")

        except Exception as e:
            print(f"❌ Reranking failed: {str(e)}")
            return assessments