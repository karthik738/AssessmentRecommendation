import os
import json
import faiss
import numpy as np
import re
from openai import OpenAI
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class SHLRecommender:
    def __init__(self, top_k=10, spell_threshold=0.7):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.index_path = os.path.join(base_path, "../data/faiss_index.idx")
        self.docstore_path = os.path.join(base_path, "../data/index_metadata.json")
        self.top_k = top_k
        self.spell_threshold = spell_threshold

        self.index = faiss.read_index(self.index_path)

        with open(self.docstore_path, "r", encoding="utf-8") as f:
            self.docstore = json.load(f)

        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.llm_model = "anthropic/claude-3-haiku"


    def _embed_query(self, query: str) -> list:
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="RETRIEVAL_QUERY"
            )
            return response["embedding"]
        except Exception as e:
            print(f"❌ Gemini embedding failed: {e}")
            return []


    def search_by_vector(self, vector):
        query_vec = np.array([vector], dtype="float32")
        _, indices = self.index.search(query_vec, self.top_k)
        return [self.docstore[i] for i in indices[0]]

    def recommend_text(self, query: str):
        embedded_vector = self._embed_query(query)
        results = self.search_by_vector(embedded_vector)

        final = []
        for doc in results:
            text_raw = doc["text"]
            text_lower = text_raw.lower()

            structured = {
                "url": doc["source"],
                "adaptive_support": "Yes" if "adaptive/irt: yes" in text_lower else "No",
                "description": self._extract_description(text_raw),
                "duration": self._extract_duration_int(text_raw),
                "remote_support": "Yes" if "remote testing: yes" in text_lower else "No",
                "test_type": self._extract_test_types(text_lower)
            }

            final.append(structured)

        reranked = self._rerank_with_openrouter(query, final)
        return reranked

    def _extract_description(self, text):
        match = re.search(r"Description:\s*(.*?)\s*Job Levels:", text, re.DOTALL)
        return match.group(1).strip().replace("\n", " ") if match else "No description found."

    def _extract_duration_int(self, text):
        match = re.search(r"Completion Time in minutes\s*=\s*(\d+)", text, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    def _extract_test_types(self, text):
        test_type_block = ""
        try:
            test_type_block = text.split("test types:")[-1].split("link:")[0].strip()
        except Exception:
            return []
        return [t.strip() for t in test_type_block.split(",") if t.strip()]

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
