# 📁 step1_rag.py — SHL RAG chatbot with LangChain memory, feedback, multi-model scoring

import os
import json
import faiss
import numpy as np
from datetime import datetime
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage

# === Config ===
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DOCSTORE_PATH = "docstore.json"
INDEX_PATH = "faiss_index.bin"
LOG_FILE = "chat_logs.json"
FEEDBACK_FILE = "feedback.json"
TOP_K = 5
SPELL_THRESHOLD = 0.7

OPENROUTER_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.5-pro-exp-03-25:free",
    "deepseek/deepseek-r1-distill-qwen-32b:free"
]

# === OpenRouter Setup ===
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "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"),
)

# === Load FAISS + Docstore ===
print("🔁 Loading FAISS index + docstore...")
index = faiss.read_index(INDEX_PATH)
with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
    docstore = json.load(f)

print("🔍 Loading embedding model...")
embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

# === Memory ===
memory = ConversationBufferMemory(return_messages=True)

# === Spell Correction ===
def fuzzy_match(query):
    names = [doc["name"] for doc in docstore]
    match = get_close_matches(query, names, n=1, cutoff=SPELL_THRESHOLD)
    return match[0] if match else None

# === Similarity Search ===
def search(query):
    embeddings = np.array(embedding_model.encode([query]))
    # print(f"🔎 FAISS index type: {type(index)}")
    _, indices = index.search(embeddings, TOP_K)
    return [docstore[i] for i in indices[0]]

# === Prompt Builder ===
def build_prompt(query, memory_text, matches):
    context = "\n\n".join(doc["text"] for doc in matches)
    return f"""
You are a helpful assistant trained on the SHL product catalog.

- Answer only using the provided context. Answer in detail.
- Prioritize answering using the provided context.
- If no relevant context is found but the query is SHL-related (e.g., CEO, history), you may answer from general knowledge.
-If the context doesn't include the answer but the query is SHL-related (like CEO or company info), use general knowledge.
- Respond to SHL product queries and general SHL company info.
- Clarify and proceed with minor spelling issues.
- For ambiguous product names, offer suggestions.
- Use long-form and bullet-pointed formatting when useful.
- Do not hallucinate facts unrelated to SHL.
-Give detailed answers, but avoid excessive verbosity.

Previous memory:
{memory_text}

Context:
{context}

User: {query}
Answer:
""".strip()

from sentence_transformers import util

def rank_responses(query, responses):
    query_embed = embedding_model.encode(query)
    scores = [
        (model, response, util.cos_sim(embedding_model.encode(response), query_embed).item())
        for model, response in responses
    ]
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[0]  # Return the best

# === OpenRouter Call ===
def call_model(prompt, model_name):
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "shl-rag-cli"
            },
            model=model_name,
            messages=[
                {"role": "system", "content": """You are a helpful assistant trained on the SHL product catalog.
You must prioritize using the provided context and memory. 
However, if the question is clearly about SHL (like its CEO or company details), you may answer from general knowledge.
Do not hallucinate facts unrelated to SHL. Politely refuse totally unrelated topics."""},
                {"role": "user", "content": prompt}
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR from {model_name}]: {str(e)}"

# === Logging ===
def log_chat(user_query, response, model_used):
    log = {
        "timestamp": datetime.now().isoformat(),
        "query": user_query,
        "model": model_used,
        "response": response
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")

def record_feedback(query, response, score, model):
    feedback = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "model": model,
        "response": response,
        "score": score
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback) + "\n")

# === Main Chat Loop ===
print("\n🤖 SHL Chatbot ready! Ask anything about SHL. Type 'exit' to quit.\n")

while True:
    query = input("👤 You: ").strip()
    if query.lower() in {"exit", "quit"}:
        break

    best_match = fuzzy_match(query)
    if best_match and best_match.lower() != query.lower():
        print(f"🤖 Bot: Did you mean **{best_match}**? Using best match.")
        query = best_match

    memory_text = "\n".join(
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Bot: {msg.content}"
        for msg in memory.buffer
    )

    top_docs = search(query)
    prompt = build_prompt(query, memory_text, top_docs)

    print("🧠 Comparing model responses...\n")
    responses = []
    for model in OPENROUTER_MODELS:
        response = call_model(prompt, model)
        responses.append((model, response))
        log_chat(query, response, model)

    best_model, best_response, best_score = rank_responses(query, responses)

    print(f"🤖 Bot (via {best_model}): {best_response}")
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(best_response)

    # for model in OPENROUTER_MODELS:
    #     print(f"📡 Model: {model}")
    #     response = call_model(prompt, model)
    #     print(f"🤖 {model.split('/')[-1]}:\n{response}\n")
    #     log_chat(query, response, model)
    #     memory.chat_memory.add_user_message(query)
    #     memory.chat_memory.add_ai_message(response)

        # feedback = input("📝 Feedback? (👍=1 / 👎=0 / enter=skip): ").strip()
        # if feedback in {"0", "1"}:
        #     record_feedback(query, response, int(feedback), model)
