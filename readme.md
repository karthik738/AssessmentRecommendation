# 🧠 SHL Generative AI Product Recommender

This project is a smart product recommendation system built for the SHL AI Intern assignment using a **Retrieval-Augmented Generation (RAG)** pipeline with an LLM-based reranking layer. It recommends SHL assessments based on natural language queries or job descriptions.

---

## 🚀 Features

- 🔍 Semantic Search over SHL product catalog using FAISS
- 🧠 Claude 3 Haiku reranking for precise results
- 📋 Duration, Test Types, Remote & Adaptive flags, Downloads
- 🎯 Precision Mode: Top 3 accurate results based on query
- 📦 FastAPI backend with `/recommend` endpoint
- 🖼️ Streamlit UI for testing and interaction
- 🧪 Bonus: Conversational RAG Chatbot (`step1_rag.py`) with Gemini / DeepSeek / LLaMA3

---

## 📁 Folder Structure
```
shl-recommender/
├── data/                    # JSON + FAISS index
├── recommender/            # Core recommendation logic
│   └── core.py             
├── streamlit_app/          # Streamlit UI
│   └── app.py              
├── api/                    # FastAPI backend
│   └── main.py             
├── step1_rag.py            # Chatbot variant (multi-model)
├── SHL_Generative_AI_Summary.pdf   # Approach document
├── requirements.txt        # Dependencies
├── README.md               # This file
```

---

## 🧰 Tech Stack

| Component      | Tool                      |
|----------------|---------------------------|
| Embeddings     | BAAI bge-small-en-v1.5    |
| Vector DB      | FAISS                     |
| LLM Reranker   | Claude 3 Haiku (OpenRouter) |
| Frontend       | Streamlit                 |
| Backend        | FastAPI                   |
| Chatbot Models | Gemini, DeepSeek, LLaMA   |
| Scraper        | Selenium + BeautifulSoup  |

---

## 🖼️ Screenshots
(Add here or link images/videos showing UI + API in action)

---

## ▶️ How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Backend (API)
```bash
uvicorn api.main:app --reload
```
Visit: `http://localhost:8000/recommend?q=productivity manager`

### 3. Run Streamlit Frontend
```bash
streamlit run streamlit_app/app.py
```

---

## 🌐 Deployment Links

| Component   | URL                  |
|-------------|----------------------|
| Live UI     | _Coming soon_        |
| API Endpoint| _Coming soon_        |
| GitHub Repo | _To be added_        |

---

## ✅ Deliverables

- ✅ Top 3 SHL product recommendations
- ✅ API + Frontend working demo
- ✅ Full metadata extraction
- ✅ RAG chatbot (optional)
- ✅ PDF report with reasoning (`SHL_Generative_AI_Summary.pdf`)

---

## 📬 Contact
For feedback or questions, feel free to reach out!

