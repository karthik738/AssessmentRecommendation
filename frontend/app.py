# 📁 frontend/app.py

import streamlit as st
import pandas as pd
import requests
import google.generativeai as genai
import os

# ===== CONFIG =====
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Replace with your actual Gemini API key
GEMINI_MODEL = "models/embedding-001"
API_URL = "https://assessmentrecommendation.onrender.com/recommend"
# API_URL="http://localhost:9000/recommend"  # For local testing

# ===== Setup =====
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🔍 SHL Assessment Recommendation System")
st.markdown("""
Enter a **natural language query** or paste a **job description**, and we'll recommend up to 10 SHL assessments.
""")

# ===== Configure Gemini =====
genai.configure(api_key=GOOGLE_API_KEY)

# ===== Input field =====
query = st.text_area("Paste Job Description or Query", height=150)

# ===== Embed query with Gemini =====
def embed_query(text):
    response = genai.embed_content(
        model=GEMINI_MODEL,
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return response["embedding"]

# ===== On button click =====
if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query or job description.")
    else:
        with st.spinner("🔍 Embedding query and fetching recommendations..."):
            try:
                vector = embed_query(query)

                # Send to backend
                response = requests.post(API_URL, json={"vector": vector})
                response.raise_for_status()
                results = response.json()

                if not results:
                    st.info("No recommendations found.")
                else:
                    df = pd.DataFrame(results)

                    if "downloads" in df.columns:
                        df["downloads"] = df["downloads"].apply(
                            lambda items: "\n".join(
                                [f"[{d['title']}]({d['url']}) ({d['language']})" for d in items if d.get("url")]
                            ) if items else ""
                        )

                    df = df.rename(columns={
                        "name": "Assessment Name",
                        "url": "Assessment URL",
                        "remote_testing": "Remote Testing",
                        "adaptive_irt": "Adaptive/IRT",
                        "duration": "Duration",
                        "test_types": "Test Types",
                        "downloads": "Downloads"
                    })

                    st.markdown("### 📋 Top Recommendations")
                    st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")