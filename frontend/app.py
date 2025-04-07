import streamlit as st
import pandas as pd
import requests
import google.generativeai as genai
import numpy as np
import os

# Constants
API_URL = "https://assessmentrecommendation.onrender.com/recommend"  # Backend POST endpoint
GEMINI_MODEL = "models/embedding-001"

# Streamlit UI setup
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🔍 SHL Assessment Recommendation System")
st.markdown("""
Enter a **natural language query** or paste a **job description**, and we'll recommend up to 10 SHL assessments.
""")

# Input field
query = st.text_area("Paste Job Description or Query", height=150)

# Load Gemini config
@st.cache_resource
def load_gemini_model():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel(GEMINI_MODEL)

# Embed query using Gemini
def embed_query(text):
    model = load_gemini_model()
    embedding_response = model.embed_content(
        content=text,
        task_type="retrieval_document"
    )
    return embedding_response["embedding"]

# When user clicks
if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query or job description.")
    else:
        with st.spinner("Embedding query with Gemini and fetching recommendations..."):
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
                                [f"[{d['title']}]({d['url']}) ({d['language']})" for d in items if d['url']]
                            )
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
