# 📁 streamlit_app/app.py

import streamlit as st
import requests
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

API_URL = "https://assessmentrecommendation.onrender.com/recommend"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Set Streamlit page config
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🔍 SHL Assessment Recommendation System")
st.markdown("""
Enter a **natural language query** or paste a **job description**, and we'll recommend up to 10 SHL assessments.
""")

# Load embedding model once with Hugging Face token
@st.cache_resource
def load_model():
    hf_token = st.secrets["HUGGINGFACE_TOKEN"]
    os.environ["HF_TOKEN"] = hf_token
    login(hf_token)
    return SentenceTransformer(EMBED_MODEL, use_auth_token=hf_token)

model = load_model()

# Text input
query = st.text_area("Paste Job Description or Query", height=150)

# Button action
if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query or job description.")
    else:
        with st.spinner("Embedding query and fetching recommendations..."):
            try:
                embedding = model.encode(query).tolist()

                response = requests.post(API_URL, json={"vector": embedding})
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
                st.error(f"❌ Failed to fetch recommendations: {str(e)}")
