# 📁 streamlit_app/app.py

import streamlit as st
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer

API_URL = "https://assessmentrecommendation.onrender.com/recommend"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Set Streamlit page config
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🔍 SHL Assessment Recommendation System")
st.markdown("""
Enter a **natural language query** or paste a **job description**, and we'll recommend up to 10 SHL assessments.
""")

# Text input
query = st.text_area("Paste Job Description or Query", height=150)

# Load embedding model once
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBED_MODEL)

model = load_model()

# When user clicks the button
if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query or job description.")
    else:
        with st.spinner("Embedding query and fetching recommendations..."):
            try:
                # Embed query using same model as FAISS
                embedding = model.encode(query).tolist()

                # Call backend with vector
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
