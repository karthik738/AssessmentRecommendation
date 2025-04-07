# 📁 streamlit_app/app.py

import streamlit as st
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer

API_URL = "https://assessmentrecommendation.onrender.com/recommend"
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"

# Set page layout
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🔍 SHL Assessment Recommendation System")
st.markdown("""
Enter a **natural language query** or paste a **job description**, and we'll recommend up to 10 SHL assessments.
""")

# Input query
query = st.text_area("Paste Job Description or Query", height=150)

# Embed model init (cache to avoid reload)
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBED_MODEL)

model = load_model()

# Button handler
if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query or job description.")
    else:
        with st.spinner("Embedding query and fetching recommendations..."):
            try:
                # Embed user query
                embedding = model.encode(query).tolist()

                # Send vector to backend
                response = requests.post(API_URL, json={"vector": embedding})
                response.raise_for_status()
                results = response.json()

                if not results:
                    st.info("No recommendations found.")
                else:
                    df = pd.DataFrame(results)

                    # Format downloads into readable links
                    if "downloads" in df.columns:
                        df["downloads"] = df["downloads"].apply(
                            lambda items: "\n".join(
                                [f"[{d['title']}]({d['url']}) ({d['language']})" for d in items if d['url']]
                            )
                        )

                    # Clean column headers
                    df = df.rename(columns={
                        "name": "Assessment Name",
                        "url": "Assessment URL",
                        "remote_testing": "Remote Testing",
                        "adaptive_irt": "Adaptive/IRT",
                        "duration": "Duration",
                        "test_types": "Test Types",
                        "downloads": "Downloads"
                    })

                    # Show recommendations
                    st.markdown("### 📋 Top Recommendations")
                    st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Failed to fetch recommendations: {str(e)}")
