# 📁 streamlit_app/app.py

import streamlit as st
import requests
import pandas as pd

API_URL = "https://assessmentrecommendation-production.up.railway.app/recommend"

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🔍 SHL Assessment Recommendation System")
st.markdown("""
Enter a **natural language query** or paste a **job description**, and we'll recommend up to 10 SHL assessments.
""")

query = st.text_area("Paste Job Description or Query", height=150)

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query or job description.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                response = requests.get(API_URL, params={"q": query})
                response.raise_for_status()
                results = response.json()

                if not results:
                    st.info("No recommendations found.")
                else:
                    df = pd.DataFrame(results)

                    # Format downloads into a readable column
                    if "downloads" in df.columns:
                        df["downloads"] = df["downloads"].apply(lambda items: "\n".join([f"[{d['title']}]({d['url']}) ({d['language']})" for d in items if d['url']]))

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
                st.error(f"Failed to fetch recommendations: {str(e)}")