# 📁 frontend/app.py

import streamlit as st
import pandas as pd
import requests
import os

# ===== CONFIG =====
API_URL = "https://assessmentrecommendation.onrender.com/recommend"

# ===== Setup =====
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🔍 SHL Assessment Recommendation System")
st.markdown("""
Enter a **natural language query** or paste a **job description**, and we'll recommend up to 10 SHL assessments.
""")

# ===== Input field =====
query = st.text_area("Paste Job Description or Query", height=150)

# ===== On button click =====
if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query or job description.")
    else:
        with st.spinner("🔍 Sending query to backend..."):
            try:
                response = requests.post(API_URL, json={"query": query})
                response.raise_for_status()
                results = response.json()["recommended_assessments"]

                if not results:
                    st.info("No recommendations found.")
                else:
                    df = pd.DataFrame(results)

                    df = df.rename(columns={
                        "description": "Assessment Description",
                        "url": "Assessment URL",
                        "remote_support": "Remote Support",
                        "adaptive_support": "Adaptive Support",
                        "duration": "Duration (mins)",
                        "test_type": "Test Types"
                    })

                    st.markdown("### 📋 Top Recommendations")
                    st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
