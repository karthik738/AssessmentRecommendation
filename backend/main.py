# 📁 backend/main.py — FastAPI backend for SHL recommender

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from recommender import SHLRecommender
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="SHL Assessment Recommender")

# CORS config to allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS + docstore
recommender = SHLRecommender()

# ✅ Input: Vector from frontend
class QueryVector(BaseModel):
    vector: List[float]

# ✅ Output structure
class Download(BaseModel):
    title: str
    url: str
    language: str

class Recommendation(BaseModel):
    name: str
    url: str
    remote_testing: str
    adaptive_irt: str
    duration: str
    test_types: List[str]
    downloads: Optional[List[Download]] = []

@app.get("/")
def read_root():
    return {"message": "Welcome to the SHL Assessment Recommender API!"}

@app.post("/recommend", response_model=List[Recommendation])
def recommend_from_vector(req: QueryVector):
    # print("✅ POST received. Vector length:", len(req.vector))
    """Recommend assessments based on pre-embedded vector input."""
    return recommender.recommend_detailed(req.vector)
