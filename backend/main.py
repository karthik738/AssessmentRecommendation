# 📁 api/main.py — FastAPI backend for SHL recommender

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from recommender import SHLRecommender
from pydantic import BaseModel
from typing import List
from typing import Optional

app = FastAPI(title="SHL Assessment Recommender")

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
recommender = SHLRecommender()

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

@app.get("/recommend", response_model=List[Recommendation])
def get_recommendations(q: str = Query(..., description="Query or Job Description")):
    """Get top 10 SHL recommendations based on a query or job description."""
    return recommender.recommend_detailed(q)
