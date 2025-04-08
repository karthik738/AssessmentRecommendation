from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from recommender import SHLRecommender

app = FastAPI(title="SHL Assessment Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = SHLRecommender()

# SHL Expected Input
class QueryText(BaseModel):
    query: str

# SHL Expected Output
class Assessment(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: List[Assessment]

@app.get("/")
def read_root():
    return {"message": "Welcome to the SHL Assessment Recommender API!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_from_query(req: QueryText):
    results = recommender.recommend_text(req.query)
    return {"recommended_assessments": results}
