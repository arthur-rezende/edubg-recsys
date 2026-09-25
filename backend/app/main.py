from fastapi import FastAPI

from .schemas import (RecommendationRequest, RecommendationResponse)

app = FastAPI(
    title="Recomendador de Jogos Educacionais",
    description="API para recomendar jogos de tabuleiro educacionais com base em critérios específicos.",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Bem-vindo à API de Recomendação de Jogos Educacionais!",
            "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/recommendations", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest): return {"recomendacoes": []}