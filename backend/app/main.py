from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .schemas import (RecommendationRequest, RecommendationResponse)
from .services.recommendation_service import RecommendationService


service: RecommendationService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = RecommendationService(Path(__file__).resolve().parents[2])
    yield
    service = None

app = FastAPI(
    title="Recomendador de Jogos Educacionais",
    description="API para recomendar jogos de tabuleiro educacionais com base em critérios específicos.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def root():
    return {"message": "Bem-vindo à API de Recomendação de Jogos Educacionais!",
            "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
@app.post("/recommendations", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Serviço de recomendação não inicializado")
    return {"recomendacoes": service.recommend(request)}