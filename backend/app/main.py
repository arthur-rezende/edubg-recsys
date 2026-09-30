from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .schemas import (RecommendationRequest, RecommendationResponse)
from .services.recommendation_service import RecommendationService
from .database import init_db
from .routers.evaluation import router as evaluation_router
from .routers.history import router as history_router


service: RecommendationService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    init_db()
    service = RecommendationService(Path(__file__).resolve().parents[1])
    app.state.service = service
    yield
    service = None

app = FastAPI(
    title="Recomendador de Jogos Educacionais",
    description="API para recomendar jogos de tabuleiro educacionais com base em critérios específicos.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(evaluation_router)
app.include_router(history_router)

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

@app.get("/home-recommendations")
def home_recommendations(user_id: str, top_k: int = 15):
    if service is None:
        raise HTTPException(status_code=503, detail="Serviço de recomendação não inicializado")
    return {"sections": service.recommend_home(user_id, top_k)}
