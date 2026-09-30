from fastapi import APIRouter, HTTPException, Request

from ..database import get_avaliacoes

router = APIRouter(prefix="/historico", tags=["historico"])


@router.get("/{user_id}")
def historico(user_id: str, request: Request):
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Serviço não inicializado")

    items: dict[int, dict] = {}

    # 1) Notas do dataset (interactions.csv), já carregadas no recomendador
    dataset_ratings = service.recommenders["item"].get_user_ratings(user_id)
    for game_id, rating in dataset_ratings.items():
        items[int(game_id)] = {
            "game_id": int(game_id),
            "rating": float(rating),
            "origem": "dataset",
            "created_at": None,
        }

    # Notas dadas pelo app (SQLite).
    for avaliacao in get_avaliacoes(user_id):
        items[avaliacao["game_id"]] = {
            "game_id": avaliacao["game_id"],
            "rating": avaliacao["rating"],
            "origem": "app",
            "created_at": avaliacao["created_at"],
        }

    # Mais recentes primeiro depois, as do dataset pela nota
    return sorted(
        items.values(),
        key=lambda item: (item["created_at"] or "", item["rating"]),
        reverse=True,
    )
