from fastapi import APIRouter, HTTPException, Request

from ..database import get_avaliacao_by_id, get_avaliacoes, insert_avaliacao
from ..schemas import CreateEvaluation, Evaluation

router = APIRouter(prefix="/avaliacoes", tags=["avaliacoes"])


@router.post("", response_model=Evaluation, status_code=201)
def criar_avaliacao(avaliacao: CreateEvaluation, request: Request):
    # Se o serviço estiver carregado, valida se o jogo existe no catálogo
    service = getattr(request.app.state, "service", None)
    if service is not None and avaliacao.game_id not in service.games_by_id.index:
        raise HTTPException(status_code=404, detail="Jogo não encontrado no catálogo")

    # nomes da API (PT) -> colunas do banco (EN)
    novo_id = insert_avaliacao(
        user_id=avaliacao.user_id,
        game_id=avaliacao.game_id,
        rating=avaliacao.rating,
        student_age=avaliacao.idade_alunos,
        student_count=avaliacao.quantidade_alunos,
        class_duration=avaliacao.tempo_disponivel,
        pedagogical_objective=avaliacao.objetivo_pedagogico,
        context=avaliacao.contexto,
    )
    return get_avaliacao_by_id(novo_id)


@router.get("", response_model=list[Evaluation])
def listar_avaliacoes(user_id: str | None = None):
    return get_avaliacoes(user_id)
