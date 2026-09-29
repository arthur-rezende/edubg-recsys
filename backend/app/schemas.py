# Modelos Pydantic usados para validar entradas e formatar respostas da API.
from typing import Literal

from pydantic import BaseModel, Field

class RecommendationRequest(BaseModel):
    user_id: str | None = Field(
        default=None,
        min_length=1,
        description="Identificador do professor/usuário, quando houver histórico"
    )

    mode: Literal["collaborative", "context"] = Field(
        default="collaborative",
        description="Modo de recomendação"
    )

    algorithm: Literal["item", "user"] = Field(
        default="item",
        description="Algoritmo colaborativo usado no modo collaborative"
    )

    # Dados numéricos da turma e da atividade, validados como positivos.
    idade_alunos: int = Field(
        ...,
        gt=0,
        description="Idade dos alunos"
    )

    quantidade_alunos: int = Field(
        ...,
        gt=0,
        description="Quantidade de alunos"
    )

    tempo_disponivel: int = Field(
        ...,
        gt=0,
        description="Tempo disponível para a atividade em minutos"
    )

    # Informações textuais usadas para representar o objetivo e o contexto.
    objetivo_pedagogico: str = Field(
        ...,
        min_length=3,
        description="Objetivo pedagógico da atividade"
    )   

    contexto: str = Field(
        ...,
        min_length=3,
        description="Contexto da atividade em que o jogo será utilizado"
    )

    top_k: int = Field(
        default=5,
        gt=0,
        le=20,
        description="Quantidade de recomendações"
    )

# Estrutura de uma recomendação individual retornada pela API.
class Recommendation(BaseModel):
    game_id: int | str
    nome: str
    score: float
    rating: float | None = None

# Estrutura da resposta contendo a lista de recomendações.
class RecommendationResponse(BaseModel):
    recomendacoes: list[Recommendation]

class CreateEvaluation(BaseModel):
    user_id: str = Field(..., min_length=1, description="Identificador do professor")
    game_id: int = Field(..., gt=0, description="ID do jogo no BGG")
    rating: float = Field(..., ge=0, le=10, description="Nota de 0 a 10")

    idade_alunos: int | None = Field(default=None, gt=0)
    quantidade_alunos: int | None = Field(default=None, gt=0)
    tempo_disponivel: int | None = Field(default=None, gt=0)
    objetivo_pedagogico: str | None = None
    contexto: str | None = None

class Evaluation(BaseModel):
    id: int
    user_id: str
    game_id: int
    rating: float
    student_age: int | None
    student_count: int | None
    class_duration: int | None
    pedagogical_objective: str | None
    context: str | None
    created_at: str