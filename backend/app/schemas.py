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
    user_id: str = Field(
        ...,
        description="Identificador do professor usuário"
    )

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