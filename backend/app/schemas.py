# Modelos Pydantic usados para validar entradas e formatar respostas da API.
from pydantic import BaseModel, Field

class RecommendationRequest(BaseModel):
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
class Recomendation(BaseModel):
    game_id: int | str
    nome: str
    similaridade: float
    rating: float | None = None

# Estrutura da resposta contendo a lista de recomendações.
class RecommendationResponse(BaseModel):
    recomendacoes: list[Recomendation]