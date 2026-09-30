# EduBG RecSys

Trabalho de Oficina de Desenvolvimento de Sistemas (UEA): sistema de recomendação de jogos de tabuleiro educacionais para professores.

O sistema recomenda jogos por:
- **Collaborative Filtering** (Item-Based e User-Based), a partir das avaliações de usuários;
- **Context-Based Filtering**, a partir do contexto da turma (idade, quantidade de alunos, tempo disponível, objetivo pedagógico). Esse modo também é usado no *cold start*, quando o usuário ainda não tem avaliações.

**Tecnologias:** Python, FastAPI (backend), Streamlit (frontend), SQLite (avaliações feitas no app), pandas, scikit-learn e sentence-transformers.

## Estrutura

```
edubg-recsys/
├── backend/
│   ├── app/
│   │   ├── main.py              # API FastAPI
│   │   ├── database.py          # acesso ao SQLite
│   │   ├── schemas.py           # validação (Pydantic)
│   │   ├── recommenders/        # item_based, user_based, context_based
│   │   ├── routers/             # rotas de avaliações
│   │   └── services/            # orquestração e embeddings
│   └── data/                    # dados (fora do Git)
│       ├── BGG_Data_Set.csv
│       ├── user_ratings.csv
│       ├── app.db               # criado automaticamente
│       └── processed/           # gerado pelos scripts
├── frontend/                    # app Streamlit
├── scripts/                     # pipeline offline e avaliação
└── requirements.txt
```

## Pré-requisitos

- Python 3.10 ou superior
- Os datasets em `backend/data/`:
  - `BGG_Data_Set.csv`: catálogo de jogos ([Board Games – Kaggle](https://www.kaggle.com/datasets/melissamonfared/board-games))
  - `user_ratings.csv`: avaliações individuais (colunas `BGGId`, `Rating`, `Username`)

> A pasta `backend/data/` está no `.gitignore` (os arquivos são grandes), então cada pessoa precisa colocar os datasets nela.

Os comandos abaixo são para o **PowerShell (Windows)** e devem ser executados **na pasta `edubg-recsys`**.

## 1. Instalação (uma vez)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- Se a ativação der erro de "execução de scripts desabilitada", rode `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e tente de novo.
- No Linux/macOS, ative com `source .venv/bin/activate`.
- A instalação demora, porque o `sentence-transformers` instala o PyTorch.

> **Sempre ative o ambiente** (`.venv\Scripts\Activate.ps1`) em cada terminal novo. O prompt deve começar com `(.venv)`.

## 2. Gerar os dados processados (uma vez)

```powershell
python scripts/interactions_creator.py
```

Gera `games.csv`, `interactions.csv` e `scaler.pkl` em `backend/data/processed/`. Esse passo processa milhões de avaliações e pode demorar. Só é preciso rodar de novo quando os datasets mudarem.

## 3. (Opcional) Capas reais dos jogos

Sem esta etapa o sistema funciona normalmente, mas com imagens genéricas no lugar das capas.

O script busca offline, na BGG XML API2, as URLs das capas dos jogos mais populares e salva o resultado em `backend/data/processed/game_images.csv`. Ele exige o token de uma aplicação registrada na BGG:

```powershell
$env:BGG_TOKEN = "token-da-aplicacao-registrada"
python scripts/fetch_game_images.py
```

- Coloque **só o token**, sem a palavra `Bearer`: o script já monta o cabeçalho `Authorization: Bearer <token>`.
- **Nunca** coloque o token no código nem faça commit dele.
- Se a execução for interrompida, rode de novo: o script continua de onde parou.

Termos de uso da BGG respeitados por este script:
- Exige o token de uma aplicação registrada e aprovada (uso não comercial).
- Salva somente as URLs; as imagens continuam hospedadas pela BGG (nada é baixado nem redistribuído).
- Espera `PAUSE_SECONDS` entre requisições para não sobrecarregar a API.

## 4. Rodar o backend (terminal 1)

```powershell
.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --reload
```

- A inicialização é lenta: ela carrega as interações e calcula os embeddings do catálogo. Aguarde a mensagem `Application startup complete`.
- Documentação interativa (Swagger): http://127.0.0.1:8000/docs
- Verificação rápida: http://127.0.0.1:8000/health deve responder `{"status": "healthy"}`

Principais rotas:

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status da API |
| `POST` | `/recommend` | Recomendações (modo `collaborative` ou `context`) |
| `POST` | `/avaliacoes` | Salva uma avaliação no SQLite |
| `GET` | `/avaliacoes?user_id=...` | Lista as avaliações salvas |

Exemplo de corpo para `POST /recommend`:

```json
{
  "user_id": "prof_teste",
  "mode": "context",
  "idade_alunos": 12,
  "quantidade_alunos": 4,
  "tempo_disponivel": 50,
  "objetivo_pedagogico": "Desenvolver raciocínio lógico",
  "contexto": "Turma do ensino fundamental",
  "top_k": 5
}
```

## 5. Rodar o frontend (terminal 2)

Com o backend rodando no terminal 1:

```powershell
.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

- Acesse http://localhost:8501
- **Login de demonstração:** o usuário aparece na tela de login; a senha é `12345`.
- O front procura a API em `http://127.0.0.1:8000`. Para usar outro endereço, defina `$env:API_URL` antes de rodar o Streamlit.
- Sem o backend, a vitrine de jogos continua funcionando, mas as recomendações e o envio de avaliações ficam indisponíveis.
- Depois de alterar o CSS ou os dados, **reinicie o Streamlit** (`Ctrl+C` e rode de novo), porque eles ficam em cache.

## 6. Avaliação dos algoritmos

Compara Item-Based e User-Based com Precision@K, Recall@K, Hit Rate@K e tempo de execução:

```powershell
python scripts/evaluate_recsys.py
```

## Problemas comuns

| Sintoma | Solução |
|---|---|
| `streamlit`/`uvicorn` não é reconhecido | Ative o `.venv` |
| `ModuleNotFoundError: No module named 'backend'` | Rode os comandos a partir da pasta `edubg-recsys` |
| `FileNotFoundError: ...processed\games.csv` | Rode `python scripts/interactions_creator.py` |
| `address already in use` na porta 8000 | Já existe um backend rodando; feche-o ou use `--port 8001` (e ajuste `API_URL`) |
| Mudanças no visual não aparecem | Reinicie o Streamlit |

## Créditos

Desenvolvido por **Arthur Marshall** e **Emanuel Ami**.

Dados e imagens dos jogos: [Powered by BoardGameGeek](https://boardgamegeek.com). Projeto acadêmico, sem fins comerciais.
