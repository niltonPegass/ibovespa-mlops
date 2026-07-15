"""
api/main.py

API REST que serve previsões de volatilidade do Ibovespa usando o modelo
"champion" registrado no MLflow (Fase 7).

Por que uma pasta `api/` separada do `src/`?
`src/` é o pipeline de TREINO (ingestão, features, treino, avaliação) -
roda periodicamente, offline, e produz um modelo. `api/` é a camada de
SERVIÇO - roda continuamente, online, e apenas CONSOME o modelo já
treinado. São responsabilidades diferentes; separar evita que o serviço
de produção dependa de bibliotecas pesadas que só o treino precisa
(ex: statsmodels, usado só na EDA).

Endpoints:
- GET  /health          -> verifica se a API está no ar
- GET  /model-info      -> qual versão do modelo está carregada
- POST /predict         -> previsão a partir de features já calculadas
- GET  /predict/latest  -> busca os dados mais recentes do Ibovespa,
                           calcula as features e prevê a volatilidade
                           de amanhã, tudo em uma chamada
"""

from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd
import mlflow
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import compute_feature_columns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
REGISTERED_MODEL_NAME = "ibovespa_volatility_rf"
MODEL_ALIAS = "champion"

FEATURE_COLUMNS = [
    "return_lag_1", "return_lag_2", "return_lag_5",
    "abs_return_lag_1", "abs_return_lag_2",
    "volatility_5d", "volatility_10d", "volatility_21d",
    "momentum_5d", "momentum_10d",
    "hl_range_lag_1",
]

TICKER = "^BVSP"

# Dicionário simples que guarda o modelo carregado e seus metadados.
# Carregar o modelo é uma operação relativamente cara (lê do MLflow),
# por isso fazemos isso UMA vez na inicialização da API, não a cada request.
model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo champion na inicialização da API (e só nela)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

    client = mlflow.tracking.MlflowClient()
    model_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)

    model_state["model"] = mlflow.pyfunc.load_model(model_uri)
    model_state["version"] = model_version.version
    model_state["alias"] = MODEL_ALIAS

    print(f"Modelo carregado: {REGISTERED_MODEL_NAME} versão {model_version.version} (alias: {MODEL_ALIAS})")
    yield
    model_state.clear()


app = FastAPI(
    title="Ibovespa Volatility Forecast API",
    description="Previsão da volatilidade (retorno absoluto) do Ibovespa para o próximo dia útil.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionInput(BaseModel):
    """Features já calculadas, na mesma ordem/definição usada no treino."""
    return_lag_1: float
    return_lag_2: float
    return_lag_5: float
    abs_return_lag_1: float = Field(ge=0)
    abs_return_lag_2: float = Field(ge=0)
    volatility_5d: float = Field(ge=0)
    volatility_10d: float = Field(ge=0)
    volatility_21d: float = Field(ge=0)
    momentum_5d: float
    momentum_10d: float
    hl_range_lag_1: float = Field(ge=0)


class PredictionOutput(BaseModel):
    predicted_volatility: float
    model_name: str
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_name": REGISTERED_MODEL_NAME,
        "version": model_state.get("version"),
        "alias": model_state.get("alias"),
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    """Recebe features já calculadas e retorna a previsão de volatilidade."""
    input_df = pd.DataFrame([payload.model_dump()])[FEATURE_COLUMNS]
    prediction = model_state["model"].predict(input_df)[0]

    return PredictionOutput(
        predicted_volatility=float(prediction),
        model_name=REGISTERED_MODEL_NAME,
        model_version=str(model_state.get("version")),
    )


@app.get("/predict/latest", response_model=PredictionOutput)
def predict_latest():
    """
    Busca os dados mais recentes do Ibovespa via yfinance, calcula as
    features com a MESMA função usada no treino (src.features), e prevê
    a volatilidade do próximo dia útil - sem o usuário precisar calcular
    nada manualmente.
    """
    df = yf.download(TICKER, period="3mo", interval="1d", auto_adjust=False, progress=False)

    if df.empty:
        raise HTTPException(status_code=503, detail="Não foi possível obter dados do Ibovespa no momento.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    features_df = compute_feature_columns(df)
    latest_features = features_df.iloc[[-1]][FEATURE_COLUMNS]

    if latest_features.isnull().any(axis=None):
        raise HTTPException(
            status_code=503,
            detail="Dados insuficientes para calcular todas as features (histórico muito curto).",
        )

    prediction = model_state["model"].predict(latest_features)[0]

    return PredictionOutput(
        predicted_volatility=float(prediction),
        model_name=REGISTERED_MODEL_NAME,
        model_version=str(model_state.get("version")),
    )
