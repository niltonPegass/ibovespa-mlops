"""
src/train.py

Treina e compara 3 abordagens para prever a VOLATILIDADE (retorno absoluto)
do Ibovespa no próximo dia útil:
1. Baseline: volatilidade média histórica (calculada só com o treino)
2. Random Forest
3. XGBoost

O split treino/teste é CRONOLÓGICO (não aleatório): os últimos 20% da
linha do tempo viram teste, o restante vira treino.

A partir da Fase 7, cada modelo treinado aqui é registrado no MLflow:
- Tracking: parâmetros, métricas e o modelo em si viram um "run" registrado
  (em vez de só aparecerem no terminal e serem esquecidos).
- Registry: o modelo vencedor da validação walk-forward (Fase 6 apontou
  Random Forest como o mais consistente) é promovido a "champion" -
  a versão oficialmente candidata à produção.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ibovespa_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = [
    "return_lag_1", "return_lag_2", "return_lag_5",
    "abs_return_lag_1", "abs_return_lag_2",
    "volatility_5d", "volatility_10d", "volatility_21d",
    "momentum_5d", "momentum_10d",
    "hl_range_lag_1",
]
TARGET_COLUMN = "target_volatility"

TEST_SIZE_FRACTION = 0.2

# --- Configuração do MLflow ---
# Usamos SQLite como backend de tracking (arquivo local mlflow.db na raiz
# do projeto) - evita o banner "Demo" que aparece quando não se especifica
# um backend de verdade, e não depende de nenhum servidor externo rodando.
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "ibovespa_volatility_forecast"
REGISTERED_MODEL_NAME = "ibovespa_volatility_rf"


def load_processed_data() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DATA_PATH)


def split_train_test(df: pd.DataFrame, test_size: float = TEST_SIZE_FRACTION):
    """
    Divide o DataFrame cronologicamente: as primeiras (1 - test_size)
    linhas viram treino, as últimas test_size viram teste.
    """
    corte = int(len(df) * (1 - test_size))
    train_df = df.iloc[:corte]
    test_df = df.iloc[corte:]

    print(f"Treino: {len(train_df)} linhas ({train_df.index.min().date()} a {train_df.index.max().date()})")
    print(f"Teste:  {len(test_df)} linhas ({test_df.index.min().date()} a {test_df.index.max().date()})")

    return train_df, test_df


def compute_metrics(y_true, y_pred) -> dict:
    """Calcula MAE e RMSE e retorna como dicionário (facilita logar no MLflow)."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def train_baseline(train_df: pd.DataFrame) -> float:
    """
    Baseline ingênuo: prever sempre a volatilidade média histórica
    (retorno absoluto médio), calculada SOMENTE com dados de treino.
    """
    media_treino = train_df[TARGET_COLUMN].mean()
    return media_treino


def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    """
    Random Forest com árvores rasas e quantidade moderada de estimadores -
    dado o sinal fraco identificado na EDA, evitamos um modelo complexo
    demais (que apenas decoraria ruído do treino).
    """
    params = dict(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    model._logged_params = params  # guardamos os params pra facilitar o log no MLflow
    return model


def train_xgboost(X_train, y_train) -> XGBRegressor:
    """
    XGBoost com profundidade baixa e learning rate conservador,
    pela mesma razão do Random Forest: sinal fraco -> regularização forte.
    """
    params = dict(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    model._logged_params = params
    return model


def log_run(run_name: str, params: dict, metrics: dict, model=None, log_fn=None):
    """
    Registra um run no MLflow: parâmetros, métricas e (opcionalmente) o
    modelo treinado como artefato. Retorna o model_uri (quando há modelo
    logado), necessário para registrar o modelo depois no Model Registry.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if model is not None and log_fn is not None:
            model_info = log_fn(model, name="model")
            return model_info.model_uri
        return None


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_processed_data()
    train_df, test_df = split_train_test(df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    print("\n--- Treinando modelos ---")

    # 1. Baseline (sem artefato de modelo - é só uma constante)
    media_treino = train_baseline(train_df)
    y_pred_baseline = np.full(shape=len(y_test), fill_value=media_treino)
    metrics_baseline = compute_metrics(y_test, y_pred_baseline)
    log_run("baseline", {"estrategia": "media_historica"}, metrics_baseline)
    print(f"Baseline        | MAE: {metrics_baseline['mae']:.5f} | RMSE: {metrics_baseline['rmse']:.5f}")

    # 2. Random Forest
    rf_model = train_random_forest(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    metrics_rf = compute_metrics(y_test, y_pred_rf)
    rf_run_model_uri = log_run("random_forest", rf_model._logged_params, metrics_rf, rf_model, mlflow.sklearn.log_model)
    print(f"Random Forest   | MAE: {metrics_rf['mae']:.5f} | RMSE: {metrics_rf['rmse']:.5f}")

    # 3. XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    metrics_xgb = compute_metrics(y_test, y_pred_xgb)
    log_run("xgboost", xgb_model._logged_params, metrics_xgb, xgb_model, mlflow.xgboost.log_model)
    print(f"XGBoost         | MAE: {metrics_xgb['mae']:.5f} | RMSE: {metrics_xgb['rmse']:.5f}")

    # --- Promovendo o Random Forest a "champion" no Model Registry ---
    # A Fase 6 (walk-forward) mostrou que o Random Forest é o mais
    # consistente entre os folds - por isso é ele o candidato oficial
    # à produção, não necessariamente o de menor erro num único split.
    model_uri = rf_run_model_uri
    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", registered.version)
    print(f"\nModelo '{REGISTERED_MODEL_NAME}' versão {registered.version} promovido a 'champion' no MLflow Registry.")

    # Mantemos também os arquivos .pkl locais - simples e portáveis,
    # úteis caso a gente precise de acesso rápido sem depender do MLflow.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.pkl")
    joblib.dump({"media_treino": media_treino}, MODELS_DIR / "baseline.pkl")
    print(f"Modelos também salvos localmente em: {MODELS_DIR}")


if __name__ == "__main__":
    main()
