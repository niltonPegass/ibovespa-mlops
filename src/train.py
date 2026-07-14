"""
src/train.py

Treina e compara 3 abordagens para prever o retorno diário do Ibovespa:
1. Baseline: média histórica do retorno (calculada só com o treino)
2. Random Forest
3. XGBoost

O split treino/teste é CRONOLÓGICO (não aleatório): os últimos 20% da
linha do tempo viram teste, o restante vira treino. Isso simula a
situação real - o modelo nunca "vê" dados do futuro durante o treino.

Esta fase entrega uma comparação RÁPIDA entre os modelos, só para termos
uma primeira leitura. A avaliação rigorosa (walk-forward validation,
métricas completas) fica para a Fase 6.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

PROCESSED_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "ibovespa_features.parquet"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

FEATURE_COLUMNS = [
    "return_lag_1", "return_lag_2", "return_lag_5",
    "volatility_5d", "volatility_10d", "volatility_21d",
    "momentum_5d", "momentum_10d",
    "hl_range_lag_1",
]
TARGET_COLUMN = "target"

TEST_SIZE_FRACTION = 0.2


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


def quick_evaluate(y_true, y_pred, name: str) -> None:
    """Calcula e imprime MAE e RMSE. Avaliação completa vem na Fase 6."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:20s} | MAE: {mae:.5f} | RMSE: {rmse:.5f}")


def train_baseline(train_df: pd.DataFrame) -> float:
    """
    Baseline ingênuo: prever sempre a média histórica do retorno,
    calculada SOMENTE com dados de treino (nunca com teste).
    """
    media_treino = train_df[TARGET_COLUMN].mean()
    return media_treino


def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    """
    Random Forest com árvores rasas e quantidade moderada de estimadores -
    dado o sinal fraco identificado na EDA, evitamos um modelo complexo
    demais (que apenas decoraria ruído do treino).
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=20,  # cada folha precisa de pelo menos 20 amostras -> reduz overfitting
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> XGBRegressor:
    """
    XGBoost com profundidade baixa e learning rate conservador,
    pela mesma razão do Random Forest: sinal fraco -> regularização forte.
    """
    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def main():
    df = load_processed_data()
    train_df, test_df = split_train_test(df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    print("\n--- Treinando modelos ---")

    # 1. Baseline
    media_treino = train_baseline(train_df)
    y_pred_baseline = np.full(shape=len(y_test), fill_value=media_treino)

    # 2. Random Forest
    rf_model = train_random_forest(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # 3. XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    print("\n--- Comparação rápida no conjunto de teste ---")
    quick_evaluate(y_test, y_pred_baseline, "Baseline (média)")
    quick_evaluate(y_test, y_pred_rf, "Random Forest")
    quick_evaluate(y_test, y_pred_xgb, "XGBoost")

    # Salvando os modelos treinados (a "média baseline" também, como um dict simples)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.pkl")
    joblib.dump({"media_treino": media_treino}, MODELS_DIR / "baseline.pkl")
    print(f"\nModelos salvos em: {MODELS_DIR}")


if __name__ == "__main__":
    main()
