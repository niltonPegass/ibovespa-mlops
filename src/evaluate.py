"""
src/evaluate.py

Validação rigorosa dos modelos via Walk-Forward Validation (TimeSeriesSplit):
em vez de um único corte treino/teste, avaliamos os modelos em múltiplos
blocos sequenciais no tempo, com a janela de treino sempre EXPANDINDO
(nunca "voltando no tempo"). Isso dá uma leitura muito mais confiável
sobre a consistência do modelo do que um único split (Fase 5).

Reaproveita as funções de treino já criadas em src/train.py - evita
duplicar lógica que já validamos.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.train import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    train_baseline,
    train_random_forest,
    train_xgboost,
    load_processed_data,
)

N_SPLITS = 5
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def evaluate_fold(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Treina os 3 modelos num fold e retorna as métricas de cada um."""
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    media_treino = train_baseline(train_df)
    y_pred_baseline = np.full(shape=len(y_test), fill_value=media_treino)

    rf_model = train_random_forest(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    xgb_model = train_xgboost(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    resultado = {}
    for nome, y_pred in [
        ("baseline", y_pred_baseline),
        ("random_forest", y_pred_rf),
        ("xgboost", y_pred_xgb),
    ]:
        resultado[f"{nome}_mae"] = mean_absolute_error(y_test, y_pred)
        resultado[f"{nome}_rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))

    return resultado


def run_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa o walk-forward validation com TimeSeriesSplit e retorna um
    DataFrame com as métricas de cada fold.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    resultados = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        print(
            f"\nFold {fold_idx}/{N_SPLITS} | "
            f"Treino: {train_df.index.min().date()} a {train_df.index.max().date()} "
            f"({len(train_df)} linhas) | "
            f"Teste: {test_df.index.min().date()} a {test_df.index.max().date()} "
            f"({len(test_df)} linhas)"
        )

        metrics = evaluate_fold(train_df, test_df)
        metrics["fold"] = fold_idx
        resultados.append(metrics)

        print(
            f"  MAE  -> baseline: {metrics['baseline_mae']:.5f} | "
            f"rf: {metrics['random_forest_mae']:.5f} | "
            f"xgb: {metrics['xgboost_mae']:.5f}"
        )

    return pd.DataFrame(resultados).set_index("fold")


def summarize_results(results_df: pd.DataFrame) -> None:
    """Imprime a média e o desvio padrão de cada métrica entre os folds."""
    print("\n--- Resumo (média ± desvio padrão entre os 5 folds) ---")
    for modelo in ["baseline", "random_forest", "xgboost"]:
        mae_mean = results_df[f"{modelo}_mae"].mean()
        mae_std = results_df[f"{modelo}_mae"].std()
        print(f"{modelo:15s} | MAE médio: {mae_mean:.5f} (+/- {mae_std:.5f})")

    print("\n--- Em quantos dos 5 folds cada modelo superou o baseline (MAE)? ---")
    for modelo in ["random_forest", "xgboost"]:
        venceu = (results_df[f"{modelo}_mae"] < results_df["baseline_mae"]).sum()
        print(f"{modelo:15s} | venceu em {venceu}/{N_SPLITS} folds")


def main():
    df = load_processed_data()
    results_df = run_walk_forward(df)
    summarize_results(results_df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "walk_forward_results.csv"
    results_df.to_csv(output_path)
    print(f"\nResultados detalhados salvos em: {output_path}")


if __name__ == "__main__":
    main()
