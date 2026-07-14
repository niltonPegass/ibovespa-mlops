"""
src/features.py

Constrói o conjunto de features para prever o retorno diário do Ibovespa
do PRÓXIMO dia útil, a partir do dado bruto (raw).

Regra de ouro deste arquivo (para evitar data leakage):
Toda feature na linha do dia `t` deve usar apenas informação disponível
ATÉ o dia `t` (inclusive). O alvo (target) é o retorno do dia `t+1`,
deslocado para trás (shift(-1)) para ficar alinhado com as features de `t`.
Isso simula fielmente a situação real: no fim do dia `t`, com os dados
que temos até ali, queremos prever o retorno de amanhã.
"""

from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ibovespa_raw.parquet"
PROCESSED_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "ibovespa_features.parquet"


def load_raw_data() -> pd.DataFrame:
    return pd.read_parquet(RAW_DATA_PATH)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe o DataFrame bruto (Open, High, Low, Close, Volume) e retorna
    um DataFrame de features + coluna 'target', pronto para treino.
    """
    out = pd.DataFrame(index=df.index)

    # Retorno diário - base de tudo. shift(1) em relação a ele = passado.
    daily_return = df["Close"].pct_change()

    # --- Lags de retorno: "qual foi o retorno N dias atrás?" ---
    out["return_lag_1"] = daily_return.shift(1)
    out["return_lag_2"] = daily_return.shift(2)
    out["return_lag_5"] = daily_return.shift(5)

    # --- Volatilidade recente (desvio padrão móvel do retorno) ---
    # .shift(1) aqui garante que a janela de cálculo termina ONTEM,
    # nunca inclui o retorno de hoje (que ainda não existe no momento da previsão).
    out["volatility_5d"] = daily_return.shift(1).rolling(window=5).std()
    out["volatility_10d"] = daily_return.shift(1).rolling(window=10).std()
    out["volatility_21d"] = daily_return.shift(1).rolling(window=21).std()

    # --- Momentum (média móvel do retorno) ---
    out["momentum_5d"] = daily_return.shift(1).rolling(window=5).mean()
    out["momentum_10d"] = daily_return.shift(1).rolling(window=10).mean()

    # --- Amplitude do dia anterior (proxy de volatilidade intradiária) ---
    hl_range = (df["High"] - df["Low"]) / df["Close"]
    out["hl_range_lag_1"] = hl_range.shift(1)

    # --- Alvo: retorno de AMANHÃ (t+1), alinhado com as features de hoje (t) ---
    out["target"] = daily_return.shift(-1)

    # Linhas iniciais (janelas incompletas) e a última linha (target inexistente,
    # pois não sabemos o retorno do dia seguinte ao último dado disponível)
    # viram NaN. Removemos.
    out = out.dropna()

    return out


def save_processed(df: pd.DataFrame) -> None:
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DATA_PATH)
    print(f"Features salvas em: {PROCESSED_DATA_PATH}")
    print(f"Formato final: {df.shape[0]} linhas x {df.shape[1]} colunas")


def main():
    df_raw = load_raw_data()
    df_features = build_features(df_raw)
    save_processed(df_features)
    print("\nPrimeiras linhas:")
    print(df_features.head())


if __name__ == "__main__":
    main()
