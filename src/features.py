"""
src/features.py

Constrói o conjunto de features para prever a VOLATILIDADE do Ibovespa
no PRÓXIMO dia útil, a partir do dado bruto (raw).

Por que volatilidade e não o retorno em si?
A EDA (Fase 3) e um primeiro teste de modelagem (Fase 5) mostraram que o
retorno diário do Ibovespa tem sinal muito fraco (mercado eficiente:
autocorrelação quase nula). Porém, a própria EDA revelou "clusters de
volatilidade" - dias turbulentos tendem a ser seguidos por outros dias
turbulentos. Esse padrão É previsível e bem documentado na literatura
financeira (é o que motiva toda a família de modelos GARCH). Por isso,
pivotamos o alvo para prever o TAMANHO do movimento de amanhã
(|retorno_{t+1}|), em vez da sua direção.

Regra de ouro deste arquivo (para evitar data leakage):
Toda feature na linha do dia `t` deve usar apenas informação disponível
ATÉ o dia `t` (inclusive). O alvo é o retorno absoluto do dia `t+1`,
deslocado para trás (shift(-1)) para ficar alinhado com as features de `t`.
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
    um DataFrame de features + coluna 'target_volatility', pronto para treino.
    """
    out = pd.DataFrame(index=df.index)

    daily_return = df["Close"].pct_change()
    abs_return = daily_return.abs()

    # --- Lags de retorno (mantidos - ainda podem carregar algum sinal de direção) ---
    out["return_lag_1"] = daily_return.shift(1)
    out["return_lag_2"] = daily_return.shift(2)
    out["return_lag_5"] = daily_return.shift(5)

    # --- Lags de retorno ABSOLUTO: o ingrediente central pra prever volatilidade ---
    # Se ontem foi um dia de movimento forte (positivo ou negativo), isso é
    # um bom preditor de que amanhã também pode ser volátil.
    out["abs_return_lag_1"] = abs_return.shift(1)
    out["abs_return_lag_2"] = abs_return.shift(2)

    # --- Volatilidade recente (desvio padrão móvel do retorno) ---
    out["volatility_5d"] = daily_return.shift(1).rolling(window=5).std()
    out["volatility_10d"] = daily_return.shift(1).rolling(window=10).std()
    out["volatility_21d"] = daily_return.shift(1).rolling(window=21).std()

    # --- Momentum (média móvel do retorno) - mantido, útil de forma secundária ---
    out["momentum_5d"] = daily_return.shift(1).rolling(window=5).mean()
    out["momentum_10d"] = daily_return.shift(1).rolling(window=10).mean()

    # --- Amplitude do dia anterior (proxy de volatilidade intradiária) ---
    hl_range = (df["High"] - df["Low"]) / df["Close"]
    out["hl_range_lag_1"] = hl_range.shift(1)

    # --- Alvo: retorno ABSOLUTO de amanhã (t+1), alinhado com as features de hoje (t) ---
    out["target_volatility"] = abs_return.shift(-1)

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
