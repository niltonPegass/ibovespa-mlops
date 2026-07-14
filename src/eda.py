"""
src/eda.py

Análise exploratória focada em série temporal:
- Cálculo do retorno diário (nosso alvo)
- Estatísticas descritivas
- Teste de estacionariedade (ADF)
- Gráficos: preço, retorno, distribuição, ACF/PACF

Esse script não "decide" nada sozinho - ele gera números e gráficos pra
VOCÊ interpretar. Isso é intencional: EDA é uma etapa de investigação
humana, não de automação. As conclusões aqui vão orientar decisões da
Fase 4 (quais features criar) e Fase 5 (qual modelo faz mais sentido).
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ibovespa_raw.parquet"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"


def load_raw_data() -> pd.DataFrame:
    """Carrega o dado bruto salvo na Fase 2."""
    df = pd.read_parquet(RAW_DATA_PATH)
    return df


def compute_daily_return(df: pd.DataFrame) -> pd.Series:
    """
    Calcula o retorno percentual diário a partir do preço de fechamento:
    retorno_t = (Close_t - Close_t-1) / Close_t-1

    O primeiro dia sempre vira NaN (não há "dia anterior" pra comparar),
    então removemos essa linha.
    """
    retorno = df["Close"].pct_change().dropna()
    retorno.name = "daily_return"
    return retorno


def print_descriptive_stats(retorno: pd.Series) -> None:
    """Imprime estatísticas básicas do retorno diário."""
    print("\n--- Estatísticas descritivas do retorno diário ---")
    print(f"Média:        {retorno.mean():.5f}")
    print(f"Desvio padrão:{retorno.std():.5f}")
    print(f"Mínimo:       {retorno.min():.5f}")
    print(f"Máximo:       {retorno.max():.5f}")
    print(f"Assimetria (skew): {retorno.skew():.3f}")
    print(f"Curtose:           {retorno.kurtosis():.3f}")


def test_stationarity(retorno: pd.Series) -> None:
    """
    Teste de Dickey-Fuller Aumentado (ADF).

    Hipótese nula (H0): a série NÃO é estacionária (tem raiz unitária).
    Se o p-valor for menor que 0.05, rejeitamos H0 -> a série É estacionária.
    """
    resultado = adfuller(retorno)
    p_valor = resultado[1]

    print("\n--- Teste de Estacionariedade (ADF) ---")
    print(f"Estatística ADF: {resultado[0]:.4f}")
    print(f"p-valor:         {p_valor:.6f}")
    if p_valor < 0.05:
        print("Conclusão: a série é estacionária (rejeitamos H0).")
    else:
        print("Conclusão: a série NÃO é estacionária (não rejeitamos H0).")


def plot_price_and_returns(df: pd.DataFrame, retorno: pd.Series) -> None:
    """Gera gráfico comparando preço bruto (com tendência) vs retorno (sem tendência)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(df.index, df["Close"])
    axes[0].set_title("Ibovespa - Preço de Fechamento (bruto)")
    axes[0].set_ylabel("Pontos")

    axes[1].plot(retorno.index, retorno.values, linewidth=0.5)
    axes[1].set_title("Ibovespa - Retorno Diário (%)")
    axes[1].set_ylabel("Retorno")
    axes[1].axhline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "price_vs_returns.png", dpi=120)
    plt.close(fig)
    print(f"\nGráfico salvo: {FIGURES_DIR / 'price_vs_returns.png'}")


def plot_return_distribution(retorno: pd.Series) -> None:
    """Histograma da distribuição do retorno diário."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(retorno, bins=80)
    ax.set_title("Distribuição do Retorno Diário")
    ax.set_xlabel("Retorno")
    ax.set_ylabel("Frequência")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "return_distribution.png", dpi=120)
    plt.close(fig)
    print(f"Gráfico salvo: {FIGURES_DIR / 'return_distribution.png'}")


def plot_autocorrelation(retorno: pd.Series) -> None:
    """Gráficos de ACF e PACF - mostram se há 'memória' na série."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    plot_acf(retorno, lags=30, ax=axes[0])
    axes[0].set_title("Autocorrelação (ACF)")

    plot_pacf(retorno, lags=30, ax=axes[1])
    axes[1].set_title("Autocorrelação Parcial (PACF)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "autocorrelation.png", dpi=120)
    plt.close(fig)
    print(f"Gráfico salvo: {FIGURES_DIR / 'autocorrelation.png'}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    retorno = compute_daily_return(df)

    print_descriptive_stats(retorno)
    test_stationarity(retorno)
    plot_price_and_returns(df, retorno)
    plot_return_distribution(retorno)
    plot_autocorrelation(retorno)

    print("\nEDA concluída. Veja os gráficos em reports/figures/")


if __name__ == "__main__":
    main()
