"""
src/ingest.py

Responsável por UMA coisa: baixar o histórico de preços do Ibovespa via
yfinance e salvar como dado bruto (raw) em formato Parquet.

Por que esse arquivo não faz mais nada além disso?
Em MLOps, cada etapa do pipeline deve ter uma responsabilidade clara.
Isso facilita testar, debugar e reaproveitar cada parte separadamente.

Sobre o retry:
O Yahoo Finance não tem API oficial - o yfinance faz engenharia reversa
do site, e o Yahoo por vezes bloqueia temporariamente esse acesso
("YFRateLimitError: Too Many Requests"). Esse bloqueio costuma durar
de minutos a algumas horas. Por isso, em vez de tentar uma segunda fonte
de dados (o que adiciona complexidade sem garantia de funcionar), fazemos
algumas tentativas com espera crescente entre elas (backoff exponencial):
se a 1ª falhar, espera 10s e tenta de novo; se falhar de novo, espera 20s;
e assim por diante.
"""

from pathlib import Path
import time

import pandas as pd
import yfinance as yf

TICKER = "^BVSP"
RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ibovespa_raw.parquet"

MAX_TENTATIVAS = 3
ESPERA_INICIAL_SEGUNDOS = 10


def _tentar_download(period_years: int) -> pd.DataFrame:
    """Uma única tentativa de download via yfinance."""
    df = yf.download(
        TICKER,
        period=f"{period_years}y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError("yfinance retornou um DataFrame vazio.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "date"
    return df


def fetch_ibovespa(period_years: int = 10) -> pd.DataFrame:
    """
    Baixa o histórico diário do Ibovespa dos últimos `period_years` anos,
    com retry e espera progressiva em caso de falha (ex: rate limit do Yahoo).
    """
    print(f"Baixando histórico do Ibovespa ({TICKER}) - últimos {period_years} anos...")

    espera = ESPERA_INICIAL_SEGUNDOS
    ultimo_erro = None

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            df = _tentar_download(period_years)
            print(f"Sucesso na tentativa {tentativa}: {len(df)} linhas.")
            print(f"Período obtido: {df.index.min().date()} até {df.index.max().date()}")
            return df
        except Exception as e:
            ultimo_erro = e
            print(f"Tentativa {tentativa}/{MAX_TENTATIVAS} falhou: {e}")
            if tentativa < MAX_TENTATIVAS:
                print(f"Aguardando {espera}s antes de tentar novamente...")
                time.sleep(espera)
                espera *= 2  # backoff exponencial

    raise RuntimeError(
        f"Não foi possível baixar os dados após {MAX_TENTATIVAS} tentativas. "
        f"Último erro: {ultimo_erro}"
    )


def save_raw(df: pd.DataFrame) -> None:
    """Salva o DataFrame bruto em Parquet, criando a pasta de destino se necessário."""
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_DATA_PATH)
    print(f"Dado bruto salvo em: {RAW_DATA_PATH}")


def main():
    df = fetch_ibovespa(period_years=10)
    save_raw(df)


if __name__ == "__main__":
    main()
