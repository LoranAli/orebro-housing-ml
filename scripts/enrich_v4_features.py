"""
A2 + A3: Mäklarfirma & Riksbankränta — Örebro Housing ML
==========================================================
Lägger till:

  A2. maklare_te   — Target-encodad mäklarfirma (extraheras ur raw_text).
      Nordå hanterar premiumvillor (5.1M median), Fastighetsbyrån budget (3.1M).
      Hjälper modellen förstå mäklar-segmentering utan att koda kategorier hårt.

  A3. riksbank_rate         — Riksbankens reporänta vid försäljningstillfället (%)
      rate_price_interact   — rate × log(boarea): stordrift mer räntekänsligt
      rate_change_6m        — räntedelta senaste 6 månader (stigande/fallande)

Körs mot orebro_housing_enriched_v4.csv (efter geocodning).
Sparar uppdaterat dataset i samma fil.

Kör:
    python scripts/enrich_v4_features.py
"""

import os
import sys
import logging
import re
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(PROJECT_DIR, 'data', 'processed', 'orebro_housing_enriched_v4.csv')
LOG_PATH    = os.path.join(PROJECT_DIR, 'logs', 'enrich_v4_features.log')

os.makedirs(os.path.join(PROJECT_DIR, 'logs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-5s %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(LOG_PATH, encoding='utf-8'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# A2: MÄKLARFIRMA
# ─────────────────────────────────────────────────────────────
MAKLARE_MAP = {
    'Bjurfors':           'Bjurfors',
    'Mäklarhuset':        'Maklarhuset',
    'LF Fastighets':      'LF_Fastighets',
    'Fastighetsbyrån':    'Fastighetsbyran',
    'Svensk Fastighets':  'Svensk_Fastighets',
    'Nordå':              'Norda',
    'Notar':              'Notar',
    'ERA ':               'ERA',
}

def extract_maklare(text: str) -> str:
    if not isinstance(text, str):
        return 'Ovrigt'
    for key, norm in MAKLARE_MAP.items():
        if key in text:
            return norm
    return 'Ovrigt'


def add_maklare(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Extraherar mäklare ur raw_text och target-encodar på träningsdata."""
    df = df.copy()
    df['maklare'] = df['raw_text'].apply(extract_maklare)

    # Target encoding: smoothed mean(slutpris) per mäklare
    SMOOTHING = 20
    global_mean = df.loc[train_mask, 'slutpris'].mean()
    stats = (
        df[train_mask]
        .groupby('maklare')['slutpris']
        .agg(['mean', 'count'])
    )
    smooth = stats['count'] / (stats['count'] + SMOOTHING)
    stats['te'] = smooth * stats['mean'] + (1 - smooth) * global_mean
    te_map = stats['te'].to_dict()

    df['maklare_te'] = df['maklare'].map(te_map).fillna(global_mean)
    log.info(f'  maklare distribution: {df.loc[train_mask, "maklare"].value_counts().to_dict()}')
    log.info(f'  maklare_te range: {df["maklare_te"].min():,.0f} – {df["maklare_te"].max():,.0f}')
    return df


# ─────────────────────────────────────────────────────────────
# A3: RIKSBANKRÄNTA
# ─────────────────────────────────────────────────────────────
# Riksbankens reporänta per månad (källa: riksbank.se)
RIKSBANK_RATE = {
    '2019-01': 0.0,  '2019-02': 0.0,  '2019-03': 0.0,  '2019-04': 0.0,
    '2019-05': 0.0,  '2019-06': 0.0,  '2019-07': 0.0,  '2019-08': 0.0,
    '2019-09': 0.0,  '2019-10': 0.0,  '2019-11': 0.0,  '2019-12': -0.25,
    '2020-01': -0.25,'2020-02': -0.25,'2020-03': -0.25,'2020-04': -0.25,
    '2020-05': -0.25,'2020-06': -0.25,'2020-07': -0.25,'2020-08': -0.25,
    '2020-09': -0.25,'2020-10': -0.25,'2020-11': -0.25,'2020-12': -0.25,
    '2021-01': 0.0,  '2021-02': 0.0,  '2021-03': 0.0,  '2021-04': 0.0,
    '2021-05': 0.0,  '2021-06': 0.0,  '2021-07': 0.0,  '2021-08': 0.0,
    '2021-09': 0.0,  '2021-10': 0.0,  '2021-11': 0.0,  '2021-12': 0.0,
    '2022-01': 0.0,  '2022-02': 0.0,  '2022-03': 0.0,  '2022-04': 0.25,
    '2022-05': 0.25, '2022-06': 0.75, '2022-07': 0.75, '2022-08': 0.75,
    '2022-09': 1.75, '2022-10': 1.75, '2022-11': 2.5,  '2022-12': 2.5,
    '2023-01': 2.5,  '2023-02': 3.0,  '2023-03': 3.0,  '2023-04': 3.5,
    '2023-05': 3.5,  '2023-06': 3.75, '2023-07': 3.75, '2023-08': 4.0,
    '2023-09': 4.0,  '2023-10': 4.0,  '2023-11': 4.0,  '2023-12': 4.0,
    '2024-01': 4.0,  '2024-02': 4.0,  '2024-03': 4.0,  '2024-04': 4.0,
    '2024-05': 4.0,  '2024-06': 3.75, '2024-07': 3.75, '2024-08': 3.5,
    '2024-09': 3.25, '2024-10': 3.25, '2024-11': 2.75, '2024-12': 2.5,
    '2025-01': 2.5,  '2025-02': 2.5,  '2025-03': 2.25, '2025-04': 2.25,
    '2025-05': 2.25, '2025-06': 2.25, '2025-07': 2.0,  '2025-08': 2.0,
    '2025-09': 2.0,  '2025-10': 2.0,  '2025-11': 2.0,  '2025-12': 2.0,
    '2026-01': 2.0,  '2026-02': 2.0,  '2026-03': 2.0,  '2026-04': 2.0,
}

def add_riksbank(df: pd.DataFrame) -> pd.DataFrame:
    """Lägger till riksbank_rate, rate_change_6m, rate_price_interact."""
    df = df.copy()
    df['sald_datum'] = pd.to_datetime(df['sald_datum'], errors='coerce')
    df['_period'] = df['sald_datum'].dt.strftime('%Y-%m')

    df['riksbank_rate'] = df['_period'].map(RIKSBANK_RATE).fillna(2.0)

    # Räntedelta senaste 6 månader (stigande = positivt = dyrare bolån framöver)
    rate_series = pd.Series(RIKSBANK_RATE)
    rate_series.index = pd.to_datetime(rate_series.index + '-01')
    rate_series = rate_series.sort_index()

    def _rate_change_6m(period_str):
        try:
            d     = pd.to_datetime(period_str + '-01')
            d_6m  = d - pd.DateOffset(months=6)
            now   = RIKSBANK_RATE.get(d.strftime('%Y-%m'), 2.0)
            then  = RIKSBANK_RATE.get(d_6m.strftime('%Y-%m'), 2.0)
            return round(now - then, 2)
        except Exception:
            return 0.0

    df['rate_change_6m'] = df['_period'].apply(_rate_change_6m)

    # Interaktion: ränta × log(boarea) — stor villa, dyrare bolån → mer räntekänslig
    log_boarea = np.log(df['boarea_kvm'].clip(lower=10))
    df['rate_boarea_interact'] = df['riksbank_rate'] * log_boarea

    df.drop(columns=['_period'], inplace=True)

    log.info(f'  riksbank_rate: {df["riksbank_rate"].min():.2f}% – {df["riksbank_rate"].max():.2f}%')
    log.info(f'  rate_change_6m unique: {sorted(df["rate_change_6m"].unique())}')
    log.info(f'  Andel med positiv rate_change (räntehöjning): '
             f'{(df["rate_change_6m"] > 0).mean()*100:.0f}%')
    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    log.info('=' * 60)
    log.info('BERIKA V4: MÄKLARFIRMA (A2) + RIKSBANKRÄNTA (A3)')
    log.info('=' * 60)

    df = pd.read_csv(DATA_PATH)
    df['sald_datum'] = pd.to_datetime(df['sald_datum'], errors='coerce')
    log.info(f'Laddat: {len(df)} rader, {df.shape[1]} kolumner')

    train_mask = df['sald_datum'] <= pd.Timestamp('2024-12-31')
    log.info(f'Träningsrader för TE-kalibrering: {train_mask.sum()}')

    # A2: Mäklare
    log.info('\n--- A2: Mäklarfirma ---')
    df = add_maklare(df, train_mask)

    # A3: Riksbank
    log.info('\n--- A3: Riksbankränta ---')
    df = add_riksbank(df)

    # Spara
    df.to_csv(DATA_PATH, index=False, encoding='utf-8-sig')
    log.info(f'\n✅ Sparad: {DATA_PATH}')
    log.info(f'   Rader: {len(df)}, Kolumner: {len(df.columns)}')
    log.info('Nya kolumner: maklare, maklare_te, riksbank_rate, rate_change_6m, rate_boarea_interact')
    log.info('\nNästa steg: python scripts/train_villa_v9.py')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
