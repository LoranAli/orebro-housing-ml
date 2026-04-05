"""
Energiklass-berikning för Örebro Housing ML
=============================================
Steg 1: Re-scrapar aktiva villa-annonser (hemnet.se/bostad/...) för att
         hämta energiklass + uppvarmning — dessa finns INTE på sålda sidor.
Steg 2: Bygger ek_proxy (A-G-skala) från byggår för historisk träningsdata.
Steg 3: Mergear båda till orebro_housing_enriched_v3.csv.

Kör:
    python scripts/enrich_energiklass.py               # Alla aktiva villor
    python scripts/enrich_energiklass.py --limit 10    # Snabbtest
    python scripts/enrich_energiklass.py --merge-only  # Bara merge, ingen scraping
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, 'scripts'))

ENRICHED_V2   = os.path.join(PROJECT_DIR, 'data', 'processed', 'orebro_housing_enriched_v2.csv')
ACTIVE_CSV    = os.path.join(PROJECT_DIR, 'data', 'processed', 'active_listings_scored.csv')
CACHE_PATH    = os.path.join(PROJECT_DIR, 'data', 'raw', 'energiklass_cache.json')
OUTPUT_PATH   = os.path.join(PROJECT_DIR, 'data', 'processed', 'orebro_housing_enriched_v3.csv')
LOG_PATH      = os.path.join(PROJECT_DIR, 'logs', 'enrich_energiklass.log')

os.makedirs(os.path.join(PROJECT_DIR, 'logs'), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-5s %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# ENERGIKLASS-PROXY (byggt på svenska BBR-normer)
# ─────────────────────────────────────────────────────────
# Mappar byggår till förväntad energiklass (numerisk 1=G ... 7=A)
# Baserat på historiska krav: SBN-67, SBN-75, NR-88, BBR-94, BBR-06, BBR-09, BBR-21
EK_BYGGAR_MAP = [
    (2020, 7),   # A — nära-nollenergibyggnader (BBR 2020+)
    (2010, 6),   # B — BBR 2009, krav ~75 kWh/m²
    (2000, 5),   # C — BBR 1994/2006, krav ~110 kWh/m²
    (1985, 4),   # D — NR 1988, krav ~150 kWh/m²
    (1975, 3),   # E — SBN-75, krav ~200 kWh/m²
    (1961, 2),   # F — SBN-67, minimikrav
    (0,    1),   # G — äldre, inga energikrav
]

EK_TO_NUM = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
NUM_TO_EK = {v: k for k, v in EK_TO_NUM.items()}


def byggar_to_ek_proxy(year):
    """Konverterar byggår till energiklass-proxy (1=G..7=A) baserat på BBR-normer."""
    if pd.isna(year) or year <= 0:
        return 3  # E som default (median för äldre svenska villor)
    y = int(year)
    for threshold, score in EK_BYGGAR_MAP:
        if y >= threshold:
            return score
    return 1


# ─────────────────────────────────────────────────────────
# UPPVARMNING → SCORE
# ─────────────────────────────────────────────────────────
UPPVARMNING_SCORE = {
    'bergvärme': 3, 'värmepump': 2, 'fjärrvärme': 2,
    'geotermisk': 3, 'jordvärme': 3, 'luftvärmepump': 2,
    'gas': 0, 'pellets': 1, 'ved': 0, 'biobränsle': 1,
    'direktel': -2, 'elpanna': -2, 'elradiatorer': -2,
    'elvärme': -2, 'oljevärme': -3, 'olja': -3,
}


def uppvarmning_to_score(text):
    if not isinstance(text, str) or not text.strip():
        return 0
    t = text.lower()
    for key, score in sorted(UPPVARMNING_SCORE.items(), key=lambda x: -abs(x[1])):
        if key in t:
            return score
    return 0


# ─────────────────────────────────────────────────────────
# STEG 1: SCRAPA AKTIVA ANNONSER
# ─────────────────────────────────────────────────────────
def scrape_active_listings(urls, cache, limit=None):
    """Scrapar aktiva villa-sidor för energiklass + uppvarmning."""
    from daily_update import parse_detail_page, create_driver

    if limit:
        urls = urls[:limit]

    need = [u for u in urls if u not in cache or cache[u].get('error')]
    log.info(f'Totalt: {len(urls)} | Cachade: {len(urls) - len(need)} | Behöver scrapa: {len(need)}')

    if not need:
        log.info('Allt redan cachat!')
        return cache

    driver = create_driver()
    errors = 0
    checkpoint_every = 25

    try:
        for i, url in enumerate(need, 1):
            try:
                driver.get(url)
                time.sleep(random.uniform(2.5, 3.5))
                details = parse_detail_page(driver.page_source)
                details['url'] = url
                details['scraped_at'] = datetime.now().isoformat()
                cache[url] = details

                ek = details.get('energiklass', '-')
                up = details.get('uppvarmning', '-')
                log.info(f'  [{i}/{len(need)}] OK | ek={ek} upp={up} | {url[-60:]}')

            except Exception as e:
                errors += 1
                cache[url] = {'url': url, 'error': str(e)[:120]}
                log.warning(f'  [{i}/{len(need)}] FEL: {e}')

            if i % checkpoint_every == 0:
                _save_cache(cache)
                ek_count = sum(1 for d in cache.values() if d.get('energiklass'))
                up_count = sum(1 for d in cache.values() if d.get('uppvarmning'))
                log.info(f'  Checkpoint: ek={ek_count}, uppvarmning={up_count}, fel={errors}')

            if i % 150 == 0:
                driver.quit()
                time.sleep(random.uniform(5, 10))
                driver = create_driver()

    finally:
        driver.quit()
        _save_cache(cache)
        log.info(f'Scraping klar: {len(need) - errors} OK, {errors} fel')

    return cache


def _save_cache(cache):
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(list(cache.values()), f, ensure_ascii=False)


# ─────────────────────────────────────────────────────────
# STEG 2: BYGG ek_proxy FÖR ALLA VILLOR
# ─────────────────────────────────────────────────────────
def add_ek_proxy(df):
    """Lägger till ek_proxy och uppvarmning_score_v2 baserat på byggår."""
    df = df.copy()
    byggar = pd.to_numeric(df.get('byggar', pd.Series(dtype=float)), errors='coerce')
    df['ek_proxy'] = byggar.apply(byggar_to_ek_proxy)
    df['ek_proxy_letter'] = df['ek_proxy'].map(NUM_TO_EK)
    return df


# ─────────────────────────────────────────────────────────
# STEG 3: MERGE
# ─────────────────────────────────────────────────────────
def merge_energiklass(df, cache):
    """
    Slår ihop scrapad energiklass (från cache) med befintlig enriched CSV.
    Lägger till:
      - ek_proxy          : energiklass-proxy från byggår (alltid tillgänglig)
      - energiklass        : riktig energiklass från Hemnet (aktiva sidor)
      - energiklass_num    : numerisk (1=G..7=A), riktigt om tillgänglig annars proxy
      - uppvarmning        : uppvarmningstyp (aktiva sidor)
      - uppvarmning_score  : numerisk score
    """
    log.info('Slår ihop energiklass-data...')

    # Bygg energiklass från cache
    cache_rows = [d for d in cache.values() if d and not d.get('error') and d.get('url')]
    if cache_rows:
        cache_df = pd.DataFrame(cache_rows)[['url', 'energiklass', 'uppvarmning']].copy()
        cache_df = cache_df.drop_duplicates(subset='url')
        log.info(f'  Cache: {len(cache_df)} poster | '
                 f'ek: {cache_df["energiklass"].notna().sum()} | '
                 f'uppv: {cache_df["uppvarmning"].notna().sum()}')

        # Merge med befintlig df
        df = df.merge(cache_df, on='url', how='left',
                      suffixes=('', '_new'))
        # Kombinera nya och gamla kolumner
        for col in ['energiklass', 'uppvarmning']:
            new_col = col + '_new'
            if new_col in df.columns:
                df[col] = df[new_col].where(df[new_col].notna(), df.get(col))
                df.drop(columns=[new_col], inplace=True)
    else:
        log.warning('Cache tom — lägger bara till ek_proxy')
        for col in ['energiklass', 'uppvarmning']:
            if col not in df.columns:
                df[col] = None

    # ek_proxy (alltid från byggår)
    df = add_ek_proxy(df)

    # energiklass_num: riktig energiklass om tillgänglig, annars proxy
    ek_num = df['energiklass'].map(EK_TO_NUM)
    df['energiklass_num'] = ek_num.where(ek_num.notna(), df['ek_proxy'])
    df['energiklass_kalla'] = 'proxy'
    df.loc[ek_num.notna(), 'energiklass_kalla'] = 'hemnet'

    # uppvarmning_score
    df['uppvarmning_score'] = df['uppvarmning'].apply(uppvarmning_to_score)

    # Rapport
    villor = df[df['bostadstyp'] == 'villor'] if 'bostadstyp' in df.columns else df
    log.info(f'\n  Rapport (villor):')
    for col in ['energiklass', 'uppvarmning', 'ek_proxy', 'energiklass_num', 'uppvarmning_score']:
        if col in villor.columns:
            cov = villor[col].notna().mean() * 100
            nonzero = (villor[col] != 0).mean() * 100 if villor[col].dtype in ['float64', 'int64'] else None
            nonzero_str = f', {nonzero:.0f}%≠0' if nonzero is not None else ''
            log.info(f'    {col}: {cov:.0f}% täckning{nonzero_str}')
    log.info(f"    energiklass_kalla: {villor['energiklass_kalla'].value_counts().to_dict()}")

    return df


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Berikar med energiklass + uppvarmning')
    parser.add_argument('--limit', type=int, default=None, help='Max aktiva URLs att scrapa (test)')
    parser.add_argument('--merge-only', action='store_true', help='Hoppa scraping, bara merge')
    args = parser.parse_args()

    log.info('=' * 60)
    log.info('ENERGIKLASS-BERIKNING v1')
    log.info('=' * 60)

    # Ladda träningsdata
    log.info(f'Laddar {ENRICHED_V2}...')
    df = pd.read_csv(ENRICHED_V2)
    log.info(f'  {len(df)} rader, {df.shape[1]} kolumner')

    # Aktiva villa-URLs från active_listings_scored.csv
    aktiva_urls = []
    if os.path.exists(ACTIVE_CSV):
        active = pd.read_csv(ACTIVE_CSV)
        aktiva_urls = active[active['bostadstyp'] == 'villor']['url'].dropna().tolist()
        log.info(f'  Aktiva villa-URLs: {len(aktiva_urls)}')

    # Ladda cache
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding='utf-8') as f:
            cache_list = json.load(f)
        cache = {d['url']: d for d in cache_list if d and d.get('url')}
        log.info(f'  Befintlig energiklass-cache: {len(cache)} poster')

    if not args.merge_only and aktiva_urls:
        cache = scrape_active_listings(aktiva_urls, cache, limit=args.limit)
    elif args.merge_only:
        log.info('--merge-only: hoppar scraping')
    else:
        log.info('Inga aktiva URLs — hoppar scraping, bygger bara proxy')

    # Merge + spara
    df_out = merge_energiklass(df, cache)
    df_out.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    log.info(f'\n✅ Sparad: {OUTPUT_PATH}')
    log.info(f'   Rader: {len(df_out)}, Kolumner: {len(df_out.columns)}')
    log.info('Nya kolumner: ek_proxy, ek_proxy_letter, energiklass_num, uppvarmning_score, energiklass_kalla')
    log.info('\nNästa steg:')
    log.info('  python scripts/train_villa_v7.py  (använder orebro_housing_enriched_v3.csv)')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
