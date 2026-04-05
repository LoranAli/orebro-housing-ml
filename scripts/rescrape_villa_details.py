"""
Re-scrapa villa-detaljsidor — Hämta saknade features
======================================================
VIKTIG NOTERING OM DATAVERKLIGHET (uppdaterad 2026-04-03):
-----------------------------------------------------------
Sålda Hemnet-sidor (/salda/villa/...) INNEHÅLLER INTE:
  - beskrivning (annonstext / fritext)
  - uppvarmning (uppvärmningstyp)
  - energiklass (A-G)
  - garage-info i fritext

Dessa fält FINNS på aktiva annonser (/bostad/villa/...) men tas
bort av Hemnet när bostaden är såld. Skriptet körde framgångsrikt
2476 sidor men fick 0% täckning för dessa fält — det är KORREKT
beteende givet Hemnets datasituation, inte ett scraping-fel.

VAD SOM FUNGERAR för sålda sidor:
  OK  byggar, biarea_kvm, tomtarea_kvm, driftkostnad_ar
  OK  antal_besok, boarea_kvm, antal_rum
  EJ  beskrivning, uppvarmning, energiklass, har_garage (alltid 0)

LÖSNING FRAMÅT:
  Spara rica features (beskrivning, uppvarmning, garage) NÄR annonsen
  är aktiv (/bostad/...). Transferera dem till sold-records vid försäljning.
  Implementera i daily_update.py -> scrape_detail_pages().

---

Scrapar alla 2 476 villa-URLs från orebro_housing_enriched.csv
med den uppdaterade parsern (v5) som extraherar vad som finns:

  Fungerar:    byggar, biarea_kvm, tomtarea_kvm, driftkostnad_ar,
               antal_besok, har_uteplats, har_balkong
  Fungerar EJ: energiklass, uppvarmning, beskrivning (saknas på sålda sidor)

Resultat sparas i:
  data/raw/villa_details_v2.json    — rådata per URL
  data/processed/orebro_housing_enriched_v2.csv — uppdaterat dataset

Tidsåtgång: ~2.5 timmar (2 476 sidor × 3.5 sek/sida)
Tips: Kör med --limit 50 för ett snabbt test.

Kör:
    cd "orebro-housing-ml 3"
    python scripts/rescrape_villa_details.py           # Alla villor
    python scripts/rescrape_villa_details.py --limit 50  # Test 50 st
    python scripts/rescrape_villa_details.py --merge-only  # Bara merge (om JSON finns)
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

# ─────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'scripts'))

DATA_PATH    = os.path.join(PROJECT_DIR, 'data', 'processed', 'orebro_housing_enriched.csv')
CACHE_PATH   = os.path.join(PROJECT_DIR, 'data', 'raw', 'villa_details_v2.json')
OUTPUT_PATH  = os.path.join(PROJECT_DIR, 'data', 'processed', 'orebro_housing_enriched_v2.csv')
LOG_PATH     = os.path.join(PROJECT_DIR, 'logs', 'rescrape_villa.log')

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

# ─────────────────────────────────────────────────────────────
# PARSER (importera från daily_update)
# ─────────────────────────────────────────────────────────────
from daily_update import parse_detail_page, create_driver


# ─────────────────────────────────────────────────────────────
# SCRAPING
# ─────────────────────────────────────────────────────────────
def scrape_urls(urls, cache, limit=None):
    """Scrapa en lista URLs och uppdatera cache."""
    if limit:
        urls = urls[:limit]

    need = [u for u in urls if u not in cache or cache[u].get('error')]
    log.info(f'Totalt: {len(urls)} | Cachade: {len(urls)-len(need)} | Behöver scrapa: {len(need)}')

    if not need:
        log.info('Allt är redan cachat!')
        return cache

    driver = create_driver()
    errors = 0
    checkpoint_every = 50

    try:
        for i, url in enumerate(need, 1):
            try:
                driver.get(url)
                time.sleep(random.uniform(2.5, 4.0))
                details = parse_detail_page(driver.page_source)
                details['url'] = url
                details['scraped_at'] = datetime.now().isoformat()
                cache[url] = details

            except Exception as e:
                errors += 1
                cache[url] = {'url': url, 'error': str(e)[:100]}
                log.warning(f'  [{i}/{len(need)}] FEL: {url[-50:]} → {e}')

            # Checkpoint var 50:e sida
            if i % checkpoint_every == 0:
                _save_cache(cache)
                log.info(
                    f'  [{i}/{len(need)}] Checkpoint sparad. '
                    f'Fel: {errors} | '
                    f'Energiklass: {sum(1 for d in cache.values() if d.get("energiklass"))} | '
                    f'Uppvarmning: {sum(1 for d in cache.values() if d.get("uppvarmning"))} | '
                    f'Beskrivning: {sum(1 for d in cache.values() if d.get("beskrivning"))}'
                )

            # Restart Selenium var 300:e sida (undviker minnesproblem)
            if i % 300 == 0:
                driver.quit()
                time.sleep(random.uniform(5, 10))
                driver = create_driver()
                log.info('  Selenium omstartad.')

    finally:
        driver.quit()
        _save_cache(cache)
        ok = len(need) - errors
        log.info(f'Scraping klar: {ok} OK, {errors} fel')

    return cache


def _save_cache(cache):
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(list(cache.values()), f, ensure_ascii=False, indent=None)


# ─────────────────────────────────────────────────────────────
# UPPVÄRMNING → NUMERISK
# ─────────────────────────────────────────────────────────────
UPPVARMNING_SCORE = {
    # Positiva (moderna, låg driftskostnad)
    'bergvärme': 3, 'värmepump': 2, 'fjärrvärme': 2,
    'geotermisk': 3, 'jordvärme': 3, 'luftvärmepump': 2,
    # Neutrala
    'gas': 0, 'pellets': 1, 'ved': 0, 'biobränsle': 1,
    # Negativa (gamla, hög driftskostnad)
    'direktel': -2, 'elpanna': -2, 'elradiatorer': -2,
    'elvärme': -2, 'oljevärme': -3, 'olja': -3,
}


def uppvarmning_to_score(text):
    if not isinstance(text, str):
        return 0
    t = text.lower()
    for key, score in sorted(UPPVARMNING_SCORE.items(), key=lambda x: -abs(x[1])):
        if key in t:
            return score
    return 0


# ─────────────────────────────────────────────────────────────
# NLP-FEATURES från beskrivning
# ─────────────────────────────────────────────────────────────
PREMIUM_KW = [
    'totalrenoverad', 'nytt kök', 'nytt badrum', 'sjöutsikt', 'havsutsikt',
    'bergvärme', 'pool', 'dubbelgarage', 'kakelugn', 'inglasad terrass',
    'öppen planlösning', 'tyst läge', 'sjötomt', 'sjönära', 'nyproduktion',
    'nyrenoverad', 'nybyggd', 'modernt kök', 'modernt badrum', 'stambytt',
    'ny installation', 'solceller', 'laddstolpe', 'elbil',
]
NEGATIVE_KW = [
    'renoveringsbehov', 'renovering krävs', 'stambyte behövs', 'fukt',
    'mögel', 'rivningsobjekt', 'direktel', 'äldre standard',
    'behov av renovering', 'eftersatt underhåll', 'elpanna',
    'ej renoverat', 'originalskick',
]


def extract_nlp(text):
    if not isinstance(text, str) or len(text) < 20:
        return {'n_premium_words': 0, 'n_negative_words': 0, 'premium_score': 0,
                'beskrivning_langd': 0}
    t = text.lower()
    pos = sum(1 for kw in PREMIUM_KW if kw in t)
    neg = sum(1 for kw in NEGATIVE_KW if kw in t)
    return {
        'n_premium_words':  pos,
        'n_negative_words': neg,
        'premium_score':    pos - neg,
        'beskrivning_langd': len(text),
    }


# ─────────────────────────────────────────────────────────────
# MERGE: Slå ihop ny data med befintlig CSV
# ─────────────────────────────────────────────────────────────
def merge_new_data(df, cache):
    """
    Slår ihop ny scrapad data (v2) med befintlig enriched CSV.
    Skapar orebro_housing_enriched_v2.csv med alla nya features.
    """
    log.info('Slår ihop data...')

    cache_df = pd.DataFrame([
        v for v in cache.values()
        if v and not v.get('error') and v.get('url')
    ])

    if cache_df.empty:
        log.error('Cache är tom — inga data att merga')
        return df

    log.info(f'  Cache: {len(cache_df)} poster med data')

    # Energiklass → numerisk
    ek_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
    if 'energiklass' in cache_df.columns:
        cache_df['energiklass_num'] = (
            cache_df['energiklass'].str.strip().str.upper().map(ek_map).fillna(0)
        )
        coverage = cache_df['energiklass'].notna().mean() * 100
        log.info(f'  energiklass täckning: {coverage:.0f}%')

    # Uppvärmning → score
    if 'uppvarmning' in cache_df.columns:
        cache_df['uppvarmning_score'] = cache_df['uppvarmning'].apply(uppvarmning_to_score)
        coverage = cache_df['uppvarmning'].notna().mean() * 100
        log.info(f'  uppvarmning täckning: {coverage:.0f}%')

    # NLP från beskrivning
    if 'beskrivning' in cache_df.columns:
        nlp_df = cache_df['beskrivning'].apply(extract_nlp).apply(pd.Series)
        cache_df = pd.concat([cache_df, nlp_df], axis=1)
        coverage = cache_df['beskrivning'].notna().mean() * 100
        log.info(f'  beskrivning täckning: {coverage:.0f}%')
        pos_pct = (cache_df['n_premium_words'] > 0).mean() * 100
        log.info(f'  NLP premium-ord träffar: {pos_pct:.0f}%')

    # Välj kolumner att merga (undvik dubbletter med befintliga)
    existing_cols = set(df.columns)
    new_cols = [c for c in cache_df.columns
                if c not in existing_cols
                or c in ['energiklass', 'uppvarmning', 'har_kallare',
                         'antal_badrum', 'renoverat_ar', 'beskrivning']]
    merge_cols = ['url'] + [c for c in new_cols if c != 'url' and c != 'scraped_at' and c != 'error']

    cache_slim = cache_df[merge_cols].drop_duplicates(subset='url')

    df_merged = df.merge(cache_slim, on='url', how='left')

    # Rapport
    new_feature_cols = [c for c in merge_cols if c != 'url']
    log.info(f'\n  Nya features tillagda: {new_feature_cols}')
    for col in new_feature_cols:
        if col in df_merged.columns:
            coverage = df_merged[col].notna().mean() * 100
            log.info(f'    {col}: {coverage:.0f}% täckning')

    return df_merged


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None,
                        help='Max antal URLs att scrapa (test-läge)')
    parser.add_argument('--merge-only', action='store_true',
                        help='Hoppa över scraping, bara merge befintlig cache')
    parser.add_argument('--bostadstyp', default='villor',
                        help='Filtrera på bostadstyp (default: villor)')
    args = parser.parse_args()

    log.info('=' * 60)
    log.info(f'VILLA DETAIL RE-SCRAPING v2 — {args.bostadstyp.upper()}')
    log.info('=' * 60)

    # Ladda data
    log.info('Laddar befintlig data...')
    df = pd.read_csv(DATA_PATH)
    if args.bostadstyp != 'alla':
        df_typ = df[df['bostadstyp'] == args.bostadstyp]
    else:
        df_typ = df
    urls = df_typ['url'].dropna().tolist()
    log.info(f'  {len(urls)} URLs för {args.bostadstyp}')

    # Ladda befintlig cache
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding='utf-8') as f:
            cache_list = json.load(f)
        cache = {d['url']: d for d in cache_list if d and d.get('url')}
        log.info(f'  Befintlig cache: {len(cache)} poster')

    if not args.merge_only:
        # Scrapa
        cache = scrape_urls(urls, cache, limit=args.limit)

    # Merge + spara
    log.info('\nMergar data...')
    df_out = merge_new_data(df, cache)
    df_out.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    log.info(f'✅ Sparad: {OUTPUT_PATH}')
    log.info(f'   Rader: {len(df_out)}, Kolumner: {len(df_out.columns)}')

    # Summera nya features för villor
    v_out = df_out[df_out['bostadstyp'] == 'villor']
    log.info(f'\n  === Villor i v2-datan ===')
    for col in ['energiklass', 'uppvarmning', 'energiklass_num',
                'uppvarmning_score', 'n_premium_words', 'har_kallare',
                'antal_badrum', 'beskrivning_langd']:
        if col in v_out.columns:
            cov = v_out[col].notna().mean() * 100
            if v_out[col].dtype in [float, int]:
                nonzero = (v_out[col] != 0).mean() * 100
                log.info(f'  {col}: {cov:.0f}% täckning, {nonzero:.0f}% ≠ 0')
            else:
                log.info(f'  {col}: {cov:.0f}% täckning')

    log.info('\n' + '=' * 60)
    log.info('KLAR!')
    log.info('Nästa steg: python scripts/train_villa_v5.py \\')
    log.info('  (ändra DATA_PATH till orebro_housing_enriched_v2.csv)')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
