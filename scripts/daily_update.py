#!/usr/bin/env python3
"""
Örebro Housing — Daglig uppdatering
=====================================
Scrapar aktiva annonser från Hemnet, kör ML-modellen,
och sparar resultaten. Kör dagligen via cron eller manuellt.

Användning:
    python scripts/daily_update.py

Schemalägg med cron (kör varje dag kl 08:00):
    crontab -e
    0 8 * * * cd /Users/loranali/Downloads/orebro-housing-ml\ 3 && .venv/bin/python scripts/daily_update.py >> logs/daily.log 2>&1
"""

import pandas as pd
import numpy as np
import time
import random
import re
import json
import os
import sys
import subprocess
import joblib
from datetime import datetime

# Lägg till projektets rot i path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

# ============================================================
# KONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(PROJECT_DIR, 'models', 'best_model.pkl')
TRAIN_DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'orebro_housing_enriched.csv')
OUTPUT_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'active_listings_scored.csv')
HISTORY_DIR = os.path.join(PROJECT_DIR, 'data', 'history')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')

# Skapa mappar
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] {msg}')

# ============================================================
# STEG 1: SCRAPA AKTIVA ANNONSER
# ============================================================

def scrape_active():
    """Scrapa alla aktiva annonser från Hemnet med requests (ingen webbläsare krävs)."""
    import requests
    from bs4 import BeautifulSoup

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept-Language': 'sv-SE,sv;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Referer': 'https://www.hemnet.se/',
    }
    session = requests.Session()
    session.headers.update(HEADERS)

    def scrape_page(url):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                log(f'    HTTP {resp.status_code} för {url}')
                return []
            soup = BeautifulSoup(resp.text, 'lxml')
            cards = soup.find_all('a', href=re.compile(r'/bostad/'))
            listings = []
            for card in cards:
                text = card.get_text(' ', strip=True)
                if len(text) < 10:
                    continue
                listing = {'raw_text': text, 'url': 'https://www.hemnet.se' + card.get('href', '')}
                price_match = re.search(r'([\d\s\xa0]+)\s*kr', text)
                if price_match:
                    p = price_match.group(1).replace(' ', '').replace('\xa0', '')
                    try:
                        listing['utgangspris'] = int(p)
                    except:
                        pass
                area_match = re.search(r'([\d,\.]+)\s*m[²2]', text)
                if area_match:
                    listing['boarea_kvm'] = float(area_match.group(1).replace(',', '.'))
                rooms_match = re.search(r'([\d,]+)\s*rum', text)
                if rooms_match:
                    listing['antal_rum'] = float(rooms_match.group(1).replace(',', '.'))
                fee_match = re.search(r'([\d\s\xa0]+)\s*kr/m[åa]n', text)
                if fee_match:
                    f = fee_match.group(1).replace(' ', '').replace('\xa0', '')
                    try:
                        listing['avgift_kr'] = int(f)
                    except:
                        pass
                if listing.get('utgangspris') and listing.get('utgangspris') > 100000:
                    listings.append(listing)
            return listings
        except Exception as e:
            log(f'    Fel vid {url}: {e}')
            return []

    URLS = {
        'lagenheter': 'https://www.hemnet.se/till-salu/lagenhet/orebro-kommun',
        'villor': 'https://www.hemnet.se/till-salu/villa/orebro-kommun',
        'radhus': 'https://www.hemnet.se/till-salu/radhus/orebro-kommun',
    }

    all_active = []
    for typ, base_url in URLS.items():
        log(f'  Scrapar {typ}...')
        for page in range(1, 11):
            url = f'{base_url}?page={page}'
            listings = scrape_page(url)
            if not listings:
                break
            for l in listings:
                l['bostadstyp'] = typ
            all_active.extend(listings)
            time.sleep(random.uniform(1, 2))
        log(f'  {typ}: {sum(1 for l in all_active if l["bostadstyp"] == typ)} annonser')

    return pd.DataFrame(all_active)


# ============================================================
# STEG 2: RENSA OCH MATCHA OMRÅDEN
# ============================================================

# Områdesmappning: URL-format → modellens format
URL_TO_AREA = {
    'sorbyangen': 'Sörbyängen', 'sodra ladugardsangen': 'Ladugårdsängen',
    'ladugardsangen': 'Ladugårdsängen', 'ormesta': 'Ormesta',
    'centralt vaster': 'Centralt Väster', 'almby': 'Almby',
    'lillan': 'Lillån', 'oster': 'Öster', 'orebro': 'Örebro',
    'vaster': 'Väster', 'adolfsberg': 'Adolfsberg',
    'rynningeasen': 'Rynningeåsen', 'centralt oster': 'Centralt Öster',
    'mellringe': 'Mellringe', 'solhaga': 'Solhaga', 'bettorp': 'Bettorp',
    'sorby': 'Sörby', 'centralt': 'Centralt', 'sodra lindhult': 'Södra Lindhult',
    'ekeby almby': 'Almby', 'hovsta': 'Hovsta', 'lundby': 'Lundby',
    'bjorkhaga': 'Björkhaga', 'nya hjarsta': 'Nya Hjärsta',
    'garphyttan': 'Garphyttan', 'glanshammar': 'Glanshammar',
    'stora mellosa': 'Stora Mellösa', 'vintrosa': 'Vintrosa',
    'odensbacken': 'Odensbacken', 'mosas': 'Mosås', 'marieberg': 'Marieberg',
    'rynninge': 'Rynninge', 'vivalla': 'Vivalla', 'brickebacken': 'Brickebacken',
    'baronbackarna': 'Baronbackarna', 'hagaby': 'Hagaby',
    'norra bro': 'Norra Bro', 'tybble': 'Tybble', 'kilsmo': 'Kilsmo',
    'asker': 'Asker', 'sodra ladugards': 'Ladugårdsängen', 'ekeby': 'Almby',
    'norr': 'Norr', 'soder': 'Söder', 'talby': 'Talby',
    'klockhammar': 'Klockhammar', 'nasta': 'Nästa', 'latorp': 'Latorp',
    'askers by': 'Asker', 'varberga': 'Varberga', 'oxhagen': 'Oxhagen',
    'markbacken': 'Markbacken', 'norrby': 'Norrby', 'brickeberg': 'Brickeberg',
    'navesta': 'Navesta', 'lillans': 'Lillån', 'gamla hjarsta': 'Gamla Hjärsta',
}

def clean_and_match(df_raw, top_areas):
    """Rensa data och matcha områden."""
    df = df_raw.copy()
    
    # Fyll avgifter
    for typ, val in [('lagenheter', df.loc[df['bostadstyp'] == 'lagenheter', 'avgift_kr'].median()),
                     ('radhus', df.loc[df['bostadstyp'] == 'radhus', 'avgift_kr'].median())]:
        df.loc[(df['bostadstyp'] == typ) & df['avgift_kr'].isna(), 'avgift_kr'] = val
    df.loc[df['bostadstyp'] == 'villor', 'avgift_kr'] = df.loc[df['bostadstyp'] == 'villor', 'avgift_kr'].fillna(0)
    
    # Dedup och dropna
    df = df.drop_duplicates(subset='url')
    df = df.dropna(subset=['utgangspris', 'boarea_kvm', 'antal_rum'])
    
    # Extrahera och matcha område
    def extract_area(url):
        try:
            parts = url.split('/')[-1]
            cleaned = re.sub(r'^(lagenhet|villa|radhus)-\d+rum-', '', parts)
            area = cleaned.split('-orebro-kommun')[0]
            return area.replace('-', ' ').title()
        except:
            return 'Örebro'
    
    def match_area(url_area):
        clean = url_area.lower().strip()
        if clean in URL_TO_AREA:
            return URL_TO_AREA[clean]
        for key, value in URL_TO_AREA.items():
            if key in clean or clean in key:
                return value
        for area in top_areas:
            area_ascii = area.lower().replace('å', 'a').replace('ä', 'a').replace('ö', 'o').replace('é', 'e')
            if area_ascii in clean or clean in area_ascii:
                return area
        return 'övrigt'
    
    df['omrade'] = df['url'].apply(extract_area).apply(match_area)
    return df


# ============================================================
# STEG 3: KÖR MODELLEN
# ============================================================

def score_listings(df_live, model_data, df_train, top_areas, feature_names):
    """Kör ML-modellen på aktiva annonser."""
    model = model_data['model']
    df_pred = df_live.copy()
    
    # Grundfeatures
    df_pred['prisforandring_pct'] = 0
    df_pred['sald_ar'] = datetime.now().year
    df_pred['sald_manad'] = datetime.now().month
    
    # Detaljfeatures
    df_pred['bostad_alder'] = df_train['bostad_alder'].median()
    df_pred['har_hiss'] = (df_pred['bostadstyp'] == 'lagenheter').astype(int)
    df_pred['har_balkong'] = (df_pred['bostadstyp'] == 'lagenheter').astype(int)
    df_pred['har_garage'] = (df_pred['bostadstyp'] == 'villor').astype(int)
    df_pred['renoverad'] = 0
    df_pred['driftkostnad_ar'] = df_pred['bostadstyp'].map({
        'lagenheter': df_train.loc[df_train['bostadstyp'] == 'lagenheter', 'driftkostnad_ar'].median(),
        'villor': df_train.loc[df_train['bostadstyp'] == 'villor', 'driftkostnad_ar'].median(),
        'radhus': df_train.loc[df_train['bostadstyp'] == 'radhus', 'driftkostnad_ar'].median(),
    })
    df_pred['tomtarea_kvm'] = df_pred['bostadstyp'].map({
        'lagenheter': 0,
        'villor': df_train.loc[df_train['bostadstyp'] == 'villor', 'tomtarea_kvm'].median(),
        'radhus': df_train.loc[df_train['bostadstyp'] == 'radhus', 'tomtarea_kvm'].median(),
    })
    df_pred['vaning'] = df_pred['bostadstyp'].apply(lambda x: 2 if x == 'lagenheter' else 0)
    df_pred['antal_besok'] = df_train['antal_besok'].median()
    
    # One-hot
    df_pred['bostadstyp_radhus'] = (df_pred['bostadstyp'] == 'radhus').astype(int)
    df_pred['bostadstyp_villor'] = (df_pred['bostadstyp'] == 'villor').astype(int)
    
    for col in feature_names:
        if col.startswith('omrade_grupp_'):
            area_name = col.replace('omrade_grupp_', '')
            df_pred[col] = (df_pred['omrade'] == area_name).astype(int)
    
    for col in feature_names:
        if col not in df_pred.columns:
            df_pred[col] = 0
    df_pred[feature_names] = df_pred[feature_names].fillna(0)
    
    # Prediktera
    X = df_pred[feature_names]
    df_live['estimerat_varde'] = model.predict(X)
    df_live['skillnad_kr'] = df_live['estimerat_varde'] - df_live['utgangspris']
    df_live['skillnad_pct'] = (
        (df_live['estimerat_varde'] - df_live['utgangspris']) / df_live['estimerat_varde'] * 100
    ).round(1)
    
    # Smart bedömning
    def bedom(pct):
        if abs(pct) > 60:
            return '⚠️ Osäkert'
        elif pct > 15:
            return '🟢 Potentiellt fynd'
        elif pct > -5:
            return '🟡 Rimligt pris'
        else:
            return '🔴 Överprissatt'
    
    df_live['bedomning'] = df_live['skillnad_pct'].apply(bedom)
    df_live['scrape_datum'] = datetime.now().strftime('%Y-%m-%d')
    
    return df_live


# ============================================================
# HUVUDPROGRAM
# ============================================================

def main():
    log('=' * 50)
    log('DAGLIG UPPDATERING STARTAR')
    log('=' * 50)
    
    # Ladda modell och träningsdata
    log('Laddar modell...')
    model_data = joblib.load(MODEL_PATH)
    feature_names = model_data['feature_names']
    log(f'  Modell: {model_data.get("model_name", "unknown")} (R²={model_data["metrics"]["R2"]})')
    
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    top_areas = df_train['omrade_clean'].value_counts().head(70).index.tolist()
    log(f'  Träningsdata: {len(df_train)} rader, {len(top_areas)} områden')
    
    # Scrapa
    log('Scrapar aktiva annonser...')
    df_raw = scrape_active()
    log(f'  Scrapade: {len(df_raw)} annonser')
    
    # Rensa
    log('Rensar och matchar områden...')
    df_clean = clean_and_match(df_raw, top_areas)
    log(f'  Rensade: {len(df_clean)} annonser')
    matched = (df_clean['omrade'] != 'övrigt').sum()
    log(f'  Matchade områden: {matched} ({matched/len(df_clean)*100:.0f}%)')
    
    # Score
    log('Kör ML-modellen...')
    df_scored = score_listings(df_clean, model_data, df_train, top_areas, feature_names)
    
    # Resultat
    log('Resultat:')
    for bed, count in df_scored['bedomning'].value_counts().items():
        log(f'  {bed}: {count}')
    
    # Spara
    save_cols = ['url', 'omrade', 'bostadstyp', 'utgangspris', 'estimerat_varde',
                 'skillnad_kr', 'skillnad_pct', 'bedomning', 'boarea_kvm',
                 'antal_rum', 'avgift_kr', 'scrape_datum']
    df_scored[save_cols].to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    log(f'Sparad: {OUTPUT_PATH}')
    
    # Spara historik (för att kunna jämföra över tid)
    history_file = os.path.join(HISTORY_DIR, f'listings_{datetime.now().strftime("%Y%m%d")}.csv')
    df_scored[save_cols].to_csv(history_file, index=False, encoding='utf-8-sig')
    log(f'Historik: {history_file}')

    # ============================================================
    # STEG 5: PUSHA TILL GITHUB → STREAMLIT UPPDATERAS
    # (Hoppas över om körning sker via GitHub Actions — Actions gör push själv)
    # ============================================================
    if os.environ.get('GITHUB_ACTIONS'):
        log('Kör på GitHub Actions — push hanteras av workflow.')
        log('=' * 50)
        log(f'KLART! {len(df_scored)} annonser bedömda.')
        log('=' * 50)
        return

    log('Pushar till GitHub...')
    try:
        commit_msg = f'Auto-update live listings {datetime.now().strftime("%Y-%m-%d %H:%M")}'
        cmds = [
            ['git', '-C', PROJECT_DIR, 'add', '-f', OUTPUT_PATH],
            ['git', '-C', PROJECT_DIR, 'commit', '-m', commit_msg],
            ['git', '-C', PROJECT_DIR, 'push'],
        ]
        for cmd in cmds:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 and 'nothing to commit' not in result.stdout:
                log(f'  Git-fel: {result.stderr.strip()}')
                break
        else:
            log('  GitHub uppdaterat — Streamlit hämtar ny data inom ~1 minut.')
    except Exception as e:
        log(f'  Push misslyckades: {e}')

    log('=' * 50)
    log(f'KLART! {len(df_scored)} annonser bedömda.')
    log('=' * 50)


if __name__ == '__main__':
    main()
# Updated Ons 18 Mar 2026 12:48:02 CET
