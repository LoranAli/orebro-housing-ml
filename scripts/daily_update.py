#!/usr/bin/env python3
"""
Örebro Housing — Daglig uppdatering v2.2
==========================================
1. Scrapar listsidor → grunddata (pris, boarea, rum, avgift)
2. Scrapar DETALJSIDOR → alla features (byggår, våning, hiss, etc.)
3. Kör ML-modell per bostadstyp med riktiga features
4. Beräknar Deal Score
5. Sparar + pushar till GitHub

v2.2: Detaljscraping löser feature mismatch-problemet.
Tar ~20-30 min pga detaljsidorna (2-3 sek/sida × 700 annonser).

Användning:
    .venv/bin/python scripts/daily_update.py
    .venv/bin/python scripts/daily_update.py --skip-details  # Snabbläge utan detaljer
"""

import pandas as pd
import numpy as np
import time
import random
import re
import os
import sys
import subprocess
import joblib
import json
import logging
import argparse
from datetime import datetime, timedelta

# ============================================================
# SÖKVÄGAR
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'scripts'))

MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
_TRAIN_V5 = os.path.join(PROJECT_DIR, 'data', 'processed', 'orebro_housing_enriched_v5.csv')
_TRAIN_V4 = os.path.join(PROJECT_DIR, 'data', 'processed', 'orebro_housing_enriched_v4.csv')
_TRAIN_V3 = os.path.join(PROJECT_DIR, 'data', 'processed', 'orebro_housing_enriched_v3.csv')
_TRAIN_V1 = os.path.join(PROJECT_DIR, 'data', 'processed', 'orebro_housing_enriched.csv')
TRAIN_DATA_PATH = (
    _TRAIN_V5 if os.path.exists(_TRAIN_V5) else
    _TRAIN_V4 if os.path.exists(_TRAIN_V4) else
    _TRAIN_V3 if os.path.exists(_TRAIN_V3) else
    _TRAIN_V1
)
OUTPUT_PATH = os.path.join(
    PROJECT_DIR, 'data', 'processed', 'active_listings_scored.csv')
DETAILS_CACHE_PATH = os.path.join(
    PROJECT_DIR, 'data', 'raw', 'live_details_cache.json')
HISTORY_DIR = os.path.join(PROJECT_DIR, 'data', 'history')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DETAILS_CACHE_PATH), exist_ok=True)

# ============================================================
# LOGGING
# ============================================================

log_file = os.path.join(
    LOG_DIR, f'daily_{datetime.now().strftime("%Y%m%d")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-5s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ============================================================
# VILLA MODEL CLASSES — måste importeras för pickle-deserialisering
# ============================================================

try:
    from villa_models import StackingVillaModel, SegmentedVillaModel  # noqa: F401
except ImportError:
    log.warning('villa_models.py ej hittad — villa-modellen kan inte laddas')

# ============================================================
# DEAL SCORE
# ============================================================

try:
    from deal_score import compute_deal_scores_batch
except ImportError:
    log.warning('deal_score.py ej hittad')
    compute_deal_scores_batch = None

# ============================================================
# AREAS
# ============================================================

URL_TO_AREA = {
    'sorbyangen': 'Sörbyängen', 'ladugardsangen': 'Ladugårdsängen',
    'ormesta': 'Ormesta', 'centralt vaster': 'Centralt Väster',
    'almby': 'Almby', 'lillan': 'Lillån', 'orebro': 'Örebro',
    'vaster': 'Väster', 'adolfsberg': 'Adolfsberg',
    'rynningeasen': 'Rynningeåsen', 'centralt oster': 'Centralt Öster',
    'mellringe': 'Mellringe', 'solhaga': 'Solhaga', 'bettorp': 'Bettorp',
    'sorby': 'Sörby', 'centralt': 'Centralt', 'hovsta': 'Hovsta',
    'lundby': 'Lundby', 'bjorkhaga': 'Björkhaga', 'nya hjarsta': 'Hjärsta',
    'garphyttan': 'Garphyttan', 'glanshammar': 'Glanshammar',
    'vintrosa': 'Vintrosa', 'odensbacken': 'Odensbacken',
    'marieberg': 'Marieberg', 'rynninge': 'Rynninge', 'vivalla': 'Vivalla',
    'brickebacken': 'Brickebacken', 'baronbackarna': 'Baronbackarna',
    'hagaby': 'Hagaby', 'tybble': 'Tybble', 'asker': 'Asker',
    'norr': 'Centralt', 'soder': 'Centralt', 'nasta': 'Adolfsberg',
    'varberga': 'Varberga', 'oxhagen': 'Oxhagen', 'markbacken': 'Markbacken',
    'norrby': 'Norrby', 'navesta': 'Navesta', 'lillans': 'Lillån',
    'gamla hjarsta': 'Hjärsta', 'karlslund': 'Adolfsberg',
    'vasastan': 'Centralt', 'nasby': 'Almby', 'eklunda': 'Sörby',
    'hjarsta': 'Hjärsta', 'lindhult': 'Lindhult', 'latorp': 'Latorp',
    'ekeby': 'Almby', 'brickeberg': 'Brickebacken',
}

# ============================================================
# TYPSPECIFIK VALIDERING
# ============================================================

VALID_RANGES = {
    'lagenheter': {'boarea': (15, 200), 'pris': (200_000, 8_000_000), 'rum': (1, 8)},
    'villor':     {'boarea': (40, 400), 'pris': (500_000, 15_000_000), 'rum': (2, 12)},
    'radhus':     {'boarea': (30, 250), 'pris': (300_000, 8_000_000), 'rum': (1, 10)},
}


def validate_listing(row):
    typ = row.get('bostadstyp', 'lagenheter')
    r = VALID_RANGES.get(typ, VALID_RANGES['lagenheter'])
    pris, bo, rum = row.get('utgangspris', 0), row.get(
        'boarea_kvm', 0), row.get('antal_rum', 0)
    if not (r['pris'][0] <= pris <= r['pris'][1]):
        return False
    if not (r['boarea'][0] <= bo <= r['boarea'][1]):
        return False
    if not (r['rum'][0] <= rum <= r['rum'][1]):
        return False
    if rum > 0 and bo > 0 and (bo / rum < 8 or bo / rum > 60):
        return False
    return True


# ============================================================
# STEG 1: SCRAPA LISTSIDOR
# ============================================================

def create_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def scrape_list_pages():
    from bs4 import BeautifulSoup

    def scrape_page(driver, url):
        driver.get(url)
        time.sleep(random.uniform(2, 4))
        soup = BeautifulSoup(driver.page_source, 'lxml')
        cards = soup.find_all('a', href=re.compile(r'/bostad/'))
        listings = []
        for card in cards:
            text = card.get_text(' ', strip=True)
            if len(text) < 10:
                continue
            listing = {'raw_text': text,
                       'url': 'https://www.hemnet.se' + card.get('href', '')}
            m = re.search(r'([\d\s\xa0]+)\s*kr', text)
            if m:
                try:
                    listing['utgangspris'] = int(
                        m.group(1).replace(' ', '').replace('\xa0', ''))
                except ValueError:
                    pass
            m = re.search(r'([\d,\.]+)\s*m[²2]', text)
            if m:
                listing['boarea_kvm'] = float(m.group(1).replace(',', '.'))
            m = re.search(r'([\d,]+)\s*rum', text)
            if m:
                listing['antal_rum'] = float(m.group(1).replace(',', '.'))
            m = re.search(r'([\d\s\xa0]+)\s*kr/m[åa]n', text)
            if m:
                try:
                    listing['avgift_kr'] = int(
                        m.group(1).replace(' ', '').replace('\xa0', ''))
                except ValueError:
                    pass
            if listing.get('utgangspris', 0) > 100000:
                listings.append(listing)
        return listings

    URLS = {
        'lagenheter': 'https://www.hemnet.se/till-salu/lagenhet/orebro-kommun',
        'villor': 'https://www.hemnet.se/till-salu/villa/orebro-kommun',
        'radhus': 'https://www.hemnet.se/till-salu/radhus/orebro-kommun',
    }

    driver = create_driver()
    all_listings = []
    try:
        for typ, base_url in URLS.items():
            log.info(f'  Scrapar {typ}...')
            count = 0
            for page in range(1, 11):
                for attempt in range(3):
                    try:
                        listings = scrape_page(
                            driver, f'{base_url}?page={page}')
                        break
                    except Exception as e:
                        time.sleep(2 ** attempt * 30)
                else:
                    listings = []
                if not listings:
                    break
                for l in listings:
                    l['bostadstyp'] = typ
                all_listings.extend(listings)
                count += len(listings)
            log.info(f'    {count} annonser')
    finally:
        driver.quit()
    return pd.DataFrame(all_listings)


# ============================================================
# STEG 2: SCRAPA DETALJSIDOR
# ============================================================

def parse_detail_page(html):
    """Extrahera features från Hemnet-detaljsida."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'lxml')
    details = {}

    text = soup.get_text('\n', strip=True)
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    for i, line in enumerate(lines):
        if i + 1 >= len(lines):
            break
        next_line = lines[i + 1]
        label = line.lower().strip()

        if label in ['byggår', 'byggnadsår']:
            m = re.search(r'(1[6-9]\d{2}|20[0-2]\d)', next_line)
            if m:
                details['byggar'] = int(m.group(1))

        elif label == 'våning':
            m = re.search(r'(\d+)', next_line)
            if m:
                details['vaning'] = int(m.group(1))
            m2 = re.search(r'av\s*(\d+)', next_line)
            if m2:
                details['antal_vaningar'] = int(m2.group(1))
            if 'hiss' in next_line.lower():
                details['har_hiss'] = 1

        elif label == 'balkong':
            details['har_balkong'] = 1 if next_line.lower().strip() in [
                'ja', 'yes'] else 0

        elif label == 'boarea':
            m = re.search(r'([\d\s,\.]+)\s*m', next_line)
            if m:
                details['boarea_detail'] = float(m.group(1).replace(
                    ' ', '').replace(',', '.').replace('\xa0', ''))

        elif label == 'biarea':
            m = re.search(r'([\d\s,\.]+)\s*m', next_line)
            if m:
                details['biarea_kvm'] = float(m.group(1).replace(
                    ' ', '').replace(',', '.').replace('\xa0', ''))

        elif label == 'tomtarea':
            m = re.search(r'([\d\s,\.]+)\s*m', next_line)
            if m:
                details['tomtarea_kvm'] = float(m.group(1).replace(
                    ' ', '').replace(',', '.').replace('\xa0', ''))

        elif label == 'avgift':
            m = re.search(r'([\d\s]+)', next_line)
            if m:
                details['avgift_detail'] = int(
                    m.group(1).replace(' ', '').replace('\xa0', ''))

        elif label in ['driftskostnad', 'driftkostnad']:
            m = re.search(r'([\d\s]+)', next_line)
            if m:
                details['driftkostnad_ar'] = int(
                    m.group(1).replace(' ', '').replace('\xa0', ''))

        elif label == 'antal rum':
            m = re.search(r'([\d,]+)', next_line)
            if m:
                details['antal_rum_detail'] = float(
                    m.group(1).replace(',', '.'))

        elif label == 'upplåtelseform':
            details['upplatelseform'] = next_line

        elif label == 'antal besök':
            m = re.search(r'([\d\s]+)', next_line)
            if m:
                details['antal_besok'] = int(
                    m.group(1).replace(' ', '').replace('\xa0', ''))

        # ── NYA FÄLT (v5) ──────────────────────────────────────

        elif label in ['energiklass', 'energiklass:']:
            # Format: "A", "B", "C" ... "G" eller "A (20 kWh/m²)"
            m = re.search(r'\b([A-G])\b', next_line.upper())
            if m:
                details['energiklass'] = m.group(1)

        elif label in ['uppvärmning', 'uppvärming', 'uppvärmningssätt']:
            details['uppvarmning'] = next_line.strip().lower()

        elif label in ['antal badrum', 'badrum']:
            m = re.search(r'(\d+)', next_line)
            if m:
                details['antal_badrum'] = int(m.group(1))

        elif label in ['renoverat', 'renoverades', 'renoveringsår']:
            m = re.search(r'(1[89]\d{2}|20[0-2]\d)', next_line)
            if m:
                details['renoverat_ar'] = int(m.group(1))

        elif label in ['källare', 'källarplan']:
            details['har_kallare'] = 1 if next_line.lower().strip() in [
                'ja', 'yes', 'finns'] else 0

        elif label in ['pantbrev', 'inteckning']:
            m = re.search(r'([\d\s]+)', next_line)
            if m:
                try:
                    details['pantbrev_kr'] = int(
                        m.group(1).replace(' ', '').replace('\xa0', ''))
                except ValueError:
                    pass

    # ── Binära features via textsökning ──────────────────────
    # NOTERING: Sålda Hemnet-sidor (/salda/...) saknar fritext-beskrivning.
    # Dessa regex fungerar bara på AKTIVA annonser (/bostad/...) där
    # mäklarens text innehåller ord som "garage", "renoverad" etc.
    # För sålda sidor ger dessa alltid 0 — det är korrekt beteende.
    full_text = text.lower()

    # Begränsa sökningen till relevant_text (efter fakta-tablån börjar)
    # för att undvika falska träffar från Hemnets navigeringsmeny.
    _start_markers = ['bostadstyp', 'upplåtelseform', 'antal rum', 'slutpris']
    _relevant_start = len(full_text)
    for marker in _start_markers:
        idx = full_text.find(marker)
        if 0 < idx < _relevant_start:
            _relevant_start = idx
    relevant_text = (
        full_text[_relevant_start:]
        if _relevant_start < len(full_text)
        else full_text
    )

    if 'har_hiss' not in details:
        details['har_hiss'] = 1 if re.search(r'\bhiss\b', relevant_text) else 0
    if 'har_balkong' not in details:
        details['har_balkong'] = 1 if re.search(r'\bbalkong\b', relevant_text) else 0
    details['har_uteplats'] = 1 if re.search(
        r'\b(uteplats|terrass|altan)\b', relevant_text) else 0
    details['har_garage'] = 1 if re.search(
        r'\b(garage|carport|dubbelgarage)\b', relevant_text) else 0
    details['renoverad'] = 1 if re.search(
        r'\b(totalrenover|nyrenovera|helrenover|genomrenover|stamby|renoverad\s+20)\b',
        relevant_text) else 0

    # ── Uppvärmning — textsökning som fallback ────────────────
    if 'uppvarmning' not in details:
        if re.search(r'bergvärme|geotermisk|jordvärme', relevant_text):
            details['uppvarmning'] = 'bergvärme'
        elif re.search(r'värmepump|luftvärmepump', relevant_text):
            details['uppvarmning'] = 'värmepump'
        elif re.search(r'fjärrvärme', relevant_text):
            details['uppvarmning'] = 'fjärrvärme'
        elif re.search(r'direktel|elpanna|elradiatorer', relevant_text):
            details['uppvarmning'] = 'direktel'
        elif re.search(r'oljevärme|oljepanna|olja', relevant_text):
            details['uppvarmning'] = 'oljevärme'
        elif re.search(r'pellets|ved\b|biobränsle', relevant_text):
            details['uppvarmning'] = 'pellets'

    # ── Källare — textsökning ─────────────────────────────────
    if 'har_kallare' not in details:
        details['har_kallare'] = 1 if re.search(
            r'\bkällare\b', relevant_text) else 0

    # ── Beskrivningstext (NLP) — extrahera fritext ────────────
    # Hemnet har beskrivning i en sektion skild från faktatablån.
    # Sålda sidor saknar denna sektion — aktiva annonser har den.
    beskrivning = _extract_description(soup)
    if beskrivning:
        details['beskrivning'] = beskrivning

    return details


def _extract_description(soup):
    """
    Extrahera annonstextens fritext från Hemnet-sida.

    Hemnet använder React med hashed class names (ex: Broker_description__mtsHN).
    Strategin är att prova generella CSS-selektorer baserade på komponentnamn,
    sedan falla tillbaka på keyword-baserad heuristik.

    NOTERING: Sålda Hemnet-sidor (/salda/...) har INTE beskrivningssektionen
    — returnerar None för dessa. Aktiva annonser (/bostad/...) har den.
    """
    # 1. Prova React-komponent-klasser (Hemnets HTML 2025–2026)
    selectors = [
        '[class*="Broker_description"]',       # Mäklarens beskrivning (2025–2026)
        '[class*="BrokerDescription"]',        # Alternativt komponentnamn
        '[class*="PropertyDescription"]',      # Objektbeskrivning
        '[class*="ListingDescription"]',       # Ny variant (2026)
        '[class*="property-description"]',     # Kebab-case variant
        '[class*="description__text"]',        # Äldre variant
        '[class*="listing-description"]',      # Äldre variant
        '[class*="broker-description"]',       # Äldre variant
        '[class*="Description_description"]',  # Ytterligare variant
        '[data-testid*="description"]',        # Test-id variant
        '[data-testid*="listing-description"]',  # Ny test-id variant
    ]
    NAV_WORDS = [
        'cookie', 'javascript', 'hoppa till', 'sök bostad',
        'annonsera', 'kundservice', 'nyheter', 'nyhet!',
    ]
    for sel in selectors:
        els = soup.select(sel)
        if els:
            text = ' '.join(el.get_text(' ', strip=True) for el in els)
            if (len(text) > 100
                    and not any(w in text.lower() for w in NAV_WORDS)):
                return text[:3000]

    # 2. Heuristisk fallback: hitta text-nod med högst property-keyword-densitet
    PROPERTY_WORDS = [
        'kök', 'badrum', 'sovrum', 'vardagsrum', 'trädgård', 'garage',
        'renoverad', 'byggd', 'tomt', 'källare', 'altan', 'terrass',
        'uppvärmning', 'bergvärme', 'fjärrvärme', 'energi', 'pool',
        'läge', 'utsikt', 'nära', 'skola', 'kommunikationer', 'boyta',
    ]
    best_text = ''
    best_score = 0.0
    for tag in soup.find_all(['p', 'div', 'section']):
        # Hoppa över strukturella containers med barn-element
        if tag.find(['p', 'div', 'ul', 'ol', 'table']):
            continue
        t = tag.get_text(' ', strip=True)
        if len(t) < 80:
            continue
        t_lower = t.lower()
        if any(w in t_lower for w in NAV_WORDS):
            continue
        score = sum(1 for w in PROPERTY_WORDS if w in t_lower)
        score += len(t) / 500.0  # Bonus för längre text
        if score > best_score:
            best_score = score
            best_text = t

    return best_text[:3000] if len(best_text) > 100 else None


def scrape_detail_pages(df_listings):
    """Scrapa detaljsidor för alla listings. Använder cache."""
    # Ladda cache
    cache = {}
    if os.path.exists(DETAILS_CACHE_PATH):
        try:
            with open(DETAILS_CACHE_PATH, 'r', encoding='utf-8') as f:
                cache_list = json.load(f)
            cache = {d['url']: d for d in cache_list}
            log.info(f'  Cache: {len(cache)} sidor')
        except Exception:
            cache = {}

    urls = df_listings['url'].tolist()

    # Rensa cache från URLs som inte längre är aktiva (håller cachen liten och aktuell)
    active_url_set = set(urls)
    old_cache_size = len(cache)
    cache = {url: data for url, data in cache.items() if url in active_url_set}
    if old_cache_size != len(cache):
        log.info(f'  Cache: rensade {old_cache_size - len(cache)} utgångna annonser ({len(cache)} kvar)')

    need_scraping = [u for u in urls if u not in cache]
    log.info(
        f'  Behöver scrapa: {len(need_scraping)} av {len(urls)} (cache: {len(urls) - len(need_scraping)})')

    if need_scraping:
        driver = create_driver()
        errors = 0
        try:
            for i, url in enumerate(need_scraping):
                try:
                    driver.get(url)
                    time.sleep(random.uniform(1.5, 3.0))
                    details = parse_detail_page(driver.page_source)
                    details['url'] = url
                    cache[url] = details
                except Exception as e:
                    errors += 1
                    cache[url] = {'url': url, 'error': str(e)}

                # Checkpoint var 50:e
                if (i + 1) % 50 == 0:
                    with open(DETAILS_CACHE_PATH, 'w', encoding='utf-8') as f:
                        json.dump(list(cache.values()), f, ensure_ascii=False)
                    log.info(
                        f'    {i+1}/{len(need_scraping)} scrapade ({errors} fel)')

                # Restart driver var 200:e
                if (i + 1) % 200 == 0:
                    driver.quit()
                    time.sleep(random.uniform(5, 10))
                    driver = create_driver()
        finally:
            driver.quit()
            with open(DETAILS_CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(list(cache.values()), f, ensure_ascii=False)
            log.info(
                f'  Detaljscraping klar: {len(need_scraping) - errors} OK, {errors} fel')

    # Merga detaljer med listings
    detail_rows = []
    for url in urls:
        detail_rows.append(cache.get(url, {}))

    df_details = pd.DataFrame(detail_rows)
    if 'url' in df_details.columns:
        df_details = df_details.drop(columns=['url', 'error'], errors='ignore')

    # Slå ihop — detalj-features skriver över listsidans vid konflikter
    for col in df_details.columns:
        if col == 'avgift_detail':
            # Detaljsidans avgift är mer pålitlig
            mask = df_details[col].notna()
            df_listings.loc[mask,
                            'avgift_kr'] = df_details.loc[mask, col].astype(int)
        elif col == 'boarea_detail':
            mask = df_details[col].notna()
            df_listings.loc[mask, 'boarea_kvm'] = df_details.loc[mask, col]
        elif col == 'antal_rum_detail':
            mask = df_details[col].notna()
            df_listings.loc[mask, 'antal_rum'] = df_details.loc[mask, col]
        elif col not in df_listings.columns:
            df_listings[col] = df_details[col]

    # Beräkna bostad_alder
    if 'byggar' in df_listings.columns:
        df_listings['bostad_alder'] = datetime.now(
        ).year - df_listings['byggar'].fillna(1980)
    else:
        df_listings['bostad_alder'] = 40  # Default

    log.info(
        f'  Features efter merge: {[c for c in df_listings.columns if c not in ["raw_text", "url"]]}')

    return df_listings


# ============================================================
# STEG 3: RENSA OCH MATCHA
# ============================================================

def clean_and_match(df_raw, top_areas):
    df = df_raw.copy()

    # Avgift sanity check
    df.loc[df['avgift_kr'] > 15000, 'avgift_kr'] = np.nan
    for typ in ['lagenheter', 'radhus']:
        med = df.loc[df['bostadstyp'] == typ, 'avgift_kr'].median()
        if pd.notna(med):
            df.loc[(df['bostadstyp'] == typ) &
                   df['avgift_kr'].isna(), 'avgift_kr'] = med
    df.loc[df['bostadstyp'] == 'villor', 'avgift_kr'] = \
        df.loc[df['bostadstyp'] == 'villor', 'avgift_kr'].fillna(0)

    df = df.drop_duplicates(subset='url')
    df = df.dropna(subset=['utgangspris', 'boarea_kvm', 'antal_rum'])

    before = len(df)
    df = df[df.apply(validate_listing, axis=1)]
    removed = before - len(df)
    if removed > 0:
        log.info(f'  Validering: {removed} borttagna')

    def extract_area(url):
        try:
            parts = url.split('/')[-1]
            cleaned = re.sub(r'^(lagenhet|villa|radhus)-\d+rum-', '', parts)
            return cleaned.split('-orebro-kommun')[0].replace('-', ' ').strip()
        except Exception:
            return 'Örebro'

    def match_area(a):
        c = a.lower().strip()
        if c in URL_TO_AREA:
            return URL_TO_AREA[c]
        for k, v in URL_TO_AREA.items():
            if k in c or c in k:
                return v
        for area in top_areas:
            aa = area.lower().replace('å', 'a').replace('ä', 'a').replace('ö', 'o')
            if aa in c or c in aa:
                return area
        return 'övrigt'

    df['omrade'] = df['url'].apply(extract_area).apply(match_area)
    matched = (df['omrade'] != 'övrigt').sum()
    log.info(f'  Områden: {matched}/{len(df)} ({matched/len(df)*100:.0f}%)')
    return df


# ============================================================
# STEG 4: COMPS
# ============================================================

def compute_live_comps(df_live, df_train, typ):
    if df_train is None or 'omrade_clean' not in df_train.columns:
        for c in ['comps_pris_kvm_90d', 'comps_antal_90d', 'comps_pristrend']:
            df_live[c] = 0
        return df_live

    dt = df_train[df_train['bostadstyp'] == typ].copy()
    dt['sald_datum'] = pd.to_datetime(dt['sald_datum'], errors='coerce')
    dt['_pkvm'] = dt['slutpris'] / dt['boarea_kvm'].clip(lower=10)
    latest = dt['sald_datum'].max()
    c90, c180 = latest - timedelta(days=90), latest - timedelta(days=180)

    p, a, t = [], [], []
    for _, row in df_live.iterrows():
        omr = row.get('omrade', 'övrigt')
        ad = dt[dt['omrade_clean'] == omr]
        rec = ad[ad['sald_datum'] >= c90]
        old = ad[(ad['sald_datum'] >= c180) & (ad['sald_datum'] < c90)]
        p.append(rec['_pkvm'].median() if len(rec) >= 2
                 else (ad['_pkvm'].median() if len(ad) > 0 else dt['_pkvm'].median()))
        a.append(len(rec))
        t.append(round((rec['_pkvm'].median() / old['_pkvm'].median() - 1) * 100, 2)
                 if len(rec) >= 2 and len(old) >= 2 else 0)

    df_live['comps_pris_kvm_90d'] = p
    df_live['comps_antal_90d'] = a
    df_live['comps_pristrend'] = t
    return df_live


# ============================================================
# STEG 5: PREDIKTION
# ============================================================

def load_v2_models():
    models = {}
    for typ in ['lagenheter', 'villor', 'radhus']:
        # Föredra versionsspecifik modell (v10 > v9 > generisk)
        candidates = [
            os.path.join(MODELS_DIR, f'model_{typ}_v10.pkl'),
            os.path.join(MODELS_DIR, f'model_{typ}_v9.pkl'),
            os.path.join(MODELS_DIR, f'model_{typ}.pkl'),
            os.path.join(MODELS_DIR, 'best_model.pkl'),  # fallback för lagenheter
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path:
            models[typ] = joblib.load(path)
            log.info(
                f'  {typ}: {os.path.basename(path)} (R²={models[typ]["metrics"].get("R2", models[typ]["metrics"].get("lgbm_test", {}).get("R2", "?"))})')
    return models


def build_features(df_typ, typ, feature_names, df_train_typ, pkg=None):
    """Bygg alla features modellen behöver."""
    df = df_typ.copy()

    # Tidfeatures
    df['sald_ar'] = datetime.now().year
    df['sald_manad'] = datetime.now().month
    df['sald_kvartal'] = (datetime.now().month - 1) // 3 + 1
    df['prisforandring_pct'] = 0
    df['budkrig'] = 0
    df['prissankt'] = 0

    # Härledda features
    df['kvm_per_rum'] = df['boarea_kvm'] / df['antal_rum'].clip(lower=1)
    df['total_yta'] = df['boarea_kvm'] + \
        df.get('biarea_kvm', pd.Series(0, index=df.index)).fillna(0)

    if typ in ['lagenheter', 'radhus']:
        df['avgift_per_kvm'] = df['avgift_kr'].fillna(
            0) / df['boarea_kvm'].clip(lower=10)
        df['avgift_andel'] = df['avgift_kr'].fillna(
            0) * 12 / df['utgangspris'].clip(lower=1) * 100
    else:
        df['avgift_per_kvm'] = 0
        df['avgift_andel'] = 0

    if 'vaning' in df.columns and 'antal_vaningar' in df.columns:
        df['relativ_vaning'] = df['vaning'].fillna(
            2) / df['antal_vaningar'].fillna(4).clip(lower=1)
        df['toppvaning'] = (df['vaning'].fillna(
            0) >= df['antal_vaningar'].fillna(99)).astype(float)
    if 'tomtarea_kvm' in df.columns and typ in ['villor', 'radhus']:
        df['tomt_per_boarea'] = df['tomtarea_kvm'].fillna(
            0) / df['boarea_kvm'].clip(lower=10)
    if 'driftkostnad_ar' in df.columns and typ == 'villor':
        df['driftkostnad_per_kvm'] = df['driftkostnad_ar'].fillna(
            0) / df['boarea_kvm'].clip(lower=10)

    # ── v5+: Target encoding, KMeans, grannskap, NLP, log-transforms ──
    # Aktiveras för alla modeller som har avancerade features (v5, v6, v7+)
    _has_advanced = pkg and (
        pkg.get('model_type') in ('lgbm_quantile', 'lgbm_mse', 'catboost')
        or pkg.get('version', '') in ('v5', 'v6', 'v7')
        or pkg.get('te_map_pris')  # fallback: har target encoding-data
    )
    if _has_advanced:
        # 1.1 Target encoding av område
        te_pris = pkg.get('te_map_pris', {})
        te_kvm  = pkg.get('te_map_kvm', {})
        df['te_omrade_pris'] = df['omrade'].map(te_pris).fillna(
            pkg.get('te_global_pris', 0))
        df['te_omrade_kvm'] = df['omrade'].map(te_kvm).fillna(
            pkg.get('te_global_kvm', 0))

        # 1.5 KMeans cluster → cluster_te
        km     = pkg.get('kmeans')
        km_sc  = pkg.get('kmeans_scaler')
        km_fs  = pkg.get('kmeans_feats', [])
        c_map  = pkg.get('cluster_te_map', {})
        c_glob = pkg.get('cluster_global', 0)
        if km and km_sc and km_fs:
            X_c = pd.DataFrame(index=df.index)
            for col in km_fs:
                fallback = df_train_typ[col].median() if col in df_train_typ.columns else 0
                X_c[col] = df[col].fillna(fallback) if col in df.columns else fallback
            df['cluster_id'] = km.predict(km_sc.transform(X_c.values))
            df['cluster_te'] = df['cluster_id'].map(c_map).fillna(c_glob)

        # 1.6 Grannskap: comps_pris_kvm_90d som proxy vid inference
        # (BallTree körs mot df_train vid inference är kostsamt → comps är korrelerat)
        _comps_fallback = df.get(
            'comps_pris_kvm_90d', pd.Series(
                pkg.get('te_global_kvm', 0), index=df.index))
        df['grannskap_median_kvm'] = _comps_fallback
        df['grannskap_vd_kvm']     = _comps_fallback  # v8: distance-weighted variant
        # v9 B2: prisvariation i grannskapet — okänt vid inference (proxy: 0)
        if 'grannskap_spread_kvm' not in df.columns:
            df['grannskap_spread_kvm'] = 0.0

        # 1.75 v9 A2: Mäklarfirma target encoding
        if pkg is not None and 'maklare_te_map' in (pkg or {}):
            _MAKLARE_MAP = {
                'Bjurfors': 'Bjurfors', 'Mäklarhuset': 'Maklarhuset',
                'LF Fastighets': 'LF_Fastighets', 'Fastighetsbyrån': 'Fastighetsbyran',
                'Svensk Fastighets': 'Svensk_Fastighets', 'Nordå': 'Norda',
                'Notar': 'Notar', 'ERA ': 'ERA',
            }
            _te_map    = pkg['maklare_te_map']
            _te_global = pkg.get('maklare_te_global', pkg.get('te_global_pris', 0))
            def _extract_maklare(text):
                if not isinstance(text, str):
                    return 'Ovrigt'
                for key, norm in _MAKLARE_MAP.items():
                    if key in text:
                        return norm
                return 'Ovrigt'
            _text_col = 'raw_text' if 'raw_text' in df.columns else None
            if _text_col:
                df['_maklare_inf'] = df[_text_col].apply(_extract_maklare)
                df['maklare_te']   = df['_maklare_inf'].map(_te_map).fillna(_te_global)
                df.drop(columns=['_maklare_inf'], inplace=True)
            elif 'maklare_te' not in df.columns:
                df['maklare_te'] = _te_global

        # 1.76 v9 A3: Riksbankränta vid inference
        _RIKSBANK_RATE = {
            '2019-01': 0.0,  '2019-12': -0.25, '2020-01': -0.25, '2021-01': 0.0,
            '2022-04': 0.25, '2022-06': 0.75,  '2022-09': 1.75,  '2022-11': 2.5,
            '2023-02': 3.0,  '2023-04': 3.5,   '2023-06': 3.75,  '2023-08': 4.0,
            '2024-06': 3.75, '2024-08': 3.5,   '2024-09': 3.25,  '2024-11': 2.75,
            '2024-12': 2.5,  '2025-01': 2.5,   '2025-03': 2.25,  '2025-07': 2.0,
            '2026-01': 2.0,
        }
        def _lookup_rate(period_str):
            """Sök bakåt i tid för att hitta närmaste rate."""
            rate_series = pd.Series(_RIKSBANK_RATE)
            rate_series.index = pd.to_datetime(
                [k + '-01' for k in rate_series.index])
            rate_series = rate_series.sort_index()
            try:
                d = pd.to_datetime(period_str + '-01')
                candidates = rate_series[rate_series.index <= d]
                return float(candidates.iloc[-1]) if len(candidates) > 0 else 2.0
            except Exception:
                return 2.0

        if any(f in (feature_names or []) for f in
               ['riksbank_rate', 'rate_change_6m', 'rate_boarea_interact']):
            _now_period = pd.Timestamp.now().strftime('%Y-%m')
            _now_rate   = _lookup_rate(_now_period)
            # 6m tillbaka
            _6m_ago = (pd.Timestamp.now() - pd.DateOffset(months=6)).strftime('%Y-%m')
            _6m_rate = _lookup_rate(_6m_ago)
            if 'riksbank_rate' not in df.columns:
                df['riksbank_rate']    = _now_rate
            if 'rate_change_6m' not in df.columns:
                df['rate_change_6m']   = round(_now_rate - _6m_rate, 2)
            if 'rate_boarea_interact' not in df.columns:
                _log_boarea = np.log(df['boarea_kvm'].clip(lower=10))
                df['rate_boarea_interact'] = _now_rate * _log_boarea

        # v9: geocode_quality_bin — 0 vid inference (live listings ej geocodade)
        if 'geocode_quality_bin' not in df.columns:
            df['geocode_quality_bin'] = 0

        # v10 C2: DeSO socioekonomiska features — slå upp via omrade_clean
        _deso_feature_cols = [
            'deso_median_ink_tkr', 'deso_lon_ink_tkr', 'deso_andel_lon_pct',
            'deso_befolkning', 'deso_median_alder', 'deso_andel_0_19', 'deso_andel_65_plus',
        ]
        if pkg is not None and 'deso_omrade_map' in (pkg or {}):
            _deso_map     = pkg['deso_omrade_map']      # {omrade_clean: {col: val}}
            _deso_global  = pkg.get('deso_global_stats', {})
            _omr_col = 'omrade_clean' if 'omrade_clean' in df.columns else \
                       'omrade'       if 'omrade'       in df.columns else None
            for _col in _deso_feature_cols:
                if _col not in df.columns:
                    _glob_val = _deso_global.get(_col, 0.0)
                    if _omr_col:
                        df[_col] = df[_omr_col].map(
                            lambda o, c=_col: _deso_map.get(str(o), {}).get(c, _glob_val)
                        ).fillna(_glob_val)
                    else:
                        df[_col] = _glob_val

        # 1.7 Global marknadstrendindex (v8) — rolling 6-mån median pris/kvm
        # Beräknas från träningsdata för aktuellt datum
        if 'marknad_trend_6m' in (feature_names or []) or 'marknad_trend_ratio' in (feature_names or []):
            _now = pd.Timestamp.now()
            _cutoff = _now - pd.Timedelta(days=180)
            if df_train_typ is not None and 'slutpris' in df_train_typ.columns and 'boarea_kvm' in df_train_typ.columns:
                _dates_tr = pd.to_datetime(df_train_typ.get('sald_datum', pd.Series(dtype='datetime64[ns]')), errors='coerce')
                _pkvm_tr  = df_train_typ['slutpris'] / df_train_typ['boarea_kvm'].clip(lower=10)
                _recent   = _pkvm_tr[_dates_tr >= _cutoff]
                _trend    = float(_recent.median()) if len(_recent) >= 3 else float(_pkvm_tr.median())
            else:
                _trend = float(df.get('comps_pris_kvm_90d', pd.Series([30000])).median())
            df['marknad_trend_6m'] = _trend
            _comps_kvm = df.get('comps_pris_kvm_90d', pd.Series(_trend, index=df.index))
            df['marknad_trend_ratio'] = (_comps_kvm / _trend).clip(0.5, 2.0).fillna(1.0)

        # 1.4 NLP-features — extrahera från beskrivning (v2-data) eller raw_text
        _PREMIUM_KW = [
            'totalrenoverad', 'nytt kök', 'nytt badrum', 'sjöutsikt', 'havsutsikt',
            'bergvärme', 'pool', 'dubbelgarage', 'kakelugn', 'fjärrvärme',
            'inglasad', 'öppen planlösning', 'tyst läge', 'sjötomt', 'sjönära',
            'nyproduktion', 'nyrenoverad', 'nybyggd', 'fräscht', 'modernt',
        ]
        _NEGATIVE_KW = [
            'renoveringsbehov', 'renovering krävs', 'stambyte', 'fukt', 'mögel',
            'rivningsobjekt', 'direktel', 'äldre standard', 'behov av renovering',
            'eftersatt underhåll', 'elpanna',
        ]
        def _nlp_extract(text):
            if not isinstance(text, str) or len(text) < 20:
                return 0, 0
            t = text.lower()
            return (sum(1 for kw in _PREMIUM_KW if kw in t),
                    sum(1 for kw in _NEGATIVE_KW if kw in t))

        # Föredra beskrivning (från rescrape) framför raw_text (listsida)
        text_col = 'beskrivning' if 'beskrivning' in df.columns else \
                   'raw_text'   if 'raw_text'   in df.columns else None
        if text_col:
            _nlp = df[text_col].apply(_nlp_extract)
            df['n_premium_words'] = _nlp.apply(lambda x: x[0])
            df['n_negative_words'] = _nlp.apply(lambda x: x[1])
        else:
            df['n_premium_words'] = 0
            df['n_negative_words'] = 0
        df['premium_score'] = df['n_premium_words'] - df['n_negative_words']

        # ── v7: Log-transforms och nya features ─────────────────
        if 'log_boarea' not in df.columns:
            df['log_boarea'] = np.log(df['boarea_kvm'].clip(lower=1))
        if 'log_tomtarea' not in df.columns:
            tomt_raw = df['tomtarea_kvm'].fillna(0) if 'tomtarea_kvm' in df.columns else 0
            df['log_tomtarea'] = np.log1p(tomt_raw)
        if 'log_driftkostnad' not in df.columns:
            drift = df['driftkostnad_ar'].fillna(0) if 'driftkostnad_ar' in df.columns else 0
            df['log_driftkostnad'] = np.log1p(drift)
        if 'byggdekad' not in df.columns:
            byggar = df['byggar'].fillna(1970) if 'byggar' in df.columns else pd.Series(1970, index=df.index)
            df['byggdekad'] = ((byggar // 10) * 10).astype(int)
        if 'driftkostnad_per_kvm' not in df.columns and typ == 'villor':
            drift = df['driftkostnad_ar'].fillna(0) if 'driftkostnad_ar' in df.columns else 0
            df['driftkostnad_per_kvm'] = drift / df['boarea_kvm'].clip(lower=10)

        # ── Pris-kontext features ─────────────────────────────
        if 'omrade_hist_pris_kvm' not in df.columns:
            # Bästa proxy vid inference: target encoding av kvm-pris
            df['omrade_hist_pris_kvm'] = df.get(
                'te_omrade_kvm',
                df.get('te_omrade_pris',
                       pd.Series(pkg.get('te_global_pris', 0), index=df.index)))
        if 'forvantat_komps_pris' not in df.columns:
            df['forvantat_komps_pris'] = df['comps_pris_kvm_90d'] * df['boarea_kvm']

        # ── Interaktionsfeatures (v3–v7) ──────────────────────
        tomt_med = df_train_typ['tomtarea_kvm'].median() if 'tomtarea_kvm' in df_train_typ.columns else 500
        avst_med = df_train_typ['avstand_centrum_km'].median() if 'avstand_centrum_km' in df_train_typ.columns else 5

        if 'tomt_boarea_interact' not in df.columns:
            tomt = df['tomtarea_kvm'].fillna(tomt_med) if 'tomtarea_kvm' in df.columns else tomt_med
            df['tomt_boarea_interact'] = (tomt * df['boarea_kvm']) / 1e4
        if 'boarea_log_tomt' not in df.columns:
            df['boarea_log_tomt'] = df['log_boarea'] * df['log_tomtarea']
        if 'avst_pris_interact' not in df.columns:
            avst = df['avstand_centrum_km'].fillna(avst_med) if 'avstand_centrum_km' in df.columns else avst_med
            area_pris = df['omrade_hist_pris_kvm'].fillna(
                df_train_typ['omrade_hist_pris_kvm'].median()
                if 'omrade_hist_pris_kvm' in df_train_typ.columns
                else pkg.get('te_global_pris', 0))
            df['avst_pris_interact'] = (avst * area_pris) / 1000
        if 'tomt_avst_interact' not in df.columns:
            avst = df['avstand_centrum_km'].fillna(avst_med) if 'avstand_centrum_km' in df.columns else avst_med
            df['tomt_avst_interact'] = df['log_tomtarea'] * avst

        if 'alder_ej_renoverad' not in df.columns:
            alder = df['bostad_alder'].fillna(40) if 'bostad_alder' in df.columns else 40
            df['alder_ej_renoverad'] = alder  # renoverad är alltid 0 för sålda villor

        # ── ek_proxy och energiklass_num (v7+) ───────────────────
        # ek_proxy: energiklass-proxy från byggår (BBR-normer 1961–2020)
        if 'ek_proxy' not in df.columns:
            _EK_MAP = [(2020, 7), (2010, 6), (2000, 5), (1985, 4), (1975, 3), (1961, 2), (0, 1)]
            def _ek_proxy(y):
                if pd.isna(y) or y <= 0:
                    return 3
                y = int(y)
                for thr, sc in _EK_MAP:
                    if y >= thr:
                        return sc
                return 1
            byggar_col = df['byggar'] if 'byggar' in df.columns else pd.Series(1970, index=df.index)
            df['ek_proxy'] = byggar_col.fillna(1970).apply(_ek_proxy)

        # energiklass_num: riktig energiklass om scrapad, annars proxy
        _EK_TO_NUM = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
        if 'energiklass_num' not in df.columns:
            if 'energiklass' in df.columns:
                ek_real = df['energiklass'].map(_EK_TO_NUM)
                df['energiklass_num'] = ek_real.where(ek_real.notna(), df['ek_proxy'])
            else:
                df['energiklass_num'] = df['ek_proxy']

        # uppvarmning_score: score för värmekälla
        if 'uppvarmning_score' not in df.columns:
            _UPP_SCORE = {
                'bergvärme': 3, 'värmepump': 2, 'fjärrvärme': 2,
                'jordvärme': 3, 'luftvärmepump': 2,
                'pellets': 1, 'biobränsle': 1,
                'direktel': -2, 'elpanna': -2, 'elvärme': -2,
                'oljevärme': -3, 'olja': -3,
            }
            def _upp_score(t):
                if not isinstance(t, str):
                    return 0
                t = t.lower()
                for k, s in sorted(_UPP_SCORE.items(), key=lambda x: -abs(x[1])):
                    if k in t:
                        return s
                return 0
            upp_col = df['uppvarmning'] if 'uppvarmning' in df.columns else pd.Series(dtype=str)
            df['uppvarmning_score'] = upp_col.apply(_upp_score)

        # Bakåtkompatibilitet: v5 använde 'avstand_tomt_interact' (annat namn)
        if 'avstand_tomt_interact' not in df.columns:
            avst  = df['avstand_centrum_km'].fillna(avst_med) if 'avstand_centrum_km' in df.columns else avst_med
            tomt2 = df['tomtarea_kvm'].fillna(tomt_med)       if 'tomtarea_kvm'       in df.columns else tomt_med
            df['avstand_tomt_interact'] = (avst * tomt2) / 1e3

        if 'biarea_var_missing' not in df.columns:
            df['biarea_var_missing'] = df['biarea_kvm'].isna().astype(int) \
                if 'biarea_kvm' in df.columns else 1
        if 'biarea_kvm' in df.columns:
            df['biarea_kvm'] = df['biarea_kvm'].fillna(
                df['total_yta'] - df['boarea_kvm']).clip(lower=0)

    # One-hot och saknade features (v1–v4 bakåtkompatibilitet)
    for col in feature_names:
        if col not in df.columns:
            if col.startswith('omrade_grupp_'):
                area_name = col.replace('omrade_grupp_', '')
                df[col] = (df['omrade'] == area_name).astype(int)
            elif col.startswith('area_'):
                # Villa-modeller v3/v4 använder prefix 'area_' (inte 'omrade_grupp_')
                area_name = col.replace('area_', '')
                df[col] = (df['omrade'] == area_name).astype(int)
            elif col.startswith('upplatelseform_'):
                if typ == 'villor' and 'Äganderätt' in col:
                    df[col] = 1
                elif typ != 'villor' and 'Bostadsrätt' in col:
                    df[col] = 1
                else:
                    df[col] = 0
            elif len(df_train_typ) > 0 and col in df_train_typ.columns:
                df[col] = df_train_typ[col].median()
            else:
                df[col] = 0

    df[feature_names] = df[feature_names].fillna(0)
    return df


def predict_all(df_live, models, df_train):
    all_scored = []

    for typ in ['lagenheter', 'villor', 'radhus']:
        if typ not in models:
            continue
        df_typ = df_live[df_live['bostadstyp'] == typ].copy()
        if len(df_typ) == 0:
            continue

        pkg = models[typ]
        feature_names = pkg['feature_names']
        df_train_typ = df_train[df_train['bostadstyp'] ==
                                typ] if df_train is not None else pd.DataFrame()

        # Comps
        df_typ = compute_live_comps(df_typ, df_train, typ)

        # Bygg features (pkg skickas för v5-stöd)
        df_typ = build_features(df_typ, typ, feature_names, df_train_typ, pkg=pkg)

        # Prediktera
        X = df_typ[feature_names].copy()
        if pkg.get('scaler'):
            X = pkg['scaler'].transform(X)

        lgbm_preds = pkg['model'].predict(X)

        # v8 stack: 50/50 blend LightGBM + CatBoost om båda finns
        if pkg.get('model_catboost') is not None and pkg.get('model_ridge') is not None:
            cb_preds   = pkg['model_catboost'].predict(X)
            coefs      = pkg['model_ridge'].coef_  # [w_lgbm, w_cb]
            preds      = coefs[0] * lgbm_preds + coefs[1] * cb_preds
        else:
            preds = lgbm_preds

        if pkg.get('log_transform', True):
            preds = np.expm1(preds)

        df_typ = df_typ.copy()
        df_typ['estimerat_varde'] = preds.astype(int)
        df_typ['skillnad_kr'] = df_typ['estimerat_varde'] - \
            df_typ['utgangspris']
        df_typ['skillnad_pct'] = (
            (df_typ['estimerat_varde'] - df_typ['utgangspris']) /
            df_typ['estimerat_varde'] * 100
        ).round(1)

        # CI: v5+ med q10/q90 (datadrivna per bostad), äldre använder residual_std_log
        if 'model_q10' in pkg and 'model_q90' in pkg:
            X_raw = df_typ[feature_names].copy()
            q10_pred = pkg['model_q10'].predict(X_raw)
            q90_pred = pkg['model_q90'].predict(X_raw)
            df_typ['ci_low']  = np.expm1(q10_pred).astype(int)
            df_typ['ci_high'] = np.expm1(q90_pred).astype(int)
        else:
            std_log  = pkg.get('confidence', {}).get('residual_std_log', 0.15)
            log_pred = np.log1p(df_typ['estimerat_varde'].clip(lower=1))
            df_typ['ci_low']  = np.expm1(log_pred - 1.96 * std_log).astype(int)
            df_typ['ci_high'] = np.expm1(log_pred + 1.96 * std_log).astype(int)

        all_scored.append(df_typ)
        log.info(f'  {typ}: {len(df_typ)} prediktioner')

    if not all_scored:
        return pd.DataFrame()

    df_scored = pd.concat(all_scored, ignore_index=True)

    # Deal Score
    if compute_deal_scores_batch is not None:
        def _conf_dict(raw):
            if isinstance(raw, dict):
                return raw
            return {'interval_pct': 20}
        conf = {t: _conf_dict(models[t].get('confidence')) for t in models}
        df_scored = compute_deal_scores_batch(df_scored, df_train, conf)
        if 'deal_kategori' in df_scored.columns:
            for kat, n in df_scored['deal_kategori'].value_counts().items():
                log.info(f'    {kat}: {n}')

    df_scored['scrape_datum'] = datetime.now().strftime('%Y-%m-%d')
    return df_scored


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-details', action='store_true',
                        help='Hoppa över detaljscraping')
    args = parser.parse_args()

    start = time.time()
    log.info('=' * 60)
    log.info(
        f'DAGLIG UPPDATERING v2.2 {"(utan detaljer)" if args.skip_details else "(med detaljer)"}')
    log.info('=' * 60)

    log.info('Laddar modeller...')
    models = load_v2_models()
    if not models:
        log.error('Inga modeller!')
        return

    log.info('Laddar träningsdata...')
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    top_areas = df_train['omrade_clean'].value_counts().head(70).index.tolist()

    log.info('Steg 1: Scrapar listsidor...')
    df_raw = scrape_list_pages()
    log.info(f'  {len(df_raw)} annonser')
    if len(df_raw) == 0:
        log.warning('Inga annonser!')
        return

    if not args.skip_details:
        log.info('Steg 2: Scrapar detaljsidor...')
        df_raw = scrape_detail_pages(df_raw)

    log.info('Steg 3: Rensar och matchar...')
    df_clean = clean_and_match(df_raw, top_areas)
    log.info(f'  {len(df_clean)} efter rensning')

    log.info('Steg 4: ML-prediktion + Deal Score...')
    df_scored = predict_all(df_clean, models, df_train)

    if len(df_scored) == 0:
        log.warning('Inga prediktioner!')
        return

    # Spara
    save_cols = [c for c in [
        'url', 'omrade', 'bostadstyp', 'utgangspris', 'estimerat_varde',
        'skillnad_kr', 'skillnad_pct', 'ci_low', 'ci_high',
        'deal_score', 'deal_kategori', 'deal_ikon', 'underval_pct', 'deal_reasons',
        'sane_estimate', 'boarea_kvm', 'antal_rum', 'avgift_kr',
        'comps_pris_kvm_90d', 'comps_antal_90d', 'scrape_datum',
    ] if c in df_scored.columns]

    df_scored[save_cols].to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    log.info(f'Sparad: {OUTPUT_PATH}')

    hist = os.path.join(
        HISTORY_DIR, f'listings_{datetime.now().strftime("%Y%m%d")}.csv')
    df_scored[save_cols].to_csv(hist, index=False, encoding='utf-8-sig')

    try:
        msg = f'Auto-update {datetime.now().strftime("%Y-%m-%d %H:%M")}'
        for cmd in [
            ['git', '-C', PROJECT_DIR, 'add', '-f', OUTPUT_PATH],
            ['git', '-C', PROJECT_DIR, 'commit', '-m', msg],
            ['git', '-C', PROJECT_DIR, 'push'],
        ]:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                if 'nothing to commit' in r.stdout or 'nothing to commit' in r.stderr:
                    log.info('Git: inget nytt att commita')
                else:
                    log.warning(f'Git fel ({" ".join(cmd[-1:])}): {r.stderr.strip()}')
                    break
    except Exception as e:
        log.warning(f'Git push misslyckades: {e}')

    log.info('=' * 60)
    log.info(f'KLART! {len(df_scored)} annonser — {time.time()-start:.0f} sek')
    log.info('=' * 60)


if __name__ == '__main__':
    main()

