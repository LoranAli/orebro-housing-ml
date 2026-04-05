"""
ValuEstate — Live URL-analys
============================
Hämtar en Hemnet-annons via URL (requests, ingen Selenium),
extraherar features och kör ML-modellen.

Används av dashboard/app.py för live prediktion per annons.
"""

import re
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Gör daily_update-funktioner tillgängliga
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'sv-SE,sv;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Cache-Control': 'no-cache',
}


# ─────────────────────────────────────────────────────────────
# 1. HÄMTA SIDA
# ─────────────────────────────────────────────────────────────
def fetch_page(url: str) -> str | None:
    """Hämtar HTML-källkod med requests. Returnerar None vid fel."""
    try:
        import requests
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────
# 2. EXTRAHERA GRUNDDATA FRÅN URL + SIDA
# ─────────────────────────────────────────────────────────────
URL_TO_AREA = {
    'adolfsberg': 'Adolfsberg', 'almby': 'Almby', 'baronbackarna': 'Baronbackarna',
    'brickebacken': 'Brickebacken', 'bsta': 'Bsta', 'charlottenborg': 'Charlottenborg',
    'ekeby': 'Ekeby', 'ekebyhov': 'Ekebyhov', 'engelbrekt': 'Engelbrekt',
    'garphyttan': 'Garphyttan', 'glanshammar': 'Glanshammar', 'hallsberg': 'Hallsberg',
    'hjalmaren': 'Hjälmaren', 'hjulsjö': 'Hjulsjö', 'hovsta': 'Hovsta',
    'hällabrottet': 'Hällabrottet', 'karlslund': 'Karlslund', 'kristinehamn': 'Kristinehamn',
    'kumla': 'Kumla', 'lekeberg': 'Lekeberg', 'lindesberg': 'Lindesberg',
    'lundby': 'Lundby', 'marieberg': 'Marieberg', 'mellringe': 'Mellringe',
    'nabbelund': 'Nabbelund', 'navesta': 'Navesta', 'norra-orebro': 'Norra Örebro',
    'norrby': 'Norrby', 'odensbacken': 'Odensbacken', 'oxhagen': 'Oxhagen',
    'pålsboda': 'Pålsboda', 'rosta': 'Rosta', 'rynninge': 'Rynninge',
    'sörbyangen': 'Sörbyängen', 'sörby': 'Sörby', 'sörbyängen': 'Sörbyängen',
    'tybble': 'Tybble', 'varberga': 'Varberga', 'vasastan': 'Vasastan',
    'vaster': 'Väster', 'vivalla': 'Vivalla', 'varmsätra': 'Värmsätra',
    'orebro': 'Örebro', 'örebro': 'Örebro',
}

BOSTADSTYP_MAP = {
    'villa': 'villor',
    'lagenhet': 'lagenheter',
    'lägenhet': 'lagenheter',
    'radhus': 'radhus',
    'kedjehus': 'radhus',
    'parhus': 'radhus',
}


def parse_url_metadata(url: str) -> dict:
    """Extrahera bostadstyp och omrade ur Hemnet-URL."""
    slug = url.rstrip('/').split('/')[-1]  # ex: villa-4rum-adolfsberg-orebro-id12345678

    # Bostadstyp
    typ = 'villor'
    for key, val in BOSTADSTYP_MAP.items():
        if slug.startswith(key):
            typ = val
            break

    # Område: ta bort prefix (villa-4rum-) och suffix (-orebro-id...)
    area_slug = re.sub(r'^[a-z]+-\d+rum-', '', slug)
    area_slug = re.sub(r'-orebro.*$', '', area_slug)
    area_slug = re.sub(r'-id\d+$', '', area_slug)

    omrade = 'Örebro'
    for key, val in URL_TO_AREA.items():
        if key in area_slug:
            omrade = val
            break

    return {'bostadstyp': typ, 'omrade': omrade, 'url': url}


def parse_listing_from_html(html: str, meta: dict) -> dict:
    """Extrahera alla features ur Hemnet-sida."""
    from bs4 import BeautifulSoup
    # Återanvänd parse_detail_page från daily_update
    from daily_update import parse_detail_page

    details = parse_detail_page(html)
    details.update(meta)

    soup = BeautifulSoup(html, 'lxml')
    text = soup.get_text(' ', strip=True)

    # Utgångspris
    m = re.search(r'([\d\s\xa0]{4,})\s*kr', text.replace('\u00a0', ' '))
    if m:
        try:
            details.setdefault('utgangspris',
                               int(m.group(1).replace(' ', '').replace('\xa0', '')))
        except ValueError:
            pass

    # Boarea
    m = re.search(r'([\d,\.]+)\s*m[²2]', text)
    if m:
        try:
            details.setdefault('boarea_kvm', float(m.group(1).replace(',', '.')))
        except ValueError:
            pass

    # Antal rum
    m = re.search(r'(\d+[\.,]?\d*)\s*rum', text)
    if m:
        try:
            details.setdefault('antal_rum', float(m.group(1).replace(',', '.')))
        except ValueError:
            pass

    # Tomtarea (villor)
    m = re.search(r'Tomtarea[^\d]*([\d\s\xa0]+)\s*m[²2]', text)
    if m:
        try:
            details.setdefault('tomtarea_kvm',
                               int(m.group(1).replace(' ', '').replace('\xa0', '')))
        except ValueError:
            pass

    # Driftkostnad
    m = re.search(r'Driftkostnad[^\d]*([\d\s\xa0]+)\s*kr', text)
    if m:
        try:
            details.setdefault('driftkostnad_ar',
                               int(m.group(1).replace(' ', '').replace('\xa0', '')))
        except ValueError:
            pass

    # Avgift (lgh/radhus)
    m = re.search(r'Avgift[^\d]*([\d\s\xa0]+)\s*kr', text)
    if m:
        try:
            details.setdefault('avgift_kr',
                               int(m.group(1).replace(' ', '').replace('\xa0', '')))
        except ValueError:
            pass

    # Antal besök
    m = re.search(r'(\d+)\s+(?:visningar|intressenter|spekulanter)', text.lower())
    if m:
        details.setdefault('antal_besok', int(m.group(1)))

    # Gatuadress (från sida-titel eller URL)
    m = re.search(r'<title>([^<]+)</title>', html)
    if m:
        title = m.group(1)
        addr_m = re.match(r'^([A-ZÅÄÖ][a-zåäö]+(?:\s+\d+[A-Za-z]?)?)', title)
        if addr_m:
            details.setdefault('gatuadress', addr_m.group(1))

    return details


# ─────────────────────────────────────────────────────────────
# 3. HUVUDFUNKTION: URL → PREDIKTION
# ─────────────────────────────────────────────────────────────
def analyze_url(url: str, models: dict, df_train: pd.DataFrame) -> dict:
    """
    Komplett pipeline: URL → scraping → features → ML → resultat.

    Returnerar dict med:
      ok: bool
      error: str (om ok=False)
      listing: dict (scrapad data)
      estimat: int
      ci_low: int
      ci_high: int
      underval_pct: float
      underval_kr: int
      deal_score: int
      deal_kategori: str
      bostadstyp: str
      omrade: str
    """
    from daily_update import build_features

    # Validera URL
    if not url or 'hemnet.se' not in url:
        return {'ok': False, 'error': 'Ogiltig URL — måste vara en hemnet.se-länk.'}
    if '/bostad/' not in url:
        return {'ok': False, 'error': 'Länken måste peka på en specifik annons (/bostad/...).'}

    # Metadata från URL
    meta = parse_url_metadata(url)
    typ = meta['bostadstyp']

    # Hämta HTML
    html = fetch_page(url)
    if not html:
        return {'ok': False, 'error': 'Kunde inte hämta sidan. Kontrollera URL och nätverksanslutning.'}

    # Parsa features
    listing = parse_listing_from_html(html, meta)

    # Kontrollera minimikrav
    if not listing.get('boarea_kvm') or not listing.get('antal_rum'):
        return {'ok': False,
                'error': 'Kunde inte extrahera boarea eller antal rum från sidan. '
                         'Hemnet kan ha blockerat förfrågan — prova igen om en stund.'}

    # Sätt defaults
    listing.setdefault('utgangspris', 0)
    listing.setdefault('tomtarea_kvm', 500 if typ == 'villor' else 0)
    listing.setdefault('driftkostnad_ar', 30000 if typ == 'villor' else 0)
    listing.setdefault('avgift_kr', 0 if typ == 'villor' else 3000)
    listing.setdefault('biarea_kvm', 0)
    listing.setdefault('antal_besok', 0)
    listing.setdefault('byggar', 1970)
    listing.setdefault('har_balkong', 0)
    listing.setdefault('har_uteplats', 0)
    listing.setdefault('har_garage', 0)
    listing.setdefault('har_hiss', 0)
    listing.setdefault('har_kallare', 0)
    listing.setdefault('renoverad', 0)

    # Bygg DataFrame
    df_row = pd.DataFrame([listing])

    # Modell
    if typ not in models:
        return {'ok': False, 'error': f'Ingen modell laddad för {typ}.'}

    pkg = models[typ]
    feature_names = pkg.get('feature_names', [])
    df_train_typ = df_train[df_train['bostadstyp'] == typ].copy() if df_train is not None else None

    try:
        df_feat = build_features(df_row, typ, feature_names, df_train_typ, pkg=pkg)
    except Exception as e:
        return {'ok': False, 'error': f'Feature-byggnad misslyckades: {e}'}

    # Prediktera
    X = df_feat.reindex(columns=feature_names).fillna(0).values
    try:
        lgbm = pkg.get('model_lgbm')
        cb   = pkg.get('model_catboost')
        w    = pkg.get('blend_weights', {'lgbm': 0.5, 'cb': 0.5})

        if lgbm and cb:
            pred_lgbm = float(lgbm.predict(X)[0])
            pred_cb   = float(cb.predict(X)[0])
            estimat   = int(w.get('lgbm', 0.5) * pred_lgbm + w.get('cb', 0.5) * pred_cb)
        elif lgbm:
            estimat = int(float(lgbm.predict(X)[0]))
        else:
            estimat = int(float(pkg['model'].predict(X)[0]))
    except Exception as e:
        return {'ok': False, 'error': f'Prediktion misslyckades: {e}'}

    # Konfidensintervall
    ci_pct = 15.0
    try:
        q10 = pkg.get('model_q10')
        q90 = pkg.get('model_q90')
        if q10 and q90:
            ci_low  = int(float(q10.predict(X)[0]))
            ci_high = int(float(q90.predict(X)[0]))
        else:
            ci_low  = int(estimat * (1 - ci_pct / 100))
            ci_high = int(estimat * (1 + ci_pct / 100))
        ci_pct = round((ci_high - ci_low) / estimat * 50, 1)
    except Exception:
        ci_low  = int(estimat * 0.85)
        ci_high = int(estimat * 1.15)

    # Undervärdering
    utgpris = listing.get('utgangspris', 0)
    if utgpris and utgpris > 100_000:
        underval_kr  = estimat - utgpris
        underval_pct = round((estimat - utgpris) / utgpris * 100, 1)
    else:
        underval_kr  = 0
        underval_pct = 0.0

    # Deal Score (förenklad)
    deal_score, deal_kategori = _simple_deal_score(underval_pct, ci_pct, typ)

    return {
        'ok':           True,
        'listing':      listing,
        'estimat':      estimat,
        'ci_low':       ci_low,
        'ci_high':      ci_high,
        'ci_pct':       ci_pct,
        'underval_kr':  underval_kr,
        'underval_pct': underval_pct,
        'deal_score':   deal_score,
        'deal_kategori': deal_kategori,
        'bostadstyp':   typ,
        'omrade':       meta['omrade'],
    }


def _simple_deal_score(underval_pct: float, ci_pct: float, typ: str) -> tuple[int, str]:
    """Enkel deal score baserad på undervärdering och konfidensintervall."""
    conf_factor = max(0.5, 1.0 - (ci_pct - 10) / 100)

    if underval_pct >= 20:
        raw = 75 + min(25, (underval_pct - 20) * 1.5)
    elif underval_pct >= 10:
        raw = 55 + (underval_pct - 10) * 2
    elif underval_pct >= 5:
        raw = 40 + (underval_pct - 5) * 3
    elif underval_pct >= 0:
        raw = 25 + underval_pct * 3
    else:
        raw = max(1, 25 + underval_pct * 1.5)

    score = int(round(raw * conf_factor))
    score = max(1, min(100, score))

    if score >= 70:
        return score, 'Exceptionellt fynd'
    elif score >= 55:
        return score, 'Bra fynd'
    elif score >= 40:
        return score, 'Potentiellt intressant'
    else:
        return score, 'Rimligt pris'
