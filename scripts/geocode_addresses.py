"""
A1: Geocodning av gatuadresser — Örebro Housing ML
====================================================
Ersätter Hemnets felaktiga stadsdels-centroids med riktiga adressnivå-koordinater.

Problem: 100% av 356 omraden har exakt 1 unik koordinat (Hemnet lagrar centroid).
         810 villor delar (59.2753, 15.2134) — BallTree och avstånd är meningslösa.

Lösning: Geocoda gatuadress + omrade/kommun via Nominatim (OpenStreetMap, gratis).
         Fallback-kedja: exakt adress → adress utan nummer → centroid behålls.

Output:
  data/raw/geocode_cache.json          — cache (url → {lat, lon, quality})
  data/processed/orebro_housing_enriched_v4.csv — uppdaterat dataset

Tidsåtgång: ~45 min (2334 unika adresser × 1 req/sek Nominatim rate limit)
Tips: Kör --limit 50 för snabbtest, --merge-only om cache redan finns.

Kör:
    cd "orebro-housing-ml 3"
    python scripts/geocode_addresses.py              # Alla adresser
    python scripts/geocode_addresses.py --limit 50   # Test
    python scripts/geocode_addresses.py --merge-only # Bara merge (cache finns)
"""

import argparse
import json
import logging
import os
import sys
import time
import re
from datetime import datetime

import pandas as pd
import numpy as np
import requests

# ─────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH  = os.path.join(PROJECT_DIR, 'data', 'processed', 'orebro_housing_enriched_v3.csv')
CACHE_PATH  = os.path.join(PROJECT_DIR, 'data', 'raw', 'geocode_cache.json')
OUTPUT_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'orebro_housing_enriched_v4.csv')
LOG_PATH    = os.path.join(PROJECT_DIR, 'logs', 'geocode_addresses.log')

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
# ÖREBRO REGION — bounding box för validering
# ─────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 58.8, 59.6
LON_MIN, LON_MAX = 14.5, 16.0

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {
    "User-Agent": "ValuEstate-ML/1.0 orebro-housing-research (educational)"
}

# Kommuner inom Örebro-datasetet
KOMMUNER = [
    "Örebro",
    "Örebro kommun",
    "Hallsberg",
    "Kumla",
    "Lekeberg",
    "Lindesberg",
]


# ─────────────────────────────────────────────────────────────
# GEOCODING-FUNKTIONER
# ─────────────────────────────────────────────────────────────
def nominatim_query(query: str, delay=1.05) -> dict | None:
    """Kör en Nominatim-fråga. Returnerar {lat, lon} eller None."""
    time.sleep(delay)
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": 1,
                    "countrycodes": "se", "addressdetails": 1},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            display = data[0].get("display_name", "")
            return {"lat": lat, "lon": lon, "display": display}
    except Exception as e:
        log.warning(f"  Nominatim fel: {e} | query={query}")
    return None


def is_valid(lat: float, lon: float) -> bool:
    """Kontrollera att koordinaterna är inom Örebro-regionen."""
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


def clean_address(addr: str) -> str:
    """Rensa adress: ta bort lägenhetsbeteckning etc."""
    if not isinstance(addr, str):
        return ""
    addr = addr.strip()
    # Ta bort "lgh XXX" eller "Lgh XXX"
    addr = re.sub(r'\s+[Ll]gh\s+\S+', '', addr)
    # Ta bort "nb", "ov" suffix
    addr = re.sub(r'\s+(nb|ov|tr\s*\d+)\b', '', addr, flags=re.IGNORECASE)
    return addr.strip()


def strip_number(addr: str) -> str:
    """Ta bort husnummer: 'Rönnvägen 8' → 'Rönnvägen'."""
    parts = addr.rsplit(' ', 1)
    if len(parts) == 2 and re.match(r'^\d+[A-Za-z]?$', parts[1]):
        return parts[0]
    return addr


def geocode_address(gatuadress: str, omrade: str, cache: dict) -> dict:
    """
    Geocoda en adress med fallback-kedja:
    1. "Gatuadress, Omrade, Örebro, Sverige"  (mest specifik)
    2. "Gatuadress, Örebro, Sverige"
    3. "Gatuadress utan nummer, Örebro, Sverige"
    4. Behåll gammal centroid (fallback)
    """
    addr_clean = clean_address(gatuadress)
    cache_key  = f"{addr_clean}|{omrade}"

    if cache_key in cache and not cache[cache_key].get("error"):
        return cache[cache_key]

    # Bygg Örebro-kontextstring från omrade
    omrade_clean = re.sub(r'^Villa\s+', '', str(omrade)).strip() if omrade else ""

    queries = [
        f"{addr_clean}, {omrade_clean}, Örebro, Sverige",
        f"{addr_clean}, Örebro, Sverige",
        f"{addr_clean}, Sverige",
        f"{strip_number(addr_clean)}, {omrade_clean}, Örebro, Sverige",
    ]

    for i, q in enumerate(queries):
        result = nominatim_query(q)
        if result and is_valid(result["lat"], result["lon"]):
            result["quality"] = ["exact", "city", "country", "street_only"][i]
            result["query"]   = q
            cache[cache_key]  = result
            return result

    # Alla queries misslyckades
    fallback = {"error": "not_found", "query": queries[0]}
    cache[cache_key] = fallback
    return fallback


# ─────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────
def save_cache(cache: dict):
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=None)


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding='utf-8') as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────
# HUVUD-GEOCODNING
# ─────────────────────────────────────────────────────────────
def geocode_all(df, cache, limit=None):
    """Geocoda alla unika (gatuadress, omrade)-kombinationer."""
    pairs = (
        df[df['gatuadress'].notna()][['gatuadress', 'omrade']]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if limit:
        pairs = pairs.head(limit)

    # Filtrera bort redan cachade
    need = []
    for _, row in pairs.iterrows():
        key = f"{clean_address(row['gatuadress'])}|{row['omrade']}"
        if key not in cache or cache[key].get("error"):
            need.append(row)
    log.info(f"  Totalt: {len(pairs)} | Cachade: {len(pairs)-len(need)} | Geocodar: {len(need)}")

    if not need:
        log.info("  Allt redan cachat!")
        return cache

    ok = err = 0
    quality_counts = {}
    checkpoint = 50

    for i, row in enumerate(need, 1):
        result = geocode_address(row['gatuadress'], row['omrade'], cache)
        if result.get("error"):
            err += 1
        else:
            ok += 1
            q = result.get("quality", "unknown")
            quality_counts[q] = quality_counts.get(q, 0) + 1

        if i % checkpoint == 0:
            save_cache(cache)
            log.info(f"  [{i}/{len(need)}] OK={ok} Fel={err} | "
                     f"Kvalitet: {quality_counts}")

    save_cache(cache)
    log.info(f"\nGeocoding klar: {ok} OK, {err} fel")
    log.info(f"Kvalitetsfördelning: {quality_counts}")
    return cache


# ─────────────────────────────────────────────────────────────
# MERGE — uppdatera koordinater i CSV
# ─────────────────────────────────────────────────────────────
def merge_coordinates(df, cache):
    """
    Ersätt Hemnet-centroider med riktiga adress-koordinater.
    Behåller gamla koordinater som fallback om geocoding misslyckades.
    """
    log.info("\nMergar geocodade koordinater...")
    df = df.copy()

    df['latitude_orig']  = df['latitude']
    df['longitude_orig'] = df['longitude']
    df['geocode_quality'] = 'hemnet_centroid'

    updated = 0
    kept_centroid = 0
    missing_addr  = 0

    for idx, row in df.iterrows():
        addr = clean_address(str(row.get('gatuadress', '')))
        omr  = str(row.get('omrade', ''))

        if not addr:
            missing_addr += 1
            continue

        key = f"{addr}|{omr}"
        result = cache.get(key)

        if result and not result.get("error") and is_valid(result["lat"], result["lon"]):
            df.at[idx, 'latitude']        = result["lat"]
            df.at[idx, 'longitude']       = result["lon"]
            df.at[idx, 'geocode_quality'] = result.get("quality", "geocoded")
            updated += 1
        else:
            kept_centroid += 1

    log.info(f"  Uppdaterade koordinater: {updated}")
    log.info(f"  Behöll centroid (fallback): {kept_centroid}")
    log.info(f"  Saknar adress: {missing_addr}")

    # Rapport: unika koordinater före/efter
    unique_before = len(df[['latitude_orig', 'longitude_orig']].drop_duplicates())
    unique_after  = len(df[['latitude', 'longitude']].drop_duplicates())
    log.info(f"\n  Unika koordinatpar FÖRE: {unique_before}")
    log.info(f"  Unika koordinatpar EFTER: {unique_after}")

    # Kvalitetsfördelning
    log.info(f"\n  geocode_quality:")
    log.info(f"  {df['geocode_quality'].value_counts().to_dict()}")

    # Kontrollera avstånd före/efter (signifikant förändring?)
    changed = df['latitude'] != df['latitude_orig']
    if changed.any():
        lat_delta = (df.loc[changed, 'latitude'] - df.loc[changed, 'latitude_orig']).abs()
        lon_delta = (df.loc[changed, 'longitude'] - df.loc[changed, 'longitude_orig']).abs()
        log.info(f"\n  Koordinatförändring (uppdaterade rader):")
        log.info(f"    Δlat median: {lat_delta.median()*111:.1f} m, max: {lat_delta.max()*111:.0f} m")
        log.info(f"    Δlon median: {lon_delta.median()*70:.1f} m, max: {lon_delta.max()*70:.0f} m")

    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Geocodar adresser via Nominatim')
    parser.add_argument('--limit',      type=int, default=None)
    parser.add_argument('--merge-only', action='store_true')
    args = parser.parse_args()

    log.info('=' * 60)
    log.info('GEOCODNING A1 — ADRESSNIVÅ-KOORDINATER')
    log.info(f'  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log.info('=' * 60)

    df = pd.read_csv(INPUT_PATH)
    log.info(f'Laddat: {INPUT_PATH} ({len(df)} rader)')

    cache = load_cache()
    log.info(f'Cache: {len(cache)} poster')

    if not args.merge_only:
        # Geocoda bara villor (de enda med centroid-problem)
        df_villor = df[df['bostadstyp'] == 'villor']
        cache = geocode_all(df_villor, cache, limit=args.limit)

    df_out = merge_coordinates(df, cache)
    df_out.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    log.info(f'\n✅ Sparad: {OUTPUT_PATH}')
    log.info(f'   Rader: {len(df_out)}, Kolumner: {len(df_out.columns)}')
    log.info('\nNästa steg:')
    log.info('  python scripts/train_villa_v9.py')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
