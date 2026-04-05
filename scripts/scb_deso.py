"""
Fas C2: SCB DeSO — Socioekonomiska grannskapsfeatures
======================================================
Kopplar varje geocodad villa till ett DeSO-område (Demografiskt
Statistikområde) och hämtar:

  deso_median_ink_tkr   — Medelinkomst netto per person (tkr/år), 2024
  deso_lon_ink_tkr      — Medelinkomst lön per person (tkr/år), 2024
  deso_andel_lon_pct    — Andel med löneinkomst (%), 2024
  deso_befolkning       — Antal invånare i DeSO-området, 2024
  deso_median_alder     — Uppskattad medianålder (viktad mid-ålder), 2024
  deso_andel_0_19       — Andel barn/unga 0–19 år (%), 2024
  deso_andel_65_plus    — Andel äldre 65+ år (%), 2024

Data:
  Gränser:    SCB WFS (geodata.scb.se) — DeSO_2025
  Inkomst:    SCB API Tab2InkDesoRegso (HE0110I)
  Befolkning: SCB API FolkmDesoAldKon (BE0101Y)

Output:
  data/processed/orebro_housing_enriched_v5.csv

Kör:
    cd "orebro-housing-ml 3"
    python scripts/scb_deso.py
    python scripts/scb_deso.py --skip-download  # Om cache finns
"""

import os
import sys
import json
import time
import logging
import argparse

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import Point

# ─────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH  = os.path.join(PROJECT_DIR, 'data', 'processed', 'orebro_housing_enriched_v4.csv')
OUTPUT_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'orebro_housing_enriched_v5.csv')
CACHE_DIR   = os.path.join(PROJECT_DIR, 'data', 'raw', 'scb_deso')
LOG_PATH    = os.path.join(PROJECT_DIR, 'logs', 'scb_deso.log')

os.makedirs(CACHE_DIR, exist_ok=True)
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

SCB_API   = 'https://api.scb.se/OV0104/v1/doris/sv/ssd'
SCB_WFS   = 'https://geodata.scb.se/geoserver/stat/ows'
LAN_KOD   = '18'   # Örebro län
STAT_YEAR = '2024' # Senaste tillgängliga år

DESO_GEO_CACHE  = os.path.join(CACHE_DIR, 'deso_orebro.geojson')
DESO_INK_CACHE  = os.path.join(CACHE_DIR, 'deso_inkomst.json')
DESO_BEF_CACHE  = os.path.join(CACHE_DIR, 'deso_befolkning.json')


# ─────────────────────────────────────────────────────────────
# 1. LADDA DeSO-GRÄNSER VIA WFS
# ─────────────────────────────────────────────────────────────
def download_deso_boundaries(force=False):
    if not force and os.path.exists(DESO_GEO_CACHE):
        log.info(f'  Läser cachade DeSO-gränser: {DESO_GEO_CACHE}')
        return gpd.read_file(DESO_GEO_CACHE)

    log.info('  Hämtar DeSO-gränser från SCB WFS (Örebro län)...')
    # WFS 1.0.0, filtrera på länskod=18, hämta alla (paginering om nödvändigt)
    all_features = []
    start = 0
    page_size = 500

    while True:
        url = (f'{SCB_WFS}?service=WFS&version=1.0.0&request=GetFeature'
               f'&typeName=stat:DeSO_2025&outputFormat=application/json'
               f'&maxFeatures={page_size}&startIndex={start}')
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        feats = data.get('features', [])
        if not feats:
            break
        # Filtrera Örebro
        orebro = [f for f in feats if f['properties'].get('lanskod') == LAN_KOD]
        all_features.extend(orebro)
        log.info(f'    Sida {start//page_size + 1}: {len(feats)} features, '
                 f'{len(orebro)} Örebro')
        if len(feats) < page_size:
            break
        start += page_size
        time.sleep(0.5)

    gdf = gpd.GeoDataFrame.from_features(all_features, crs='EPSG:4326')
    gdf.to_file(DESO_GEO_CACHE, driver='GeoJSON')
    log.info(f'  Sparade {len(gdf)} DeSO-områden → {DESO_GEO_CACHE}')
    return gdf


# ─────────────────────────────────────────────────────────────
# 2. LADDA INKOMSTDATA FRÅN SCB API
# ─────────────────────────────────────────────────────────────
def _scb_post(table_path, query_body):
    url = f'{SCB_API}/{table_path}'
    resp = requests.post(url, json=query_body, timeout=60)
    resp.raise_for_status()
    return resp.json()


def download_income_data(deso_codes, force=False):
    """
    Hämtar medelinkomst netto + löneinkomst + andel med lön per DeSO.
    SCB-tabell: HE0110I/Tab2InkDesoRegso
    """
    if not force and os.path.exists(DESO_INK_CACHE):
        log.info(f'  Läser cachad inkomstdata')
        with open(DESO_INK_CACHE, encoding='utf-8') as f:
            return json.load(f)

    log.info(f'  Hämtar inkomstdata för {len(deso_codes)} DeSO-områden...')

    # API-koder: 240=nettoinkomst, 10=löneinkomst
    # ContentsCode: 000008A4=medelvärde tkr, 000008A2=andel%
    query = {
        'query': [
            {'code': 'Region',
             'selection': {'filter': 'item', 'values': deso_codes}},
            {'code': 'Inkomstkomponenter',
             'selection': {'filter': 'item', 'values': ['240', '10']}},
            {'code': 'Kon',
             'selection': {'filter': 'item', 'values': ['1+2']}},
            {'code': 'ContentsCode',
             'selection': {'filter': 'item', 'values': ['000008A4', '000008A2']}},
            {'code': 'Tid',
             'selection': {'filter': 'item', 'values': [STAT_YEAR]}},
        ],
        'response': {'format': 'json'},
    }
    raw = _scb_post('START/HE/HE0110/HE0110I/Tab2InkDesoRegso', query)

    # Strukturera: {deso_code: {nettoinkomst_tkr, lon_tkr, andel_lon_pct}}
    # API returnerar: key=[Region, Inkomstkomponenter, Kon, Tid], values=[medelvärde_tkr, andel_pct]
    result = {}
    for row in raw.get('data', []):
        region_raw = row['key'][0]          # '1814A0010_DeSO2025'
        deso_bare  = region_raw.split('_')[0]
        komponent  = row['key'][1]          # '10' (lön) eller '240' (netto)

        def _float(v):
            try:
                return float(v) if v not in ('', '..') else np.nan
            except (ValueError, TypeError):
                return np.nan

        val_medel = _float(row['values'][0])   # 000008A4: medelvärde tkr
        val_andel = _float(row['values'][1]) if len(row['values']) > 1 else np.nan  # 000008A2: andel %

        if deso_bare not in result:
            result[deso_bare] = {}

        if komponent == '240':
            result[deso_bare]['nettoinkomst_tkr'] = val_medel
        elif komponent == '10':
            result[deso_bare]['lon_tkr']      = val_medel
            result[deso_bare]['andel_lon_pct'] = val_andel

    with open(DESO_INK_CACHE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    log.info(f'  Inkomstdata: {len(result)} DeSO-områden')
    return result


# ─────────────────────────────────────────────────────────────
# 3. LADDA BEFOLKNINGSDATA (ålder) FRÅN SCB API
# ─────────────────────────────────────────────────────────────
def download_population_data(deso_codes, force=False):
    """
    Hämtar befolkning och åldersfördelning per DeSO.
    SCB-tabell: BE0101Y/FolkmDesoAldKon
    Åldersgrupper: totalt, 0-4, 5-9, ..., 85+
    """
    if not force and os.path.exists(DESO_BEF_CACHE):
        log.info(f'  Läser cachad befolkningsdata')
        with open(DESO_BEF_CACHE, encoding='utf-8') as f:
            return json.load(f)

    log.info(f'  Hämtar befolkningsdata för {len(deso_codes)} DeSO-områden...')

    query = {
        'query': [
            {'code': 'Region',
             'selection': {'filter': 'item', 'values': deso_codes}},
            {'code': 'Alder',
             'selection': {'filter': 'item',
                           'values': ['totalt', '-4', '5-9', '10-14', '15-19',
                                      '20-24', '25-29', '30-34', '35-39',
                                      '40-44', '45-49', '50-54', '55-59',
                                      '60-64', '65-69', '70-74', '75-79',
                                      '80-']}},
            {'code': 'Kon',
             'selection': {'filter': 'item', 'values': ['1+2']}},
            {'code': 'Tid',
             'selection': {'filter': 'item', 'values': [STAT_YEAR]}},
        ],
        'response': {'format': 'json'},
    }
    raw = _scb_post('START/BE/BE0101/BE0101Y/FolkmDesoAldKon', query)

    # Åldersgruppernas mittvärden för medianuppskattning
    AGE_MID = {
        'totalt': None,
        '-4': 2,    '5-9': 7,    '10-14': 12, '15-19': 17,
        '20-24': 22, '25-29': 27, '30-34': 32, '35-39': 37,
        '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57,
        '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77,
        '80-': 85,
    }

    # Bygg: {deso: {totalt, age_groups: {alder: antal}}}
    raw_data = {}
    for row in raw.get('data', []):
        region_raw = row['key'][0]
        deso_bare  = region_raw.split('_')[0]
        alder      = row['key'][1]
        try:
            val = int(float(row['values'][0])) if row['values'][0] not in ('', '..') else 0
        except (ValueError, IndexError):
            val = 0

        if deso_bare not in raw_data:
            raw_data[deso_bare] = {'totalt': 0, 'age_groups': {}}

        if alder == 'totalt':
            raw_data[deso_bare]['totalt'] = val
        else:
            raw_data[deso_bare]['age_groups'][alder] = val

    # Beräkna medianålder (viktad) och åldersandelar
    result = {}
    for deso, dat in raw_data.items():
        tot    = dat['totalt'] or 1
        groups = dat['age_groups']

        # Andel 0-19 och 65+
        young  = sum(groups.get(a, 0) for a in ['-4','5-9','10-14','15-19'])
        old    = sum(groups.get(a, 0) for a in ['65-69','70-74','75-79','80-'])
        andel_ung  = round(young / tot * 100, 1)
        andel_ald  = round(old   / tot * 100, 1)

        # Uppskattad medianålder (viktad midpoint)
        weighted_sum = sum(groups.get(a, 0) * mid
                           for a, mid in AGE_MID.items()
                           if mid is not None and a in groups)
        median_alder = round(weighted_sum / tot, 1) if tot > 0 else np.nan

        result[deso] = {
            'befolkning':    tot,
            'median_alder':  median_alder,
            'andel_0_19':    andel_ung,
            'andel_65_plus': andel_ald,
        }

    with open(DESO_BEF_CACHE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    log.info(f'  Befolkningsdata: {len(result)} DeSO-områden')
    return result


# ─────────────────────────────────────────────────────────────
# 4. SPATIAL JOIN: koordinater → DeSO-kod
# ─────────────────────────────────────────────────────────────
def spatial_join(df, gdf_deso):
    """
    Punkt-i-polygon: matchar varje villa (lat/lon) mot ett DeSO-område.
    Kräver geopandas + shapely.
    """
    log.info('\nSpatial join: koordinater → DeSO-kod...')
    has_coords = df['latitude'].notna() & df['longitude'].notna()
    df_geo = df[has_coords].copy()

    geometry = [Point(lon, lat) for lat, lon in
                zip(df_geo['latitude'], df_geo['longitude'])]
    gdf_pts = gpd.GeoDataFrame(df_geo[['latitude', 'longitude']],
                                geometry=geometry, crs='EPSG:4326')

    # Säkerställ samma CRS
    gdf_deso_proj = gdf_deso.to_crs('EPSG:4326')

    # Spatial join (punkt inom polygon)
    joined = gpd.sjoin(gdf_pts, gdf_deso_proj[['desokod', 'geometry']],
                       how='left', predicate='within')

    df['deso_kod'] = np.nan
    df.loc[has_coords, 'deso_kod'] = joined['desokod'].values

    matched   = df['deso_kod'].notna().sum()
    unmatched = has_coords.sum() - matched
    log.info(f'  Matchade: {matched}/{has_coords.sum()} '
             f'({matched/has_coords.sum()*100:.0f}%)')
    if unmatched > 0:
        # Fallback för punkter precis utanför polygon (nearest)
        log.info(f'  {unmatched} punkter utanför polygon → nearest-fallback')
        missing_mask = has_coords & df['deso_kod'].isna()
        df_miss = df[missing_mask].copy()
        gdf_miss = gpd.GeoDataFrame(
            df_miss,
            geometry=[Point(lon, lat) for lat, lon in
                      zip(df_miss['latitude'], df_miss['longitude'])],
            crs='EPSG:4326',
        )
        joined_nn = gpd.sjoin_nearest(gdf_miss, gdf_deso_proj[['desokod', 'geometry']],
                                       how='left', max_distance=0.05)
        df.loc[missing_mask, 'deso_kod'] = joined_nn['desokod'].values

    final_matched = df['deso_kod'].notna().sum()
    log.info(f'  Slutlig match: {final_matched}/{len(df)} '
             f'({final_matched/len(df)*100:.0f}%)')
    return df


# ─────────────────────────────────────────────────────────────
# 5. MERGA STATISTIK
# ─────────────────────────────────────────────────────────────
def merge_deso_stats(df, income_data, pop_data):
    log.info('\nMergar DeSO-statistik...')

    # Fallback-värden (riksmedel om DeSO saknas)
    all_ink   = [v.get('nettoinkomst_tkr', np.nan) for v in income_data.values()
                 if not np.isnan(v.get('nettoinkomst_tkr', np.nan))]
    all_lon   = [v.get('lon_tkr', np.nan) for v in income_data.values()
                 if not np.isnan(v.get('lon_tkr', np.nan))]
    all_alon  = [v.get('andel_lon_pct', np.nan) for v in income_data.values()
                 if not np.isnan(v.get('andel_lon_pct', np.nan))]
    all_bef   = [v.get('befolkning', 0) for v in pop_data.values()]
    all_alder = [v.get('median_alder', np.nan) for v in pop_data.values()
                 if not np.isnan(v.get('median_alder', np.nan))]

    fb_ink  = float(np.median(all_ink))  if all_ink  else 250.0
    fb_lon  = float(np.median(all_lon))  if all_lon  else 200.0
    fb_alon = float(np.median(all_alon)) if all_alon else 70.0
    fb_bef  = float(np.median(all_bef))  if all_bef  else 1000.0
    fb_ald  = float(np.median(all_alder)) if all_alder else 42.0

    log.info(f'  Riksmedian nettoinkomst: {fb_ink:.0f} tkr | lön: {fb_lon:.0f} tkr')
    log.info(f'  Riksmedian medianålder: {fb_ald:.1f} år | befolkning: {fb_bef:.0f}')

    def get_ink(code, key, fallback):
        if pd.isna(code): return fallback
        v = income_data.get(str(code), {}).get(key, np.nan)
        return v if not (isinstance(v, float) and np.isnan(v)) else fallback

    def get_pop(code, key, fallback):
        if pd.isna(code): return fallback
        v = pop_data.get(str(code), {}).get(key, np.nan)
        return v if not (isinstance(v, float) and np.isnan(v)) else fallback

    df['deso_median_ink_tkr']  = df['deso_kod'].apply(lambda c: get_ink(c, 'nettoinkomst_tkr', fb_ink))
    df['deso_lon_ink_tkr']     = df['deso_kod'].apply(lambda c: get_ink(c, 'lon_tkr',          fb_lon))
    df['deso_andel_lon_pct']   = df['deso_kod'].apply(lambda c: get_ink(c, 'andel_lon_pct',    fb_alon))
    df['deso_befolkning']      = df['deso_kod'].apply(lambda c: get_pop(c, 'befolkning',       fb_bef))
    df['deso_median_alder']    = df['deso_kod'].apply(lambda c: get_pop(c, 'median_alder',     fb_ald))
    df['deso_andel_0_19']      = df['deso_kod'].apply(lambda c: get_pop(c, 'andel_0_19',       20.0))
    df['deso_andel_65_plus']   = df['deso_kod'].apply(lambda c: get_pop(c, 'andel_65_plus',    20.0))

    # Rapport
    for feat in ['deso_median_ink_tkr', 'deso_lon_ink_tkr', 'deso_andel_lon_pct',
                 'deso_befolkning', 'deso_median_alder']:
        s = df[feat]
        log.info(f'  {feat}: {s.min():.1f} – {s.max():.1f} (median {s.median():.1f})')

    # Korrelation med slutpris (villor)
    vill = df[df['bostadstyp'] == 'villor']
    for feat in ['deso_median_ink_tkr', 'deso_lon_ink_tkr']:
        corr = vill[feat].corr(vill['slutpris'])
        log.info(f'  Korrelation {feat} × slutpris: {corr:.3f}')

    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-download', action='store_true',
                        help='Använd cachad data (hoppar WFS/API-anrop)')
    parser.add_argument('--force', action='store_true',
                        help='Tvinga ny nedladdning även om cache finns')
    args = parser.parse_args()

    log.info('=' * 60)
    log.info('FAS C2: SCB DeSO — SOCIOEKONOMISKA GRANNSKAPSFEATURES')
    log.info('=' * 60)

    # Ladda bostadsdata
    df = pd.read_csv(INPUT_PATH)
    log.info(f'Laddat: {len(df)} rader, {df.shape[1]} kolumner')
    villor = df[df['bostadstyp'] == 'villor']
    has_coords = villor['latitude'].notna() & villor['longitude'].notna()
    log.info(f'Villor med koordinater: {has_coords.sum()}/{len(villor)}')

    force = args.force and not args.skip_download

    # 1. DeSO-gränser
    log.info('\n--- Steg 1: DeSO-gränser ---')
    gdf = download_deso_boundaries(force=force)
    log.info(f'  DeSO-områden Örebro: {len(gdf)}')

    # 2. Hitta alla Örebro DeSO-koder för API-anrop
    deso_codes_api = [f'{code}_DeSO2025' for code in gdf['desokod'].unique()]
    log.info(f'  API-koder: {len(deso_codes_api)} DeSO-områden')

    # 3. Inkomstdata
    log.info('\n--- Steg 2: Inkomstdata (SCB API) ---')
    income_data = download_income_data(deso_codes_api, force=force)

    # 4. Befolkningsdata
    log.info('\n--- Steg 3: Befolkningsdata (SCB API) ---')
    pop_data = download_population_data(deso_codes_api, force=force)

    # 5. Spatial join
    log.info('\n--- Steg 4: Spatial join ---')
    df = spatial_join(df, gdf)

    # 6. Merga statistik
    log.info('\n--- Steg 5: Merga DeSO-statistik ---')
    df = merge_deso_stats(df, income_data, pop_data)

    # 7. Spara
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    log.info(f'\n✅ Sparad: {OUTPUT_PATH}')
    log.info(f'   Rader: {len(df)}, Kolumner: {len(df.columns)}')
    log.info('Nya kolumner: deso_kod, deso_median_ink_tkr, deso_lon_ink_tkr,')
    log.info('              deso_andel_lon_pct, deso_befolkning,')
    log.info('              deso_median_alder, deso_andel_0_19, deso_andel_65_plus')
    log.info('\nNästa steg: python scripts/train_villa_v10.py')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
