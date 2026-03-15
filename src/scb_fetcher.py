"""
SCB API Datahämtare — Örebro kommun
====================================
Hämtar kontextuell data från SCB:s öppna API:
- Medelinkomst per kommun/DeSO-område
- Befolkningsstatistik
- Fastighetspriser (officiell statistik)

SCB API dokumentation: https://www.scb.se/vara-tjanster/oppna-data/api-for-statistikdatabasen/
API-bas: https://api.scb.se/OV0104/v1/doris/sv/ssd/

Användning:
    python src/scb_fetcher.py
"""

import requests
import pandas as pd
import json
import os
import time
from datetime import datetime

# ============================================================
# KONFIGURATION
# ============================================================

SCB_API_BASE = "https://api.scb.se/OV0104/v1/doris/sv/ssd"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "external")

# Örebro kommun kod i SCB
OREBRO_KOMMUN_KOD = "1880"
OREBRO_LAN_KOD = "18"

# Rate limiting — SCB tillåter max 10 anrop per 10 sekunder
SCB_DELAY = 1.5  # sekunder mellan anrop


# ============================================================
# SCB API-FUNKTIONER
# ============================================================

def scb_query(table_path: str, query_body: dict) -> pd.DataFrame | None:
    """
    Gör en POST-förfrågan till SCB:s API och returnerar resultatet som DataFrame.
    
    Args:
        table_path: Sökväg till tabellen, t.ex. "BE/BE0101/BE0101A/BesijningFmansKommunN"
        query_body: JSON-body med query-parametrar
    
    Returns:
        DataFrame med resultatet, eller None vid fel
    """
    url = f"{SCB_API_BASE}/{table_path}"
    
    try:
        time.sleep(SCB_DELAY)
        
        # Hämta som JSON-stat
        response = requests.post(
            url,
            json=query_body,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        
        data = response.json()
        return parse_jsonstat(data)
        
    except requests.RequestException as e:
        print(f"  ⚠ API-fel för {table_path}: {e}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  ⚠ Parse-fel för {table_path}: {e}")
        return None


def parse_jsonstat(data: dict) -> pd.DataFrame:
    """
    Parsa SCB:s JSON-stat format till en pandas DataFrame.
    
    SCB returnerar data i JSON-stat format som ser ut så här:
    {
        "columns": [{"code": "Region", "text": "region"}, ...],
        "data": [{"key": ["1880", "2023"], "values": ["12345"]}, ...]
    }
    """
    columns = [col["text"] for col in data.get("columns", [])]
    
    rows = []
    for item in data.get("data", []):
        row = item.get("key", []) + item.get("values", [])
        rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    
    # Kolumnnamn: nyckelkolumner + värdekolumner
    value_cols = [col["text"] for col in data.get("columns", []) 
                  if col.get("type") == "c"]
    
    df = pd.DataFrame(rows, columns=columns if len(columns) == len(rows[0]) else None)
    return df


def get_table_metadata(table_path: str) -> dict | None:
    """Hämta metadata om en SCB-tabell (vilka variabler och värden som finns)."""
    url = f"{SCB_API_BASE}/{table_path}"
    
    try:
        time.sleep(SCB_DELAY)
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  ⚠ Metadata-fel: {e}")
        return None


# ============================================================
# SPECIFIKA DATAHÄMTNINGAR
# ============================================================

def fetch_income_data() -> pd.DataFrame | None:
    """
    Hämta medelinkomst per kommun i Örebro län.
    Tabell: HE0110/HE0110A/SamijInk1
    """
    print("\n📊 Hämtar inkomstdata från SCB...")
    
    # Sammanräknad förvärvsinkomst, medelvärde, per kommun
    table_path = "HE/HE0110/HE0110A/SamijInk1"
    
    query = {
        "query": [
            {
                "code": "Region",
                "selection": {
                    "filter": "vs:RegionKommun07",
                    "values": [OREBRO_KOMMUN_KOD]
                }
            },
            {
                "code": "ContentsCode",
                "selection": {
                    "filter": "item",
                    "values": ["HE0110J7"]  # Medelinkomst
                }
            }
        ],
        "response": {
            "format": "json"
        }
    }
    
    df = scb_query(table_path, query)
    if df is not None and not df.empty:
        print(f"  ✓ Inkomstdata: {len(df)} rader hämtade")
    else:
        print("  ⚠ Kunde inte hämta inkomstdata")
        print("    Tips: Kontrollera tabellsökvägen på scb.se")
    
    return df


def fetch_population_data() -> pd.DataFrame | None:
    """
    Hämta befolkningsstatistik per kommun.
    Tabell: BE/BE0101/BE0101A/FolsijingFmansKommunN
    """
    print("\n👥 Hämtar befolkningsdata från SCB...")
    
    table_path = "BE/BE0101/BE0101A/FolsijingFmansKommunN"
    
    query = {
        "query": [
            {
                "code": "Region",
                "selection": {
                    "filter": "vs:RegionKommun07",
                    "values": [OREBRO_KOMMUN_KOD]
                }
            },
            {
                "code": "ContentsCode",
                "selection": {
                    "filter": "item",
                    "values": ["BE0101N1"]  # Folkmängd
                }
            }
        ],
        "response": {
            "format": "json"
        }
    }
    
    df = scb_query(table_path, query)
    if df is not None and not df.empty:
        print(f"  ✓ Befolkningsdata: {len(df)} rader hämtade")
    else:
        print("  ⚠ Kunde inte hämta befolkningsdata")
    
    return df


def fetch_property_price_index() -> pd.DataFrame | None:
    """
    Hämta fastighetsprisindex för Örebro län.
    Tabell: BO/BO0501/BO0501D/SmijhijusPSRegAr
    """
    print("\n🏠 Hämtar fastighetsprisindex från SCB...")
    
    table_path = "BO/BO0501/BO0501D/SmijhijusPSRegAr"
    
    query = {
        "query": [
            {
                "code": "Region",
                "selection": {
                    "filter": "item",
                    "values": [OREBRO_LAN_KOD]
                }
            }
        ],
        "response": {
            "format": "json"
        }
    }
    
    df = scb_query(table_path, query)
    if df is not None and not df.empty:
        print(f"  ✓ Prisindex: {len(df)} rader hämtade")
    else:
        print("  ⚠ Kunde inte hämta prisindex")
        print("    Tips: Tabellnamn kan ha ändrats — kolla statistikdatabasen.scb.se")
    
    return df


def fetch_housing_data() -> pd.DataFrame | None:
    """
    Hämta bostadsbeståndsdata per kommun.
    """
    print("\n🏢 Hämtar bostadsbeståndsdata från SCB...")
    
    table_path = "BO/BO0104/BO0104D/BO0104T03"
    
    query = {
        "query": [
            {
                "code": "Region",
                "selection": {
                    "filter": "vs:RegionKommun07",
                    "values": [OREBRO_KOMMUN_KOD]
                }
            }
        ],
        "response": {
            "format": "json"
        }
    }
    
    df = scb_query(table_path, query)
    if df is not None and not df.empty:
        print(f"  ✓ Bostadsbestånd: {len(df)} rader hämtade")
    else:
        print("  ⚠ Kunde inte hämta bostadsbeståndsdata")
    
    return df


# ============================================================
# EXPLORE-FUNKTION — hitta rätt tabeller
# ============================================================

def explore_scb_tables(path: str = ""):
    """
    Utforska SCB:s tabellstruktur interaktivt.
    
    Användning:
        explore_scb_tables()             # Se toppnivå
        explore_scb_tables("BO")         # Se boende-tabeller
        explore_scb_tables("BO/BO0501")  # Gå djupare
    """
    url = f"{SCB_API_BASE}/{path}" if path else SCB_API_BASE
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        print(f"\n📂 SCB Tabell: /{path}")
        print("-" * 50)
        
        for item in data:
            item_type = item.get("type", "?")
            item_id = item.get("id", "?")
            item_text = item.get("text", "?")
            
            if item_type == "l":
                # Det är en tabell (leaf)
                print(f"  📄 {item_id}: {item_text}")
            else:
                # Det är en mapp
                print(f"  📁 {item_id}: {item_text}")
        
        return data
        
    except Exception as e:
        print(f"  ⚠ Fel: {e}")
        return None


# ============================================================
# MAIN
# ============================================================

def main():
    """Hämta all SCB-data och spara."""
    
    print("=" * 60)
    print("  SCB DATA HÄMTARE — ÖREBRO KOMMUN")
    print(f"  Startar: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    
    datasets = {
        "inkomst": fetch_income_data,
        "befolkning": fetch_population_data,
        "prisindex": fetch_property_price_index,
        "bostadsbestand": fetch_housing_data,
    }
    
    results = {}
    for name, fetch_fn in datasets.items():
        df = fetch_fn()
        if df is not None and not df.empty:
            path = os.path.join(OUTPUT_DIR, f"scb_{name}_{timestamp}.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            results[name] = {"rows": len(df), "path": path}
            print(f"  💾 Sparad: {path}")
    
    # Sammanfattning
    print(f"\n{'='*60}")
    print(f"  SAMMANFATTNING SCB-DATA")
    print(f"{'='*60}")
    for name, info in results.items():
        print(f"  {name}: {info['rows']} rader → {info['path']}")
    
    if not results:
        print("\n  ⚠ Ingen data hämtad!")
        print("  Tips: SCB:s tabellnamn ändras ibland.")
        print("  Kör explore_scb_tables() för att hitta rätt tabeller.")
        print("  Eller besök: https://www.statistikdatabasen.scb.se/")
    
    print(f"\n  Nästa steg: Kör notebooks/01_data_collection.ipynb")
    print(f"  för att kombinera Hemnet- och SCB-data.")


if __name__ == "__main__":
    main()
