"""
Preprocessing — Datarensning och Feature Engineering
=====================================================
Rensa, kombinera och berika bostadsdata från Hemnet + SCB.

Användning:
    from src.preprocessing import load_and_clean, engineer_features
"""

import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import os
import warnings

warnings.filterwarnings("ignore")

# Örebro centrum (Stortorget) — referenspunkt för avstånd
OREBRO_CENTRUM = (59.2753, 15.2134)

# Kända områden i Örebro med ungefärlig kategorisering
OMRADE_KATEGORI = {
    # Centrala
    "centralt": "centrum", "vasastan": "centrum", "city": "centrum",
    "centralt norr": "centrum", "centralt öster": "centrum",
    "centralt / vasastan": "centrum", "centralt / väster": "centrum",
    
    # Populära bostadsområden
    "sörbyängen": "populärt", "adolfsberg": "populärt",
    "södra ladugårdsängen": "populärt", "ladugårdsängen": "populärt",
    "wadköping": "populärt", "örnsro": "populärt",
    "tybble": "populärt", "baronbackarna": "populärt",
    "södra lindhult": "populärt", "norra lindhult": "populärt",
    
    # Ytterområden
    "brickebacken": "yttre", "vivalla": "yttre",
    "varberga": "yttre", "oxhagen": "yttre",
    "markbacken": "yttre", "björkhaga": "yttre",
    "hovsta": "yttre", "mosås": "yttre",
    "mellringe": "yttre", "rosta": "yttre",
    
    # Lantligt
    "garphyttan": "lantligt", "glanshammar": "lantligt",
    "lillkyrka": "lantligt", "vintrosa": "lantligt",
    "ekeby-almby": "lantligt", "norrbyås": "lantligt",
}


def load_and_clean(raw_data_path: str) -> pd.DataFrame:
    """
    Ladda rå Hemnet-data och utför grundläggande rensning.
    
    Steg:
    1. Ladda CSV
    2. Ta bort dubbletter
    3. Hantera saknade värden
    4. Konvertera datatyper
    5. Filtrera outliers
    """
    print("📋 Laddar och rensar data...")
    
    df = pd.read_csv(raw_data_path)
    original_count = len(df)
    print(f"  Rader laddade: {original_count}")
    
    # 1. Ta bort dubbletter (baserat på adress + slutpris + datum)
    df = df.drop_duplicates(
        subset=["adress", "slutpris", "sald_datum"],
        keep="first"
    )
    print(f"  Efter dedup: {len(df)} (-{original_count - len(df)} dubbla)")
    
    # 2. Ta bort rader utan slutpris
    df = df.dropna(subset=["slutpris"])
    df = df[df["slutpris"] > 0]
    
    # 3. Konvertera datatyper
    df["slutpris"] = pd.to_numeric(df["slutpris"], errors="coerce")
    df["boarea_kvm"] = pd.to_numeric(df["boarea_kvm"], errors="coerce")
    df["antal_rum"] = pd.to_numeric(df["antal_rum"], errors="coerce")
    df["avgift_kr"] = pd.to_numeric(df.get("avgift_kr"), errors="coerce")
    
    # 4. Parsera datum
    if "sald_datum" in df.columns:
        df["sald_datum"] = pd.to_datetime(df["sald_datum"], errors="coerce")
        df["sald_ar"] = df["sald_datum"].dt.year
        df["sald_manad"] = df["sald_datum"].dt.month
        df["sald_kvartal"] = df["sald_datum"].dt.quarter
    
    # 5. Filtrera extremvärden (outliers)
    df = remove_outliers(df)
    
    print(f"  Slutresultat: {len(df)} rader")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Ta bort extremvärden som sannolikt är felaktiga."""
    
    before = len(df)
    
    # Rimliga gränser för Örebro
    filters = {
        "slutpris": (100_000, 15_000_000),       # 100k - 15M kr
        "boarea_kvm": (10, 400),                   # 10 - 400 m²
        "antal_rum": (1, 12),                      # 1-12 rum
        "avgift_kr": (500, 15_000),                # 500 - 15000 kr/mån
    }
    
    for col, (min_val, max_val) in filters.items():
        if col in df.columns:
            mask = df[col].isna() | ((df[col] >= min_val) & (df[col] <= max_val))
            df = df[mask]
    
    print(f"  Outliers borttagna: {before - len(df)}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Skapa nya features för ML-modellering.
    """
    print("\n🔧 Skapar features...")
    
    # --- Prisrelaterade ---
    if "slutpris" in df.columns and "boarea_kvm" in df.columns:
        df["pris_per_kvm"] = (df["slutpris"] / df["boarea_kvm"]).round(0)
    
    if "avgift_kr" in df.columns and "boarea_kvm" in df.columns:
        df["avgift_per_kvm"] = (df["avgift_kr"] / df["boarea_kvm"]).round(1)
    
    # --- Storlek ---
    if "boarea_kvm" in df.columns and "antal_rum" in df.columns:
        df["kvm_per_rum"] = (df["boarea_kvm"] / df["antal_rum"]).round(1)
    
    # Storlekskategorier
    if "boarea_kvm" in df.columns:
        df["storlek_kategori"] = pd.cut(
            df["boarea_kvm"],
            bins=[0, 40, 60, 80, 100, 150, 500],
            labels=["mini", "liten", "medel", "stor", "villa", "lyxvilla"]
        )
    
    # --- Område ---
    if "omrade" in df.columns:
        df["omrade_clean"] = df["omrade"].str.lower().str.strip()
        df["omrade_kategori"] = df["omrade_clean"].map(OMRADE_KATEGORI).fillna("övrigt")
    
    # --- Tid ---
    if "sald_manad" in df.columns:
        df["sasong"] = df["sald_manad"].map({
            1: "vinter", 2: "vinter", 3: "vår",
            4: "vår", 5: "vår", 6: "sommar",
            7: "sommar", 8: "sommar", 9: "höst",
            10: "höst", 11: "höst", 12: "vinter"
        })
    
    # --- Prisförändring (utgångspris vs slutpris) ---
    if "prisforandring_raw" in df.columns:
        df["budkrig"] = df["prisforandring_raw"].str.contains(
            r"\+", na=False
        ).astype(int)
        
        df["prissankt"] = df["prisforandring_raw"].str.contains(
            r"\-", na=False
        ).astype(int)
    
    # --- Avgift-kvot (för bostadsrätter) ---
    if "avgift_kr" in df.columns and "slutpris" in df.columns:
        df["avgift_andel"] = (
            (df["avgift_kr"] * 12) / df["slutpris"] * 100
        ).round(2)  # Årsavgift som % av slutpris
    
    print(f"  ✓ {len(df.columns)} features totalt")
    return df


def geocode_addresses(df: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
    """
    Geokoda adresser för att beräkna avstånd till centrum.
    
    OBS: Långsam process — kör bara på ett urval.
    Nominatim har rate limits (1 request/sekund).
    """
    print(f"\n📍 Geokodar adresser (urval: {sample_size})...")
    
    geolocator = Nominatim(user_agent="orebro-housing-ml-project")
    
    # Ta ett urval om datan är stor
    if len(df) > sample_size:
        sample_idx = df.sample(sample_size, random_state=42).index
    else:
        sample_idx = df.index
    
    df["latitude"] = np.nan
    df["longitude"] = np.nan
    df["avstand_centrum_km"] = np.nan
    
    for idx in sample_idx:
        address = df.loc[idx, "adress"]
        if pd.isna(address):
            continue
        
        try:
            time.sleep(1.1)  # Respektera Nominatims rate limit
            location = geolocator.geocode(f"{address}, Örebro, Sverige")
            
            if location:
                df.loc[idx, "latitude"] = location.latitude
                df.loc[idx, "longitude"] = location.longitude
                df.loc[idx, "avstand_centrum_km"] = round(
                    geodesic(
                        OREBRO_CENTRUM,
                        (location.latitude, location.longitude)
                    ).km,
                    2
                )
        except Exception:
            continue
    
    geocoded = df["latitude"].notna().sum()
    print(f"  ✓ Geokodade: {geocoded}/{sample_size}")
    
    return df


def merge_with_scb(
    df: pd.DataFrame,
    income_path: str = None,
    population_path: str = None,
) -> pd.DataFrame:
    """
    Berika bostadsdata med SCB-data.
    
    Eftersom SCB-data är på kommunnivå (inte områdesnivå)
    läggs det till som konstanta kolumner.
    """
    print("\n🔗 Kombinerar med SCB-data...")
    
    if income_path and os.path.exists(income_path):
        income_df = pd.read_csv(income_path)
        # Lägg till senaste medelinkomsten som feature
        if not income_df.empty:
            latest_income = income_df.iloc[-1]  # Senaste året
            df["kommun_medelinkomst"] = pd.to_numeric(
                latest_income.iloc[-1], errors="coerce"
            )
            print(f"  ✓ Inkomstdata tillagd")
    
    if population_path and os.path.exists(population_path):
        pop_df = pd.read_csv(population_path)
        if not pop_df.empty:
            latest_pop = pop_df.iloc[-1]
            df["kommun_befolkning"] = pd.to_numeric(
                latest_pop.iloc[-1], errors="coerce"
            )
            print(f"  ✓ Befolkningsdata tillagd")
    
    return df


def prepare_for_modeling(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Förbered data för ML-modellering.
    
    Returns:
        X: Feature-matris
        y: Målvariabel (slutpris)
    """
    print("\n🎯 Förbereder data för modellering...")
    
    # Features att använda
    numeric_features = [
        "boarea_kvm", "antal_rum", "avgift_kr",
        "pris_per_kvm", "avgift_per_kvm", "kvm_per_rum",
        "avstand_centrum_km", "sald_manad", "sald_kvartal",
    ]
    
    categorical_features = [
        "bostadstyp", "omrade_kategori", "sasong",
        "storlek_kategori",
    ]
    
    # Filtrera till features som faktiskt finns
    available_numeric = [f for f in numeric_features if f in df.columns]
    available_categorical = [f for f in categorical_features if f in df.columns]
    
    # One-hot encoding för kategoriska features
    X = df[available_numeric + available_categorical].copy()
    X = pd.get_dummies(X, columns=available_categorical, drop_first=True)
    
    # Fyll saknade värden med median
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())
    
    y = df["slutpris"]
    
    print(f"  ✓ Features: {X.shape[1]}")
    print(f"  ✓ Samples: {X.shape[0]}")
    print(f"  ✓ Målvariabel: slutpris (median: {y.median():,.0f} kr)")
    
    return X, y
