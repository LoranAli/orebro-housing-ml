"""
Villa-modell v10 — SCB DeSO socioekonomiska grannskapsfeatures (Fas C2)
=======================================================================
Förbättringar vs v9 (R²_test=0.7551):

  C2. SCB DeSO FEATURES (7 nya grannskapsfeatures)
      deso_median_ink_tkr   — Medelinkomst netto per person (tkr/år), 2024
      deso_lon_ink_tkr      — Medelinkomst lön per person (tkr/år), 2024
      deso_andel_lon_pct    — Andel med löneinkomst (%), 2024
      deso_befolkning       — Antal invånare i DeSO-området, 2024
      deso_median_alder     — Uppskattad medianålder (viktad mid-ålder), 2024
      deso_andel_0_19       — Andel barn/unga 0–19 år (%), 2024
      deso_andel_65_plus    — Andel äldre 65+ år (%), 2024

      Korrelation inkomst × slutpris: 0.31 (stark socioekonomisk signal)
      Kräver: scb_deso.py körd → orebro_housing_enriched_v5.csv

Kör:
    cd "orebro-housing-ml 3"
    python scripts/train_villa_v10.py
    python scripts/train_villa_v10.py --no-optuna    # snabb test
    python scripts/train_villa_v10.py --no-cb-optuna # hoppa CB-tuning
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.neighbors import BallTree
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# KONSTANTER
# ─────────────────────────────────────────────────────────────
_DATA_V5 = "data/processed/orebro_housing_enriched_v5.csv"
_DATA_V4 = "data/processed/orebro_housing_enriched_v4.csv"
_DATA_V3 = "data/processed/orebro_housing_enriched_v3.csv"
_DATA_V2 = "data/processed/orebro_housing_enriched_v2.csv"
_DATA_V1 = "data/processed/orebro_housing_enriched.csv"
DATA_PATH = (
    _DATA_V5 if os.path.exists(_DATA_V5) else
    _DATA_V4 if os.path.exists(_DATA_V4) else
    _DATA_V3 if os.path.exists(_DATA_V3) else
    _DATA_V2 if os.path.exists(_DATA_V2) else
    _DATA_V1
)
MODEL_OUT     = "models/model_villor_v10.pkl"
METADATA_PATH = "models/model_metadata_v10.json"

# 3-way split med retrain
TRAIN_END = pd.Timestamp("2024-12-31")
VAL_END   = pd.Timestamp("2025-06-30")
TARGET    = "slutpris"

AREA_MIN_SAMPLES = 3
TE_SMOOTHING     = 10
N_NEIGHBORS      = 25   # B2: upp från 10
NEIGHBOR_DAYS    = 548  # B2: ~18 månader upp från 365
N_CLUSTERS       = 12

OPTUNA_TRIALS    = 80
CB_OPTUNA_TRIALS = 30   # B1: CatBoost CV-tuning

# ─────────────────────────────────────────────────────────────
# CORE FEATURES v9
# ─────────────────────────────────────────────────────────────
CORE_FEATURES = [
    # == FYSISKA EGENSKAPER ==
    "boarea_kvm",
    "log_boarea",
    "antal_rum",
    "kvm_per_rum",
    "total_yta",
    "tomtarea_kvm",
    "log_tomtarea",
    "tomt_per_boarea",
    "biarea_kvm",
    "biarea_var_missing",

    # == ÅLDER & STANDARD ==
    "bostad_alder",
    "byggdekad",
    "alder_ej_renoverad",

    # == PRIS-KONTEXT ==
    "omrade_hist_pris_kvm",
    "comps_pris_kvm_90d",
    "forvantat_komps_pris",

    # == MARKNADSTRENDINDEX ==
    "marknad_trend_6m",
    "marknad_trend_ratio",

    # == OMRÅDE / GEO ==
    "te_omrade_pris",
    "grannskap_median_kvm",
    "grannskap_vd_kvm",
    "cluster_te",
    "avstand_centrum_km",
    "avstand_station_km",
    "avstand_sjukhus_km",

    # == INTERAKTIONER ==
    "tomt_boarea_interact",
    "boarea_log_tomt",
    "avst_pris_interact",
    "tomt_avst_interact",

    # == VILLA-SPECIFIKA ==
    "driftkostnad_ar",
    "log_driftkostnad",
    "driftkostnad_per_kvm",
    "ek_proxy",
    "har_uteplats",

    # == MARKNAD & EFTERFRÅGAN ==
    "antal_besok",
    "comps_antal_90d",
    "comps_pristrend",
    "prisforandring_pct",
    "budkrig",
    "prissankt",

    # == TID ==
    "sald_ar",
    "sald_manad",
    "sald_kvartal",
]

# Inkluderas automatiskt om de finns i datasetet
OPTIONAL_FEATURES = [
    "har_balkong",
    "avstand_marieberg_km",
    "avstand_universitet_km",
    "uppvarmning_score",
    "energiklass_num",
    # A2: Mäklarfirma (kräver v4 + enrich_v4_features.py)
    "maklare_te",
    # A3: Riksbankränta (kräver v4 + enrich_v4_features.py)
    "riksbank_rate",
    "rate_change_6m",
    "rate_boarea_interact",
    # B2: Prisvariation i grannskapet (bara meningsfull med riktiga koordinater)
    "grannskap_spread_kvm",
    # Kvalitetsmarkör geocodning
    "geocode_quality_bin",
    # NLP-features (finns i v3/v4)
    "premium_score",
    # C2: SCB DeSO socioekonomiska features (kräver v5 + scb_deso.py)
    "deso_median_ink_tkr",
    "deso_lon_ink_tkr",
    "deso_andel_lon_pct",
    "deso_befolkning",
    "deso_median_alder",
    "deso_andel_0_19",
    "deso_andel_65_plus",
]


# ─────────────────────────────────────────────────────────────
# 1. LADDA DATA
# ─────────────────────────────────────────────────────────────
def load_data():
    print("Laddar data...")
    print(f"  Fil: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["sald_datum"] = pd.to_datetime(df["sald_datum"], errors="coerce")
    v = df[df["bostadstyp"] == "villor"].copy()
    v = v.dropna(subset=["sald_datum", TARGET, "boarea_kvm"])
    print(f"  Villor: {len(v)}")
    print(f"  Datum:  {v['sald_datum'].min().date()} -> {v['sald_datum'].max().date()}")
    print(f"  Slutpris median: {v[TARGET].median():,.0f} kr  std: {v[TARGET].std():,.0f} kr")
    # Kontrollera A1/A2/A3 features
    v4_cols = ["maklare_te", "riksbank_rate", "rate_change_6m", "rate_boarea_interact"]
    found_v4 = [c for c in v4_cols if c in v.columns]
    missing_v4 = [c for c in v4_cols if c not in v.columns]
    if found_v4:
        print(f"  v4-features (A2/A3): {found_v4}")
    if missing_v4:
        print(f"  SAKNAR v4-features: {missing_v4} — kör enrich_v4_features.py")
    # C2: DeSO-features
    deso_cols = ["deso_median_ink_tkr", "deso_lon_ink_tkr", "deso_andel_lon_pct",
                 "deso_befolkning", "deso_median_alder", "deso_andel_0_19", "deso_andel_65_plus"]
    found_deso = [c for c in deso_cols if c in v.columns]
    if found_deso:
        print(f"  DeSO-features (C2): {len(found_deso)}/7 hittade")
        print(f"  deso_median_ink_tkr: {v['deso_median_ink_tkr'].min():.0f}–{v['deso_median_ink_tkr'].max():.0f} tkr  (korr slutpris: {v['deso_median_ink_tkr'].corr(v[TARGET]):.3f})")
    else:
        print("  SAKNAR DeSO-features — kör scb_deso.py")
    # geocode_quality
    if "geocode_quality" in v.columns:
        qc = v["geocode_quality"].value_counts().to_dict()
        print(f"  geocode_quality: {qc}")
        geocoded = sum(v for k, v in qc.items() if k != "hemnet_centroid")
        print(f"  Geocodade adresser: {geocoded}/{len(v)} ({geocoded/len(v)*100:.0f}%)")
    return v


# ─────────────────────────────────────────────────────────────
# 2. COMPS-FEATURES
# ─────────────────────────────────────────────────────────────
def compute_comps(df, window_days=90):
    print("\nBeraknar comps-features...")
    df = df.sort_values("sald_datum").copy()
    df["_pkvm"] = df[TARGET] / df["boarea_kvm"].clip(lower=10)

    cp, ca, ct = {}, {}, {}
    for area, grp in df.groupby("omrade_clean", sort=False):
        grp = grp.sort_values("sald_datum")
        dates  = grp["sald_datum"].values
        prices = grp["_pkvm"].values
        for d, ix in zip(dates, grp.index):
            lo  = d - np.timedelta64(window_days, "D")
            lo2 = d - np.timedelta64(window_days * 2, "D")
            rec        = prices[(dates >= lo)  & (dates < d)]
            old        = prices[(dates >= lo2) & (dates < lo)]
            all_before = prices[dates < d]
            cp[ix] = (np.median(rec) if len(rec) >= 2
                      else np.median(all_before) if len(all_before) > 0
                      else np.nan)
            ca[ix] = int((dates >= lo).sum() if (dates < d).any() else 0)
            ct[ix] = (round((np.median(rec) / np.median(old) - 1) * 100, 2)
                      if len(rec) >= 2 and len(old) >= 2 else 0.0)

    df["comps_pris_kvm_90d"] = pd.Series(cp)
    df["comps_antal_90d"]    = pd.Series(ca)
    df["comps_pristrend"]    = pd.Series(ct)
    glob_med = df["comps_pris_kvm_90d"].median()
    df["comps_pris_kvm_90d"] = df["comps_pris_kvm_90d"].fillna(glob_med)
    df.drop("_pkvm", axis=1, inplace=True)
    print(f"  comps median: {df['comps_pris_kvm_90d'].median():,.0f} kr/m2")
    return df


# ─────────────────────────────────────────────────────────────
# 3. GLOBAL MARKNADSTRENDINDEX
# ─────────────────────────────────────────────────────────────
def compute_marknad_trend(df):
    print("\nBeraknar global marknadstrendindex (6-man rolling)...")
    df = df.copy().sort_values("sald_datum")
    df["_pkvm"] = df[TARGET] / df["boarea_kvm"].clip(lower=10)
    dates  = df["sald_datum"].values
    prices = df["_pkvm"].values
    window = np.timedelta64(180, "D")

    trend_vals = []
    for i, d in enumerate(dates):
        lo  = d - window
        rec = prices[(dates >= lo) & (dates < d)]
        trend_vals.append(float(np.median(rec)) if len(rec) >= 3 else np.nan)

    df["marknad_trend_6m"] = trend_vals
    glob = df["marknad_trend_6m"].median()
    df["marknad_trend_6m"] = df["marknad_trend_6m"].fillna(glob)

    df["marknad_trend_ratio"] = (
        df["comps_pris_kvm_90d"] / df["marknad_trend_6m"].replace(0, np.nan)
    ).fillna(1.0)

    df.drop("_pkvm", axis=1, inplace=True)
    print(f"  marknad_trend_6m: {df['marknad_trend_6m'].min():,.0f} -> "
          f"{df['marknad_trend_6m'].max():,.0f} kr/m2")
    return df


# ─────────────────────────────────────────────────────────────
# 4. TARGET ENCODING
# ─────────────────────────────────────────────────────────────
def fit_target_encoder(df_train, col, target, smoothing=TE_SMOOTHING):
    global_mean = df_train[target].mean()
    stats  = df_train.groupby(col)[target].agg(["mean", "count"])
    smooth = stats["count"] / (stats["count"] + smoothing)
    stats["encoded"] = smooth * stats["mean"] + (1 - smooth) * global_mean
    return stats["encoded"].to_dict(), global_mean


def apply_target_encoder(df, col, te_map, global_mean):
    return df[col].map(te_map).fillna(global_mean)


# ─────────────────────────────────────────────────────────────
# 5. GRANNSKAP via BallTree (v9: 25 grannar, 18m, + spread-feature)
# ─────────────────────────────────────────────────────────────
def compute_grannskap(df, n_neighbors=N_NEIGHBORS, window_days=NEIGHBOR_DAYS):
    print(f"\nBeraknar grannskap-features (BallTree, n={n_neighbors}, {window_days}d)...")
    df = df.copy()
    has_coords = df["latitude"].notna() & df["longitude"].notna()
    n_coords   = has_coords.sum()

    # Kontrollera geocodningskvalitet
    if "geocode_quality" in df.columns:
        real_coords = df["geocode_quality"].isin(["exact", "city", "street_only", "country"])
        print(f"  Koordinater totalt: {n_coords} | Geocodade (ej centroid): {real_coords.sum()}")
    else:
        print(f"  Koordinater: {n_coords}/{len(df)} ({has_coords.mean()*100:.0f}%)")

    df["grannskap_median_kvm"] = np.nan
    df["grannskap_vd_kvm"]     = np.nan
    df["grannskap_spread_kvm"] = np.nan  # NY v9

    if n_coords < n_neighbors + 1:
        df["grannskap_median_kvm"] = df["comps_pris_kvm_90d"]
        df["grannskap_vd_kvm"]     = df["comps_pris_kvm_90d"]
        df["grannskap_spread_kvm"] = 0.0
        return df

    df_w = df[has_coords].copy().sort_values("sald_datum")
    pkvm  = (df_w[TARGET] / df_w["boarea_kvm"].clip(lower=10)).values
    dates = df_w["sald_datum"].values

    coords_rad = np.radians(df_w[["latitude", "longitude"]].values)
    tree       = BallTree(coords_rad, metric="haversine")
    k_search   = min(n_neighbors * 3, n_coords - 1)
    all_dists, all_idxs = tree.query(coords_rad, k=k_search + 1)
    all_dists_km = all_dists * 6371.0

    med_vals    = []
    vd_vals     = []
    spread_vals = []

    for i in range(len(df_w)):
        d      = dates[i]
        cutoff = d - np.timedelta64(window_days, "D")
        nbr_ix = all_idxs[i, 1:]
        nbr_d  = all_dists_km[i, 1:]

        # Samla in grannar inom tidsperiod
        valid = []
        for j, ni in enumerate(nbr_ix):
            if dates[ni] < d and dates[ni] >= cutoff:
                valid.append((ni, nbr_d[j]))
                if len(valid) >= n_neighbors:
                    break

        # Fallback: utöka om för få grannar
        if len(valid) < 2:
            for j, ni in enumerate(nbr_ix):
                if dates[ni] < d and (ni, nbr_d[j]) not in valid:
                    valid.append((ni, nbr_d[j]))
                    if len(valid) >= n_neighbors:
                        break

        if valid:
            idxs_v  = [v[0] for v in valid]
            dists_v = np.array([v[1] for v in valid])
            nbr_prices = pkvm[idxs_v]

            # Distance-weighted (exp(-d/0.5km))
            weights = np.exp(-dists_v / 0.5)
            weights = weights / weights.sum()

            med = float(np.median(nbr_prices))
            vd  = float(np.dot(weights, nbr_prices))

            # Spread: std/median (prisvariation i grannskapet, 0 om homogent)
            spread = float(np.std(nbr_prices) / med) if med > 0 and len(nbr_prices) >= 3 else 0.0

            med_vals.append(med)
            vd_vals.append(vd)
            spread_vals.append(spread)
        else:
            med_vals.append(np.nan)
            vd_vals.append(np.nan)
            spread_vals.append(np.nan)

    df.loc[df_w.index, "grannskap_median_kvm"] = med_vals
    df.loc[df_w.index, "grannskap_vd_kvm"]     = vd_vals
    df.loc[df_w.index, "grannskap_spread_kvm"] = spread_vals

    glob_med = df["grannskap_median_kvm"].median()
    fallback = df["comps_pris_kvm_90d"].where(df["comps_pris_kvm_90d"].notna(), glob_med)
    df["grannskap_median_kvm"] = df["grannskap_median_kvm"].fillna(fallback)
    df["grannskap_vd_kvm"]     = df["grannskap_vd_kvm"].fillna(fallback)
    df["grannskap_spread_kvm"] = df["grannskap_spread_kvm"].fillna(0.0)

    print(f"  grannskap_median_kvm: {df['grannskap_median_kvm'].median():,.0f} kr/m2")
    print(f"  grannskap_vd_kvm:     {df['grannskap_vd_kvm'].median():,.0f} kr/m2")
    print(f"  grannskap_spread_kvm: {df['grannskap_spread_kvm'].median():.3f} (std/median ratio)")
    return df


# ─────────────────────────────────────────────────────────────
# 6. AREA-KONSOLIDERING
# ─────────────────────────────────────────────────────────────
def consolidate_areas(df, train_mask):
    print(f"\nArea-konsolidering (min {AREA_MIN_SAMPLES} obs per omrade)...")
    counts = df.loc[train_mask, "omrade_clean"].value_counts()
    sparse = set(counts[counts < AREA_MIN_SAMPLES].index)
    print(f"  {len(sparse)} omraden -> 'Ovrigt' (av {len(counts)} totalt i traning)")
    df = df.copy()
    df["omrade_v7"] = df["omrade_clean"].where(
        ~df["omrade_clean"].isin(sparse), other="Ovrigt"
    )
    remaining = df.loc[train_mask, "omrade_v7"].nunique()
    print(f"  Kvar: {remaining} unika omraden")
    return df


# ─────────────────────────────────────────────────────────────
# 7. HISTORISK AREA-MEDIAN
# ─────────────────────────────────────────────────────────────
def compute_omrade_hist(df, train_mask):
    area_hist = (
        df[train_mask]
        .assign(_pkvm=lambda d: d[TARGET] / d["boarea_kvm"].clip(lower=10))
        .groupby("omrade_v7")["_pkvm"].median()
        .rename("omrade_hist_pris_kvm")
    )
    df = df.join(area_hist, on="omrade_v7")
    global_med = df.loc[train_mask, "omrade_hist_pris_kvm"].median()
    df["omrade_hist_pris_kvm"] = df["omrade_hist_pris_kvm"].fillna(global_med)
    return df


# ─────────────────────────────────────────────────────────────
# 8. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(df, train_mask):
    print("\nFeature engineering (v9)...")
    df = df.copy()

    # biarea proxy
    miss = df["biarea_kvm"].isna()
    potential = (df["total_yta"] - df["boarea_kvm"]).clip(lower=0)
    df.loc[miss & (potential > 5), "biarea_kvm"] = potential[miss & (potential > 5)]
    df["biarea_kvm"] = df["biarea_kvm"].fillna(0)
    df["biarea_var_missing"] = miss.astype(int)

    # Log-transforms
    df["log_boarea"]       = np.log(df["boarea_kvm"].clip(lower=1))
    df["log_tomtarea"]     = np.log1p(df["tomtarea_kvm"].fillna(0))
    df["log_driftkostnad"] = np.log1p(df["driftkostnad_ar"].fillna(0))

    # Ålder och byggdekad
    df["byggdekad"]        = ((df["byggar"].fillna(1970) // 10) * 10).astype(int)
    df["alder_ej_renoverad"] = df["bostad_alder"].fillna(50)

    # Interaktioner
    tomt_med = df.loc[train_mask, "tomtarea_kvm"].median()
    avst_med = df.loc[train_mask, "avstand_centrum_km"].median()

    df["tomt_boarea_interact"] = (
        df["tomtarea_kvm"].fillna(tomt_med) * df["boarea_kvm"]
    ) / 1e4

    df["boarea_log_tomt"] = df["log_boarea"] * df["log_tomtarea"]

    area_pris_med = df.loc[train_mask, "omrade_hist_pris_kvm"].median()
    df["avst_pris_interact"] = (
        df["avstand_centrum_km"].fillna(avst_med)
        * df["omrade_hist_pris_kvm"].fillna(area_pris_med)
    ) / 1000

    df["tomt_avst_interact"] = (
        df["log_tomtarea"] * df["avstand_centrum_km"].fillna(avst_med)
    )

    df["forvantat_komps_pris"] = df["comps_pris_kvm_90d"] * df["boarea_kvm"]

    # ek_proxy
    if "ek_proxy" not in df.columns:
        EK_BYGGAR_MAP = [(2020,7),(2010,6),(2000,5),(1985,4),(1975,3),(1961,2),(0,1)]
        def _ek(y):
            if pd.isna(y) or y <= 0:
                return 3
            y = int(y)
            for thr, sc in EK_BYGGAR_MAP:
                if y >= thr:
                    return sc
            return 1
        df["ek_proxy"] = df["byggar"].fillna(1970).apply(_ek)

    if "energiklass_num" not in df.columns:
        ek_map = {"A":7,"B":6,"C":5,"D":4,"E":3,"F":2,"G":1}
        ek_num = df.get("energiklass", pd.Series(dtype=str)).map(ek_map)
        df["energiklass_num"] = ek_num.where(ek_num.notna(), df["ek_proxy"])

    if "uppvarmning_score" not in df.columns:
        df["uppvarmning_score"] = 0

    # geocode_quality_bin: 1 om riktigt geocodad, 0 om Hemnet-centroid
    if "geocode_quality" in df.columns:
        df["geocode_quality_bin"] = (df["geocode_quality"] != "hemnet_centroid").astype(int)
        print(f"  geocode_quality_bin: {df['geocode_quality_bin'].mean()*100:.0f}% geocodade")

    return df


# ─────────────────────────────────────────────────────────────
# 9. KLUSTRING
# ─────────────────────────────────────────────────────────────
def fit_kmeans(df_train, n_clusters=N_CLUSTERS):
    print(f"\nKMeans klustring ({n_clusters} kluster med lat/lon)...")
    cluster_feats = [
        "latitude", "longitude",
        "avstand_centrum_km", "boarea_kvm",
        "comps_pris_kvm_90d", "tomtarea_kvm",
    ]
    X_c = df_train[cluster_feats].copy()
    for col in cluster_feats:
        X_c[col] = X_c[col].fillna(X_c[col].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_c)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X_scaled)
    labels = km.labels_
    for cid in range(n_clusters):
        n   = (labels == cid).sum()
        med = df_train.loc[df_train.index[labels == cid], TARGET].median()
        print(f"  Kluster {cid:2d}: {n:4d} villor, median {med:>10,.0f} kr")
    return km, scaler, cluster_feats


def apply_kmeans(df, km, scaler, cluster_feats):
    X_c = df[cluster_feats].copy()
    for col in cluster_feats:
        X_c[col] = X_c[col].fillna(X_c[col].median())
    return km.predict(scaler.transform(X_c))


# ─────────────────────────────────────────────────────────────
# 10. FEATURE-MATRIS
# ─────────────────────────────────────────────────────────────
def build_feature_matrix(df):
    feats = [f for f in CORE_FEATURES if f in df.columns]
    missing = [f for f in CORE_FEATURES if f not in df.columns]
    if missing:
        print(f"  Saknade core-features: {missing}")

    opt_raw = [f for f in OPTIONAL_FEATURES if f in df.columns and f not in feats]
    opt = [f for f in opt_raw if df[f].nunique() > 1]
    skipped = [f for f in opt_raw if f not in opt]
    if skipped:
        print(f"  Valfria features hoppad (noll-varians): {skipped}")
    if opt:
        print(f"  Valfria features inkluderade: {opt}")

    all_feats = feats + opt
    X = df[all_feats].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)

    print(f"  Feature-matris: {X.shape[1]} features ({len(feats)} core + {len(opt)} valfria)")
    return X, all_feats


# ─────────────────────────────────────────────────────────────
# 11. TIDSSPLIT
# ─────────────────────────────────────────────────────────────
def time_split(df, X):
    train_m = df["sald_datum"] <= TRAIN_END
    val_m   = (df["sald_datum"] > TRAIN_END) & (df["sald_datum"] <= VAL_END)
    test_m  = df["sald_datum"] > VAL_END

    X_train, X_val, X_test = X[train_m], X[val_m], X[test_m]
    y_train = np.log1p(df.loc[train_m, TARGET])
    y_val   = np.log1p(df.loc[val_m,   TARGET])
    y_test  = np.log1p(df.loc[test_m,  TARGET])

    X_full_train = X[train_m | val_m]
    y_full_train = np.log1p(df.loc[train_m | val_m, TARGET])

    print(f"\nTidssplit:")
    print(f"  Träning:      {len(X_train):>5} rader (-> {TRAIN_END.date()})")
    print(f"  Val (ES):     {len(X_val):>5} rader ({TRAIN_END.date()} -> {VAL_END.date()})")
    print(f"  Test:         {len(X_test):>5} rader ({VAL_END.date()} ->)")
    print(f"  Full retrain: {len(X_full_train):>5} rader (train+val)")

    return X_train, X_val, X_test, y_train, y_val, y_test, X_full_train, y_full_train


# ─────────────────────────────────────────────────────────────
# 12. UTVÄRDERING
# ─────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred_log):
    y_true_kr = np.expm1(y_true)
    y_pred_kr = np.expm1(y_pred_log)
    r2   = r2_score(y_true, y_pred_log)
    mae  = mean_absolute_error(y_true_kr, y_pred_kr)
    mape = float(np.mean(np.abs((y_true_kr - y_pred_kr) / y_true_kr.clip(lower=1))) * 100)
    print(f"  {name:<42} R2={r2:.4f}  MAE={mae:>9,.0f} kr  MAPE={mape:.1f}%")
    return {"R2": round(r2, 4), "MAE": round(mae, 0), "MAPE": round(mape, 1)}


def print_feature_importance(model, feature_names, top_n=20):
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"\nTop {top_n} viktigaste features:")
    for feat, val in imp.head(top_n).items():
        bar = "#" * int(val / imp.max() * 25)
        print(f"  {feat:<35} {bar} {val:.0f}")


# ─────────────────────────────────────────────────────────────
# 13. OPTUNA HP-TUNING LGBM
# ─────────────────────────────────────────────────────────────
def tune_lgbm_optuna(X_train_full, y_train_full, train_dates, n_trials=OPTUNA_TRIALS):
    print(f"\nOptuna HP-tuning LightGBM ({n_trials} trials, TimeSeriesSplit n=5)...")
    tscv = TimeSeriesSplit(n_splits=5)
    sort_idx = np.argsort(train_dates)
    X_s = X_train_full.iloc[sort_idx]
    y_s = y_train_full.iloc[sort_idx]

    def objective(trial):
        use_unlimited_depth = trial.suggest_categorical("unlimited_depth", [True, False])
        max_depth = -1 if use_unlimited_depth else trial.suggest_int("max_depth", 4, 10)
        params = {
            "objective":          "regression",
            "metric":             "rmse",
            "n_estimators":       800,
            "learning_rate":      trial.suggest_float("lr", 0.02, 0.12, log=True),
            "num_leaves":         trial.suggest_int("num_leaves", 63, 255),
            "max_depth":          max_depth,
            "min_child_samples":  trial.suggest_int("min_child", 5, 40),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample", 0.6, 1.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":         trial.suggest_float("reg_lambda", 0.5, 10.0),
            "n_jobs":             -1,
            "verbose":            -1,
            "random_state":       42,
        }
        fold_r2s = []
        for tr_idx, vl_idx in tscv.split(X_s):
            X_tr, y_tr = X_s.iloc[tr_idx], y_s.iloc[tr_idx]
            X_vl, y_vl = X_s.iloc[vl_idx], y_s.iloc[vl_idx]
            m = lgb.LGBMRegressor(**params)
            m.fit(X_tr, y_tr,
                  eval_set=[(X_vl, y_vl)],
                  callbacks=[lgb.early_stopping(40, verbose=False),
                              lgb.log_evaluation(period=-1)])
            fold_r2s.append(r2_score(y_vl, m.predict(X_vl)))
        return np.mean(fold_r2s)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best_r2 = study.best_value
    depth_str = "-1 (unlimited)" if best.get("unlimited_depth") else str(best.get("max_depth", -1))
    print(f"  Basta CV R2={best_r2:.4f}")
    print(f"  Params: lr={best['lr']:.4f}, leaves={best['num_leaves']}, "
          f"depth={depth_str}, min_child={best['min_child']}, "
          f"sub={best['subsample']:.2f}, col={best['colsample']:.2f}")

    use_unlimited = best.get("unlimited_depth", False)
    final_max_depth = -1 if use_unlimited else best.get("max_depth", -1)
    final_params = {
        "objective":          "regression",
        "metric":             "rmse",
        "n_estimators":       3000,
        "learning_rate":      best["lr"],
        "num_leaves":         best["num_leaves"],
        "max_depth":          final_max_depth,
        "min_child_samples":  best["min_child"],
        "subsample":          best["subsample"],
        "colsample_bytree":   best["colsample"],
        "reg_alpha":          best["reg_alpha"],
        "reg_lambda":         best["reg_lambda"],
        "n_jobs":             -1,
        "verbose":            -1,
        "random_state":       42,
    }
    return final_params


# ─────────────────────────────────────────────────────────────
# 14. TRÄNA SLUTLIG LGBM
# ─────────────────────────────────────────────────────────────
def train_final_lgbm(X_train, y_train, X_val, y_val, params, n_estimators_fixed=None):
    if n_estimators_fixed is not None:
        p = {**params, "n_estimators": n_estimators_fixed}
        model = lgb.LGBMRegressor(**p)
        model.fit(X_train, y_train,
                  callbacks=[lgb.log_evaluation(period=-1)])
        print(f"  LGBM retrain (train+val): {n_estimators_fixed} trad (fixerade)")
    else:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        n_trees = model.best_iteration_ or model.n_estimators_
        print(f"  LGBM (train only): {n_trees} trad")
    return model


# ─────────────────────────────────────────────────────────────
# 15. B1: CATBOOST CV-TUNING (TimeSeriesSplit, ej direkt val)
# ─────────────────────────────────────────────────────────────
def tune_catboost_cv(X_train_full, y_train_full, train_dates, n_trials=CB_OPTUNA_TRIALS):
    """
    B1: CatBoost HP-tuning via TimeSeriesSplit(n_splits=3) på träningsdata.
    Undviker val-overfit som uppstod i v8 (val R²=0.807 men test R²=0.738).
    """
    print(f"\nB1: Optuna HP-tuning CatBoost ({n_trials} trials, TimeSeriesSplit n=3)...")
    tscv = TimeSeriesSplit(n_splits=3)
    sort_idx = np.argsort(train_dates)
    X_s = X_train_full.iloc[sort_idx]
    y_s = y_train_full.iloc[sort_idx]

    def objective(trial):
        params = {
            "iterations":            600,
            "learning_rate":         trial.suggest_float("lr", 0.01, 0.1, log=True),
            "depth":                 trial.suggest_int("depth", 5, 10),
            "l2_leaf_reg":           trial.suggest_float("l2", 1.0, 15.0),
            "min_data_in_leaf":      trial.suggest_int("min_data", 5, 30),
            "subsample":             trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel":     trial.suggest_float("colsample", 0.6, 1.0),
            "loss_function":         "RMSE",
            "eval_metric":           "R2",
            "random_seed":           42,
            "verbose":               False,
            "early_stopping_rounds": 50,
        }
        fold_r2s = []
        for tr_idx, vl_idx in tscv.split(X_s):
            m = CatBoostRegressor(**params)
            m.fit(X_s.iloc[tr_idx], y_s.iloc[tr_idx],
                  eval_set=(X_s.iloc[vl_idx], y_s.iloc[vl_idx]))
            fold_r2s.append(m.best_score_["validation"]["R2"])
        return np.mean(fold_r2s)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best_r2 = study.best_value
    print(f"  Basta CV R2={best_r2:.4f}")
    print(f"  Params: lr={best['lr']:.4f}, depth={best['depth']}, "
          f"l2={best['l2']:.2f}, min_data={best['min_data']}, "
          f"sub={best['subsample']:.2f}, col={best['colsample']:.2f}")
    return best


def train_catboost(X_train, y_train, X_val=None, y_val=None,
                   cb_params=None, iterations_fixed=None):
    base = dict(iterations=3000, learning_rate=0.03, depth=6,
                l2_leaf_reg=5, min_data_in_leaf=20,
                subsample=0.8, colsample_bylevel=0.8)
    if cb_params is not None:
        base = dict(iterations=3000, learning_rate=cb_params["lr"],
                    depth=cb_params["depth"], l2_leaf_reg=cb_params["l2"],
                    min_data_in_leaf=cb_params["min_data"],
                    subsample=cb_params["subsample"],
                    colsample_bylevel=cb_params["colsample"])

    if iterations_fixed is not None:
        base["iterations"] = iterations_fixed
        cb = CatBoostRegressor(**base, loss_function="RMSE", eval_metric="R2",
                               random_seed=42, verbose=False)
        cb.fit(X_train, y_train)
        print(f"  CatBoost retrain (train+val): {iterations_fixed} iterationer (fixerade)")
    else:
        print("\nTranar CatBoost (med ES)...")
        cb = CatBoostRegressor(**base, loss_function="RMSE", eval_metric="R2",
                               random_seed=42, verbose=False,
                               early_stopping_rounds=100)
        cb.fit(X_train, y_train, eval_set=(X_val, y_val))
        print(f"  CatBoost: {cb.best_iteration_} iterationer (ES)")
    return cb


# ─────────────────────────────────────────────────────────────
# 16. B3: CONSTRAINED VAL-BLEND
# ─────────────────────────────────────────────────────────────
def optimize_blend_weights_val(es_lgbm, es_cb, X_val, y_val):
    """
    B3: Bestäm blend-vikter från val-prediktioner med hårda bounds [0.3, 0.7].

    Varför inte OOF: Tidiga TimeSeriesSplit-folds (~400 rader) med fasta träd
    ger R²=-133 (underfitted modell på liten fold → garbage OOF predictions).
    Constraints [0.3, 0.7] förhindrar kollaps till en-modell-dominans.

    Vikter baseras på ES-modellernas val-prediktioner (ej retrained) för
    att undvika att retrained-modellens memorerade val påverkar viktestimering.
    """
    val_lgbm = es_lgbm.predict(X_val)
    val_cb   = es_cb.predict(X_val)

    r2_lgbm = r2_score(y_val, val_lgbm)
    r2_cb   = r2_score(y_val, val_cb)
    print(f"\nB3: Blend-vikter (constrained val, ES-modeller)...")
    print(f"  Val R2: LGBM={r2_lgbm:.4f}  CB={r2_cb:.4f}")

    def neg_r2(w):
        return -r2_score(y_val, w[0] * val_lgbm + w[1] * val_cb)

    result = minimize(
        neg_r2,
        x0=[0.5, 0.5],
        method="SLSQP",
        constraints=[{"type": "eq", "fun": lambda w: w[0] + w[1] - 1}],
        bounds=[(0.4, 0.8), (0.2, 0.6)],
    )
    w_lgbm, w_cb = result.x
    blend_r2 = r2_score(y_val, w_lgbm * val_lgbm + w_cb * val_cb)
    print(f"  Blend-vikter: w_lgbm={w_lgbm:.3f}, w_cb={w_cb:.3f}, Val R2={blend_r2:.4f}")
    return w_lgbm, w_cb


# ─────────────────────────────────────────────────────────────
# 17. STACKING + RETRAIN
# ─────────────────────────────────────────────────────────────
def train_stacking(X_train, y_train, X_val, y_val, X_test,
                   lgbm_params, feature_names, cb_params=None,
                   X_full_train=None, y_full_train=None):
    """
    Steg 1: Träna på train, early-stop på val → bestäm n_iter + blend-vikter.
    Steg 2 (RETRAIN): Träna om på train+val med fixerade iterationer.

    Val-metrics rapporteras från ES-modeller (ej retrained) för att undvika
    att retrained-modellens memorerade val ger falskt hög val-R².
    """
    print("\nSteg 1: Träna base models (train+ES på val)...")

    es_lgbm = train_final_lgbm(X_train, y_train, X_val, y_val, lgbm_params)
    es_cb   = train_catboost(X_train, y_train, X_val, y_val, cb_params=cb_params)
    n_lgbm  = es_lgbm.best_iteration_ or es_lgbm.n_estimators_
    n_cb    = es_cb.best_iteration_

    # B3: blend-vikter från ES-modeller (ej retrained → ärlig val-signal)
    w_lgbm, w_cb = optimize_blend_weights_val(es_lgbm, es_cb, X_val, y_val)

    if X_full_train is not None and y_full_train is not None:
        print(f"\nSteg 2: RETRAIN på train+val ({len(X_full_train)} rader)...")
        final_lgbm = train_final_lgbm(X_full_train, y_full_train, None, None,
                                       lgbm_params, n_estimators_fixed=n_lgbm)
        final_cb   = train_catboost(X_full_train, y_full_train,
                                     cb_params=cb_params, iterations_fixed=n_cb)
    else:
        final_lgbm = es_lgbm
        final_cb   = es_cb

    # Ärliga val-predictions: från ES-modeller (tränade bara på train)
    val_lgbm_es  = es_lgbm.predict(X_val)
    val_cb_es    = es_cb.predict(X_val)
    y_val_blend  = w_lgbm * val_lgbm_es + w_cb * val_cb_es

    # Test-predictions: från final (retrained) modeller
    test_lgbm = final_lgbm.predict(X_test)
    test_cb   = final_cb.predict(X_test)
    y_test_blend = w_lgbm * test_lgbm + w_cb * test_cb

    ridge = Ridge(alpha=1.0)
    ridge.coef_ = np.array([w_lgbm, w_cb])
    ridge.intercept_ = 0.0

    return final_lgbm, final_cb, ridge, y_val_blend, y_test_blend, es_lgbm, es_cb


# ─────────────────────────────────────────────────────────────
# 18. CI-MODELLER (q10/q90)
# ─────────────────────────────────────────────────────────────
def train_ci_models(X_train, y_train, X_val, y_val):
    def _train_q(alpha, label):
        params = {
            "objective": "quantile", "alpha": alpha,
            "n_estimators": 2000, "learning_rate": 0.03,
            "num_leaves": 63, "min_child_samples": 15,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_lambda": 2.0, "n_jobs": -1, "verbose": -1, "random_state": 42,
        }
        m = lgb.LGBMRegressor(**params)
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(60, verbose=False),
                          lgb.log_evaluation(period=-1)])
        print(f"  q{alpha*100:.0f} {label}: {m.best_iteration_ or m.n_estimators_} trad")
        return m

    print("\nTranar CI-modeller (q10/q90)...")
    return _train_q(0.1, "CI-lower"), _train_q(0.9, "CI-upper")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-optuna",    action="store_true",
                        help="Hoppa Optuna helt (1 trial snabb test)")
    parser.add_argument("--no-cb-optuna", action="store_true",
                        help="Hoppa CatBoost Optuna, kör fasta params")
    args = parser.parse_args()

    n_trials_lgbm = 1 if args.no_optuna else OPTUNA_TRIALS
    run_cb_optuna = not (args.no_optuna or args.no_cb_optuna)

    print("=" * 65)
    print("  VILLA-MODELL v9 — GEOCODNING + MÄKLARE + RIKSBANK")
    print("  + BÄTTRE BALLTREE + CV-TUNING + OOF-BLEND")
    print("=" * 65)
    if args.no_optuna:
        print("  (--no-optuna: snabb test)")
    if args.no_cb_optuna:
        print("  (--no-cb-optuna: CatBoost med fasta params)")

    # 1. Ladda
    df = load_data()

    # 2. Comps
    df = compute_comps(df)

    # 3. Global marknadstrendindex
    df = compute_marknad_trend(df)

    # 4. Grannskap BallTree (v9: 25 grannar, 18m, + spread)
    df = compute_grannskap(df)

    # 5. Tidsmask
    train_mask = df["sald_datum"] <= TRAIN_END

    # 6. Area-konsolidering
    df = consolidate_areas(df, train_mask)

    # 7. Historisk area-median
    df = compute_omrade_hist(df, train_mask)

    # 8. Feature engineering
    df = engineer_features(df, train_mask)

    # 9. Target encoding
    print("\nTarget encoding av omrade_v7...")
    te_map_pris, te_global_pris = fit_target_encoder(df[train_mask], "omrade_v7", TARGET)
    df["te_omrade_pris"] = apply_target_encoder(df, "omrade_v7", te_map_pris, te_global_pris)
    print(f"  te_omrade_pris: {df['te_omrade_pris'].min():,.0f} -> {df['te_omrade_pris'].max():,.0f}")

    # 9b. Mäklare TE-karta för inference (sparas i modellen → daily_update.py)
    maklare_te_map, maklare_te_global = {}, df[TARGET].mean()
    if "maklare" in df.columns:
        maklare_te_map, maklare_te_global = fit_target_encoder(
            df[train_mask], "maklare", TARGET, smoothing=20)
        print(f"  maklare_te: {len(maklare_te_map)} unika mäklare i TE-kartan")

    # 9c. DeSO-tabell för inference: omrade → median DeSO-stats
    # Live-annonser har ingen geocodning men har omrade → slå upp via omrade-median
    _deso_cols = ["deso_median_ink_tkr", "deso_lon_ink_tkr", "deso_andel_lon_pct",
                  "deso_befolkning", "deso_median_alder", "deso_andel_0_19", "deso_andel_65_plus"]
    deso_omrade_map = {}
    deso_global_stats = {}
    _have_deso = all(c in df.columns for c in _deso_cols)
    if _have_deso:
        _tr = df[train_mask].copy()
        _grp = _tr.groupby("omrade_clean")[_deso_cols].median()
        deso_omrade_map = _grp.to_dict(orient="index")   # {omrade: {col: val, ...}}
        deso_global_stats = {c: float(df[train_mask][c].median()) for c in _deso_cols}
        print(f"  DeSO omrade-tabell: {len(deso_omrade_map)} områden sparade för inference")

    # 10. KMeans (12 kluster, lat/lon)
    km, km_scaler, cluster_feats = fit_kmeans(df[train_mask])
    df["cluster_id"] = apply_kmeans(df, km, km_scaler, cluster_feats)
    cluster_te_map, cluster_global = fit_target_encoder(
        df[train_mask], "cluster_id", TARGET, smoothing=5)
    df["cluster_te"] = df["cluster_id"].map(cluster_te_map).fillna(cluster_global)

    # 11. Feature-matris
    print("\nBygger feature-matris...")
    X, feature_names = build_feature_matrix(df)

    # 12. Tidssplit
    X_train, X_val, X_test, y_train, y_val, y_test, X_full, y_full = time_split(df, X)

    # 13. Optuna HP-tuning LGBM
    train_dates = df.loc[X_train.index, "sald_datum"].values
    lgbm_params = tune_lgbm_optuna(X_train, y_train, train_dates, n_trials=n_trials_lgbm)

    # 14. B1: CatBoost CV-tuning
    cb_params = None
    if run_cb_optuna:
        cb_params = tune_catboost_cv(X_train, y_train, train_dates, n_trials=CB_OPTUNA_TRIALS)
    else:
        print("\nCatBoost: kör fasta params (hoppar CV-tuning)")

    # 15. Stacking: blend-vikter bestäms inuti (B3 constrained val-blend)
    print("\nTranar modeller (constrained val-blend + retrain på train+val)...")
    final_lgbm, final_cb, ridge_meta, y_val_stack, y_test_stack, es_lgbm, es_cb = train_stacking(
        X_train, y_train, X_val, y_val, X_test, lgbm_params, feature_names,
        cb_params=cb_params,
        X_full_train=X_full, y_full_train=y_full,
    )
    w_lgbm = ridge_meta.coef_[0]
    w_cb   = ridge_meta.coef_[1]

    # 17. Utvardering
    # Val: från ES-modeller (ärlig, train-only) — undviker retrain-leakage
    # Test: från final retrained-modeller (mer träningsdata)
    print("\n" + "=" * 65)
    print("RESULTAT (val=ES-modeller, test=retrained):")
    val_lgbm   = es_lgbm.predict(X_val)
    test_lgbm  = final_lgbm.predict(X_test)
    val_cb     = es_cb.predict(X_val)
    test_cb    = final_cb.predict(X_test)

    m_lgbm_val  = evaluate("LightGBM Val (ES)",    y_val,  val_lgbm)
    m_lgbm_test = evaluate("LightGBM Test",        y_test, test_lgbm)
    m_cb_val    = evaluate("CatBoost Val (ES)",     y_val,  val_cb)
    m_cb_test   = evaluate("CatBoost Test",         y_test, test_cb)
    m_stack_val  = evaluate("STACK Val (ES)",       y_val,  y_val_stack)
    m_stack_test = evaluate("STACK Test",           y_test, y_test_stack)

    best_test_r2 = max(m_lgbm_test["R2"], m_cb_test["R2"], m_stack_test["R2"])
    print(f"\n  Basta Test R2: {best_test_r2:.4f} (v8 var 0.7589)")
    delta = best_test_r2 - 0.7589
    print(f"  Delta vs v8:   {delta:+.4f}")

    # 18. CI-modeller
    q10, q90 = train_ci_models(X_train, y_train, X_val, y_val)

    # 19. Feature importance (LightGBM)
    print_feature_importance(final_lgbm, feature_names)

    # 20. Välja bästa modell för produktion
    scores = {
        "lgbm":  m_lgbm_test["R2"],
        "cb":    m_cb_test["R2"],
        "stack": m_stack_test["R2"],
    }
    best_name = max(scores, key=scores.get)
    print(f"\n  Basta modell: {best_name} (R2={scores[best_name]:.4f})")

    # 21. Spara modell
    print(f"\nSparar {MODEL_OUT}...")
    model_obj = {
        "model":          final_lgbm,
        "model_catboost": final_cb,
        "model_ridge":    ridge_meta,
        "model_q10":      q10,
        "model_q90":      q90,
        "model_type":     f"stack_{best_name}",
        "feature_names":  feature_names,
        "te_map_pris":        te_map_pris,
        "te_global_pris":     te_global_pris,
        "maklare_te_map":     maklare_te_map,
        "maklare_te_global":  maklare_te_global,
        "deso_omrade_map":    deso_omrade_map,
        "deso_global_stats":  deso_global_stats,
        "kmeans":         km,
        "kmeans_scaler":  km_scaler,
        "kmeans_feats":   cluster_feats,
        "cluster_te_map": cluster_te_map,
        "cluster_global": cluster_global,
        "confidence":     0.75,
        "blend_weights":  {"lgbm": round(w_lgbm, 4), "cb": round(w_cb, 4)},
        "metrics": {
            "lgbm_val":   m_lgbm_val,
            "lgbm_test":  m_lgbm_test,
            "cb_val":     m_cb_val,
            "cb_test":    m_cb_test,
            "stack_val":  m_stack_val,
            "stack_test": m_stack_test,
        },
        "improvements": [
            "A1_geocoded_coordinates",
            "A2_maklare_target_encoding",
            "A3_riksbank_rate_features",
            "B1_catboost_cv_tuning",
            f"B2_balltree_n{N_NEIGHBORS}_{NEIGHBOR_DAYS}d_spread",
            "B3_constrained_val_blend",
            "retrain_train_plus_val",
            f"kmeans_{N_CLUSTERS}_clusters_latlon",
            "C2_scb_deso_socioeconomic_features",
        ],
    }
    joblib.dump(model_obj, MODEL_OUT)
    print(f"  OK: {MODEL_OUT}")

    # 22. Uppdatera metadata
    try:
        with open(METADATA_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        meta = {"models": {}}

    meta.setdefault("models", {})["villor"] = {
        "R2_test":    scores[best_name],
        "R2_val":     m_stack_val["R2"],
        "MAE_test":   m_stack_test["MAE"],
        "n_features": len(feature_names),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "version":    "v9",
        "best_model": best_name,
        "blend_weights": {"lgbm": round(w_lgbm, 4), "cb": round(w_cb, 4)},
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"  KLAR! Best Test R2={scores[best_name]:.4f} ({best_name.upper()})")
    print(f"  Stack Test R2={m_stack_test['R2']:.4f} | LGBM={m_lgbm_test['R2']:.4f} | CB={m_cb_test['R2']:.4f}")
    print(f"  Blend: w_lgbm={w_lgbm:.3f}, w_cb={w_cb:.3f}")
    print("=" * 65)
    print("\nNästa steg om R² < 0.83: python scripts/train_villa_v9.py --no-cb-optuna")
    print("Fas C: python scripts/daily_update.py (beskrivning/uppvarmning-pipeline)")
    print("=" * 65)


if __name__ == "__main__":
    main()
