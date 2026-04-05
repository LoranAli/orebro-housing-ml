"""
Radhus-modell v2 — LightGBM + CatBoost blend med DeSO-features
===============================================================
Samma pipeline som villa v10 men anpassad för radhus:
  - avgift_kr istf driftkostnad_ar / tomtarea
  - Färre optuna trials (1625 rader data)
  - Utdata: models/model_radhus_v2.pkl

Kör:
    cd "orebro-housing-ml 3"
    python scripts/train_radhus_v2.py
    python scripts/train_radhus_v2.py --no-optuna   # snabbtest
"""

import os, sys, json, argparse, warnings
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
for _p in ["data/processed/orebro_housing_enriched_v5.csv",
           "data/processed/orebro_housing_enriched_v4.csv",
           "data/processed/orebro_housing_enriched.csv"]:
    if os.path.exists(_p):
        DATA_PATH = _p
        break

MODEL_OUT     = "models/model_radhus_v2.pkl"
METADATA_PATH = "models/model_metadata_v10.json"
TARGET        = "slutpris"

TRAIN_END = pd.Timestamp("2024-06-30")   # kortare val-period p.g.a. lite data
VAL_END   = pd.Timestamp("2025-03-31")

AREA_MIN_SAMPLES = 3
TE_SMOOTHING     = 8
N_NEIGHBORS      = 15
NEIGHBOR_DAYS    = 548
N_CLUSTERS       = 8
OPTUNA_TRIALS    = 50
CB_OPTUNA_TRIALS = 20

CORE_FEATURES = [
    # Fysiska
    "boarea_kvm", "log_boarea", "antal_rum", "kvm_per_rum",
    # Ålder
    "bostad_alder", "byggdekad",
    # Avgift (radhus-specifik)
    "avgift_kr", "avgift_per_kvm",
    # Pris-kontext
    "omrade_hist_pris_kvm", "comps_pris_kvm_90d", "forvantat_komps_pris",
    # Marknadstrend
    "marknad_trend_6m", "marknad_trend_ratio",
    # Område / geo
    "te_omrade_pris", "grannskap_median_kvm", "grannskap_vd_kvm",
    "cluster_te", "avstand_centrum_km", "avstand_station_km",
    # Interaktion
    "boarea_avgift_interact",
    # Efterfrågan
    "antal_besok", "comps_antal_90d", "comps_pristrend",
    "prisforandring_pct", "budkrig", "prissankt",
    # Tid
    "sald_ar", "sald_manad", "sald_kvartal",
]

OPTIONAL_FEATURES = [
    "har_balkong", "har_uteplats", "vaning", "har_hiss",
    "geocode_quality_bin", "premium_score",
    "avstand_marieberg_km", "avstand_universitet_km",
    "riksbank_rate", "rate_change_6m",
    # DeSO
    "deso_median_ink_tkr", "deso_lon_ink_tkr", "deso_andel_lon_pct",
    "deso_befolkning", "deso_median_alder", "deso_andel_0_19", "deso_andel_65_plus",
]


# ─────────────────────────────────────────────────────────────
# 1. LADDA DATA
# ─────────────────────────────────────────────────────────────
def load_data():
    print(f"Laddar data... ({DATA_PATH})")
    df = pd.read_csv(DATA_PATH)
    df["sald_datum"] = pd.to_datetime(df["sald_datum"], errors="coerce")
    r = df[df["bostadstyp"] == "radhus"].copy()
    r = r.dropna(subset=["sald_datum", TARGET, "boarea_kvm"])
    print(f"  Radhus: {len(r)} rader")
    print(f"  Datum:  {r['sald_datum'].min().date()} -> {r['sald_datum'].max().date()}")
    print(f"  Slutpris median: {r[TARGET].median():,.0f} kr")
    deso_cols = ["deso_median_ink_tkr", "deso_lon_ink_tkr", "deso_andel_lon_pct",
                 "deso_befolkning", "deso_median_alder", "deso_andel_0_19", "deso_andel_65_plus"]
    found = [c for c in deso_cols if c in r.columns]
    print(f"  DeSO-features: {len(found)}/7")
    return r


# ─────────────────────────────────────────────────────────────
# 2. COMPS
# ─────────────────────────────────────────────────────────────
def compute_comps(df, window_days=90):
    print("\nBeräknar comps-features...")
    df = df.sort_values("sald_datum").copy()
    df["_pkvm"] = df[TARGET] / df["boarea_kvm"].clip(lower=10)
    cp, ca, ct = {}, {}, {}
    for area, grp in df.groupby("omrade_clean", sort=False):
        grp   = grp.sort_values("sald_datum")
        dates = grp["sald_datum"].values
        prices = grp["_pkvm"].values
        for d, ix in zip(dates, grp.index):
            lo  = d - np.timedelta64(window_days, "D")
            lo2 = d - np.timedelta64(window_days * 2, "D")
            rec = prices[(dates >= lo) & (dates < d)]
            old = prices[(dates >= lo2) & (dates < lo)]
            all_b = prices[dates < d]
            cp[ix] = np.median(rec) if len(rec) >= 2 else (np.median(all_b) if len(all_b) > 0 else np.nan)
            ca[ix] = int((dates >= lo).sum() if (dates < d).any() else 0)
            ct[ix] = round((np.median(rec) / np.median(old) - 1) * 100, 2) if len(rec) >= 2 and len(old) >= 2 else 0.0
    df["comps_pris_kvm_90d"] = pd.Series(cp)
    df["comps_antal_90d"]    = pd.Series(ca)
    df["comps_pristrend"]    = pd.Series(ct)
    df["comps_pris_kvm_90d"] = df["comps_pris_kvm_90d"].fillna(df["comps_pris_kvm_90d"].median())
    df.drop("_pkvm", axis=1, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# 3. MARKNADSTRENDINDEX
# ─────────────────────────────────────────────────────────────
def compute_marknad_trend(df):
    print("Beräknar marknadstrendindex...")
    df = df.copy().sort_values("sald_datum")
    df["_pkvm"] = df[TARGET] / df["boarea_kvm"].clip(lower=10)
    dates  = df["sald_datum"].values
    prices = df["_pkvm"].values
    window = np.timedelta64(180, "D")
    trend_vals = []
    for i, d in enumerate(dates):
        rec = prices[(dates >= d - window) & (dates < d)]
        trend_vals.append(float(np.median(rec)) if len(rec) >= 3 else np.nan)
    df["marknad_trend_6m"] = trend_vals
    df["marknad_trend_6m"] = df["marknad_trend_6m"].fillna(df["marknad_trend_6m"].median())
    df["marknad_trend_ratio"] = (
        df["comps_pris_kvm_90d"] / df["marknad_trend_6m"].replace(0, np.nan)
    ).fillna(1.0)
    df.drop("_pkvm", axis=1, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# 4. GRANNSKAP (BallTree)
# ─────────────────────────────────────────────────────────────
def compute_grannskap(df):
    print(f"Beräknar grannskap (BallTree, n={N_NEIGHBORS}, {NEIGHBOR_DAYS}d)...")
    df = df.copy()
    has_coords = df["latitude"].notna() & df["longitude"].notna()
    df["grannskap_median_kvm"] = np.nan
    df["grannskap_vd_kvm"]     = np.nan
    n_coords = has_coords.sum()
    if n_coords < N_NEIGHBORS + 1:
        df["grannskap_median_kvm"] = df["comps_pris_kvm_90d"]
        df["grannskap_vd_kvm"]     = df["comps_pris_kvm_90d"]
        print(f"  För få koordinater ({n_coords}) — fallback till comps")
        return df
    df_w  = df[has_coords].copy().sort_values("sald_datum")
    pkvm  = (df_w[TARGET] / df_w["boarea_kvm"].clip(lower=10)).values
    dates = df_w["sald_datum"].values
    coords_rad = np.radians(df_w[["latitude", "longitude"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    k_search = min(N_NEIGHBORS * 3, n_coords - 1)
    all_dists, all_idxs = tree.query(coords_rad, k=k_search + 1)
    all_dists_km = all_dists * 6371.0
    med_vals, vd_vals = [], []
    for i in range(len(df_w)):
        d = dates[i]
        cutoff = d - np.timedelta64(NEIGHBOR_DAYS, "D")
        valid = []
        for j, ni in enumerate(all_idxs[i, 1:]):
            if dates[ni] < d and dates[ni] >= cutoff:
                valid.append((ni, all_dists_km[i, j + 1]))
                if len(valid) >= N_NEIGHBORS:
                    break
        if len(valid) < 2:
            for j, ni in enumerate(all_idxs[i, 1:]):
                if dates[ni] < d and (ni, all_dists_km[i, j + 1]) not in valid:
                    valid.append((ni, all_dists_km[i, j + 1]))
                    if len(valid) >= N_NEIGHBORS:
                        break
        if valid:
            idxs_v  = [v[0] for v in valid]
            dists_v = np.array([v[1] for v in valid])
            nbr_prices = pkvm[idxs_v]
            weights = np.exp(-dists_v / 0.5)
            weights = weights / weights.sum()
            med_vals.append(float(np.median(nbr_prices)))
            vd_vals.append(float(np.dot(weights, nbr_prices)))
        else:
            med_vals.append(np.nan)
            vd_vals.append(np.nan)
    df.loc[df_w.index, "grannskap_median_kvm"] = med_vals
    df.loc[df_w.index, "grannskap_vd_kvm"]     = vd_vals
    fallback = df["comps_pris_kvm_90d"]
    df["grannskap_median_kvm"] = df["grannskap_median_kvm"].fillna(fallback)
    df["grannskap_vd_kvm"]     = df["grannskap_vd_kvm"].fillna(fallback)
    print(f"  grannskap_median_kvm: {df['grannskap_median_kvm'].median():,.0f} kr/m²")
    return df


# ─────────────────────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(df, train_mask):
    print("Feature engineering...")
    df = df.copy()
    df["log_boarea"]  = np.log(df["boarea_kvm"].clip(lower=1))
    df["byggdekad"]   = ((df["byggar"].fillna(1970) // 10) * 10).astype(int)
    df["avgift_kr"]   = df["avgift_kr"].fillna(df.loc[train_mask, "avgift_kr"].median())
    df["avgift_per_kvm"] = df["avgift_kr"] / df["boarea_kvm"].clip(lower=10)
    df["boarea_avgift_interact"] = df["boarea_kvm"] * df["avgift_kr"] / 1e6
    df["forvantat_komps_pris"]   = df["comps_pris_kvm_90d"] * df["boarea_kvm"]
    if "geocode_quality" in df.columns:
        df["geocode_quality_bin"] = (df["geocode_quality"] != "hemnet_centroid").astype(int)
    return df


# ─────────────────────────────────────────────────────────────
# 6. HJÄLPFUNKTIONER
# ─────────────────────────────────────────────────────────────
def fit_target_encoder(df_train, col, target, smoothing=TE_SMOOTHING):
    global_mean = df_train[target].mean()
    stats  = df_train.groupby(col)[target].agg(["mean", "count"])
    smooth = stats["count"] / (stats["count"] + smoothing)
    stats["encoded"] = smooth * stats["mean"] + (1 - smooth) * global_mean
    return stats["encoded"].to_dict(), global_mean

def apply_target_encoder(df, col, te_map, global_mean):
    return df[col].map(te_map).fillna(global_mean)

def consolidate_areas(df, train_mask):
    counts = df.loc[train_mask, "omrade_clean"].value_counts()
    sparse = set(counts[counts < AREA_MIN_SAMPLES].index)
    df = df.copy()
    df["omrade_v7"] = df["omrade_clean"].where(~df["omrade_clean"].isin(sparse), "Ovrigt")
    print(f"  {len(sparse)} glesa områden -> 'Ovrigt'  kvar: {df.loc[train_mask,'omrade_v7'].nunique()}")
    return df

def compute_omrade_hist(df, train_mask):
    area_hist = (df[train_mask]
                 .assign(_pkvm=lambda d: d[TARGET] / d["boarea_kvm"].clip(lower=10))
                 .groupby("omrade_v7")["_pkvm"].median()
                 .rename("omrade_hist_pris_kvm"))
    df = df.join(area_hist, on="omrade_v7")
    df["omrade_hist_pris_kvm"] = df["omrade_hist_pris_kvm"].fillna(
        df.loc[train_mask, "omrade_hist_pris_kvm"].median())
    return df

def fit_kmeans(df_train):
    cluster_feats = ["latitude", "longitude", "avstand_centrum_km",
                     "boarea_kvm", "comps_pris_kvm_90d"]
    X_c = df_train[cluster_feats].copy()
    for col in cluster_feats:
        X_c[col] = X_c[col].fillna(X_c[col].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_c)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    km.fit(X_scaled)
    print(f"  KMeans {N_CLUSTERS} kluster")
    return km, scaler, cluster_feats

def apply_kmeans(df, km, scaler, cluster_feats):
    X_c = df[cluster_feats].copy()
    for col in cluster_feats:
        X_c[col] = X_c[col].fillna(X_c[col].median())
    return km.predict(scaler.transform(X_c))

def build_feature_matrix(df):
    feats = [f for f in CORE_FEATURES if f in df.columns]
    opt   = [f for f in OPTIONAL_FEATURES if f in df.columns and f not in feats
             and df[f].nunique() > 1]
    all_feats = feats + opt
    X = df[all_feats].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)
    print(f"  Feature-matris: {X.shape[1]} features ({len(feats)} core + {len(opt)} valfria)")
    return X, all_feats

def time_split(df, X):
    train_m = df["sald_datum"] <= TRAIN_END
    val_m   = (df["sald_datum"] > TRAIN_END) & (df["sald_datum"] <= VAL_END)
    test_m  = df["sald_datum"] > VAL_END
    X_train, X_val, X_test = X[train_m], X[val_m], X[test_m]
    y_train = np.log1p(df.loc[train_m, TARGET])
    y_val   = np.log1p(df.loc[val_m,   TARGET])
    y_test  = np.log1p(df.loc[test_m,  TARGET])
    X_full  = X[train_m | val_m]
    y_full  = np.log1p(df.loc[train_m | val_m, TARGET])
    print(f"\nTidssplit: träning={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, X_full, y_full

def evaluate(name, y_true, y_pred_log):
    y_true_kr = np.expm1(y_true)
    y_pred_kr = np.expm1(y_pred_log)
    r2   = r2_score(y_true, y_pred_log)
    mae  = mean_absolute_error(y_true_kr, y_pred_kr)
    mape = float(np.mean(np.abs((y_true_kr - y_pred_kr) / y_true_kr.clip(lower=1))) * 100)
    print(f"  {name:<40} R2={r2:.4f}  MAE={mae:>8,.0f} kr  MAPE={mape:.1f}%")
    return {"R2": round(r2, 4), "MAE": round(mae, 0), "MAPE": round(mape, 1)}


# ─────────────────────────────────────────────────────────────
# 7. OPTUNA LGBM
# ─────────────────────────────────────────────────────────────
def tune_lgbm(X_train, y_train, train_dates, n_trials):
    print(f"\nOptuna LGBM ({n_trials} trials)...")
    tscv = TimeSeriesSplit(n_splits=4)
    sort_idx = np.argsort(train_dates)
    X_s, y_s = X_train.iloc[sort_idx], y_train.iloc[sort_idx]

    def objective(trial):
        params = {
            "objective": "regression", "metric": "rmse",
            "n_estimators": 600,
            "learning_rate":     trial.suggest_float("lr", 0.02, 0.12, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 31, 127),
            "max_depth":         trial.suggest_int("max_depth", 4, 8),
            "min_child_samples": trial.suggest_int("min_child", 5, 30),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 8.0),
            "n_jobs": -1, "verbose": -1, "random_state": 42,
        }
        fold_r2s = []
        for tr_idx, vl_idx in tscv.split(X_s):
            m = lgb.LGBMRegressor(**params)
            m.fit(X_s.iloc[tr_idx], y_s.iloc[tr_idx],
                  eval_set=[(X_s.iloc[vl_idx], y_s.iloc[vl_idx])],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                              lgb.log_evaluation(period=-1)])
            fold_r2s.append(r2_score(y_s.iloc[vl_idx], m.predict(X_s.iloc[vl_idx])))
        return np.mean(fold_r2s)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"  Bästa CV R2={study.best_value:.4f}  lr={best['lr']:.4f}, "
          f"leaves={best['num_leaves']}, depth={best['max_depth']}")
    return {
        "objective": "regression", "metric": "rmse", "n_estimators": 2000,
        "learning_rate": best["lr"], "num_leaves": best["num_leaves"],
        "max_depth": best["max_depth"], "min_child_samples": best["min_child"],
        "subsample": best["subsample"], "colsample_bytree": best["colsample"],
        "reg_alpha": best["reg_alpha"], "reg_lambda": best["reg_lambda"],
        "n_jobs": -1, "verbose": -1, "random_state": 42,
    }


# ─────────────────────────────────────────────────────────────
# 8. OPTUNA CATBOOST
# ─────────────────────────────────────────────────────────────
def tune_catboost(X_train, y_train, train_dates, n_trials):
    print(f"\nOptuna CatBoost ({n_trials} trials)...")
    tscv = TimeSeriesSplit(n_splits=3)
    sort_idx = np.argsort(train_dates)
    X_s, y_s = X_train.iloc[sort_idx], y_train.iloc[sort_idx]

    def objective(trial):
        params = {
            "iterations": 400,
            "learning_rate":     trial.suggest_float("lr", 0.01, 0.1, log=True),
            "depth":             trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg":       trial.suggest_float("l2", 1.0, 12.0),
            "min_data_in_leaf":  trial.suggest_int("min_data", 5, 25),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample", 0.6, 1.0),
            "loss_function": "RMSE", "eval_metric": "R2",
            "random_seed": 42, "verbose": False, "early_stopping_rounds": 40,
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
    print(f"  Bästa CV R2={study.best_value:.4f}  lr={best['lr']:.4f}, depth={best['depth']}")
    return best


# ─────────────────────────────────────────────────────────────
# 9. TRÄNA + BLEND
# ─────────────────────────────────────────────────────────────
def train_models(X_train, y_train, X_val, y_val, X_test,
                 lgbm_params, cb_params, X_full, y_full):
    print("\nSteg 1: Träna base models (ES på val)...")

    # LGBM med ES
    es_lgbm = lgb.LGBMRegressor(**lgbm_params)
    es_lgbm.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(80, verbose=False),
                            lgb.log_evaluation(period=-1)])
    n_lgbm = es_lgbm.best_iteration_ or es_lgbm.n_estimators_
    print(f"  LGBM ES: {n_lgbm} träd")

    # CatBoost med ES
    cb_base = dict(iterations=2000, learning_rate=cb_params["lr"],
                   depth=cb_params["depth"], l2_leaf_reg=cb_params["l2"],
                   min_data_in_leaf=cb_params["min_data"],
                   subsample=cb_params["subsample"],
                   colsample_bylevel=cb_params["colsample"])
    es_cb = CatBoostRegressor(**cb_base, loss_function="RMSE", eval_metric="R2",
                               random_seed=42, verbose=False, early_stopping_rounds=80)
    es_cb.fit(X_train, y_train, eval_set=(X_val, y_val))
    n_cb = es_cb.best_iteration_
    print(f"  CatBoost ES: {n_cb} iterationer")

    # Blend-vikter (constrained, val)
    val_lgbm = es_lgbm.predict(X_val)
    val_cb   = es_cb.predict(X_val)
    r2_l = r2_score(y_val, val_lgbm)
    r2_c = r2_score(y_val, val_cb)
    print(f"\nBlend-vikter (val): LGBM R2={r2_l:.4f}  CB R2={r2_c:.4f}")

    def neg_r2(w):
        return -r2_score(y_val, w[0] * val_lgbm + w[1] * val_cb)
    res = minimize(neg_r2, [0.5, 0.5], method="SLSQP",
                   constraints=[{"type": "eq", "fun": lambda w: w[0]+w[1]-1}],
                   bounds=[(0.3, 0.7), (0.3, 0.7)])
    w_lgbm, w_cb = res.x
    blend_r2 = r2_score(y_val, w_lgbm * val_lgbm + w_cb * val_cb)
    print(f"  w_lgbm={w_lgbm:.3f}, w_cb={w_cb:.3f}, Val R2={blend_r2:.4f}")

    print(f"\nSteg 2: Retrain på train+val ({len(X_full)} rader)...")
    final_lgbm = lgb.LGBMRegressor(**{**lgbm_params, "n_estimators": n_lgbm})
    final_lgbm.fit(X_full, y_full, callbacks=[lgb.log_evaluation(period=-1)])

    final_cb = CatBoostRegressor(**{**cb_base, "iterations": n_cb},
                                   loss_function="RMSE", eval_metric="R2",
                                   random_seed=42, verbose=False)
    final_cb.fit(X_full, y_full)

    # Val-predictions från ES-modeller (ärliga)
    y_val_blend  = w_lgbm * val_lgbm + w_cb * val_cb
    # Test-predictions från retrained
    test_lgbm    = final_lgbm.predict(X_test)
    test_cb      = final_cb.predict(X_test)
    y_test_blend = w_lgbm * test_lgbm + w_cb * test_cb

    return final_lgbm, final_cb, es_lgbm, es_cb, w_lgbm, w_cb, y_val_blend, y_test_blend


def train_ci(X_train, y_train, X_val, y_val):
    print("\nCI-modeller (q10/q90)...")
    def _q(alpha):
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha,
                               n_estimators=1500, learning_rate=0.03,
                               num_leaves=63, min_child_samples=15,
                               subsample=0.8, colsample_bytree=0.8,
                               n_jobs=-1, verbose=-1, random_state=42)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                          lgb.log_evaluation(period=-1)])
        return m
    return _q(0.1), _q(0.9)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-optuna", action="store_true")
    args = parser.parse_args()
    n_trials_lgbm = 1 if args.no_optuna else OPTUNA_TRIALS
    n_trials_cb   = 1 if args.no_optuna else CB_OPTUNA_TRIALS

    print("=" * 65)
    print("  RADHUS-MODELL v2 — LightGBM + CatBoost + DeSO")
    print("=" * 65)

    df = load_data()
    df = compute_comps(df)
    df = compute_marknad_trend(df)
    df = compute_grannskap(df)

    train_mask = df["sald_datum"] <= TRAIN_END

    df = consolidate_areas(df, train_mask)
    df = compute_omrade_hist(df, train_mask)
    df = engineer_features(df, train_mask)

    # Target encoding
    te_map, te_global = fit_target_encoder(df[train_mask], "omrade_v7", TARGET)
    df["te_omrade_pris"] = apply_target_encoder(df, "omrade_v7", te_map, te_global)

    # DeSO-tabell för inference
    _deso_cols = ["deso_median_ink_tkr", "deso_lon_ink_tkr", "deso_andel_lon_pct",
                  "deso_befolkning", "deso_median_alder", "deso_andel_0_19", "deso_andel_65_plus"]
    deso_omrade_map, deso_global_stats = {}, {}
    if all(c in df.columns for c in _deso_cols):
        _grp = df[train_mask].groupby("omrade_clean")[_deso_cols].median()
        deso_omrade_map   = _grp.to_dict(orient="index")
        deso_global_stats = {c: float(df[train_mask][c].median()) for c in _deso_cols}
        print(f"  DeSO omrade-tabell: {len(deso_omrade_map)} områden")

    # KMeans
    km, km_scaler, cluster_feats = fit_kmeans(df[train_mask])
    df["cluster_id"] = apply_kmeans(df, km, km_scaler, cluster_feats)
    cluster_te_map, cluster_global = fit_target_encoder(df[train_mask], "cluster_id", TARGET, smoothing=5)
    df["cluster_te"] = df["cluster_id"].map(cluster_te_map).fillna(cluster_global)

    X, feature_names = build_feature_matrix(df)
    X_train, X_val, X_test, y_train, y_val, y_test, X_full, y_full = time_split(df, X)
    train_dates = df.loc[X_train.index, "sald_datum"].values

    lgbm_params = tune_lgbm(X_train, y_train, train_dates, n_trials_lgbm)
    cb_params   = tune_catboost(X_train, y_train, train_dates, n_trials_cb)

    final_lgbm, final_cb, es_lgbm, es_cb, w_lgbm, w_cb, y_val_blend, y_test_blend = train_models(
        X_train, y_train, X_val, y_val, X_test, lgbm_params, cb_params, X_full, y_full
    )

    print("\n" + "=" * 65)
    print("RESULTAT:")
    m_lgbm_val  = evaluate("LightGBM Val (ES)",  y_val,  es_lgbm.predict(X_val))
    m_lgbm_test = evaluate("LightGBM Test",      y_test, final_lgbm.predict(X_test))
    m_cb_val    = evaluate("CatBoost Val (ES)",  y_val,  es_cb.predict(X_val))
    m_cb_test   = evaluate("CatBoost Test",      y_test, final_cb.predict(X_test))
    m_stack_val  = evaluate("STACK Val",         y_val,  y_val_blend)
    m_stack_test = evaluate("STACK Test",        y_test, y_test_blend)

    best_test_r2 = max(m_lgbm_test["R2"], m_cb_test["R2"], m_stack_test["R2"])
    print(f"\n  Bästa Test R2: {best_test_r2:.4f}  (nuv. model_radhus.pkl: 0.9047)")

    q10, q90 = train_ci(X_train, y_train, X_val, y_val)

    # Spara
    print(f"\nSparar {MODEL_OUT}...")
    model_obj = {
        "model_lgbm":      final_lgbm,
        "model_catboost":  final_cb,
        "model_q10":       q10,
        "model_q90":       q90,
        "model_name":      "LightGBM + CatBoost (radhus v2)",
        "bostadstyp":      "radhus",
        "feature_names":   feature_names,
        "te_map_pris":     te_map,
        "te_global_pris":  te_global,
        "deso_omrade_map": deso_omrade_map,
        "deso_global_stats": deso_global_stats,
        "kmeans":          km,
        "kmeans_scaler":   km_scaler,
        "kmeans_feats":    cluster_feats,
        "cluster_te_map":  cluster_te_map,
        "cluster_global":  cluster_global,
        "blend_weights":   {"lgbm": round(w_lgbm, 4), "cb": round(w_cb, 4)},
        "confidence":      0.75,
        "metrics": {
            "lgbm_val":   m_lgbm_val,  "lgbm_test":  m_lgbm_test,
            "cb_val":     m_cb_val,    "cb_test":    m_cb_test,
            "stack_val":  m_stack_val, "stack_test": m_stack_test,
        },
    }
    joblib.dump(model_obj, MODEL_OUT)
    print(f"  OK: {MODEL_OUT}")

    # Uppdatera metadata
    try:
        with open(METADATA_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        meta = {"models": {}}
    meta.setdefault("models", {})["radhus"] = {
        "R2_test": best_test_r2, "MAE_test": m_stack_test["MAE"],
        "n_features": len(feature_names),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "blend_weights": {"lgbm": round(w_lgbm, 4), "cb": round(w_cb, 4)},
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"  KLAR!  Stack Test R2={m_stack_test['R2']:.4f} | "
          f"LGBM={m_lgbm_test['R2']:.4f} | CB={m_cb_test['R2']:.4f}")
    print(f"  Blend: w_lgbm={w_lgbm:.3f}, w_cb={w_cb:.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
