"""
Villa-modell v11 — CatBoost depth=4–6 för att minska overfit
=============================================================
Identisk med v10 utom:
  - CB depth söks i [4, 6] istf [5, 10]
  - CB iterations begränsat till 400 i tuning (ger mer regularisering)
  - Utdata: models/model_villor_v11.pkl

Kör:
    python scripts/train_villa_v11.py
    python scripts/train_villa_v11.py --no-optuna   # snabbtest
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Importera allt från v10
import train_villa_v10 as _v10

import numpy as np
import joblib
import optuna
import argparse
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

# Override
MODEL_OUT     = "models/model_villor_v11.pkl"
METADATA_PATH = _v10.METADATA_PATH


def tune_catboost_cv_v11(X_train_full, y_train_full, train_dates,
                          n_trials=_v10.CB_OPTUNA_TRIALS):
    """v11: depth 4–6 + max 400 iterations för att minska overfit."""
    print(f"\nv11: Optuna CB ({n_trials} trials, depth 4–6, iter≤400)...")
    tscv = TimeSeriesSplit(n_splits=3)
    sort_idx = np.argsort(train_dates)
    X_s = X_train_full.iloc[sort_idx]
    y_s = y_train_full.iloc[sort_idx]

    def objective(trial):
        params = {
            "iterations":            400,
            "learning_rate":         trial.suggest_float("lr", 0.01, 0.1, log=True),
            "depth":                 trial.suggest_int("depth", 4, 6),
            "l2_leaf_reg":           trial.suggest_float("l2", 2.0, 20.0),
            "min_data_in_leaf":      trial.suggest_int("min_data", 10, 40),
            "subsample":             trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bylevel":     trial.suggest_float("colsample", 0.6, 0.9),
            "loss_function":         "RMSE",
            "eval_metric":           "R2",
            "random_seed":           42,
            "verbose":               False,
            "early_stopping_rounds": 40,
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
    print(f"  Bästa CV R2={study.best_value:.4f}  depth={best['depth']}, lr={best['lr']:.4f}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-optuna",    action="store_true")
    parser.add_argument("--no-cb-optuna", action="store_true")
    args = parser.parse_args()

    n_trials_lgbm = 1 if args.no_optuna else _v10.OPTUNA_TRIALS
    run_cb_optuna = not (args.no_optuna or args.no_cb_optuna)

    print("=" * 65)
    print("  VILLA-MODELL v11 — CatBoost depth=4–6 (anti-overfit)")
    print("=" * 65)

    df = _v10.load_data()
    df = _v10.compute_comps(df)
    df = _v10.compute_marknad_trend(df)
    df = _v10.compute_grannskap(df)

    train_mask = df["sald_datum"] <= _v10.TRAIN_END

    df = _v10.consolidate_areas(df, train_mask)
    df = _v10.compute_omrade_hist(df, train_mask)
    df = _v10.engineer_features(df, train_mask)

    te_map_pris, te_global_pris = _v10.fit_target_encoder(
        df[train_mask], "omrade_v7", _v10.TARGET)
    df["te_omrade_pris"] = _v10.apply_target_encoder(
        df, "omrade_v7", te_map_pris, te_global_pris)

    maklare_te_map, maklare_te_global = {}, df[_v10.TARGET].mean()
    if "maklare" in df.columns:
        maklare_te_map, maklare_te_global = _v10.fit_target_encoder(
            df[train_mask], "maklare", _v10.TARGET, smoothing=20)

    _deso_cols = ["deso_median_ink_tkr", "deso_lon_ink_tkr", "deso_andel_lon_pct",
                  "deso_befolkning", "deso_median_alder", "deso_andel_0_19", "deso_andel_65_plus"]
    deso_omrade_map, deso_global_stats = {}, {}
    if all(c in df.columns for c in _deso_cols):
        _tr = df[train_mask].copy()
        _grp = _tr.groupby("omrade_clean")[_deso_cols].median()
        deso_omrade_map   = _grp.to_dict(orient="index")
        deso_global_stats = {c: float(df[train_mask][c].median()) for c in _deso_cols}

    km, km_scaler, cluster_feats = _v10.fit_kmeans(df[train_mask])
    df["cluster_id"] = _v10.apply_kmeans(df, km, km_scaler, cluster_feats)
    cluster_te_map, cluster_global = _v10.fit_target_encoder(
        df[train_mask], "cluster_id", _v10.TARGET, smoothing=5)
    df["cluster_te"] = df["cluster_id"].map(cluster_te_map).fillna(cluster_global)

    print("\nBygger feature-matris...")
    X, feature_names = _v10.build_feature_matrix(df)
    X_train, X_val, X_test, y_train, y_val, y_test, X_full, y_full = _v10.time_split(df, X)

    train_dates = df.loc[X_train.index, "sald_datum"].values
    lgbm_params = _v10.tune_lgbm_optuna(X_train, y_train, train_dates, n_trials=n_trials_lgbm)

    cb_params = None
    if run_cb_optuna:
        cb_params = tune_catboost_cv_v11(X_train, y_train, train_dates,
                                          n_trials=_v10.CB_OPTUNA_TRIALS)
    else:
        print("\nCatBoost: fasta params (depth=4)")
        cb_params = {"lr": 0.03, "depth": 4, "l2": 5.0,
                     "min_data": 20, "subsample": 0.8, "colsample": 0.8}

    print("\nTränar modeller...")
    final_lgbm, final_cb, ridge_meta, y_val_stack, y_test_stack, es_lgbm, es_cb = \
        _v10.train_stacking(
            X_train, y_train, X_val, y_val, X_test, lgbm_params, feature_names,
            cb_params=cb_params, X_full_train=X_full, y_full_train=y_full,
        )
    w_lgbm = ridge_meta.coef_[0]
    w_cb   = ridge_meta.coef_[1]

    print("\n" + "=" * 65)
    print("RESULTAT (v11 vs v10 baseline 0.7597):")
    val_lgbm  = es_lgbm.predict(X_val)
    test_lgbm = final_lgbm.predict(X_test)
    val_cb    = es_cb.predict(X_val)
    test_cb   = final_cb.predict(X_test)

    m_lgbm_val  = _v10.evaluate("LightGBM Val (ES)",  y_val,  val_lgbm)
    m_lgbm_test = _v10.evaluate("LightGBM Test",      y_test, test_lgbm)
    m_cb_val    = _v10.evaluate("CatBoost Val (ES)",   y_val,  val_cb)
    m_cb_test   = _v10.evaluate("CatBoost Test",       y_test, test_cb)
    m_stack_val  = _v10.evaluate("STACK Val",          y_val,  y_val_stack)
    m_stack_test = _v10.evaluate("STACK Test",         y_test, y_test_stack)

    best_test_r2 = max(m_lgbm_test["R2"], m_cb_test["R2"], m_stack_test["R2"])
    delta = best_test_r2 - 0.7597
    print(f"\n  Bästa Test R2: {best_test_r2:.4f}  delta vs v10: {delta:+.4f}")

    q10, q90 = _v10.train_ci_models(X_train, y_train, X_val, y_val)
    _v10.print_feature_importance(final_lgbm, feature_names)

    print(f"\nSparar {MODEL_OUT}...")
    model_obj = {
        "model":          final_lgbm,
        "model_catboost": final_cb,
        "model_ridge":    ridge_meta,
        "model_q10":      q10,
        "model_q90":      q90,
        "model_name":     "LightGBM + CatBoost v11 (depth≤6)",
        "model_type":     "stack",
        "bostadstyp":     "villor",
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
            "lgbm_val":  m_lgbm_val,  "lgbm_test": m_lgbm_test,
            "cb_val":    m_cb_val,    "cb_test":   m_cb_test,
            "stack_val": m_stack_val, "stack_test": m_stack_test,
        },
    }
    joblib.dump(model_obj, MODEL_OUT)
    print(f"  OK: {MODEL_OUT}")

    try:
        import json
        with open(METADATA_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        meta = {"models": {}}
    meta.setdefault("models", {})["villor_v11"] = {
        "R2_test": best_test_r2, "MAE_test": m_stack_test["MAE"],
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "blend_weights": {"lgbm": round(w_lgbm, 4), "cb": round(w_cb, 4)},
        "cb_depth": cb_params.get("depth", "?") if cb_params else "?",
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"  KLAR! Stack Test R2={m_stack_test['R2']:.4f} | "
          f"LGBM={m_lgbm_test['R2']:.4f} | CB={m_cb_test['R2']:.4f}")
    print(f"  Delta vs v10: {delta:+.4f}  (v10 baseline=0.7597)")
    print(f"  Blend: w_lgbm={w_lgbm:.3f}, w_cb={w_cb:.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
