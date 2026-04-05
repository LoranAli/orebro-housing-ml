"""
ValuEstate — Villa Model Classes
=================================
Gemensam modul för StackingVillaModel (v3) och SegmentedVillaModel (v4).
Måste importeras i daily_update.py och app.py för att joblib/pickle
ska kunna deserialisera model_villor.pkl.

OBS: Ändra ALDRIG klassnamnen eller flytta denna fil utan att också
     spara om modellerna med den nya strukturen.
"""

import numpy as np
from sklearn.metrics import r2_score


class StackingVillaModel:
    """
    Stacking-ensemble: RF + XGBoost + CatBoost med val-R²-baserade vikter.
    Används av villa-modell v3.
    Interface: predict(X) → np.ndarray
    """
    def __init__(self):
        self.base_models = {}
        self.weights = {}

    def fit(self, X_train, y_train, X_val, y_val, n_iter=20):
        raise NotImplementedError("Träning sker i train_villa_v3.py")

    def predict(self, X):
        X_arr = X.values if hasattr(X, "values") else X
        weighted_sum = np.zeros(len(X_arr))
        for mname, model in self.base_models.items():
            w = self.weights[mname]
            weighted_sum += w * model.predict(X_arr)
        return weighted_sum


class SegmentedVillaModel:
    """
    Wrapper som kombinerar budget- och high-end-modell.
    Routing baseras på forvantat_komps_pris (comps × boarea).
    Används av villa-modell v4.
    Interface: predict(X) → np.ndarray
    """
    def __init__(self, budget_seg, highend_seg, routing_split, feat_names):
        self.budget = budget_seg
        self.highend = highend_seg
        self.routing_split = routing_split
        self.feature_names = feat_names

    def _predict_seg(self, seg, X_arr):
        return sum(w * seg["models"][n].predict(X_arr) for n, w in seg["weights"].items())

    def predict(self, X):
        X_arr = X.values if hasattr(X, "values") else X
        feat = self.feature_names

        if "forvantat_komps_pris" in feat:
            ridx = feat.index("forvantat_komps_pris")
            route_vals = X_arr[:, ridx]
        else:
            route_vals = np.full(len(X_arr), self.routing_split)

        preds = np.zeros(len(X_arr))
        bud_mask = route_vals < self.routing_split
        high_mask = route_vals >= self.routing_split

        if bud_mask.any():
            preds[bud_mask] = self._predict_seg(self.budget, X_arr[bud_mask])
        if high_mask.any():
            preds[high_mask] = self._predict_seg(self.highend, X_arr[high_mask])
        return preds
