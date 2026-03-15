"""
ML-modeller — Prisprediktering, Fynd-detektor & Marknadsanalys
================================================================
Tre modeller som bygger på varandra:
1. Prisprediktering (XGBoost / Random Forest)
2. Fynd-detektor (residualanalys)
3. Marknadsanalys (feature importance + trender)

Användning:
    from src.models import PricePredictor, DealDetector, MarketAnalyzer
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import StandardScaler
import joblib
import os

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost inte installerat — använder GradientBoosting istället")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠ SHAP inte installerat — feature importance via modellens inbyggda")


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


# ============================================================
# 1. PRISPREDIKTERING
# ============================================================

class PricePredictor:
    """
    Predikterar slutpris baserat på bostadsfeatures.
    
    Testar flera modeller och väljer den bästa:
    - Linear Regression (baseline)
    - Ridge Regression
    - Random Forest
    - XGBoost (om installerat)
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.results = {}
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict:
        """
        Träna och jämför flera modeller.
        
        Returns:
            dict med resultat per modell
        """
        print("\n🤖 Tränar modeller...")
        print(f"  Dataset: {X.shape[0]} rader, {X.shape[1]} features")
        
        self.feature_names = X.columns.tolist()
        
        # Dela upp i träning och test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Skala features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Definiera modeller
        model_configs = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),
        }
        
        if HAS_XGBOOST:
            model_configs["XGBoost"] = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            model_configs["Gradient Boosting"] = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=random_state,
            )
        
        # Träna och utvärdera
        best_r2 = -np.inf
        
        for name, model in model_configs.items():
            print(f"\n  📈 Tränar {name}...")
            
            # Linjära modeller behöver skalad data
            if name in ["Linear Regression", "Ridge"]:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Utvärdera
            metrics = {
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R²": r2_score(y_test, y_pred),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred) * 100,
            }
            
            self.models[name] = model
            self.results[name] = metrics
            
            print(f"    MAE:  {metrics['MAE']:>12,.0f} kr")
            print(f"    RMSE: {metrics['RMSE']:>12,.0f} kr")
            print(f"    R²:   {metrics['R²']:>12.4f}")
            print(f"    MAPE: {metrics['MAPE']:>12.1f}%")
            
            if metrics["R²"] > best_r2:
                best_r2 = metrics["R²"]
                self.best_model = model
                self.best_model_name = name
        
        print(f"\n  🏆 Bästa modell: {self.best_model_name} (R² = {best_r2:.4f})")
        
        # Cross-validation för bästa modell
        if self.best_model_name not in ["Linear Regression", "Ridge"]:
            cv_scores = cross_val_score(
                self.best_model, X, y, cv=5, scoring="r2", n_jobs=-1
            )
            print(f"  Cross-val R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return self.results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prediktera slutpris."""
        if self.best_model is None:
            raise ValueError("Modellen är inte tränad! Kör .train() först.")
        
        if self.best_model_name in ["Linear Regression", "Ridge"]:
            X_scaled = self.scaler.transform(X)
            return self.best_model.predict(X_scaled)
        
        return self.best_model.predict(X)
    
    def save(self, filename: str = "price_model.pkl"):
        """Spara modellen till disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump({
            "model": self.best_model,
            "model_name": self.best_model_name,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "results": self.results,
        }, path)
        print(f"  💾 Modell sparad: {path}")
    
    @classmethod
    def load(cls, filename: str = "price_model.pkl"):
        """Ladda sparad modell."""
        path = os.path.join(MODEL_DIR, filename)
        data = joblib.load(path)
        
        predictor = cls()
        predictor.best_model = data["model"]
        predictor.best_model_name = data["model_name"]
        predictor.scaler = data["scaler"]
        predictor.feature_names = data["feature_names"]
        predictor.results = data["results"]
        
        return predictor


# ============================================================
# 2. FYND-DETEKTOR
# ============================================================

class DealDetector:
    """
    Identifierar undervärderade bostäder genom att jämföra
    modellens estimat med det faktiska utgångspriset/slutpriset.
    
    En bostad är ett "fynd" om:
    - Modellens estimat >> utgångspris (annonserat under marknadsvärde)
    - Slutpris < modellens estimat (såldes billigare än förväntat)
    """
    
    def __init__(self, predictor: PricePredictor):
        self.predictor = predictor
    
    def find_deals(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        threshold_pct: float = 10.0,
    ) -> pd.DataFrame:
        """
        Hitta undervärderade bostäder.
        
        Args:
            df: Originaldatan med priser
            X: Feature-matris (samma som användes för modellen)
            threshold_pct: Minsta procentuella undervärdering
        
        Returns:
            DataFrame med potentiella fynd, sorterade efter undervärdering
        """
        print(f"\n🔍 Letar efter fynd (tröskel: {threshold_pct}%)...")
        
        # Prediktera estimerat värde
        df = df.copy()
        df["estimerat_varde"] = self.predictor.predict(X)
        
        # Beräkna avvikelse
        df["avvikelse_kr"] = df["estimerat_varde"] - df["slutpris"]
        df["avvikelse_pct"] = (
            (df["estimerat_varde"] - df["slutpris"]) / df["estimerat_varde"] * 100
        ).round(1)
        
        # Filtrera fynd
        deals = df[df["avvikelse_pct"] >= threshold_pct].copy()
        deals = deals.sort_values("avvikelse_pct", ascending=False)
        
        print(f"  ✓ Hittade {len(deals)} potentiella fynd")
        
        if not deals.empty:
            print(f"\n  Topp 5 fynd:")
            for _, row in deals.head(5).iterrows():
                addr = row.get("adress", "Okänd")[:40]
                est = row["estimerat_varde"]
                actual = row["slutpris"]
                diff = row["avvikelse_pct"]
                print(f"    {addr}")
                print(f"      Estimat: {est:,.0f} kr | Slutpris: {actual:,.0f} kr | "
                      f"Undervärderat: {diff:.1f}%")
        
        return deals
    
    def score_listing(
        self,
        features: dict,
        asking_price: int,
    ) -> dict:
        """
        Bedöm en enskild bostad.
        
        Args:
            features: Dict med bostadsfeatures
            asking_price: Utgångspris i kr
        
        Returns:
            Dict med bedömning
        """
        X = pd.DataFrame([features])[self.predictor.feature_names]
        estimated = self.predictor.predict(X)[0]
        
        diff_kr = estimated - asking_price
        diff_pct = (diff_kr / estimated) * 100
        
        if diff_pct > 15:
            verdict = "🟢 Starkt fynd"
        elif diff_pct > 5:
            verdict = "🟡 Möjligt fynd"
        elif diff_pct > -5:
            verdict = "⚪ Rimligt pris"
        else:
            verdict = "🔴 Överprissatt"
        
        return {
            "estimerat_varde": round(estimated),
            "utgangspris": asking_price,
            "skillnad_kr": round(diff_kr),
            "skillnad_pct": round(diff_pct, 1),
            "bedomning": verdict,
        }


# ============================================================
# 3. MARKNADSANALYS
# ============================================================

class MarketAnalyzer:
    """
    Analysera Örebros bostadsmarknad:
    - Feature importance (vad driver priset?)
    - Pristrender över tid
    - Säsongsvariation
    - Områdesjämförelse
    """
    
    def __init__(self, predictor: PricePredictor):
        self.predictor = predictor
    
    def feature_importance(self, X: pd.DataFrame = None) -> pd.DataFrame:
        """
        Beräkna feature importance.
        Använder SHAP om tillgängligt, annars modellens inbyggda.
        """
        print("\n📊 Beräknar feature importance...")
        
        model = self.predictor.best_model
        features = self.predictor.feature_names
        
        # Prova SHAP först
        if HAS_SHAP and X is not None:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X.head(500))  # Urval
                importance = np.abs(shap_values).mean(axis=0)
                
                fi_df = pd.DataFrame({
                    "feature": features,
                    "importance": importance,
                    "method": "SHAP",
                }).sort_values("importance", ascending=False)
                
                print("  ✓ SHAP-baserad feature importance")
                return fi_df
                
            except Exception:
                pass
        
        # Fallback: modellens inbyggda feature importance
        if hasattr(model, "feature_importances_"):
            fi_df = pd.DataFrame({
                "feature": features,
                "importance": model.feature_importances_,
                "method": "model_builtin",
            }).sort_values("importance", ascending=False)
        else:
            # Linjär modell — använd koefficienter
            fi_df = pd.DataFrame({
                "feature": features,
                "importance": np.abs(model.coef_),
                "method": "coefficients",
            }).sort_values("importance", ascending=False)
        
        print("  ✓ Feature importance beräknad")
        print(f"\n  Topp 10 viktigaste features:")
        for _, row in fi_df.head(10).iterrows():
            bar = "█" * int(row["importance"] / fi_df["importance"].max() * 20)
            print(f"    {row['feature']:<30} {bar} {row['importance']:.4f}")
        
        return fi_df
    
    def price_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analysera pristrender per kvartal och bostadstyp."""
        print("\n📈 Analyserar pristrender...")
        
        if "sald_datum" not in df.columns:
            print("  ⚠ Inget sålddatum i datan")
            return pd.DataFrame()
        
        trends = df.groupby(
            [pd.Grouper(key="sald_datum", freq="QE"), "bostadstyp"]
        ).agg(
            medianpris=("slutpris", "median"),
            medelpris=("slutpris", "mean"),
            antal_forsaljningar=("slutpris", "count"),
            median_kvm_pris=("pris_per_kvm", "median"),
        ).reset_index()
        
        print(f"  ✓ Trenddata: {len(trends)} kvartal × bostadstyper")
        return trends
    
    def seasonal_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analysera säsongsvariation i priser."""
        print("\n🗓️ Analyserar säsongsvariation...")
        
        if "sald_manad" not in df.columns:
            return pd.DataFrame()
        
        seasonal = df.groupby("sald_manad").agg(
            medianpris=("slutpris", "median"),
            antal=("slutpris", "count"),
            median_kvm=("pris_per_kvm", "median"),
            andel_budkrig=("budkrig", "mean") if "budkrig" in df.columns else ("slutpris", "count"),
        ).reset_index()
        
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
            5: "Maj", 6: "Jun", 7: "Jul", 8: "Aug",
            9: "Sep", 10: "Okt", 11: "Nov", 12: "Dec",
        }
        seasonal["manad_namn"] = seasonal["sald_manad"].map(month_names)
        
        print(f"  ✓ Säsongsdata klar")
        return seasonal
    
    def area_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """Jämför bostadspriser per område."""
        print("\n🗺️ Jämför områden...")
        
        area_col = "omrade_kategori" if "omrade_kategori" in df.columns else "omrade"
        
        if area_col not in df.columns:
            return pd.DataFrame()
        
        areas = df.groupby(area_col).agg(
            medianpris=("slutpris", "median"),
            medelpris=("slutpris", "mean"),
            antal=("slutpris", "count"),
            median_kvm_pris=("pris_per_kvm", "median"),
            median_boarea=("boarea_kvm", "median"),
        ).reset_index()
        
        areas = areas.sort_values("medianpris", ascending=False)
        
        print(f"  ✓ {len(areas)} områden jämförda")
        return areas
    
    def generate_report(self, df: pd.DataFrame, X: pd.DataFrame = None) -> dict:
        """Generera en komplett marknadsrapport."""
        
        print("\n" + "=" * 60)
        print("  MARKNADSRAPPORT — ÖREBRO KOMMUN")
        print("=" * 60)
        
        report = {
            "feature_importance": self.feature_importance(X),
            "price_trends": self.price_trends(df),
            "seasonal": self.seasonal_analysis(df),
            "area_comparison": self.area_comparison(df),
            "summary": {
                "total_listings": len(df),
                "median_price": df["slutpris"].median(),
                "mean_price": df["slutpris"].mean(),
                "price_range": (df["slutpris"].min(), df["slutpris"].max()),
                "median_kvm_price": df.get("pris_per_kvm", pd.Series()).median(),
            }
        }
        
        print(f"\n  Sammanfattning:")
        print(f"    Totalt antal försäljningar: {report['summary']['total_listings']}")
        print(f"    Medianpris: {report['summary']['median_price']:,.0f} kr")
        print(f"    Prisspann: {report['summary']['price_range'][0]:,.0f} - "
              f"{report['summary']['price_range'][1]:,.0f} kr")
        
        return report
