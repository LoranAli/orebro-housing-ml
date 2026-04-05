# ValuEstate — Örebro Housing Intelligence

ML-drivet bostadsanalysverktyg för Örebro kommun. Prisprediktering, live fynd-detektor, SHAP-analys och email-notiser via en Streamlit-dashboard.

---

## Funktioner

**Marknadsöversikt** — KPI:er, pristrender och fördelningar baserade på 6 600+ sålda bostäder i Örebro (2013–2026).

**Prisprediktering** — Ange bostadsparametrar och få ett ML-estimat med konfidensintervall (q10/q90) och jämförbara sålda objekt.

**Analysera URL** — Klistra in en Hemnet-URL och få direkt ML-estimat, SHAP waterfall-diagram (vad driver priset?) och deal score.

**Live Fynd-detektor** — Daglig scraping av aktiva Hemnet-annonser. Varje annons poängsätts (0–100 deal score) och flaggas som fynd, rimligt eller överprissatt.

**Interaktiv karta** — Alla bostäder på karta, färgkodade efter deal score eller pris per m².

**Marknadsanalys** — Pristrender, säsongsvariation, omradesanalys, budkrigsstatistik.

**Scenarioanalys** — Prisprognos 5–15 år med tre scenarier baserade på historiska Örebro-trender.

**Bevakningar & Email-notiser** — Spara sökfilter (typ, område, maxpris, antal rum, min deal score) och få email när matchande annonser dyker upp.

---

## Modeller

Tre separata modeller — en per bostadstyp — tränade med tidsbaserad train/val/test-split:

| Modell | Typ | Test R² | Test MAE |
|--------|-----|---------|----------|
| Lägenheter | LightGBM + CatBoost stack | ~0.87 | ~180 000 kr |
| Villor | LightGBM + CatBoost stack (v11) | ~0.76 | ~350 000 kr |
| Radhus | LightGBM + CatBoost stack (v2) | ~0.72 | ~365 000 kr |

Alla modeller använder:
- Optuna hyperparameter-tuning (80 trials LGBM + 30 trials CatBoost)
- Target encoding per område med smoothing
- Spatial neighbor-features via BallTree (grannskap median/vd pris/kvm)
- SCB DeSO socioekonomiska features (medianinkomst, befolkning m.m.)
- KMeans geografisk klustring
- Konfidensintervall via separata q10/q90 quantile-modeller
- SHAP TreeExplainer för förklarbarhet

---

## Projektstruktur

```
orebro-housing-ml/
├── dashboard/
│   └── app.py                  # Streamlit-app (huvud-entrypoint)
├── scripts/
│   ├── daily_update.py         # Daglig scraping + scoring + email-notiser
│   ├── url_analyzer.py         # Hemnet URL → ML-estimat + SHAP
│   ├── deal_score.py           # Deal score (0–100) per annons
│   ├── email_alerts.py         # Bevakningsfilter + Gmail SMTP
│   ├── train_villa_v10.py      # Villor-pipeline (huvud)
│   ├── train_villa_v11.py      # Villor v11 (CatBoost depth≤6)
│   └── train_radhus_v2.py      # Radhus-pipeline med DeSO + upplåtelseform
├── models/
│   ├── model_lagenheter.pkl    # Lägenheter-modell
│   ├── model_villor.pkl        # Villor-modell
│   └── model_radhus.pkl        # Radhus-modell
├── data/
│   ├── raw/                    # Scrapad rådata
│   └── processed/              # Enrichad dataset (v5)
├── requirements.txt
└── runtime.txt
```

---

## Kom igång

### Krav

- Python 3.11
- Gmail App Password (för email-notiser, valfritt)

### Installation

```bash
git clone https://github.com/LoranAli/orebro-housing-ml.git
cd orebro-housing-ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Kör dashboarden lokalt

```bash
streamlit run dashboard/app.py
```

### Daglig uppdatering (scraping + scoring)

```bash
python scripts/daily_update.py
```

Schemaläggs med cron eller GitHub Actions för automatisk körning.

---

## Miljövariabler

Email-notiser kräver:

```bash
EMAIL_SENDER=dinemail@gmail.com
EMAIL_PASSWORD=xxxx xxxx xxxx xxxx   # Gmail App Password (16 tecken)
```

Lokalt: skapa `.env` i projektroten.  
Streamlit Cloud: lägg till under Settings → Secrets.

---

## Träna om modeller

```bash
# Villor
python scripts/train_villa_v11.py

# Radhus
python scripts/train_radhus_v2.py

# Snabbtest utan Optuna
python scripts/train_villa_v11.py --no-optuna
python scripts/train_radhus_v2.py --no-optuna
```

---

## Tech Stack

| Kategori | Verktyg |
|----------|---------|
| ML | LightGBM, CatBoost, scikit-learn |
| Hyperparameter-tuning | Optuna |
| Förklarbarhet | SHAP |
| Geospatial | BallTree, geopy |
| Data | pandas, NumPy |
| Dashboard | Streamlit, Plotly, Folium |
| Email | smtplib (Gmail SMTP) |
| Datakälla | Hemnet, SCB DeSO |

---

## Deployment

Projektet är deployat på Streamlit Cloud. Modellfilerna (.pkl) är committade till repot och laddas direkt vid start.

```
runtime.txt  →  python-3.11
```
