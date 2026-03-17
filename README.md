# 🏠 Örebro Housing Intelligence

**ML-drivet bostadsanalysverktyg för Örebro kommun** — prisprediktering, live fynd-detektor, interaktiv karta och scenarioanalys.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Data](https://img.shields.io/badge/Data-6%2C600%20bostäder-orange)

---

## Vad är detta?

En komplett plattform som hjälper bostadsköpare, mäklare och investerare i Örebro att fatta bättre beslut. Systemet scrapar bostadsdata från Hemnet, tränar ML-modeller och presenterar insikter i en interaktiv dashboard.

### Funktioner

**📊 Marknadsöversikt** — KPI:er, prisfördelning och trender baserade på 6 600+ sålda bostäder i Örebro kommun (2013–2026).

**💰 Prisprediktering** — Skriv in bostadsfeatures och få ett AI-baserat prisestimat med jämförelse mot liknande sålda bostäder.

**🔍 Live Fynd-detektor** — Scrapar aktiva annonser på Hemnet dagligen, kör ML-modellen på varje annons och identifierar undervärderade bostäder. Inkluderar smart rimlighetscheck.

**🗺️ Interaktiv karta** — Alla sålda och aktiva bostäder på en karta, färgkodade efter pris per m² eller ML-bedömning (fynd/rimligt/överprissatt).

**📈 Marknadsanalys** — Pristrender, säsongsvariation, områdesjämförelse och budkrigsanalys.

**🔮 Scenarioanalys** — Prisprognos 5–15 år framåt med tre scenarier: snabb tillväxt, stabil marknad och recession. Baserat på historiska trender i Örebro.

---

## Resultat

### Modellprestanda

| Modell | MAE (kr) | R² | MAPE |
|--------|----------|-----|------|
| Linear Regression | 568 000 | 0.700 | 26.0% |
| Ridge Regression | 568 000 | 0.700 | 26.0% |
| Random Forest | 553 000 | 0.703 | 25.2% |
| **XGBoost (tuned)** | **504 500** | **0.748** | **22.3%** |

Hyperparameter-tuning via RandomizedSearchCV (50 kombinationer, 5-fold CV).

### Viktiga insikter

- **Boarea** är starkaste prisdrivaren (SHAP importance)
- **Rynninge** är dyraste området (median 5.4M kr)
- **Villor** har mest budkrig (53%), lägenheter minst (18%)
- **Våren** är hetaste säsongen — flest försäljningar och högst priser
- Priserna dippade 2022–2023 (räntehöjningar) men har återhämtat sig

---

## Data

| Källa | Beskrivning | Antal |
|-------|-------------|-------|
| **Hemnet** (Selenium) | Slutpriser: boarea, rum, avgift, område, datum | 6 622 listningar |
| **Hemnet detaljsidor** | Byggår, våning, hiss, balkong, driftkostnad, antal besök | 6 600 sidor |
| **Hemnet aktiva** | Bostäder till salu just nu — scrapad dagligen | ~700 annonser |
| **SCB API** | Medelinkomst, befolkning, fastighetspriser | 1 132 rader |
| **Nominatim/OSM** | Geokodning — koordinater per område | 40 områden |

---

## Tech Stack

| Kategori | Verktyg |
|----------|---------|
| **Scraping** | Selenium, BeautifulSoup, requests |
| **Data** | pandas, NumPy |
| **ML** | scikit-learn, XGBoost, SHAP, RandomizedSearchCV |
| **Geokodning** | geopy, Nominatim (OpenStreetMap) |
| **Visualisering** | Plotly, matplotlib, seaborn |
| **Dashboard** | Streamlit |
| **API** | SCB öppna data |
| **Automatisering** | cron (daglig scraping) |

---

## Projektstruktur

```
orebro-housing-ml/
├── dashboard/
│   └── app.py                          # Streamlit dashboard (6 vyer)
├── notebooks/
│   ├── 01_data_collection.ipynb        # Hemnet scraping + SCB API
│   ├── 02_eda.ipynb                    # Utforskande dataanalys
│   ├── 03_modeling.ipynb               # ML-modellering + SHAP + tuning
│   ├── 04_detail_scraper.ipynb         # Scrapa detaljsidor (byggår, hiss etc.)
│   ├── 05_live_deals.ipynb             # Live fynd-detektor
│   └── 06_geokodning.ipynb            # Geokodning av områden
├── scripts/
│   └── daily_update.py                 # Automatisk daglig uppdatering
├── data/
│   ├── raw/                            # Rå data från Hemnet
│   ├── processed/                      # Rensad och berikad data
│   ├── external/                       # SCB-data
│   └── history/                        # Dagliga snapshots
├── models/
│   └── best_model.pkl                  # Tunad XGBoost-modell
├── logs/                               # Loggfiler från daglig uppdatering
├── src/                                # Hjälpskript
├── requirements.txt
└── README.md
```

---

## Kom igång

```bash
# 1. Klona
git clone https://github.com/LoranAli/orebro-housing-ml.git
cd orebro-housing-ml

# 2. Virtuell miljö
python -m venv .venv
source .venv/bin/activate

# 3. Installera
pip install -r requirements.txt

# 4. Kör notebooks (01 → 06)

# 5. Starta dashboard
streamlit run dashboard/app.py

# 6. Daglig uppdatering (manuellt)
python scripts/daily_update.py

# 7. Schemalägg (automatiskt kl 08:00)
crontab -e
# Lägg till: 0 8 * * * cd /path/to/project && .venv/bin/python scripts/daily_update.py >> logs/daily.log 2>&1
```

---

## Features (48 st)

**Från Hemnet listningar:** slutpris, boarea, antal_rum, avgift, prisförändring, pris_per_kvm, område, bostadstyp, säljdatum

**Från detaljsidor:** byggår, våning, antal_våningar, hiss, balkong, uteplats, garage, renoverad, driftkostnad, tomtarea, biarea, antal_besök, upplåtelseform

**Engineerade:** bostad_ålder, kvm_per_rum, avgift_per_kvm, avgift_andel, säsong, budkrig, storlek_kategori, omrade_grupp (70 områden)

**Geokodning:** latitude, longitude, avstånd till centrum/station/sjukhus/universitet

---

## Live Fynd-detektor

Systemet scrapar dagligen aktiva annonser från Hemnet och bedömer varje annons:

| Bedömning | Betydelse |
|-----------|-----------|
| 🟢 Potentiellt fynd | Utgångspris >15% under modellens estimat |
| 🟡 Rimligt pris | Inom ±5-15% av estimat |
| 🔴 Överprissatt | Utgångspris >5% över estimat |
| ⚠️ Osäkert | Avvikelse >60% — modellen saknar data för området |

Smart områdesmatchning (91% matchning) översätter URL-format till modellens 70 omrade_grupp.

---

## Scenarioanalys

Tre scenarier baserade på historiska trender i Örebro:

| Scenario | Årlig tillväxt | Baserat på |
|----------|---------------|------------|
| 📈 Snabb tillväxt | +5–7%/år | Bästa åren 2019–2021 |
| ➡️ Stabil marknad | +2–3%/år | Historisk median |
| 📉 Recession | -2–5%/år | Räntechocken 2022–2023 |

---

## Författare

**Loran Ali** — Statistik, Data Analys & BI

[![GitHub](https://img.shields.io/badge/GitHub-LoranAli-black)](https://github.com/LoranAli)

---

## Licens

MIT
