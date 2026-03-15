# 🏠 Örebro Housing ML — Bostadsprisanalys för Örebro kommun

En komplett ML-driven bostadsanalys för Örebro kommun med **prisprediktering**, **fynd-detektor** och **marknadsanalys** — byggd med riktig data från Hemnet och SCB.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

---

## Projektöversikt

| Komponent | Beskrivning |
|-----------|-------------|
| **Prisprediktering** | Förutspår slutpris baserat på boarea, rum, område, avgift m.m. |
| **Fynd-detektor** | Identifierar undervärderade bostäder (slutpris < modellens estimat) |
| **Marknadsanalys** | Pristrender, säsongsvariation, områdesjämförelse och budkrigsanalys |
| **Dashboard** | Interaktiv Streamlit-app för att utforska data och få prisestimat |

---

## Resultat

### Modellprestanda

| Modell | MAE (kr) | R² | MAPE |
|--------|----------|-----|------|
| Linear Regression | ~578 000 | 0.712 | ~23% |
| Ridge Regression | ~578 000 | 0.712 | ~23% |
| Random Forest | ~537 000 | 0.713 | ~21% |
| **XGBoost** | **~502 000** | **0.739** | **~19%** |

Modell: XGBoost (R² = 0.739)

### Viktiga insikter från analysen

- **Boarea** är den starkaste prisdrivaren (SHAP importance ~750k kr)
- **Rynninge** är dyraste området (median 5.4M kr), **Mellringe** bland de billigaste
- **Villor** har mest budkrig (53%), **lägenheter** minst (18%)
- **Våren** är hetaste säsongen — både flest försäljningar och högst priser
- Priserna dippade 2022-2023 (räntehöjningar) men har återhämtat sig

---

## Data

| Källa | Data | Rader |
|-------|------|-------|
| **Hemnet** (Selenium scraping) | Slutpriser, boarea, rum, avgift, område, datum | 6 622 |
| **SCB API** | Medelinkomst per åldersgrupp och år | 546 |
| **SCB API** | Befolkningsstatistik per månad | 300 |
| **SCB API** | Fastighetspriser och lagfarter | 286 |

Datainsamling: Mars 2026. Period: 2013–2026. Område: Örebro kommun.

---

## Tech Stack

- **Scraping:** Selenium, BeautifulSoup, requests
- **Data:** pandas, NumPy
- **ML:** scikit-learn, XGBoost, SHAP
- **Visualisering:** matplotlib, seaborn, Plotly
- **Dashboard:** Streamlit
- **API:** SCB öppna data (CC0-licens)

---

## Projektstruktur

```
orebro-housing-ml/
├── data/
│   ├── raw/                  # Rå data från Hemnet och SCB
│   ├── processed/            # Rensad och berikad data
│   └── external/             # SCB-data (inkomst, befolkning, priser)
├── notebooks/
│   ├── 01_data_collection.ipynb   # Fas 1: Scraping + API
│   ├── 02_eda.ipynb               # Fas 2: EDA + datarensning
│   └── 03_modeling.ipynb          # Fas 3: ML-modellering + SHAP
├── dashboard/
│   └── app.py                # Fas 4: Streamlit dashboard
├── src/
│   ├── scraper.py            # Hemnet-scraper
│   ├── scb_fetcher.py        # SCB API-klient
│   ├── preprocessing.py      # Datarensning + feature engineering
│   └── models.py             # ML-modeller
├── models/
│   └── best_model.pkl        # Sparad XGBoost-modell
├── requirements.txt
└── README.md
```

---

## Kom igång

```bash
# 1. Klona repot
git clone https://github.com/LoranAli/orebro-housing-ml.git
cd orebro-housing-ml

# 2. Skapa virtuell miljö
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 3. Installera beroenden
pip install -r requirements.txt

# 4. Kör notebooks i ordning (01 → 03)

# 5. Starta dashboard
streamlit run dashboard/app.py
```

---

## Features (22 kolumner)

**Från Hemnet:** slutpris, boarea_kvm, antal_rum, avgift_kr, sald_datum, prisforandring_pct, pris_per_kvm, omrade, bostadstyp

**Engineerade:** kvm_per_rum, avgift_per_kvm, avgift_andel, sald_ar, sald_manad, sald_kvartal, sasong, budkrig, prissankt, storlek_kategori, omrade_clean, omrade_grupp

---

## Möjliga förbättringar

- Lägga till fler områdesfeatures (avstånd till centrum via geokodning)
- Inkludera byggår och våning från Hemnets detaljsidor
- Testa GAM (Generalized Additive Model) för tolkbara icke-linjära samband
- Hyperparameter-tuning med Optuna/GridSearch

---

## Författare

**Loran Ali** — Statistik, Data Analys & BI

---

## Licens

MIT
