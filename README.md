# Örebro Housing ML — Bostadsprisanalys för Örebro-regionen

## Projektöversikt

En komplett ML-driven bostadsanalys för Örebro kommun som kombinerar tre verktyg:

1. **Prisprediktering** — Förutspå slutpris baserat på boarea, rum, område, avgift m.m.
2. **Fynd-detektor** — Identifiera undervärderade bostäder (utgångspris < modellens estimat)
3. **Marknadsanalys** — Pristrender, säsongsvariation och feature importance

## Datakällor

| Källa | Data | Typ |
|-------|------|-----|
| Hemnet (scraping) | Slutpriser, boarea, rum, avgift, område, bostadstyp | Primär |
| SCB API | Medelinkomst, befolkning, demografi per kommun/DeSO | Kontextuell |
| OpenStreetMap | Avstånd till centrum, skolor, kommunikation | Geografisk |

## Tech Stack

- **Scraping**: `requests`, `BeautifulSoup4`, `time` (rate limiting)
- **Data**: `pandas`, `numpy`
- **ML**: `scikit-learn`, `xgboost`, `shap`
- **Visualisering**: `plotly`, `matplotlib`, `seaborn`
- **Dashboard**: `streamlit`
- **Geo**: `geopy`, `folium`

## Projektstruktur

```
orebro-housing-ml/
├── data/
│   ├── raw/              # Rå data från scraping och API:er
│   ├── processed/        # Rensad och sammanslagen data
│   └── external/         # SCB, geodata
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_analysis.ipynb
├── src/
│   ├── scraper.py        # Hemnet-scraper
│   ├── scb_fetcher.py    # SCB API-klient
│   ├── preprocessing.py  # Datarensning
│   ├── features.py       # Feature engineering
│   └── models.py         # ML-modeller
├── dashboard/
│   └── app.py            # Streamlit-dashboard
├── models/               # Sparade modeller (.pkl)
├── docs/                 # Dokumentation
├── requirements.txt
└── README.md
```

## Kom igång

```bash
# 1. Klona repot
git clone https://github.com/ditt-användarnamn/orebro-housing-ml.git
cd orebro-housing-ml

# 2. Skapa virtuell miljö
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Installera beroenden
pip install -r requirements.txt

# 4. Kör scraper (samla data)
python src/scraper.py

# 5. Hämta SCB-data
python src/scb_fetcher.py

# 6. Kör notebooks i ordning (01 → 04)

# 7. Starta dashboard
streamlit run dashboard/app.py
```

## Etik och juridik

- Scraping sker med respektfull rate limiting (2-3 sek mellan requests)
- Ingen kommersiell användning av scrapad data
- SCB-data är öppen under CC0-licens
- Projektet är för portfolio/utbildningssyfte

## Författare

[Ditt namn] — Statistik, Data Analys & BI

## Licens

MIT
