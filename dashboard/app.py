"""
Örebro Housing Intelligence — Dashboard
=========================================
Professionellt ML-drivet bostadsanalysverktyg för Örebro kommun.

6 vyer:
1. Översikt — KPI:er och marknadsöversikt
2. Prisprediktering — ML-baserat prisestimat
3. Live Fynd — Aktiva annonser bedömda av AI
4. Karta — Alla bostäder på karta
5. Marknadsanalys — Trender och säsonger
6. Scenarioanalys — Prisprognos 5-10 år

Kör: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime

# ============================================================
# KONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Örebro Housing Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — mörk, modern, professionell
st.markdown("""
<style>
    /* Huvudfärger */
    :root {
        --primary: #00D4AA;
        --secondary: #667eea;
        --danger: #ff6b6b;
        --warning: #feca57;
        --bg-dark: #0e1117;
        --card-bg: #1a1f2e;
        --text-main: #e0e0e0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #1a1f2e 100%);
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #252d3d 100%);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 12px;
        padding: 16px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }
    
    /* Headers */
    h1 { letter-spacing: -0.5px; }
    h2 { letter-spacing: -0.3px; }
    
    /* Dataframe styling */
    .stDataFrame { border-radius: 8px; }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00D4AA, transparent);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SÖKVÄGAR
# ============================================================

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, '..', 'data',
                         'processed', 'orebro_housing_enriched.csv')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'best_model.pkl')
ACTIVE_PATH = os.path.join(BASE_DIR, '..', 'data',
                           'processed', 'active_listings_scored.csv')
COORDS_PATH = os.path.join(BASE_DIR, '..', 'data',
                           'processed', 'area_coordinates.csv')

# Fallbacks om enriched inte finns
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, '..', 'data',
                             'processed', 'orebro_housing_clean.csv')

# ============================================================
# LADDA DATA
# ============================================================


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['sald_datum'] = pd.to_datetime(df.get('sald_datum'), errors='coerce')
    return df


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_active():
    if os.path.exists(ACTIVE_PATH):
        return pd.read_csv(ACTIVE_PATH)
    return None


@st.cache_data
def load_coords():
    if os.path.exists(COORDS_PATH):
        return pd.read_csv(COORDS_PATH)
    return None


FALLBACK_LAT, FALLBACK_LON = 59.2753, 15.2134
TYP_LABELS = {'lagenheter': 'Lägenheter',
              'villor': 'Villor', 'radhus': 'Radhus'}
COLOR_MAP = {'Lägenheter': '#00D4AA', 'Villor': '#667eea', 'Radhus': '#ff6b6b'}


@st.cache_data
def get_map_df(_df, _df_coords):
    """Returnerar df med förbättrade koordinater (cachad så loopen bara körs en gång)."""
    map_df = _df[_df['latitude'].notna() & (_df['latitude'] != 0)].copy()
    if _df_coords is None or 'omrade_clean' not in map_df.columns:
        return map_df
    good_coords = _df_coords[
        ~((_df_coords['lat'].round(4) == round(FALLBACK_LAT, 4)) &
          (_df_coords['lon'].round(4) == round(FALLBACK_LON, 4)))
    ]

    def lookup(area_name):
        if pd.isna(area_name):
            return None, None
        match = good_coords[good_coords['omrade'].str.lower()
                            == str(area_name).lower()]
        if len(match) > 0:
            return match.iloc[0]['lat'], match.iloc[0]['lon']
        prefixes = ['Radhus ', 'Lägenhet ', 'Villa ', 'Fritidshus ', 'Gård/Skog ',
                    'a Radhus ', 'b Radhus ', 'c Radhus ',
                    'a Lägenhet ', 'b Lägenhet ', 'c Lägenhet ']
        for p in prefixes:
            if str(area_name).startswith(p):
                s = area_name[len(p):]
                m = good_coords[good_coords['omrade'].str.lower() == s.lower()]
                if len(m) > 0:
                    return m.iloc[0]['lat'], m.iloc[0]['lon']
        for _, row in good_coords.iterrows():
            if len(row['omrade']) > 4 and row['omrade'].lower() in str(area_name).lower():
                return row['lat'], row['lon']
        return None, None

    fallback_mask = (
        (map_df['latitude'].round(4) == round(FALLBACK_LAT, 4)) &
        (map_df['longitude'].round(4) == round(FALLBACK_LON, 4))
    )
    for idx in map_df[fallback_mask].index:
        lat, lon = lookup(map_df.at[idx, 'omrade_clean'])
        if lat is not None:
            map_df.at[idx, 'latitude'] = lat
            map_df.at[idx, 'longitude'] = lon
    return map_df


df = load_data()
model_data = load_model()
df_active = load_active()
df_coords = load_coords()
map_df_cached = get_map_df(df, df_coords)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🏠 Örebro Housing")
    st.markdown("#### Intelligence Platform")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Översikt", "💰 Prisprediktering", "🔍 Live Fynd",
         "🗺️ Karta", "📈 Marknadsanalys", "🔮 Scenarioanalys",
         "🏦 Köpkalkyl", "💼 Investeringskalkyl"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Statistik-sidebar
    st.markdown(f"**Data:** {len(df):,} sålda bostäder")
    if df_active is not None:
        st.markdown(f"**Live:** {len(df_active)} aktiva annonser")
    if model_data:
        r2 = model_data['metrics']['R2']
        st.markdown(f"**Modell:** XGBoost (R² = {r2:.3f})")
    st.markdown(f"**Område:** Örebro kommun")
    st.markdown(f"**Period:** 2013–2026")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("*Byggd av Loran Ali*")


# ============================================================
# 1. ÖVERSIKT
# ============================================================

if page == "📊 Översikt":
    st.markdown("# 📊 Örebro Bostadsmarknad")
    st.markdown("Realtidsöversikt baserad på 6 600+ sålda bostäder")

    # KPI:er
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Medianpris", f"{df['slutpris'].median()/1e6:.1f}M kr")
    col2.metric("Median kr/m²", f"{df['pris_per_kvm'].median():,.0f}")
    col3.metric("Antal sålda", f"{len(df):,}")
    col4.metric("Median boarea", f"{df['boarea_kvm'].median():.0f} m²")
    if df_active is not None:
        fynd = len(df_active[df_active['bedomning'].str.contains(
            'fynd', case=False, na=False)])
        col5.metric("Live fynd", f"{fynd} st")
    else:
        col5.metric("Median rum", f"{df['antal_rum'].median():.0f}")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        df_plot = df.copy()
        df_plot['bostadstyp'] = df_plot['bostadstyp'].map(TYP_LABELS)
        fig = px.histogram(
            df_plot, x="slutpris", color="bostadstyp", nbins=50,
            title="Prisfördelning per bostadstyp",
            labels={
                "slutpris": "Slutpris (kr)", "count": "Antal", "bostadstyp": "Typ"},
            barmode="overlay", opacity=0.7,
            color_discrete_map=COLOR_MAP,
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            yaxis=dict(title="Antal"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Prisutveckling senaste åren (lättförståelig linjegraf)
        if 'sald_ar' in df.columns:
            yearly = df.copy()
            yearly['bostadstyp'] = yearly['bostadstyp'].map(TYP_LABELS)
            yearly = yearly.groupby(['sald_ar', 'bostadstyp'])[
                'slutpris'].median().reset_index()
            yearly = yearly[yearly['sald_ar'] >= 2018]
            fig = px.line(
                yearly, x="sald_ar", y="slutpris", color="bostadstyp",
                title="Prisutveckling per år",
                labels={"sald_ar": "År",
                        "slutpris": "Medianpris (kr)", "bostadstyp": "Typ"},
                color_discrete_map=COLOR_MAP,
                markers=True,
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)

    # Antal sålda per månad (senaste 12 månaderna)
    if 'sald_datum' in df.columns:
        recent = df[df['sald_datum'] >= '2025-01-01'].copy()
        if len(recent) > 0:
            monthly = recent.groupby(pd.Grouper(
                key='sald_datum', freq='ME')).size().reset_index(name='antal')
            fig = px.bar(
                monthly, x='sald_datum', y='antal',
                title="Antal sålda per månad (2025–2026)",
                labels={"sald_datum": "", "antal": "Antal sålda"},
                color_discrete_sequence=['#667eea'],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)

    # Topp områden
    if 'omrade_clean' in df.columns:
        area_stats = df.groupby('omrade_clean').agg(
            medianpris=('slutpris', 'median'),
            antal=('slutpris', 'count'),
        ).reset_index()
        area_stats = area_stats[area_stats['antal']
                                >= 20].nlargest(15, 'medianpris')

        fig = px.bar(
            area_stats, x="medianpris", y="omrade_clean", orientation="h",
            title="Topp 15 dyraste områden (min. 20 försäljningar)",
            labels={"medianpris": "Medianpris (kr)", "omrade_clean": ""},
            color="medianpris",
            color_continuous_scale=["#1a1f2e", "#00D4AA"]
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis={'categoryorder': 'total ascending'},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Senaste försäljningar
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📋 Senaste försäljningar")
    if 'sald_datum' in df.columns and 'omrade_clean' in df.columns:
        dagar = st.slider("Visa senaste X dagar", 7,
                          90, 30, key="senaste_dagar")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=dagar)
        senaste = df[df['sald_datum'] >= cutoff].copy()
        senaste['bostadstyp_label'] = senaste['bostadstyp'].map(TYP_LABELS)
        if len(senaste) > 0:
            st.caption(
                f"{len(senaste)} försäljningar de senaste {dagar} dagarna")
            visa_cols = ['sald_datum', 'omrade_clean', 'bostadstyp_label',
                         'slutpris', 'pris_per_kvm', 'boarea_kvm', 'antal_rum', 'prisforandring_pct']
            visa_cols = [c for c in visa_cols if c in senaste.columns]
            st.dataframe(
                senaste[visa_cols].sort_values('sald_datum', ascending=False).rename(columns={
                    'sald_datum': 'Datum', 'omrade_clean': 'Område',
                    'bostadstyp_label': 'Typ', 'slutpris': 'Slutpris',
                    'pris_per_kvm': 'kr/m²', 'boarea_kvm': 'Boarea m²',
                    'antal_rum': 'Rum', 'prisforandring_pct': 'Prisskillnad %'
                }),
                use_container_width=True, height=350,
                column_config={
                    "Slutpris": st.column_config.NumberColumn(format="%d kr"),
                    "kr/m²": st.column_config.NumberColumn(format="%d kr"),
                    "Prisskillnad %": st.column_config.NumberColumn(format="%.1f%%"),
                }
            )
        else:
            st.info(
                f"Inga försäljningar registrerade de senaste {dagar} dagarna.")


# ============================================================
# 2. PRISPREDIKTERING
# ============================================================

elif page == "💰 Prisprediktering":
    st.markdown("# 💰 Prisprediktering")
    st.markdown("Få ett prisestimat för valfri bostad i Örebro")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        boarea = st.slider("Boarea (m²)", 20, 300, 80)
        antal_rum = st.selectbox(
            "Antal rum", [1, 1.5, 2, 3, 4, 5, 6, 7, 8], index=3)
        bostad_alder = st.slider("Byggnadsår", 1950, 2026, 1990,
                                 help="Årets datum - byggnadsår = bostadens ålder")

    with col2:
        avgift = st.slider("Månadsavgift (kr)", 0, 12000, 4000, step=100,
                           help="Sätt till 0 för villor/äganderätt")
        bostadstyp = st.selectbox(
            "Bostadstyp", ["Lägenhet", "Villa", "Radhus"])
        vaning = st.number_input("Våning", min_value=0, max_value=20, value=2,
                                 help="0 för villa/radhus")

    with col3:
        if model_data is not None:
            area_groups = sorted([
                f.replace('omrade_grupp_', '')
                for f in model_data['feature_names']
                if f.startswith('omrade_grupp_')
            ])
            omrade = st.selectbox("Område", ['övrigt'] + area_groups)
        elif 'omrade_clean' in df.columns:
            top_areas = df['omrade_clean'].value_counts().head(
                70).index.tolist()
            omrade = st.selectbox("Område", ['övrigt'] + sorted(top_areas))
        else:
            omrade = st.selectbox("Område", ["Örebro"])
        sald_ar = st.selectbox("År (för estimat)", [2024, 2025, 2026], index=2)

    col4, col5, col6 = st.columns(3)
    with col4:
        har_balkong = st.checkbox(
            "🏗️ Balkong/uteplats", value=bostadstyp == "Lägenhet")
    with col5:
        har_garage = st.checkbox(
            "🚗 Garage/carport", value=bostadstyp == "Villa")
    with col6:
        renoverad = st.checkbox("🔨 Renoverad")

    if st.button("🔮 Beräkna prisestimat", type="primary", use_container_width=True):
        if model_data is not None:
            features = {
                'boarea_kvm': boarea,
                'antal_rum': antal_rum,
                'avgift_kr': avgift,
                'prisforandring_pct': 0,
                'sald_ar': sald_ar,
                'sald_manad': datetime.now().month,
            }

            features['bostad_alder'] = datetime.now().year - bostad_alder
            features['har_hiss'] = 1 if bostadstyp == "Lägenhet" else 0
            features['har_balkong'] = int(har_balkong)
            features['har_garage'] = int(har_garage)
            features['renoverad'] = int(renoverad)
            features['driftkostnad_ar'] = df['driftkostnad_ar'].median(
            ) if 'driftkostnad_ar' in df.columns else 15000
            features['tomtarea_kvm'] = 800 if bostadstyp == "Villa" else 0
            features['vaning'] = vaning
            features['antal_besok'] = df['antal_besok'].median(
            ) if 'antal_besok' in df.columns else 3500

            # One-hot bostadstyp
            features['bostadstyp_radhus'] = 1 if bostadstyp == "Radhus" else 0
            features['bostadstyp_villor'] = 1 if bostadstyp == "Villa" else 0

            # One-hot område
            feature_names = model_data['feature_names']
            for fname in feature_names:
                if fname.startswith('omrade_grupp_') and fname not in features:
                    area_name = fname.replace('omrade_grupp_', '')
                    features[fname] = 1 if omrade == area_name else 0

            X_pred = pd.DataFrame([features])
            for col in feature_names:
                if col not in X_pred.columns:
                    X_pred[col] = 0
            X_pred = X_pred[feature_names]

            estimate = int(model_data['model'].predict(X_pred)[0])

            st.markdown('<div class="custom-divider"></div>',
                        unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Estimerat pris", f"{estimate:,} kr")
            col2.metric("📐 Pris per m²", f"{int(estimate/boarea):,} kr/m²")
            low, high = int(estimate * 0.85), int(estimate * 1.15)
            col3.metric("📊 Intervall (±15%)", f"{low:,} – {high:,} kr")

            # Jämförelse
            st.markdown("---")
            typ_map = {'Lägenhet': 'lagenheter',
                       'Villa': 'villor', 'Radhus': 'radhus'}
            similar = df[
                (df['boarea_kvm'].between(boarea - 15, boarea + 15)) &
                (df['bostadstyp'] == typ_map[bostadstyp])
            ]
            if len(similar) > 0:
                col1, col2 = st.columns(2)
                col1.metric("Liknande i datan", f"{len(similar)} st")
                col2.metric("Deras medianpris",
                            f"{similar['slutpris'].median():,.0f} kr")

                fig = px.histogram(similar, x="slutpris", nbins=30,
                                   title=f"Prisfördelning — {bostadstyp.lower()} ({boarea}±15 m²)",
                                   labels={
                                       "slutpris": "Slutpris (kr)", "count": "Antal"},
                                   color_discrete_sequence=['#00D4AA'])
                fig.add_vline(x=estimate, line_dash="dash", line_color="#ff6b6b",
                              annotation_text=f"Ditt estimat: {estimate:,} kr")
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis_title="Antal")
                st.plotly_chart(fig, use_container_width=True)

            # Områdesinfo
            if omrade != 'övrigt' and 'omrade_clean' in df.columns:
                omr_data = df[df['omrade_clean'] == omrade]
                if len(omr_data) > 5:
                    st.markdown("---")
                    st.markdown(f"#### 📍 Områdesinfo — {omrade}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Medianpris",
                              f"{omr_data['slutpris'].median()/1e6:.2f}M kr")
                    c2.metric("Median kr/m²",
                              f"{omr_data['pris_per_kvm'].median():,.0f}")
                    if 'prisforandring_pct' in omr_data.columns:
                        budkrig_pct = (
                            omr_data['prisforandring_pct'] > 0).mean() * 100
                        c3.metric("Budkrigsfrekvens", f"{budkrig_pct:.0f}%")
                    if 'sald_ar' in omr_data.columns:
                        recent_median = omr_data[omr_data['sald_ar']
                                                 >= 2023]['slutpris'].median()
                        old_median = omr_data[omr_data['sald_ar']
                                              <= 2020]['slutpris'].median()
                        if old_median > 0:
                            trend = (recent_median / old_median - 1) * 100
                            c4.metric("Pristrend (2020→2023+)",
                                      f"{trend:+.0f}%")
                    if 'avstand_centrum_km' in omr_data.columns:
                        st.caption(
                            f"Avstånd centrum: ~{omr_data['avstand_centrum_km'].median():.1f} km  |  "
                            f"Station: ~{omr_data['avstand_station_km'].median():.1f} km  |  "
                            f"Universitetet: ~{omr_data['avstand_universitet_km'].median():.1f} km"
                        )

            # Ladda ner rapport
            st.markdown("---")
            rapport_html = f"""
            <html><head><meta charset='utf-8'>
            <style>body{{font-family:Arial;margin:40px;color:#222}}
            h1{{color:#00a080}}table{{border-collapse:collapse;width:100%}}
            td,th{{border:1px solid #ddd;padding:8px}}th{{background:#f0f0f0}}</style></head>
            <body>
            <h1>Örebro Housing Intelligence — Analysrapport</h1>
            <p><b>Datum:</b> {pd.Timestamp.now().strftime('%Y-%m-%d')}</p>
            <h2>Indata</h2>
            <table><tr><th>Parameter</th><th>Värde</th></tr>
            <tr><td>Bostadstyp</td><td>{bostadstyp}</td></tr>
            <tr><td>Boarea</td><td>{boarea} m²</td></tr>
            <tr><td>Antal rum</td><td>{antal_rum}</td></tr>
            <tr><td>Månadsavgift</td><td>{avgift:,} kr</td></tr>
            <tr><td>Område</td><td>{omrade}</td></tr>
            <tr><td>År</td><td>{sald_ar}</td></tr>
            </table>
            <h2>Resultat</h2>
            <table><tr><th>Mått</th><th>Värde</th></tr>
            <tr><td>Estimerat pris</td><td><b>{estimate:,} kr</b></td></tr>
            <tr><td>Pris per m²</td><td>{int(estimate/boarea):,} kr/m²</td></tr>
            <tr><td>Intervall (±15%)</td><td>{low:,} – {high:,} kr</td></tr>
            </table>
            <p style='color:#888;font-size:12px;margin-top:40px'>
            Genererad av Örebro Housing Intelligence | Byggd av Loran Ali<br>
            Observera: Estimat baseras på ML-modell och ska inte ses som finansiell rådgivning.</p>
            </body></html>"""
            st.download_button(
                "📄 Ladda ner analysrapport (HTML)",
                data=rapport_html.encode('utf-8'),
                file_name=f"bostadsanalys_{omrade}_{boarea}m2.html",
                mime="text/html",
            )
        else:
            st.error("Modellen kunde inte laddas.")


# ============================================================
# 3. LIVE FYND
# ============================================================

elif page == "🔍 Live Fynd":
    st.markdown("# 🔍 Live Fynd-detektor")
    st.markdown("Bostäder till salu just nu — bedömda av vår ML-modell")

    if df_active is not None:
        st.markdown('<div class="custom-divider"></div>',
                    unsafe_allow_html=True)

        # Datum + automatisk uppdatering-info
        if 'scrape_datum' in df_active.columns:
            senast = df_active['scrape_datum'].iloc[0]
            st.success(
                f"✅ Senast uppdaterad: **{senast}** — uppdateras automatiskt varje dag kl 08:00")

        # KPI:er
        col1, col2, col3, col4 = st.columns(4)
        fynd = df_active[df_active['bedomning'].str.contains(
            'fynd', case=False, na=False)]
        rimligt = df_active[df_active['bedomning'].str.contains(
            'Rimligt', case=False, na=False)]
        overp = df_active[df_active['bedomning'].str.contains(
            'Överprissatt', case=False, na=False)]
        osakert = df_active[df_active['bedomning'].str.contains(
            'Osäkert', case=False, na=False)]

        col1.metric("🟢 Fynd", f"{len(fynd)}")
        col2.metric("🟡 Rimligt pris", f"{len(rimligt)}")
        col3.metric("🔴 Överprissatt", f"{len(overp)}")
        col4.metric("⚠️ Osäkert", f"{len(osakert)}")

        st.markdown('<div class="custom-divider"></div>',
                    unsafe_allow_html=True)

        # Filter
        col1, col2 = st.columns(2)
        with col1:
            bed_filter = st.multiselect(
                "Bedömning",
                df_active['bedomning'].unique().tolist(),
                default=[b for b in df_active['bedomning'].unique(
                ) if 'fynd' in b.lower() or 'Rimligt' in b]
            )
        with col2:
            typ_filter = st.multiselect(
                "Bostadstyp",
                df_active['bostadstyp'].unique().tolist(),
                default=df_active['bostadstyp'].unique().tolist()
            )

        # Filtrera
        filtered = df_active[
            (df_active['bedomning'].isin(bed_filter)) &
            (df_active['bostadstyp'].isin(typ_filter))

        ].sort_values('skillnad_pct', ascending=False)

        st.markdown(f"**Visar {len(filtered)} av {len(df_active)} annonser**")

        # Visa tabell
        has_url = 'url' in filtered.columns
        display_cols = (['url'] if has_url else []) + [
            'omrade', 'bostadstyp', 'utgangspris', 'estimerat_varde',
            'skillnad_pct', 'boarea_kvm', 'antal_rum', 'bedomning']

        df_display = filtered[display_cols].rename(columns={
            'url': 'Hemnet', 'omrade': 'Område', 'bostadstyp': 'Typ',
            'utgangspris': 'Utgångspris', 'estimerat_varde': 'ML-estimat',
            'skillnad_pct': 'Avvikelse %', 'boarea_kvm': 'Boarea m²',
            'antal_rum': 'Rum', 'bedomning': 'Bedömning'
        })

        col_config = {
            "Utgångspris": st.column_config.NumberColumn(format="%d kr"),
            "ML-estimat": st.column_config.NumberColumn(format="%d kr"),
            "Avvikelse %": st.column_config.NumberColumn(format="%.1f%%"),
        }
        if has_url:
            col_config["Hemnet"] = st.column_config.LinkColumn(
                label="Hemnet 🔗", display_text="Öppna")

        st.dataframe(
            df_display,
            use_container_width=True,
            height=500,
            column_config=col_config,
        )

        st.download_button(
            label="⬇️ Ladda ner som CSV",
            data=df_display.to_csv(index=False).encode('utf-8'),
            file_name="live_fynd.csv",
            mime="text/csv",
        )


# ============================================================
# 4. KARTA
# ============================================================

elif page == "🗺️ Karta":
    st.markdown("# 🗺️ Bostadskarta — Örebro")

    if 'latitude' in df.columns and 'longitude' in df.columns:
        tab1, tab2 = st.tabs(["Sålda bostäder", "Aktiva annonser"])

        with tab1:
            st.markdown("Sålda bostäder färgkodade efter pris per m²")

            # Filter
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                typ_karta = st.multiselect(
                    "Bostadstyp", list(TYP_LABELS.values()),
                    default=list(TYP_LABELS.values()), key="karta_typ")
            with col_f2:
                if 'sald_ar' in map_df_cached.columns:
                    ar_min = int(map_df_cached['sald_ar'].min())
                    ar_max = int(map_df_cached['sald_ar'].max())
                    ar_range = st.slider("Försäljningsår", ar_min, ar_max,
                                         (ar_min, ar_max), key="karta_ar")
            with col_f3:
                pris_max = int(map_df_cached['slutpris'].quantile(0.99))
                pris_filter = st.slider("Max pris (kr)", 500000, pris_max,
                                        pris_max, step=100000, key="karta_pris",
                                        format="%d kr")

            # Applicera filter
            map_df = map_df_cached.copy()
            map_df['bostadstyp_label'] = map_df['bostadstyp'].map(TYP_LABELS)
            map_df = map_df[map_df['bostadstyp_label'].isin(typ_karta)]
            if 'sald_ar' in map_df.columns:
                map_df = map_df[
                    (map_df['sald_ar'] >= ar_range[0]) &
                    (map_df['sald_ar'] <= ar_range[1])
                ]
            map_df = map_df[map_df['slutpris'] <= pris_filter]

            st.caption(
                f"Visar {len(map_df):,} av {len(map_df_cached):,} bostäder")

            map_df = map_df.copy()
            map_df['boarea_kvm'] = map_df['boarea_kvm'].fillna(
                map_df['boarea_kvm'].median())
            map_df['pris_per_kvm'] = map_df['pris_per_kvm'].fillna(
                map_df['pris_per_kvm'].median())
            map_df['latitude'] = map_df['latitude'] + \
                np.random.uniform(-0.002, 0.002, len(map_df))
            map_df['longitude'] = map_df['longitude'] + \
                np.random.uniform(-0.002, 0.002, len(map_df))
            map_df['dot_size'] = map_df['boarea_kvm'].clip(lower=60)

            fig = px.scatter_mapbox(
                map_df, lat="latitude", lon="longitude",
                color="pris_per_kvm", size="dot_size",
                color_continuous_scale=[
                    "#1a1f2e", "#667eea", "#00D4AA", "#feca57", "#ff6b6b"],
                size_max=15, zoom=10,
                hover_name="omrade_clean" if 'omrade_clean' in map_df.columns else None,
                hover_data={'slutpris': ':,.0f',
                            'boarea_kvm': ':.0f', 'bostadstyp': True},
                title="Sålda bostäder — pris per m²",
                mapbox_style="carto-darkmatter",
            )
            fig.update_layout(
                height=600, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if df_active is not None and df_coords is not None:
                st.markdown("Aktiva annonser färgkodade efter bedömning")

                # Koppla koordinater till aktiva annonser
                active_map = df_active.merge(
                    df_coords.rename(
                        columns={'omrade': 'omrade_coord', 'lat': 'latitude', 'lon': 'longitude'}),
                    left_on='omrade', right_on='omrade_coord', how='left'
                )
                active_map = active_map[active_map['latitude'].notna()]
                active_map['boarea_kvm'] = active_map['boarea_kvm'].fillna(70)
                active_map['latitude'] = active_map['latitude'] + \
                    np.random.uniform(-0.003, 0.003, len(active_map))
                active_map['longitude'] = active_map['longitude'] + \
                    np.random.uniform(-0.003, 0.003, len(active_map))

                # Färgkoda efter bedömning
                color_map = {
                    '🟢 Potentiellt fynd': '#00D4AA',
                    '🟡 Rimligt pris': '#feca57',
                    '🔴 Överprissatt': '#ff6b6b',
                    '⚠️ Osäkert': '#888888',
                }
                active_map['color'] = active_map['bedomning'].map(
                    color_map).fillna('#888888')

                active_map['dot_size'] = active_map['boarea_kvm'].fillna(
                    70).clip(lower=60)

                fig = px.scatter_mapbox(
                    active_map, lat="latitude", lon="longitude",
                    color="bedomning",
                    size="dot_size",
                    color_discrete_map=color_map,
                    size_max=15, zoom=10,
                    hover_data={'utgangspris': ':,.0f',
                                'estimerat_varde': ':,.0f', 'boarea_kvm': ':.0f'},
                    title="Aktiva annonser — bedömda av ML-modellen",
                    mapbox_style="carto-darkmatter",
                )
                fig.update_layout(
                    height=600,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e0e0'),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "Kör 05_live_deals.ipynb och 06_geokodning.ipynb för att aktivera kartan.")
    else:
        st.info("Kör 06_geokodning.ipynb för att lägga till koordinater.")


# ============================================================
# 5. MARKNADSANALYS
# ============================================================

elif page == "📈 Marknadsanalys":
    st.markdown("# 📈 Marknadsanalys")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Pristrender", "Säsongsvariation", "Områdesjämförelse", "Budkrig", "Områdesguide"])

    with tab1:
        if 'sald_datum' in df.columns:
            typ_filter = st.multiselect(
                "Bostadstyp", list(TYP_LABELS.values()),
                default=list(TYP_LABELS.values()), key="trend_typ")
            df_trend = df.copy()
            df_trend['bostadstyp'] = df_trend['bostadstyp'].map(TYP_LABELS)
            filtered = df_trend[df_trend['bostadstyp'].isin(typ_filter)]
            monthly = filtered.groupby(
                [pd.Grouper(key='sald_datum', freq='ME'), 'bostadstyp']
            )['slutpris'].median().reset_index()

            fig = px.line(monthly, x='sald_datum', y='slutpris', color='bostadstyp',
                          title='Prisutveckling — medianpris per månad (alla områden)',
                          labels={
                              'sald_datum': '', 'slutpris': 'Medianpris (kr)', 'bostadstyp': 'Typ'},
                          color_discrete_map=COLOR_MAP)
            fig.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            # Prishistorik per specifikt område
            if 'omrade_clean' in df.columns:
                st.markdown("#### Prishistorik per område")
                omr_lista = df['omrade_clean'].value_counts(
                )[lambda x: x >= 15].index.tolist()
                valt_omrade = st.selectbox(
                    "Välj område", sorted(omr_lista), key="trend_omrade")
                omr_monthly = df[df['omrade_clean'] == valt_omrade].groupby(
                    pd.Grouper(key='sald_datum', freq='QE')
                )['slutpris'].median().reset_index()
                fig2 = px.line(omr_monthly, x='sald_datum', y='slutpris',
                               title=f'Prisutveckling — {valt_omrade}',
                               labels={'sald_datum': '',
                                       'slutpris': 'Medianpris (kr)'},
                               color_discrete_sequence=['#00D4AA'], markers=True)
                fig2.update_layout(
                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if 'sald_manad' in df.columns:
            seasonal = df.groupby('sald_manad').agg(
                medianpris=('slutpris', 'median'),
                antal=('slutpris', 'count')
            ).reset_index()
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'Maj', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec']
            seasonal['manad'] = seasonal['sald_manad'].map(
                dict(enumerate(months, 1)))

            fig = go.Figure()
            fig.add_bar(x=seasonal['manad'], y=seasonal['antal'],
                        name='Antal', marker_color='#00D4AA', opacity=0.5)
            fig.add_scatter(x=seasonal['manad'], y=seasonal['medianpris'],
                            name='Medianpris', yaxis='y2', line=dict(color='#ff6b6b', width=3))
            fig.update_layout(
                title='Säsongsvariation', template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(title='Antal'), yaxis2=dict(title='Medianpris', overlaying='y', side='right'),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if 'omrade_clean' in df.columns:
            min_antal = st.slider(
                "Minsta antal försäljningar", 5, 50, 20, key="area_min")
            area_stats = df.groupby('omrade_clean').agg(
                medianpris=('slutpris', 'median'), antal=('slutpris', 'count'),
                median_kvm=('pris_per_kvm', 'median')
            ).reset_index()
            area_stats = area_stats[area_stats['antal'] >= min_antal].sort_values(
                'medianpris', ascending=True)

            fig = px.bar(area_stats, x='medianpris', y='omrade_clean', orientation='h',
                         color='median_kvm', color_continuous_scale=["#1a1f2e", "#00D4AA"],
                         title=f'Medianpris per område (min. {min_antal} försäljningar)',
                         labels={'medianpris': 'Medianpris (kr)', 'omrade_clean': '', 'median_kvm': 'kr/m²'})
            fig.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if 'prisforandring_pct' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                df_bk = df.copy()
                df_bk['bostadstyp'] = df_bk['bostadstyp'].map(TYP_LABELS)
                bk_typ = df_bk.groupby('bostadstyp').apply(
                    lambda x: (x['prisforandring_pct'] > 0).mean() * 100
                ).reset_index(name='andel')
                fig = px.bar(bk_typ, x='bostadstyp', y='andel', title='Andel budkrig per typ',
                             labels={'andel': 'Andel (%)', 'bostadstyp': ''},
                             color='andel', color_continuous_scale=['#1a1f2e', '#ff6b6b'])
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'sasong' in df.columns:
                    bk_sas = df.groupby('sasong').apply(
                        lambda x: (x['prisforandring_pct'] > 0).mean() * 100
                    ).reset_index(name='andel')
                    fig = px.bar(bk_sas, x='sasong', y='andel', title='Andel budkrig per säsong',
                                 labels={'andel': 'Andel (%)', 'sasong': ''},
                                 color='andel', color_continuous_scale=['#1a1f2e', '#00D4AA'])
                    fig.update_layout(
                        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown("### 🏆 Områdesguide — jämför och poängsätt")
        if 'omrade_clean' in df.columns and 'sald_ar' in df.columns:
            omr_stats = df.groupby('omrade_clean').agg(
                medianpris=('slutpris', 'median'),
                antal=('slutpris', 'count'),
                median_kvm=('pris_per_kvm', 'median'),
                budkrig=('prisforandring_pct', lambda x: (x > 0).mean() * 100),
            ).reset_index()
            # Pristrend: median 2023+ vs 2019-
            trend_ny = df[df['sald_ar'] >= 2023].groupby('omrade_clean')[
                'slutpris'].median()
            trend_gammal = df[df['sald_ar'] <= 2019].groupby('omrade_clean')[
                'slutpris'].median()
            omr_stats['pristrend'] = (
                (trend_ny / trend_gammal - 1) * 100).reindex(omr_stats['omrade_clean'].values).values
            omr_stats = omr_stats[omr_stats['antal']
                                  >= 20].dropna(subset=['pristrend'])

            # Poängsätt 0-10 per dimension
            def poang(serie, hog_ar_bra=True):
                mn, mx = serie.min(), serie.max()
                if mx == mn:
                    return pd.Series([5.0] * len(serie), index=serie.index)
                norm = (serie - mn) / (mx - mn) * 10
                return norm if hog_ar_bra else 10 - norm

            omr_stats['p_pris'] = poang(
                omr_stats['medianpris'], hog_ar_bra=False).values
            omr_stats['p_trend'] = poang(
                omr_stats['pristrend'], hog_ar_bra=True).values
            omr_stats['p_budkrig'] = poang(
                omr_stats['budkrig'], hog_ar_bra=True).values
            omr_stats['p_aktivitet'] = poang(
                omr_stats['antal'], hog_ar_bra=True).values
            omr_stats['totalscore'] = (
                omr_stats[['p_pris', 'p_trend', 'p_budkrig',
                           'p_aktivitet']].mean(axis=1)
            ).round(1)

            col1, col2 = st.columns([1, 2])
            with col1:
                top_n = st.slider("Visa topp N områden",
                                  5, 40, 20, key="score_n")
                visa_omr = omr_stats.nlargest(top_n, 'totalscore')[
                    ['omrade_clean', 'totalscore',
                        'medianpris', 'pristrend', 'budkrig']
                ].rename(columns={
                    'omrade_clean': 'Område', 'totalscore': 'Score',
                    'medianpris': 'Medianpris', 'pristrend': 'Trend %', 'budkrig': 'Budkrig %'
                })
                st.dataframe(visa_omr, use_container_width=True, hide_index=True,
                             column_config={
                                 "Medianpris": st.column_config.NumberColumn(format="%d kr"),
                                 "Trend %": st.column_config.NumberColumn(format="%.0f%%"),
                                 "Budkrig %": st.column_config.NumberColumn(format="%.0f%%"),
                             })

            with col2:
                # Radarchart för valt område
                valt_score_omr = st.selectbox(
                    "Radardiagram för område",
                    omr_stats.nlargest(top_n, 'totalscore')[
                        'omrade_clean'].tolist(),
                    key="score_omr")
                row = omr_stats[omr_stats['omrade_clean']
                                == valt_score_omr].iloc[0]
                kategorier = ['Prisvärdhet', 'Pristrend',
                              'Budkrigsfrekvens', 'Marknadsaktivitet']
                varden = [row['p_pris'], row['p_trend'],
                          row['p_budkrig'], row['p_aktivitet']]
                fig = go.Figure(go.Scatterpolar(
                    r=varden + [varden[0]],
                    theta=kategorier + [kategorier[0]],
                    fill='toself', fillcolor='rgba(0,212,170,0.2)',
                    line=dict(color='#00D4AA', width=2),
                    name=valt_score_omr,
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 10],
                                               gridcolor='#333', linecolor='#333'),
                               angularaxis=dict(gridcolor='#333', linecolor='#555')),
                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                    title=f"Scorecard — {valt_score_omr}",
                    height=400, showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 6. SCENARIOANALYS
# ============================================================

elif page == "🔮 Scenarioanalys":
    st.markdown("# 🔮 Scenarioanalys — Prisprognos")
    st.markdown(
        "Hur kan bostadspriserna i Örebro utvecklas de kommande 5-10 åren?")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Välj bostadens nuvarande värde
    col1, col2 = st.columns(2)
    with col1:
        nuvarande_pris = st.number_input("Nuvarande bostadens värde (kr)",
                                         value=2500000, step=100000, format="%d")
    with col2:
        ar_framat = st.slider("Antal år framåt", 1, 15, 10)

    # Beräkna historiska trender från datan
    if 'sald_ar' in df.columns:
        yearly = df.groupby('sald_ar')['slutpris'].median().reset_index()
        yearly = yearly[yearly['sald_ar'] >= 2015]  # Senaste 10 åren

        if len(yearly) > 2:
            # Beräkna årlig förändring
            yearly['forandring'] = yearly['slutpris'].pct_change()
            avg_growth = yearly['forandring'].median()
            boom_growth = yearly['forandring'].quantile(0.8)
            bust_growth = yearly['forandring'].quantile(0.2)
    else:
        avg_growth = 0.03
        boom_growth = 0.07
        bust_growth = -0.03

    # Tre scenarier
    scenarios = {
        '📈 Snabb tillväxt': {
            'rate': max(boom_growth, 0.05),
            'color': '#00D4AA',
            'desc': 'Stark ekonomi, låga räntor, hög efterfrågan. Baserat på bästa åren 2019-2021.'
        },
        '➡️ Stabil marknad': {
            'rate': max(avg_growth, 0.02),
            'color': '#667eea',
            'desc': 'Normal utveckling, historisk median. Balanserad marknad.'
        },
        '📉 Recession': {
            'rate': min(bust_growth, -0.02),
            'color': '#ff6b6b',
            'desc': 'Höga räntor, ekonomisk nedgång. Baserat på 2022-2023 (räntechocken).'
        }
    }

    # Beräkna framtida priser
    years = list(range(2026, 2026 + ar_framat + 1))

    fig = go.Figure()

    result_table = []

    for name, scenario in scenarios.items():
        prices = [nuvarande_pris]
        for y in range(ar_framat):
            # Recession: första 2 åren ner, sen återhämtning
            if 'Recession' in name and y < 2:
                rate = scenario['rate']
            elif 'Recession' in name:
                rate = abs(scenario['rate']) * 0.5  # Långsam återhämtning
            else:
                rate = scenario['rate']
            prices.append(prices[-1] * (1 + rate))

        fig.add_trace(go.Scatter(
            x=years, y=prices, name=name.split(' ', 1)[1],
            line=dict(color=scenario['color'], width=3),
            fill='tonexty' if 'Recession' not in name else None,
        ))

        result_table.append({
            'Scenario': name,
            f'Pris {years[-1]}': f"{prices[-1]:,.0f} kr",
            'Förändring': f"{((prices[-1]/nuvarande_pris)-1)*100:+.0f}%",
            'Årlig tillväxt': f"{scenario['rate']*100:+.1f}%/år",
        })

    fig.update_layout(
        title=f"Prisprognos {years[0]}–{years[-1]}",
        xaxis_title="År",
        yaxis_title="Estimerat värde (kr)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)

    # Resultattabell
    st.markdown("### Scenarioresultat")
    result_df = pd.DataFrame(result_table)
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    # Antaganden
    with st.expander("📋 Antaganden bakom scenarierna"):
        for name, scenario in scenarios.items():
            st.markdown(f"**{name}** ({scenario['rate']*100:+.1f}%/år)")
            st.markdown(f"_{scenario['desc']}_")
            st.markdown("")
        st.markdown("""
        **Viktigt:** Dessa prognoser är baserade på historiska trender i Örebro kommun
        och ska inte ses som finansiell rådgivning. Faktiska prisförändringar beror på
        räntor, ekonomisk politik, befolkningsutveckling och andra faktorer.
        """)


# ============================================================
# 7. KÖPKALKYL
# ============================================================

elif page == "🏦 Köpkalkyl":
    st.markdown("# 🏦 Köpkalkyl")
    st.markdown(
        "Beräkna din totala köpkostnad och vad bostaden kostar per månad")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💳 Bolånekalkylator", "🧾 Köpkostnadskalkyl"])

    with tab1:
        st.markdown("#### Vad kostar bolånet per månad?")
        col1, col2, col3 = st.columns(3)
        with col1:
            kop_pris = st.number_input(
                "Köpeskilling (kr)", value=2500000, step=50000, format="%d")
            kontant_pct = st.slider("Kontantinsats (%)", 10, 50, 15)
        with col2:
            ranta = st.slider("Bolåneränta (%)", 1.0, 10.0, 4.5, step=0.1)
            amort_pct = st.slider("Amortering (%/år)", 0.0, 3.0, 1.0, step=0.1)
        with col3:
            lan_tid = st.slider("Löptid (år)", 5, 50, 30)
            avgift_ko = st.number_input(
                "Månadsavgift (kr)", value=3500, step=100, format="%d")

        kontant = int(kop_pris * kontant_pct / 100)
        lan = kop_pris - kontant
        r = ranta / 100 / 12
        n = lan_tid * 12
        if r > 0:
            manads_ranta = lan * r * (1 + r)**n / ((1 + r)**n - 1)
        else:
            manads_ranta = lan / n
        manads_amort = lan * amort_pct / 100 / 12
        total_manad = manads_ranta + avgift_ko

        st.markdown('<div class="custom-divider"></div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Kontantinsats", f"{kontant:,} kr")
        c2.metric("Lånebelopp", f"{lan:,} kr")
        c3.metric("Månadsbetalning (ränta)", f"{int(manads_ranta):,} kr")
        c4.metric("Total månadskostnad", f"{int(total_manad):,} kr")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            total_betalt = manads_ranta * n
            rante_kostnad = total_betalt - lan
            st.metric("Total räntekostnad (hela lånet)",
                      f"{int(rante_kostnad):,} kr")
            st.metric("Totalt betalt", f"{int(total_betalt + kontant):,} kr")
        with col2:
            # Amorteringsplan — saldo per år
            saldo = [lan]
            for _ in range(lan_tid):
                saldo.append(
                    max(0, saldo[-1] * (1 + ranta/100) - manads_ranta * 12))
            fig = px.area(
                x=list(range(2026, 2026 + lan_tid + 1)), y=saldo,
                title="Lånesaldo över tid",
                labels={"x": "År", "y": "Kvarvarande lån (kr)"},
                color_discrete_sequence=['#667eea'],
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Totala köpkostnader utöver köpeskillingen")
        col1, col2 = st.columns(2)
        with col1:
            kop2 = st.number_input("Köpeskilling (kr)", value=2500000, step=50000,
                                   format="%d", key="kop2")
            lan2 = st.number_input("Nytt bolån (kr)", value=2000000, step=50000,
                                   format="%d", key="lan2")
            befintligt_pantbrev = st.number_input("Befintliga pantbrev (kr)", value=0,
                                                  step=100000, format="%d")
            akop = st.selectbox("Köpare", ["Privatperson", "Juridisk person"])
        with col2:
            # Beräkningar
            stampelskatt_pct = 0.015 if akop == "Privatperson" else 0.0425
            stampelskatt = round(kop2 * stampelskatt_pct /
                                 1000 + 0.5) * 1000  # avrundat uppåt
            lagfart = 825
            nytt_pantbrev = max(0, lan2 - befintligt_pantbrev)
            pantbrev_kostnad = int(nytt_pantbrev * 0.02) + \
                (375 if nytt_pantbrev > 0 else 0)
            maklararv = int(kop2 * 0.025)
            total_extra = stampelskatt + lagfart + pantbrev_kostnad

            st.markdown("##### Kostnadsspecifikation")
            poster = {
                "Stämpelskatt": stampelskatt,
                "Lagfartsavgift": lagfart,
                "Pantbrev (2% av nytt lån)": pantbrev_kostnad,
            }
            for namn, belopp in poster.items():
                st.markdown(f"**{namn}:** {belopp:,} kr")
            st.markdown(
                f"*Mäklararvode (ca 2.5%, betalas av säljare):* ~{maklararv:,} kr*")
            st.markdown(f"### Totalt extra: **{total_extra:,} kr**")
            st.markdown(f"### Total kostnad: **{kop2 + total_extra:,} kr**")
            st.caption(
                "*Mäklararvodet betalas normalt av säljaren och ingår ej i totalen.")


# ============================================================
# 8. INVESTERINGSKALKYL
# ============================================================

elif page == "💼 Investeringskalkyl":
    st.markdown("# 💼 Investeringskalkyl")
    st.markdown(
        "Beräkna avkastning och lönsamhet för en investeringsfastighet i Örebro")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        inv_pris = st.number_input(
            "Köpeskilling (kr)", value=2000000, step=50000, format="%d")
        kontant_inv = st.slider("Kontantinsats (%)", 10,
                                50, 25, key="inv_kontant")
        ranta_inv = st.slider("Bolåneränta (%)", 1.0, 10.0,
                              4.5, step=0.1, key="inv_ranta")
    with col2:
        hyra = st.number_input("Hyresintäkt/månad (kr)",
                               value=9000, step=500, format="%d")
        avgift_inv = st.number_input("Månadsavgift (kr)", value=3500, step=100,
                                     format="%d", key="inv_avgift")
        drift = st.number_input("Driftkostnad/år (kr)",
                                value=15000, step=1000, format="%d")
    with col3:
        vakans_pct = st.slider("Vakansgrad (%)", 0, 20,
                               5, help="Andel av året utan hyresgäst")
        vardeok_pct = st.slider(
            "Förväntad värdeökning (%/år)", 0.0, 8.0, 3.0, step=0.5)
        horisont = st.slider("Investeringshorisont (år)", 1, 20, 10)

    lan_inv = int(inv_pris * (1 - kontant_inv / 100))
    kontant_kr = inv_pris - lan_inv
    r_inv = ranta_inv / 100 / 12
    n_inv = 30 * 12
    manads_lan = lan_inv * r_inv * \
        (1 + r_inv)**n_inv / ((1 + r_inv)**n_inv -
                              1) if r_inv > 0 else lan_inv / n_inv

    effektiv_hyra = hyra * 12 * (1 - vakans_pct / 100)
    arliga_kostnader = avgift_inv * 12 + drift + manads_lan * 12
    netto_cashflow_ar = effektiv_hyra - arliga_kostnader
    brutto_yield = effektiv_hyra / inv_pris * 100
    netto_yield = netto_cashflow_ar / inv_pris * 100
    aterbetalning = kontant_kr / \
        netto_cashflow_ar if netto_cashflow_ar > 0 else float('inf')

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bruttoavkastning", f"{brutto_yield:.1f}%")
    c2.metric("Nettoavkastning", f"{netto_yield:.1f}%",
              delta="positivt" if netto_yield > 0 else "negativt cashflow")
    c3.metric("Månads-cashflow", f"{int(netto_cashflow_ar/12):,} kr",
              delta=f"{'+' if netto_cashflow_ar > 0 else ''}{int(netto_cashflow_ar/12):,} kr/mån")
    if aterbetalning < 50:
        c4.metric("Återbetalningstid", f"{aterbetalning:.1f} år")
    else:
        c4.metric("Återbetalningstid", "Ej lönsamt")

    st.markdown("---")
    # Värdeutveckling + total avkastning över horisont
    years_inv = list(range(2026, 2026 + horisont + 1))
    varden = [inv_pris * (1 + vardeok_pct / 100) **
              i for i in range(horisont + 1)]
    ack_cashflow = [max(0, netto_cashflow_ar * i) for i in range(horisont + 1)]
    total_avk = [v + c - inv_pris for v, c in zip(varden, ack_cashflow)]

    fig = go.Figure()
    fig.add_scatter(x=years_inv, y=varden, name="Fastighetsvärde",
                    line=dict(color='#00D4AA', width=3))
    fig.add_scatter(x=years_inv, y=[inv_pris] * len(years_inv), name="Köpeskilling",
                    line=dict(color='#888', width=1, dash='dash'))
    fig.add_bar(x=years_inv, y=ack_cashflow, name="Ackumulerat cashflow",
                marker_color='rgba(102,126,234,0.5)')
    fig.update_layout(
        title=f"Värdeutveckling & cashflow över {horisont} år",
        xaxis_title="År", yaxis_title="Kronor",
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', height=400, hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)

    tot_vinst = total_avk[-1]
    roi = tot_vinst / kontant_kr * 100
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Fastighetsvärde år {2026+horisont}", f"{int(varden[-1]):,} kr",
                delta=f"+{int(varden[-1]-inv_pris):,} kr")
    col2.metric(f"Ackumulerat cashflow", f"{int(ack_cashflow[-1]):,} kr")
    col3.metric(f"Total avkastning på insats", f"{roi:.0f}%",
                delta=f"{int(tot_vinst):,} kr vinst")

    st.caption("⚠️ Kalkylen är en uppskattning och inkluderar ej skatter, försäljningskostnader "
               "eller oförutsedda utgifter. Ska inte ses som finansiell rådgivning.")
