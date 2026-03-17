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


df = load_data()
model_data = load_model()
df_active = load_active()
df_coords = load_coords()

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
         "🗺️ Karta", "📈 Marknadsanalys", "🔮 Scenarioanalys"],
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

    # Prisfördelning + Boxplot
    TYP_LABELS = {'lagenheter': 'Lägenheter', 'villor': 'Villor', 'radhus': 'Radhus'}
    COLOR_MAP = {'Lägenheter': '#00D4AA', 'Villor': '#667eea', 'Radhus': '#ff6b6b'}

    col1, col2 = st.columns(2)

    with col1:
        df_plot = df.copy()
        df_plot['bostadstyp'] = df_plot['bostadstyp'].map(TYP_LABELS)
        fig = px.histogram(
            df_plot, x="slutpris", color="bostadstyp", nbins=50,
            title="Prisfördelning per bostadstyp",
            labels={"slutpris": "Slutpris (kr)", "count": "Antal", "bostadstyp": "Typ"},
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
                labels={"sald_ar": "År", "slutpris": "Medianpris (kr)", "bostadstyp": "Typ"},
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


# ============================================================
# 2. PRISPREDIKTERING
# ============================================================

elif page == "💰 Prisprediktering":
    st.markdown("# 💰 Prisprediktering")
    st.markdown("Få ett AI-baserat prisestimat för valfri bostad i Örebro")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        boarea = st.slider("Boarea (m²)", 20, 300, 80)
        antal_rum = st.selectbox(
            "Antal rum", [1, 1.5, 2, 3, 4, 5, 6, 7, 8], index=3)

    with col2:
        avgift = st.slider("Månadsavgift (kr)", 0, 12000, 4000, step=100,
                           help="Sätt till 0 för villor/äganderätt")
        bostadstyp = st.selectbox(
            "Bostadstyp", ["Lägenhet", "Villa", "Radhus"])

    with col3:
        if 'omrade_clean' in df.columns:
            top_areas = df['omrade_clean'].value_counts().head(
                70).index.tolist()
            omrade = st.selectbox("Område", ['övrigt'] + sorted(top_areas))
        else:
            omrade = st.selectbox("Område", ["Örebro"])
        sald_ar = st.selectbox("År (för estimat)", [2024, 2025, 2026], index=2)

    if st.button("🔮 Beräkna prisestimat", type="primary", use_container_width=True):
        if model_data is not None:
            features = {
                'boarea_kvm': boarea,
                'antal_rum': antal_rum,
                'avgift_kr': avgift,
                'prisforandring_pct': 0,
                'sald_ar': sald_ar,
                'sald_manad': 6,
            }

            # Nya features med defaults
            features['bostad_alder'] = df['bostad_alder'].median(
            ) if 'bostad_alder' in df.columns else 48
            features['har_hiss'] = 1 if bostadstyp == "Lägenhet" else 0
            features['har_balkong'] = 1 if bostadstyp == "Lägenhet" else 0
            features['har_garage'] = 1 if bostadstyp == "Villa" else 0
            features['renoverad'] = 0
            features['driftkostnad_ar'] = df['driftkostnad_ar'].median(
            ) if 'driftkostnad_ar' in df.columns else 15000
            features['tomtarea_kvm'] = 800 if bostadstyp == "Villa" else 0
            features['vaning'] = 2 if bostadstyp == "Lägenhet" else 0
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
                                   labels={"slutpris": "Slutpris (kr)", "count": "Antal"},
                                   color_discrete_sequence=['#00D4AA'])
                fig.add_vline(x=estimate, line_dash="dash", line_color="#ff6b6b",
                              annotation_text=f"Ditt estimat: {estimate:,} kr")
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis_title="Antal")
                st.plotly_chart(fig, use_container_width=True)
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

        # Datum
        if 'scrape_datum' in df_active.columns:
            st.caption(
                f"Senast uppdaterad: {df_active['scrape_datum'].iloc[0]}")

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
        display_cols = ['omrade', 'bostadstyp', 'utgangspris', 'estimerat_varde',
                        'skillnad_pct', 'boarea_kvm', 'antal_rum', 'bedomning']

        st.dataframe(
            filtered[display_cols].rename(columns={
                'omrade': 'Område', 'bostadstyp': 'Typ',
                'utgangspris': 'Utgångspris', 'estimerat_varde': 'ML-estimat',
                'skillnad_pct': 'Avvikelse %', 'boarea_kvm': 'Boarea m²',
                'antal_rum': 'Rum', 'bedomning': 'Bedömning'
            }),
            use_container_width=True,
            height=500,
            column_config={
                "Utgångspris": st.column_config.NumberColumn(format="%d kr"),
                "ML-estimat": st.column_config.NumberColumn(format="%d kr"),
                "Avvikelse %": st.column_config.NumberColumn(format="%.1f%%"),
            }
        )


# ============================================================
# 4. KARTA
# ============================================================

elif page == "🗺️ Karta":
    st.markdown("# 🗺️ Bostadskarta — Örebro")

    FALLBACK_LAT, FALLBACK_LON = 59.2753, 15.2134

    # Hjälpfunktion: slå upp koordinat med prefix-strippning
    def lookup_coords(area_name, coords_df):
        if coords_df is None or pd.isna(area_name):
            return None, None
        # Filtrera bort poster med fallback-koordinat så de inte blockerar prefix-match
        good_coords = coords_df[
            ~((coords_df['lat'].round(4) == round(FALLBACK_LAT, 4)) &
              (coords_df['lon'].round(4) == round(FALLBACK_LON, 4)))
        ]
        # Direkt match (utan fallback-poster)
        match = good_coords[good_coords['omrade'].str.lower() == str(area_name).lower()]
        if len(match) > 0:
            return match.iloc[0]['lat'], match.iloc[0]['lon']
        # Strippa vanliga prefix och försök igen
        prefixes = ['Radhus ', 'Lägenhet ', 'Villa ', 'Fritidshus ',
                    'Gård/Skog ', 'a Radhus ', 'b Radhus ', 'c Radhus ',
                    'a Lägenhet ', 'b Lägenhet ', 'c Lägenhet ']
        for prefix in prefixes:
            if str(area_name).startswith(prefix):
                stripped = area_name[len(prefix):]
                match = good_coords[good_coords['omrade'].str.lower() == stripped.lower()]
                if len(match) > 0:
                    return match.iloc[0]['lat'], match.iloc[0]['lon']
        # Partiell match — hitta känt område som ingår i namnet
        for _, row in good_coords.iterrows():
            if len(row['omrade']) > 4 and row['omrade'].lower() in str(area_name).lower():
                return row['lat'], row['lon']
        return None, None

    # Karta med historiska data
    if 'latitude' in df.columns and 'longitude' in df.columns:
        tab1, tab2 = st.tabs(["Sålda bostäder", "Aktiva annonser"])

        with tab1:
            st.markdown("Alla sålda bostäder färgkodade efter pris per m²")

            map_df = df[df['latitude'].notna() & (df['latitude'] != 0)].copy()

            # Förbättra koordinater för bostäder med fallback-koordinat
            if df_coords is not None and 'omrade_clean' in map_df.columns:
                fallback_mask = (
                    (map_df['latitude'].round(4) == round(FALLBACK_LAT, 4)) &
                    (map_df['longitude'].round(4) == round(FALLBACK_LON, 4))
                )
                for idx in map_df[fallback_mask].index:
                    area = map_df.at[idx, 'omrade_clean']
                    lat, lon = lookup_coords(area, df_coords)
                    if lat is not None:
                        map_df.at[idx, 'latitude'] = lat
                        map_df.at[idx, 'longitude'] = lon

            # Fyll NaN i size och color kolumner
            map_df['boarea_kvm'] = map_df['boarea_kvm'].fillna(
                map_df['boarea_kvm'].median())
            map_df['pris_per_kvm'] = map_df['pris_per_kvm'].fillna(
                map_df['pris_per_kvm'].median())
            # Undvik att alla hamnar på exakt samma punkt
            map_df['latitude'] = map_df['latitude'] + \
                np.random.uniform(-0.002, 0.002, len(map_df))
            map_df['longitude'] = map_df['longitude'] + \
                np.random.uniform(-0.002, 0.002, len(map_df))

            fig = px.scatter_mapbox(
                map_df, lat="latitude", lon="longitude",
                color="pris_per_kvm",
                size="boarea_kvm",
                color_continuous_scale=[
                    "#1a1f2e", "#667eea", "#00D4AA", "#feca57", "#ff6b6b"],
                size_max=15, zoom=10,
                hover_name="omrade_clean" if 'omrade_clean' in map_df.columns else None,
                hover_data={'slutpris': ':,.0f',
                            'boarea_kvm': ':.0f', 'bostadstyp': True},
                title="Sålda bostäder i Örebro kommun",
                mapbox_style="carto-darkmatter",
            )
            fig.update_layout(
                height=600,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'),
            )
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

                fig = px.scatter_mapbox(
                    active_map, lat="latitude", lon="longitude",
                    color="bedomning",
                    size="boarea_kvm",
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

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Pristrender", "Säsongsvariation", "Områdesjämförelse", "Budkrig"])

    with tab1:
        if 'sald_datum' in df.columns:
            typ_filter = st.multiselect("Bostadstyp", df['bostadstyp'].unique().tolist(),
                                        default=df['bostadstyp'].unique().tolist(), key="trend_typ")
            filtered = df[df['bostadstyp'].isin(typ_filter)]
            monthly = filtered.groupby(
                [pd.Grouper(key='sald_datum', freq='ME'), 'bostadstyp']
            )['slutpris'].median().reset_index()

            fig = px.line(monthly, x='sald_datum', y='slutpris', color='bostadstyp',
                          title='Prisutveckling — medianpris per månad',
                          labels={'sald_datum': '',
                                  'slutpris': 'Medianpris (kr)'},
                          color_discrete_sequence=['#00D4AA', '#667eea', '#ff6b6b'])
            fig.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

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
                bk_typ = df.groupby('bostadstyp').apply(
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
