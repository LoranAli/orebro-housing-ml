"""
Streamlit Dashboard — Örebro Bostadsanalys
============================================
Interaktivt dashboard med fyra vyer:
1. Översikt — KPI:er och prisfördelning
2. Prisprediktering — skriv in features, få prisestimat
3. Fynd-detektor — undervärderade bostäder
4. Marknadsanalys — trender, säsonger, områden

Kör: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ============================================================
# KONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Örebro Bostadsanalys",
    page_icon="🏠",
    layout="wide",
)

DATA_PATH = os.path.join(os.path.dirname(
    __file__), '..', 'data', 'processed', 'orebro_housing_clean.csv')
MODEL_PATH = os.path.join(os.path.dirname(
    __file__), '..', 'models', 'best_model.pkl')


# ============================================================
# LADDA DATA OCH MODELL
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['sald_datum'] = pd.to_datetime(df['sald_datum'], errors='coerce')
    return df


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


df = load_data()
model_data = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏠 Örebro Bostadsanalys")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Välj vy:",
    ["📊 Översikt", "💰 Prisprediktering", "🔍 Fynd-detektor", "📈 Marknadsanalys"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data:** 6 600 bostäder från Hemnet")
st.sidebar.markdown("**Period:** 2013–2026")
st.sidebar.markdown("**Modell:** XGBoost (R² = 0.739)")
st.sidebar.markdown("**Område:** Örebro kommun")


# ============================================================
# ÖVERSIKT
# ============================================================

if page == "📊 Översikt":
    st.title("📊 Örebro Bostadsmarknad — Översikt")

    # KPI:er
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Medianpris", f"{df['slutpris'].median():,.0f} kr")
    col2.metric("Median kr/m²", f"{df['pris_per_kvm'].median():,.0f} kr")
    col3.metric("Antal försäljningar", f"{len(df):,}")
    col4.metric("Median boarea", f"{df['boarea_kvm'].median():.0f} m²")

    st.markdown("---")

    # Prisfördelning
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x="slutpris", color="bostadstyp", nbins=50,
            title="Prisfördelning per bostadstyp",
            labels={"slutpris": "Slutpris (kr)", "count": "Antal"},
            barmode="overlay", opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df, x="bostadstyp", y="slutpris", color="bostadstyp",
            title="Prisspridning per bostadstyp",
            labels={"bostadstyp": "Typ", "slutpris": "Slutpris (kr)"},
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
            color="medianpris", color_continuous_scale="Viridis"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PRISPREDIKTERING
# ============================================================

elif page == "💰 Prisprediktering":
    st.title("💰 Prisprediktering")
    st.markdown(
        "Skriv in bostadsfeatures och få ett prisestimat från ML-modellen.")

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
        # Hämta tillgängliga områden
        if 'omrade_clean' in df.columns:
            top_areas = df['omrade_clean'].value_counts().head(
                15).index.tolist()
            omrade = st.selectbox("Område", ['övrigt'] + sorted(top_areas))
        else:
            omrade = st.selectbox("Område", ["Örebro"])

        sald_ar = st.selectbox("År (för estimat)", [2024, 2025, 2026], index=2)

    if st.button("🔮 Beräkna prisestimat", type="primary", use_container_width=True):

        if model_data is not None:
            # Bygg feature-vektor som matchar modellens features
            features = {
                'boarea_kvm': boarea,
                'antal_rum': antal_rum,
                'avgift_kr': avgift,
                'prisforandring_pct': 0,  # Vet ej ännu
                'sald_ar': sald_ar,
                'sald_manad': 6,  # Default till juni
            }

            # One-hot encoding för bostadstyp
            features['bostadstyp_radhus'] = 1 if bostadstyp == "Radhus" else 0
            features['bostadstyp_villor'] = 1 if bostadstyp == "Villa" else 0

            # One-hot encoding för område
            feature_names = model_data['feature_names']
            for fname in feature_names:
                if fname.startswith('omrade_grupp_') and fname not in features:
                    area_name = fname.replace('omrade_grupp_', '')
                    features[fname] = 1 if omrade == area_name else 0

            # Skapa DataFrame med rätt kolumner
            X_pred = pd.DataFrame([features])

            # Se till att alla kolumner finns
            for col in feature_names:
                if col not in X_pred.columns:
                    X_pred[col] = 0

            X_pred = X_pred[feature_names]

            # Prediktera
            model = model_data['model']
            estimate = int(model.predict(X_pred)[0])

            st.markdown("---")

            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Estimerat slutpris", f"{estimate:,} kr")
            col2.metric("📐 Pris per m²", f"{int(estimate / boarea):,} kr/m²")

            low = int(estimate * 0.85)
            high = int(estimate * 1.15)
            col3.metric("📊 Intervall (±15%)", f"{low:,} – {high:,} kr")

            # Jämför med liknande bostäder
            st.markdown("---")
            st.subheader("Jämförelse med liknande bostäder")

            similar = df[
                (df['boarea_kvm'].between(boarea - 15, boarea + 15)) &
                (df['bostadstyp'] == {
                 'Lägenhet': 'lagenheter', 'Villa': 'villor', 'Radhus': 'radhus'}[bostadstyp])
            ]

            if len(similar) > 0:
                col1, col2 = st.columns(2)
                col1.metric("Liknande bostäder i datan", f"{len(similar)} st")
                col2.metric("Deras medianpris",
                            f"{similar['slutpris'].median():,.0f} kr")

                fig = px.histogram(similar, x="slutpris", nbins=30,
                                   title=f"Prisfördelning — liknande {bostadstyp.lower()} ({boarea}±15 m²)",
                                   labels={"slutpris": "Slutpris (kr)"})
                fig.add_vline(x=estimate, line_dash="dash", line_color="red",
                              annotation_text=f"Ditt estimat: {estimate:,} kr")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Inga liknande bostäder hittade i datan.")

        else:
            st.error("Modellen kunde inte laddas. Kör fas 3 först!")


# ============================================================
# FYND-DETEKTOR
# ============================================================

elif page == "🔍 Fynd-detektor":
    st.title("🔍 Fynd-detektor")
    st.markdown("Bostäder som såldes under sitt estimerade marknadsvärde.")

    threshold = st.slider("Minsta undervärdering (%)", 5, 40, 15, step=5)

    if model_data is not None and 'omrade_clean' in df.columns:
        # Förbered data för prediktion
        top_areas = df['omrade_clean'].value_counts().head(15).index.tolist()
        df_copy = df.copy()
        df_copy['omrade_grupp'] = df_copy['omrade_clean'].apply(
            lambda x: x if x in top_areas else 'övrigt')

        numeric_features = ['boarea_kvm', 'antal_rum', 'avgift_kr',
                            'prisforandring_pct', 'sald_ar', 'sald_manad']
        categorical_features = ['bostadstyp', 'omrade_grupp']

        model_input = df_copy[numeric_features +
                              categorical_features + ['slutpris']].dropna()
        model_encoded = pd.get_dummies(
            model_input, columns=categorical_features, drop_first=True)

        feature_names = model_data['feature_names']
        X_all = model_encoded.drop('slutpris', axis=1)

        for col in feature_names:
            if col not in X_all.columns:
                X_all[col] = 0
        X_all = X_all[feature_names]

        model_encoded['estimerat'] = model_data['model'].predict(X_all)
        model_encoded['avvikelse_pct'] = (
            (model_encoded['estimerat'] - model_encoded['slutpris']
             ) / model_encoded['estimerat'] * 100
        ).round(1)

        deals = model_encoded[model_encoded['avvikelse_pct'] >= threshold].sort_values(
            'avvikelse_pct', ascending=False)

        col1, col2 = st.columns(2)
        col1.metric("Potentiella fynd", f"{len(deals)} bostäder")
        col2.metric("Av totalt", f"{len(model_encoded)} analyserade")

        if not deals.empty:
            display_df = deals[['slutpris', 'estimerat',
                                'avvikelse_pct', 'boarea_kvm', 'antal_rum']].head(20)
            display_df.columns = [
                'Slutpris', 'Estimerat värde', 'Undervärdering %', 'Boarea m²', 'Rum']

            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "Slutpris": st.column_config.NumberColumn(format="%d kr"),
                    "Estimerat värde": st.column_config.NumberColumn(format="%d kr"),
                    "Undervärdering %": st.column_config.NumberColumn(format="%.1f%%"),
                }
            )

            fig = px.histogram(model_encoded, x="avvikelse_pct", nbins=50,
                               title="Fördelning av prisavvikelser",
                               labels={"avvikelse_pct": "Avvikelse (%)"})
            fig.add_vline(x=threshold, line_dash="dash", line_color="green",
                          annotation_text=f"Tröskel {threshold}%")
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Modell eller data saknas.")


# ============================================================
# MARKNADSANALYS
# ============================================================

elif page == "📈 Marknadsanalys":
    st.title("📈 Marknadsanalys — Örebro kommun")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Pristrender", "Säsongsvariation", "Områdesjämförelse", "Budkrig"])

    with tab1:
        if 'sald_datum' in df.columns:
            # Filter
            typ_filter = st.multiselect("Bostadstyp", df['bostadstyp'].unique().tolist(),
                                        default=df['bostadstyp'].unique().tolist())
            filtered = df[df['bostadstyp'].isin(typ_filter)]

            monthly = filtered.groupby(
                [pd.Grouper(key='sald_datum', freq='ME'), 'bostadstyp']
            )['slutpris'].median().reset_index()

            fig = px.line(monthly, x='sald_datum', y='slutpris', color='bostadstyp',
                          title='Prisutveckling — medianpris per månad',
                          labels={'sald_datum': 'Datum', 'slutpris': 'Medianpris (kr)'})
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
                        name='Antal försäljningar', marker_color='#1D9E75', opacity=0.5)
            fig.add_scatter(x=seasonal['manad'], y=seasonal['medianpris'],
                            name='Medianpris', yaxis='y2', line=dict(color='#D85A30', width=3))

            fig.update_layout(
                title='Säsongsvariation',
                yaxis=dict(title='Antal'),
                yaxis2=dict(title='Medianpris (kr)',
                            overlaying='y', side='right'),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if 'omrade_clean' in df.columns:
            min_antal = st.slider("Minsta antal försäljningar", 5, 50, 20)

            area_stats = df.groupby('omrade_clean').agg(
                medianpris=('slutpris', 'median'),
                antal=('slutpris', 'count'),
                median_kvm=('pris_per_kvm', 'median')
            ).reset_index()

            area_stats = area_stats[area_stats['antal'] >= min_antal]
            area_stats = area_stats.sort_values('medianpris', ascending=True)

            fig = px.bar(area_stats, x='medianpris', y='omrade_clean', orientation='h',
                         color='median_kvm', color_continuous_scale='Viridis',
                         title=f'Medianpris per område (min. {min_antal} försäljningar)',
                         labels={'medianpris': 'Medianpris (kr)', 'omrade_clean': '',
                                 'median_kvm': 'kr/m²'})
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if 'prisforandring_pct' in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                budkrig_typ = df.groupby('bostadstyp').apply(
                    lambda x: (x['prisforandring_pct'] > 0).mean() * 100
                ).reset_index(name='andel_budkrig')

                fig = px.bar(budkrig_typ, x='bostadstyp', y='andel_budkrig',
                             title='Andel budkrig per bostadstyp',
                             labels={
                                 'andel_budkrig': 'Andel budkrig (%)', 'bostadstyp': ''},
                             color='andel_budkrig', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'sasong' in df.columns:
                    budkrig_sas = df.groupby('sasong').apply(
                        lambda x: (x['prisforandring_pct'] > 0).mean() * 100
                    ).reset_index(name='andel_budkrig')

                    fig = px.bar(budkrig_sas, x='sasong', y='andel_budkrig',
                                 title='Andel budkrig per säsong',
                                 labels={
                                     'andel_budkrig': 'Andel budkrig (%)', 'sasong': ''},
                                 color='andel_budkrig', color_continuous_scale='Greens')
                    st.plotly_chart(fig, use_container_width=True)
