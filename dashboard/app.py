"""
ValuEstate — Bostadsanalys med AI
====================================
Professionellt ML-drivet bostadsanalysverktyg.

8 vyer: Live Fynd | Prisprediktering | Karta | Marknadsanalys
        Scenarioanalys | Köpkalkyl | Investeringskalkyl | Om modellen

Kör: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
import json
import glob
from datetime import datetime, date

def _model_r2(m):
    """Hämtar bästa test-R² oavsett om modellen använder 'R2' (v8) eller nästlad dict (v10)."""
    met = m.get('metrics', {})
    if 'R2' in met:
        return met['R2']
    # v10: ta max av alla test-nycklar
    candidates = [met[k]['R2'] for k in ('lgbm_test', 'cb_test', 'stack_test')
                  if k in met and isinstance(met[k], dict) and 'R2' in met[k]]
    return max(candidates) if candidates else 0.0

# Villa model classes — måste importeras för pickle-deserialisering
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
try:
    from villa_models import (  # noqa: F401 — krävs för pickle-deserialisering
        StackingVillaModel, SegmentedVillaModel
    )
except ImportError:
    pass

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="ValuEstate",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');

    :root {
        --accent: #10b981;
        --accent-dim: rgba(16, 185, 129, 0.15);
        --surface: #111827;
        --surface-2: #1f2937;
        --surface-3: #374151;
        --text: #f9fafb;
        --text-dim: #9ca3af;
        --danger: #ef4444;
        --warning: #f59e0b;
        --gold: #fbbf24;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #1f2937;
    }

    [data-testid="metric-container"] {
        background: var(--surface);
        border: 1px solid var(--surface-3);
        border-radius: 12px;
        padding: 16px;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    h1, h2, h3 { font-family: 'DM Sans', sans-serif; font-weight: 700; }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #374151, transparent);
        margin: 24px 0;
    }

    .deal-card {
        background: var(--surface);
        border: 1px solid var(--surface-3);
        border-radius: 14px;
        padding: 20px;
        margin: 8px 0;
        transition: border-color 0.2s;
    }
    .deal-card:hover { border-color: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'orebro_housing_enriched.csv')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
ACTIVE_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'active_listings_scored.csv')
COORDS_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'area_coordinates.csv')
HISTORY_DIR = os.path.join(BASE_DIR, '..', 'data', 'history')
WATCHLIST_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'watchlist.json')

if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'orebro_housing_clean.csv')

TYP_LABELS = {'lagenheter': 'Lägenheter', 'villor': 'Villor', 'radhus': 'Radhus'}
TYP_MAP = {'Lägenhet': 'lagenheter', 'Villa': 'villor', 'Radhus': 'radhus'}
COLOR_MAP = {'Lägenheter': '#10b981', 'Villor': '#6366f1', 'Radhus': '#f43f5e'}
DEAL_COLORS = {
    'Exceptionellt fynd': '#fbbf24',
    'Bra fynd': '#10b981',
    'Potentiellt intressant': '#f59e0b',
    'Rimligt pris': '#6b7280',
}
FALLBACK_LAT, FALLBACK_LON = 59.2753, 15.2134

# ============================================================
# DATA LOADING
# ============================================================


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['sald_datum'] = pd.to_datetime(df.get('sald_datum'), errors='coerce')
    return df


@st.cache_resource
def load_v2_models():
    models = {}
    for typ in ['lagenheter', 'villor', 'radhus']:
        candidates = [
            os.path.join(MODELS_DIR, f'model_{typ}_v10.pkl'),
            os.path.join(MODELS_DIR, f'model_{typ}_v9.pkl'),
            os.path.join(MODELS_DIR, f'model_{typ}.pkl'),
            os.path.join(MODELS_DIR, 'best_model.pkl'),  # fallback lagenheter
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path:
            try:
                models[typ] = joblib.load(path)
            except Exception as e:
                st.warning(f"Kunde inte ladda {typ}-modell: {type(e).__name__}. "
                           f"Prediktion ej tillgänglig — övriga vyer fungerar normalt.")
    return models if models else None


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


@st.cache_data
def load_history():
    """Läs alla historikfiler och returnera first_seen + ursprungspris per URL."""
    files = sorted(glob.glob(os.path.join(HISTORY_DIR, 'listings_*.csv')))
    if not files:
        return None
    dfs = []
    for f in files:
        date_str = os.path.basename(f).replace('listings_', '').replace('.csv', '')
        try:
            d = datetime.strptime(date_str, '%Y%m%d').date()
            tmp = pd.read_csv(f, usecols=lambda c: c in ['url', 'utgangspris'])
            tmp['history_date'] = d
            dfs.append(tmp)
        except Exception:
            pass
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def enrich_with_history(df_act, df_hist):
    """Lägg till dagar_pa_marknaden, pris_sankts, pris_sank_kr på df_act."""
    if df_hist is None or df_act is None:
        return df_act
    today = date.today()
    first_seen = (df_hist.groupby('url')['history_date'].min().reset_index()
                  .rename(columns={'history_date': 'forsta_sedd'}))
    orig_price = (df_hist.sort_values('history_date').groupby('url').first()[['utgangspris']]
                  .reset_index().rename(columns={'utgangspris': 'ursprungspris'}))
    df_act = df_act.merge(first_seen, on='url', how='left')
    df_act = df_act.merge(orig_price, on='url', how='left')
    df_act['dagar_pa_marknaden'] = df_act['forsta_sedd'].apply(
        lambda x: (today - x).days if pd.notna(x) else None)
    df_act['pris_sankts'] = df_act.apply(
        lambda r: r['ursprungspris'] > r['utgangspris'] if pd.notna(r.get('ursprungspris')) else False, axis=1)
    df_act['pris_sank_kr'] = df_act.apply(
        lambda r: int(r['ursprungspris'] - r['utgangspris']) if r['pris_sankts'] else 0, axis=1)
    return df_act


def load_watchlist():
    if os.path.exists(WATCHLIST_PATH):
        try:
            with open(WATCHLIST_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_watchlist(wl):
    try:
        with open(WATCHLIST_PATH, 'w') as f:
            json.dump(wl, f)
    except Exception:
        pass


@st.cache_data
def get_map_df(_df, _df_coords):
    map_df = _df[_df['latitude'].notna() & (_df['latitude'] != 0)].copy()
    if _df_coords is None or 'omrade_clean' not in map_df.columns:
        return map_df
    good = _df_coords[~((_df_coords['lat'].round(4) == round(FALLBACK_LAT, 4)) &
                         (_df_coords['lon'].round(4) == round(FALLBACK_LON, 4)))]

    def lookup(name):
        if pd.isna(name):
            return None, None
        m = good[good['omrade'].str.lower() == str(name).lower()]
        if len(m) > 0:
            return m.iloc[0]['lat'], m.iloc[0]['lon']
        for p in ['Radhus ', 'Lägenhet ', 'Villa ']:
            if str(name).startswith(p):
                m2 = good[good['omrade'].str.lower() == name[len(p):].lower()]
                if len(m2) > 0:
                    return m2.iloc[0]['lat'], m2.iloc[0]['lon']
        return None, None

    fb = ((map_df['latitude'].round(4) == round(FALLBACK_LAT, 4)) &
          (map_df['longitude'].round(4) == round(FALLBACK_LON, 4)))
    for idx in map_df[fb].index:
        lat, lon = lookup(map_df.at[idx, 'omrade_clean'])
        if lat:
            map_df.at[idx, 'latitude'] = lat
            map_df.at[idx, 'longitude'] = lon
    return map_df


# Load everything
df = load_data()
v2_models = load_v2_models()
df_active = load_active()
df_coords = load_coords()
df_history = load_history()
map_df_cached = get_map_df(df, df_coords) if 'latitude' in df.columns else df

# Enrich active listings with history data (dagar, prissänkning)
if df_active is not None and df_history is not None:
    df_active = enrich_with_history(df_active, df_history)

# Session state: watchlist + jämförlista
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()
if 'compare_urls' not in st.session_state:
    st.session_state.compare_urls = []

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 📊 ValuEstate")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Nav",
        ["🏠 Översikt", "🔥 Live Fynd", "🔍 Analysera URL",
         "💰 Prisprediktering", "🗺️ Karta",
         "📈 Marknadsanalys", "🔮 Scenarioanalys",
         "🏦 Köpkalkyl", "💼 Investeringskalkyl", "ℹ️ Om modellen"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.caption(f"📦 {len(df):,} sålda bostäder")
    if df_active is not None:
        st.caption(f"📡 {len(df_active)} aktiva annonser")
    if v2_models:
        for typ, m in v2_models.items():
            st.caption(f"🤖 {TYP_LABELS[typ]}: R²={_model_r2(m):.3f}")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.caption("Örebro kommun · 2013–2026")
    st.caption("*Byggd av Loran Ali*")


# ============================================================
# 0. ÖVERSIKT
# ============================================================

if page == "🏠 Översikt":
    st.markdown("## Örebro bostadsmarknad")
    st.caption("Historik & nyckeltal · 2013–2026")

    OV_TYPES = [
        ('lagenheter', 'Lägenheter', '#10b981'),
        ('villor',     'Villor',     '#3b82f6'),
        ('radhus',     'Radhus',     '#f59e0b'),
    ]

    # ── Delade diagram (alla typer tillsammans) ───────────────
    ch_l, ch_r = st.columns(2)

    with ch_l:
        st.markdown("**Hur har priserna förändrats över tid?**")
        st.caption("Typiskt försäljningspris per år, i miljoner kronor")
        if 'sald_datum' in df.columns and 'slutpris' in df.columns:
            df_yy = df.copy()
            df_yy['ar'] = pd.to_datetime(df_yy['sald_datum'], errors='coerce').dt.year
            df_yy = df_yy.dropna(subset=['ar'])
            df_yy['ar'] = df_yy['ar'].astype(int)
            yr_med = (
                df_yy.groupby(['ar', 'bostadstyp'])['slutpris']
                .median().reset_index()
            )
            yr_med.columns = ['År', 'bostadstyp', 'pris']
            yr_med['Typ'] = yr_med['bostadstyp'].map(
                {'lagenheter': 'Lägenheter', 'villor': 'Villor', 'radhus': 'Radhus'}
            )
            yr_med['Pris (Mkr)'] = (yr_med['pris'] / 1_000_000).round(2)
            fig_trend = px.line(
                yr_med, x='År', y='Pris (Mkr)', color='Typ',
                color_discrete_map={'Lägenheter': '#10b981', 'Villor': '#3b82f6', 'Radhus': '#f59e0b'},
                markers=True,
            )
            fig_trend.update_traces(line_width=2.5)
            fig_trend.update_layout(
                height=260, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#9ca3af', legend_title_text='',
                legend=dict(orientation='h', y=1.18),
                yaxis_title='Miljoner kr',
                yaxis_ticksuffix=' Mkr',
                xaxis_dtick=1,
            )
            fig_trend.update_xaxes(showgrid=False)
            fig_trend.update_yaxes(showgrid=True, gridcolor='#1f2937')
            st.plotly_chart(fig_trend, use_container_width=True, key="ov_trend")
        else:
            st.info("Ingen historikdata.")

    with ch_r:
        st.markdown("**Vad kostar en typisk bostad?**")
        st.caption("Typiskt pris (mittpris) och vanligt prisintervall per bostadstyp")
        if 'slutpris' in df.columns and 'bostadstyp' in df.columns:
            range_rows = []
            for tk, tl, tc in OV_TYPES:
                d = df[df['bostadstyp'] == tk]['slutpris'].dropna()
                if len(d) == 0:
                    continue
                range_rows.append({
                    'Typ': tl,
                    'color': tc,
                    'low':  int(d.quantile(0.25)),
                    'mid':  int(d.median()),
                    'high': int(d.quantile(0.75)),
                })
            if range_rows:
                fig_range = go.Figure()
                for i, row in enumerate(range_rows):
                    # Intervallbar (Q1–Q3)
                    fig_range.add_trace(go.Bar(
                        name=row['Typ'],
                        x=[row['Typ']],
                        y=[(row['high'] - row['low']) / 1_000_000],
                        base=row['low'] / 1_000_000,
                        marker_color=row['color'],
                        marker_opacity=0.35,
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{row['Typ']}</b><br>"
                            f"Billigaste (25%): {row['low']/1e6:.2f} Mkr<br>"
                            f"Typiskt pris: {row['mid']/1e6:.2f} Mkr<br>"
                            f"Dyraste (75%): {row['high']/1e6:.2f} Mkr<extra></extra>"
                        ),
                    ))
                    # Mittlinje (median)
                    fig_range.add_trace(go.Scatter(
                        x=[row['Typ']],
                        y=[row['mid'] / 1_000_000],
                        mode='markers+text',
                        marker=dict(color=row['color'], size=12, symbol='diamond'),
                        text=[f"{row['mid']/1e6:.1f} Mkr"],
                        textposition='top center',
                        textfont=dict(color=row['color'], size=12),
                        showlegend=False,
                        hoverinfo='skip',
                    ))
                fig_range.update_layout(
                    height=260, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#9ca3af', barmode='overlay',
                    yaxis_title='Miljoner kr', yaxis_ticksuffix=' Mkr',
                    xaxis_title='',
                )
                fig_range.update_xaxes(showgrid=False)
                fig_range.update_yaxes(showgrid=True, gridcolor='#1f2937')
                st.plotly_chart(fig_range, use_container_width=True, key="ov_range")
        else:
            st.info("Ingen data.")

    st.markdown("---")

    # ── KPI-kort per typ ──────────────────────────────────────
    for typ_key, typ_label, typ_color in OV_TYPES:
        df_typ = df[df['bostadstyp'] == typ_key] if 'bostadstyp' in df.columns else pd.DataFrame()
        df_act_typ = (
            df_active[df_active['bostadstyp'] == typ_key]
            if df_active is not None and 'bostadstyp' in df_active.columns
            else None
        )

        # Beräkna värden
        n_sold   = len(df_typ)
        med_pris = int(df_typ['slutpris'].median()) if n_sold > 0 and 'slutpris' in df_typ.columns else 0
        if n_sold > 0 and 'slutpris' in df_typ.columns and 'boarea_kvm' in df_typ.columns:
            kp = df_typ[df_typ['boarea_kvm'] > 0]
            med_kvm = int((kp['slutpris'] / kp['boarea_kvm']).median()) if len(kp) > 0 else 0
        else:
            med_kvm = 0
        med_rum  = round(df_typ['antal_rum'].median(), 1) if n_sold > 0 and 'antal_rum' in df_typ.columns else 0
        med_bo   = int(df_typ['boarea_kvm'].median()) if n_sold > 0 and 'boarea_kvm' in df_typ.columns else 0
        n_active = len(df_act_typ) if df_act_typ is not None else 0
        score_col_top = 'deal_score_pct' if (df_act_typ is not None and 'deal_score_pct' in df_act_typ.columns) else 'deal_score'
        top_score = int(df_act_typ[score_col_top].max()) if (df_act_typ is not None and score_col_top in df_act_typ.columns and len(df_act_typ) > 0) else None

        # Typ-specifik extra
        if typ_key == 'lagenheter':
            extra_label = "Medianavgift/mån"
            extra_val   = f"{int(df_typ['avgift_kr'].median()):,} kr" if n_sold > 0 and 'avgift_kr' in df_typ.columns else "–"
        elif typ_key == 'villor':
            extra_label = "Mediantomtarea"
            extra_val   = f"{int(df_typ['tomtarea_kvm'].median())} kvm" if n_sold > 0 and 'tomtarea_kvm' in df_typ.columns and df_typ['tomtarea_kvm'].notna().any() else "–"
        else:
            extra_label = "Medianavgift/mån"
            extra_val   = f"{int(df_typ['avgift_kr'].median()):,} kr" if n_sold > 0 and 'avgift_kr' in df_typ.columns else "–"

        score_html = (
            f'<div style="text-align:center"><div style="font-size:1.5em;font-weight:700;color:{typ_color}">{top_score}</div>'
            f'<div style="font-size:0.75em;color:#6b7280">Bästa score</div></div>'
        ) if top_score is not None else (
            f'<div style="text-align:center"><div style="font-size:1.5em;font-weight:700;color:#374151">–</div>'
            f'<div style="font-size:0.75em;color:#6b7280">Bästa score</div></div>'
        )

        st.markdown(
            f'<div style="background:#111827;border:1px solid #1f2937;border-radius:14px;'
            f'padding:20px 24px;margin-bottom:16px;border-top:3px solid {typ_color};">'
            f'<div style="font-size:1.05em;font-weight:700;color:#f9fafb;margin-bottom:16px">{typ_label}</div>'
            f'<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:8px;text-align:center">'
            f'<div><div style="font-size:1.4em;font-weight:700;color:{typ_color}">{n_sold:,}</div><div style="font-size:0.72em;color:#6b7280">Sålda</div></div>'
            f'<div><div style="font-size:1.4em;font-weight:700;color:#f9fafb">{med_pris/1e6:.2f}</div><div style="font-size:0.72em;color:#6b7280">Medianpris Mkr</div></div>'
            f'<div><div style="font-size:1.4em;font-weight:700;color:#f9fafb">{med_kvm:,}</div><div style="font-size:0.72em;color:#6b7280">Kr/kvm</div></div>'
            f'<div><div style="font-size:1.4em;font-weight:700;color:#f9fafb">{med_rum}</div><div style="font-size:0.72em;color:#6b7280">Median rum</div></div>'
            f'<div><div style="font-size:1.4em;font-weight:700;color:#f9fafb">{med_bo}</div><div style="font-size:0.72em;color:#6b7280">Medianboarea kvm</div></div>'
            f'<div><div style="font-size:1.4em;font-weight:700;color:#f9fafb">{extra_val}</div><div style="font-size:0.72em;color:#6b7280">{extra_label}</div></div>'
            f'<div><div style="font-size:1.4em;font-weight:700;color:{typ_color}">{n_active}</div><div style="font-size:0.72em;color:#6b7280">Aktiva annonser</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )



# ============================================================
# 1. LIVE FYND
# ============================================================

if page == "🔥 Live Fynd":
    st.markdown("# 🔥 Fynd just nu")

    if df_active is None:
        st.info("Kör `scripts/daily_update.py` för att aktivera.")
    else:
        has_deal = 'deal_kategori' in df_active.columns

        if 'scrape_datum' in df_active.columns:
            last_update_str = df_active['scrape_datum'].iloc[0]
            try:
                last_update = pd.to_datetime(last_update_str)
                hours_old = (datetime.now() - last_update).total_seconds() / 3600
                if hours_old > 36:
                    st.warning(f"⚠️ Data är {int(hours_old/24)} dagar gammal (senast uppdaterad: {last_update_str}). Kör `scripts/daily_update.py`.")
                else:
                    st.caption(f"✅ Uppdaterad: {last_update_str}")
            except Exception:
                st.caption(f"Uppdaterad: {last_update_str}")

        # KPIs
        if has_deal:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🔥 Exceptionella", (df_active['deal_kategori'] == 'Exceptionellt fynd').sum())
            c2.metric("🟢 Bra fynd", (df_active['deal_kategori'] == 'Bra fynd').sum())
            c3.metric("🟡 Intressanta", (df_active['deal_kategori'] == 'Potentiellt intressant').sum())
            c4.metric("⚪ Rimligt pris", (df_active['deal_kategori'] == 'Rimligt pris').sum())

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        def render_fynd(data, prefix):
            if len(data) == 0:
                st.info("Inga annonser.")
                return

            # Dölj listings med insane estimates (score=20 default, modellen litar inte på dem)
            if 'sane_estimate' in data.columns:
                n_insane = (~data['sane_estimate'].astype(bool)).sum()
                data = data[data['sane_estimate'].astype(bool)]
                if n_insane > 0:
                    st.caption(f"ℹ️ {n_insane} annonser dolda (modellen kunde inte estimera rimligt pris)")

            # Filters
            c1, c2 = st.columns([2, 1])
            with c1:
                if has_deal:
                    cats = sorted(data['deal_kategori'].unique().tolist())
                    sel = st.multiselect("Kategori", cats, default=cats, key=f"cat_{prefix}")
                    data = data[data['deal_kategori'].isin(sel)]
            with c2:
                score_col = 'deal_score_pct' if 'deal_score_pct' in data.columns else 'deal_score'
                if score_col in data.columns:
                    ms = st.slider("Min score (percentil)", 0, 100, 0, key=f"ms_{prefix}")
                    data = data[data[score_col] >= ms]

            data = data.sort_values(
                'deal_score_pct' if 'deal_score_pct' in data.columns else
                'deal_score' if 'deal_score' in data.columns else 'skillnad_pct',
                ascending=False)
            st.caption(f"{len(data)} annonser")

            # Listing cards (top 5)
            for card_idx, (_, r) in enumerate(data.head(5).iterrows()):
                # deal_score_pct = percentilrang inom typ (0-100, använder hela skalan)
                # deal_score     = absolut råpoäng (bas för kategorier)
                score_pct = int(r.get('deal_score_pct', r.get('deal_score', 0)))
                score_raw = int(r.get('deal_score', 0))
                icon = r.get('deal_ikon', '⚪')
                kat = r.get('deal_kategori', '')
                color = DEAL_COLORS.get(kat, '#6b7280')
                typ_l = TYP_LABELS.get(r.get('bostadstyp', ''), '')
                bo = r.get('boarea_kvm', 0)
                rum = r.get('antal_rum', 0)
                avg = r.get('avgift_kr', 0)
                pris = r.get('utgangspris', 0)
                est = r.get('estimerat_varde', 0)
                omr = r.get('omrade', '')
                url = r.get('url', '#')
                # Fix: hantera NaN i deal_reasons
                reasons_raw = str(r.get('deal_reasons', '') or '')
                ci_lo = r.get('ci_low', None)
                ci_hi = r.get('ci_high', None)
                dagar = r.get('dagar_pa_marknaden', None)
                pris_sankts = r.get('pris_sankts', False)
                pris_sank_kr = r.get('pris_sank_kr', 0)
                bostadstyp_r = r.get('bostadstyp', '')

                avg_txt = f" · {int(avg):,} kr/mån" if avg and avg > 0 else ""

                # CI — visa som % istället för stora kr-tal (villor har stor range)
                if ci_lo and ci_hi and est and est > 0:
                    try:
                        ci_range_pct = int((float(ci_hi) - float(ci_lo)) / float(est) * 100)
                        ci_half = ci_range_pct // 2
                        ci_color = '#ef4444' if ci_half > 20 else '#f59e0b' if ci_half > 12 else '#9ca3af'
                        ci_txt = (f"<div style='color:{ci_color};font-size:12px;margin-top:4px;'>"
                                  f"Modellens osäkerhet: ±{ci_half}%"
                                  f"<span style='color:#6b7280;'> ({int(float(ci_lo)):,}–{int(float(ci_hi)):,} kr)</span></div>")
                        # Extra varning för villor (CI > 40% = modellen är svag)
                        if bostadstyp_r == 'villor' and ci_half > 35:
                            ci_txt += ("<div style='color:#f59e0b;font-size:11px;margin-top:2px;'>"
                                       "⚠️ Villamodellen har hög osäkerhet — verifiera mot liknande försäljningar i området</div>")
                    except Exception:
                        ci_txt = ""
                else:
                    ci_txt = ""

                # Badges — pre-compute utanför f-string
                badges_parts = []
                if pd.notna(dagar) and dagar is not None:
                    try:
                        d_int = int(dagar)
                        dag_color = '#ef4444' if d_int >= 60 else '#f59e0b' if d_int >= 30 else '#6b7280'
                        badges_parts.append(
                            f"<span style='background:{dag_color}22;color:{dag_color};"
                            f"border:1px solid {dag_color}55;border-radius:6px;"
                            f"padding:2px 8px;font-size:11px;font-weight:600;'>🕐 {d_int} dagar</span>")
                    except Exception:
                        pass
                if pris_sankts and pris_sank_kr and pris_sank_kr > 0:
                    try:
                        badges_parts.append(
                            f"<span style='background:#10b98122;color:#10b981;"
                            f"border:1px solid #10b98155;border-radius:6px;"
                            f"padding:2px 8px;font-size:11px;font-weight:600;'>"
                            f"↓ Prissänkt {int(pris_sank_kr):,} kr</span>")
                    except Exception:
                        pass
                badges_html = ("<div style='display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;'>"
                               + "".join(badges_parts) + "</div>") if badges_parts else ""

                # Top 3 faktorer — safe split
                top3 = [x.strip() for x in reasons_raw.split('|') if x.strip() and x.strip() != 'nan'][:3]
                faktor_rows = ""
                for fi, ftext in enumerate(top3):
                    faktor_rows += (f"<div style='display:flex;align-items:center;gap:6px;margin-top:4px;'>"
                                    f"<span style='color:{color};font-size:11px;font-weight:700;'>{fi+1}.</span>"
                                    f"<span style='color:#d1d5db;font-size:12px;'>{ftext}</span></div>")
                if faktor_rows:
                    faktorer_section = ("<div style='margin-top:10px;border-top:1px solid #374151;padding-top:8px;'>"
                                        "<div style='color:#6b7280;font-size:10px;letter-spacing:0.08em;"
                                        "margin-bottom:2px;'>VARFÖR DENNA BEDÖMNING</div>"
                                        + faktor_rows + "</div>")
                else:
                    faktorer_section = ""

                typ_label_short = {'lagenheter': 'lägenheter', 'villor': 'villor', 'radhus': 'radhus'}.get(bostadstyp_r, 'objekt')
                score_tooltip = f"Bättre än {score_pct}% av aktiva {typ_label_short} · Råpoäng: {score_raw}/100"
                st.markdown(
                    f'<div class="deal-card" style="border-left: 3px solid {color};">'
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
                    f'<div><span style="font-size:20px;font-weight:700;" title="{score_tooltip}">'
                    f'{icon} {score_pct}/100</span>'
                    f'<span style="color:#6b7280;font-size:11px;margin-left:6px;">percentil</span>'
                    f'<span style="color:{color};font-size:13px;margin-left:8px;">{kat}</span></div>'
                    f'<span style="color:#9ca3af;">{omr}</span></div>'
                    f'<div style="color:#9ca3af;font-size:14px;margin:6px 0;">{typ_l} · {bo:.0f} m² · {rum:.0f} rum{avg_txt}</div>'
                    + badges_html
                    + '<div style="display:flex;gap:32px;margin-top:12px;">'
                    f'<div><div style="color:#6b7280;font-size:11px;">UTGÅNGSPRIS</div>'
                    f'<div style="font-size:17px;font-weight:600;">{pris:,} kr</div></div>'
                    f'<div><div style="color:#6b7280;font-size:11px;">ML-ESTIMAT</div>'
                    f'<div style="font-size:17px;font-weight:600;color:#10b981;">{est:,} kr</div></div>'
                    f'</div>'
                    + ci_txt
                    + faktorer_section
                    + f'<a href="{url}" target="_blank" style="color:#6366f1;text-decoration:none;'
                    f'font-size:13px;margin-top:10px;display:inline-block;">Visa på Hemnet →</a>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Watchlist + Jämför knappar
                btn_c1, btn_c2, _ = st.columns([1, 1, 4])
                url_key = str(card_idx) + "_" + prefix
                is_watched = url in st.session_state.watchlist
                is_compared = url in st.session_state.compare_urls
                if btn_c1.button(
                    "★ Bevakad" if is_watched else "☆ Bevaka",
                    key=f"wl_{url_key}",
                    help="Spara i bevakningslistan"
                ):
                    if is_watched:
                        st.session_state.watchlist.remove(url)
                    else:
                        st.session_state.watchlist.append(url)
                    save_watchlist(st.session_state.watchlist)
                    st.rerun()
                if btn_c2.button(
                    "✓ Jämförs" if is_compared else "+ Jämför",
                    key=f"cmp_{url_key}",
                    help="Lägg till i jämförvyn (max 3)"
                ):
                    if is_compared:
                        st.session_state.compare_urls.remove(url)
                    elif len(st.session_state.compare_urls) < 3:
                        st.session_state.compare_urls.append(url)
                    st.rerun()

            # Table
            with st.expander(f"📋 Alla {len(data)} annonser"):
                tbl = data.copy()
                if 'deal_reasons' in tbl.columns:
                    tbl['top_3_faktorer'] = tbl['deal_reasons'].apply(
                        lambda x: ' | '.join([r.strip() for r in str(x).split('|') if r.strip()][:3])
                    )
                cols = [c for c in ['deal_score_pct', 'deal_score', 'deal_kategori', 'omrade', 'bostadstyp',
                        'utgangspris', 'estimerat_varde', 'skillnad_pct', 'boarea_kvm',
                        'antal_rum', 'dagar_pa_marknaden', 'pris_sank_kr',
                        'top_3_faktorer', 'url'] if c in tbl.columns]
                cc = {
                    "utgangspris": st.column_config.NumberColumn("Pris", format="%d kr"),
                    "estimerat_varde": st.column_config.NumberColumn("Estimat", format="%d kr"),
                    "skillnad_pct": st.column_config.NumberColumn("Diff %", format="%.1f%%"),
                    "deal_score_pct": st.column_config.ProgressColumn("Score (percentil)", min_value=0, max_value=100, help="Bättre än X% av aktiva annonser av samma typ"),
                    "deal_score": st.column_config.NumberColumn("Råpoäng", format="%d", help="Absolut poäng 0-100 (bas för kategorier)"),
                    "deal_kategori": st.column_config.TextColumn("Kategori"),
                    "dagar_pa_marknaden": st.column_config.NumberColumn("Dagar ute", format="%d d"),
                    "pris_sank_kr": st.column_config.NumberColumn("Prissänkt", format="%d kr"),
                    "top_3_faktorer": st.column_config.TextColumn("Top 3 faktorer", width="large"),
                    "url": st.column_config.LinkColumn("Hemnet", display_text="Öppna"),
                }
                st.dataframe(tbl[cols], use_container_width=True, height=400, column_config=cc)

        # Jämför-vy (visas om URLs valda)
        if st.session_state.compare_urls and df_active is not None:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### 🔍 Jämförelse")
            cmp_data = df_active[df_active['url'].isin(st.session_state.compare_urls)]
            if len(cmp_data) > 0:
                cols_cmp = st.columns(len(cmp_data))
                for col_el, (_, r) in zip(cols_cmp, cmp_data.iterrows()):
                    with col_el:
                        kat = r.get('deal_kategori', '')
                        color = DEAL_COLORS.get(kat, '#6b7280')
                        icon = r.get('deal_ikon', '⚪')
                        score = int(r.get('deal_score', 0))
                        pris = int(r.get('utgangspris', 0))
                        est = int(r.get('estimerat_varde', 0))
                        diff = float(r.get('skillnad_pct', 0))
                        bo = float(r.get('boarea_kvm', 0))
                        rum = float(r.get('antal_rum', 0))
                        omr = str(r.get('omrade', ''))
                        url = str(r.get('url', '#'))
                        dagar = r.get('dagar_pa_marknaden', None)
                        reasons_raw = str(r.get('deal_reasons', '') or '')
                        top3 = [x.strip() for x in reasons_raw.split('|')
                                if x.strip() and x.strip() != 'nan'][:3]
                        pris_sank = r.get('pris_sank_kr', 0)

                        # Pre-compute alla html-delar
                        dagar_html = ""
                        if pd.notna(dagar) and dagar is not None:
                            try:
                                dagar_html = (f"<div style='color:#f59e0b;font-size:11px;"
                                              f"margin-top:4px;'>🕐 {int(dagar)} dagar ute</div>")
                            except Exception:
                                pass
                        sank_html = ""
                        if pris_sank and pris_sank > 0:
                            try:
                                sank_html = (f"<div style='color:#10b981;font-size:11px;'>"
                                             f"↓ Prissänkt {int(pris_sank):,} kr</div>")
                            except Exception:
                                pass
                        faktor_lines = ""
                        for fi, ftext in enumerate(top3):
                            faktor_lines += (f"<div style='font-size:11px;color:#d1d5db;"
                                             f"margin-top:3px;'>{fi+1}. {ftext}</div>")

                        html = (
                            f'<div class="deal-card" style="border-left:3px solid {color};">'
                            f'<div style="font-size:18px;font-weight:700;">{icon} {score}/100</div>'
                            f'<div style="color:{color};font-size:12px;">{kat}</div>'
                            f'<div style="color:#9ca3af;font-size:13px;margin:4px 0;">{omr}</div>'
                            f'<div style="color:#9ca3af;font-size:12px;">{bo:.0f} m² · {rum:.0f} rum</div>'
                            + dagar_html + sank_html
                            + '<hr style="border-color:#374151;margin:10px 0;">'
                            f'<div style="font-size:12px;color:#6b7280;">UTGÅNGSPRIS</div>'
                            f'<div style="font-size:16px;font-weight:600;">{pris:,} kr</div>'
                            f'<div style="font-size:12px;color:#6b7280;margin-top:6px;">ML-ESTIMAT</div>'
                            f'<div style="font-size:16px;font-weight:600;color:#10b981;">{est:,} kr</div>'
                            f'<div style="font-size:12px;color:#9ca3af;">Diff: {diff:+.1f}%</div>'
                            + '<hr style="border-color:#374151;margin:10px 0;">'
                            + '<div style="font-size:10px;color:#6b7280;letter-spacing:0.08em;">TOP 3 FAKTORER</div>'
                            + faktor_lines
                            + f'<a href="{url}" target="_blank" style="color:#6366f1;font-size:12px;'
                            f'display:inline-block;margin-top:10px;">Visa på Hemnet →</a>'
                            f'</div>'
                        )
                        st.markdown(html, unsafe_allow_html=True)
            if st.button("🗑️ Rensa jämförelse"):
                st.session_state.compare_urls = []
                st.rerun()

        # Tabs per typ + Bevakningslista
        tabs = st.tabs(["🏠 Alla", "🏢 Lägenheter", "🏡 Villor", "🏘️ Radhus", "★ Bevakade"])

        with tabs[0]:
            render_fynd(df_active.copy(), "all")
        with tabs[1]:
            render_fynd(df_active[df_active['bostadstyp'] == 'lagenheter'].copy(), "lag")
        with tabs[2]:
            render_fynd(df_active[df_active['bostadstyp'] == 'villor'].copy(), "vil")
        with tabs[3]:
            render_fynd(df_active[df_active['bostadstyp'] == 'radhus'].copy(), "rad")
        with tabs[4]:
            if not st.session_state.watchlist:
                st.info("Du har inga bevakade bostäder. Klicka ☆ Bevaka på en annons.")
            else:
                wl_data = df_active[df_active['url'].isin(st.session_state.watchlist)]
                if len(wl_data) == 0:
                    st.warning("Bevakade annonser hittades inte i aktiva data (kan ha utgått).")
                else:
                    render_fynd(wl_data.copy(), "wl")
                if st.button("🗑️ Rensa bevakningslista"):
                    st.session_state.watchlist = []
                    save_watchlist([])
                    st.rerun()


# ============================================================
# 2. PRISPREDIKTERING
# ============================================================

elif page == "🔍 Analysera URL":
    st.markdown("# 🔍 Analysera en Hemnet-annons")
    st.markdown("Klistra in valfri Hemnet-länk — vi hämtar annonsen live och kör ML-modellen.")

    hemnet_url_page = st.text_input(
        "Hemnet-URL",
        placeholder="https://www.hemnet.se/bostad/villa-4rum-adolfsberg-orebro-id12345678",
        key="url_page_input"
    )

    if hemnet_url_page and st.button("🔍 Analysera nu", type="primary", use_container_width=False):
        cached = None
        if df_active is not None and 'url' in df_active.columns:
            m = df_active[df_active['url'] == hemnet_url_page]
            if len(m) > 0:
                cached = m.iloc[0]

        if cached is not None:
            r     = cached
            score = r.get('deal_score', 0)
            kat   = r.get('deal_kategori', 'Okänd')
            color = DEAL_COLORS.get(kat, '#6b7280')
            icon  = r.get('deal_ikon', '⚪')
            st.success("✅ Hittad i dagens scraping")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Deal Score",  f"{score}/100")
            c2.metric("Utgångspris", f"{int(r.get('utgangspris',0)):,} kr")
            c3.metric("ML-estimat",  f"{int(r.get('estimerat_varde',0)):,} kr")
            c4.metric("Avvikelse",   f"{r.get('skillnad_pct',0):.1f}%")
            st.markdown(f"""<div class="deal-card" style="border-left:3px solid {color};">
                <span style="font-size:22px;font-weight:700;">{icon} {kat}</span>
                <p style="color:#9ca3af;margin-top:8px;">{r.get('deal_reasons','')}</p>
            </div>""", unsafe_allow_html=True)
            if r.get('ci_low') and r.get('ci_high'):
                st.caption(f"Konfidensintervall: {int(r['ci_low']):,} – {int(r['ci_high']):,} kr")
        else:
            if v2_models is None:
                st.error("Modeller ej laddade — live-analys kräver att ML-modellerna är tillgängliga.")
            else:
                with st.spinner("Hämtar annons och kör modellen..."):
                    try:
                        sys.path.insert(0, os.path.join(BASE_DIR, '..', 'scripts'))
                        from url_analyzer import analyze_url
                        result = analyze_url(hemnet_url_page, v2_models, df)
                    except Exception as e:
                        result = {'ok': False, 'error': str(e)}

                if not result['ok']:
                    st.error(f"❌ {result['error']}")
                    st.info("Tips: Hemnet kan ibland blockera automatiska förfrågningar. Prova igen om en stund eller använd manuell input under Prisprediktering.")
                else:
                    typ     = result['bostadstyp']
                    estimat = result['estimat']
                    utgpris = result['listing'].get('utgangspris', 0)
                    upct    = result['underval_pct']
                    score   = result['deal_score']
                    kat     = result['deal_kategori']
                    color   = DEAL_COLORS.get(kat, '#6b7280')
                    icon    = {'Exceptionellt fynd':'🔥','Bra fynd':'⭐',
                               'Potentiellt intressant':'👀','Rimligt pris':'✅'}.get(kat,'⚪')

                    st.markdown(f"**Område:** {result['omrade']} &nbsp;·&nbsp; **Typ:** {typ}")
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Deal Score",     f"{score}/100")
                    c2.metric("Utgångspris",    f"{utgpris:,} kr" if utgpris else "—")
                    c3.metric("ML-estimat",     f"{estimat:,} kr")
                    c4.metric("Undervärdering", f"{upct:+.1f}%" if utgpris else "—",
                              delta_color="normal" if upct >= 0 else "inverse")

                    st.markdown(f"""<div class="deal-card" style="border-left:3px solid {color};margin:16px 0;">
                        <span style="font-size:22px;font-weight:700;">{icon} {kat}</span>
                        <p style="color:#9ca3af;margin-top:8px;">
                            Modellen uppskattar marknadsvärdet till <strong>{estimat:,} kr</strong>.
                            {'Annonsen verkar undervärde rad med ' + str(abs(upct)) + '%.' if upct > 5
                             else 'Priset är i linje med modellens estimat.' if abs(upct) <= 5
                             else f'Annonsen kan vara överprissatt med {abs(upct):.0f}%.'}
                        </p>
                    </div>""", unsafe_allow_html=True)

                    st.caption(
                        f"Konfidensintervall: {result['ci_low']:,} – {result['ci_high']:,} kr "
                        f"(±{result['ci_pct']:.0f}%)")

                    with st.expander("📋 Extraherade bostadinformation"):
                        lst = result['listing']
                        cols_info = st.columns(3)
                        info_items = [
                            ("Bostadstyp",   typ),
                            ("Område",       result['omrade']),
                            ("Boarea",       f"{lst.get('boarea_kvm','?')} m²"),
                            ("Antal rum",    lst.get('antal_rum','?')),
                            ("Byggår",       lst.get('byggar','?')),
                            ("Tomtarea",     f"{lst.get('tomtarea_kvm','?')} m²" if typ=='villor' else '—'),
                            ("Driftkostnad", f"{lst.get('driftkostnad_ar',0):,} kr/år" if typ=='villor' else '—'),
                            ("Månadsavgift", f"{lst.get('avgift_kr',0):,} kr/mån" if typ!='villor' else '—'),
                            ("Energiklass",  lst.get('energiklass','—')),
                            ("Uppvärmning",  lst.get('uppvarmning','—')),
                            ("Balkong",      "Ja" if lst.get('har_balkong') else "Nej"),
                            ("Garage",       "Ja" if lst.get('har_garage') else "Nej"),
                        ]
                        for i, (k, v) in enumerate(info_items):
                            cols_info[i % 3].write(f"**{k}:** {v}")

    elif not hemnet_url_page:
        st.info("💡 Fungerar med alla aktiva Hemnet-annonser i Örebro — villa, lägenhet och radhus.")

# ============================================================

elif page == "💰 Prisprediktering":
    st.markdown("# 💰 Vad är bostaden värd?")
    st.markdown("Fyll i bostadens egenskaper för ett prisestimat. Vill du analysera en annons direkt? Använd **🔍 Analysera URL** i menyn.")

    if True:
        c1, c2, c3 = st.columns(3)
        with c1:
            bostadstyp = st.selectbox("Bostadstyp", ["Lägenhet", "Villa", "Radhus"])
            boarea = st.number_input("Boarea (m²)", 20, 400, 80)
            antal_rum = st.number_input("Antal rum", 1, 12, 3)
        with c2:
            typ_key = TYP_MAP[bostadstyp]
            if v2_models and typ_key in v2_models:
                _pkg = v2_models[typ_key]
                # v7+ använder target encoding (te_map_pris) istf omrade_grupp_-dummies
                if _pkg.get('te_map_pris'):
                    areas = sorted(_pkg['te_map_pris'].keys())
                else:
                    areas = sorted([f.replace('omrade_grupp_', '')
                                    for f in _pkg['feature_names']
                                    if f.startswith('omrade_grupp_')])
            else:
                areas = sorted(df['omrade_clean'].value_counts().head(70).index.tolist()) if 'omrade_clean' in df.columns else ['Örebro']
            omrade = st.selectbox("Område", ['övrigt'] + areas)
            byggar = st.number_input("Byggnadsår", 1900, 2026, 1990)
            # Avgift visas bara för lägenhet/radhus
            if bostadstyp != "Villa":
                avgift = st.number_input("Månadsavgift (kr)", 0, 15000, 3500, step=100)
            else:
                avgift = 0
                st.caption("Ingen månadsavgift för villa")
        with c3:
            # Typspecifika features — de som faktiskt påverkar priset mest
            if bostadstyp == "Lägenhet":
                vaning = st.number_input("Våning", 0, 20, 2)
                har_balkong = st.toggle("Balkong", value=True)
                har_hiss = False
                har_garage = False
                har_uteplats = har_balkong
                tomtarea = 0
                st.caption("💡 Avgift och område påverkar lägenhetspriset mest")
            elif bostadstyp == "Villa":
                tomtarea = st.number_input("Tomtarea (m²)", 100, 5000, 800, step=50)
                har_garage = st.toggle("Garage/carport", value=True)
                har_uteplats = st.toggle("Uteplats/altan", value=True)
                vaning = 0
                har_balkong = har_uteplats
                har_hiss = False
                st.caption("💡 Boarea, tomtarea och område påverkar villapriset mest")
            else:  # Radhus
                vaning = st.number_input("Våning", 0, 5, 0)
                har_balkong = st.toggle("Balkong/uteplats", value=True)
                har_uteplats = har_balkong
                har_hiss = False
                har_garage = False
                tomtarea = 0
                st.caption("💡 Boarea och område påverkar radhuspriset mest")

        if st.button("🔮 Beräkna", key="calc_manual", type="primary", use_container_width=True):
            if v2_models and typ_key in v2_models:
                pkg = v2_models[typ_key]
                fnames = pkg['feature_names']
                df_typ = df[df['bostadstyp'] == typ_key]

                feat = {f: 0 for f in fnames}
                feat.update({
                    'boarea_kvm': boarea, 'antal_rum': antal_rum, 'avgift_kr': avgift,
                    'sald_ar': 2026, 'sald_manad': datetime.now().month,
                    'sald_kvartal': (datetime.now().month - 1) // 3 + 1,
                    'bostad_alder': 2026 - byggar, 'vaning': vaning,
                    'har_hiss': int(har_hiss), 'har_balkong': int(har_balkong),
                    'har_garage': int(har_garage), 'har_uteplats': int(har_uteplats),
                    'tomtarea_kvm': tomtarea,
                    'kvm_per_rum': boarea / max(antal_rum, 1),
                    'total_yta': boarea + tomtarea * 0.1,
                    'prisforandring_pct': 0,
                    'budkrig': 0, 'prissankt': 0, 'renoverad': 0,
                })
                if avgift > 0:
                    feat['avgift_per_kvm'] = avgift / max(boarea, 10)
                    feat['avgift_andel'] = avgift * 12 / max(boarea * 25000, 1) * 100

                # ── v7+ avancerade features ──────────────────────
                _is_advanced = (
                    pkg.get('model_type') in ('lgbm_quantile', 'lgbm_mse', 'catboost')
                    or pkg.get('version', '') in ('v5', 'v6', 'v7')
                    or pkg.get('te_map_pris')
                )
                if _is_advanced:
                    # Target encoding
                    te_pris = pkg.get('te_map_pris', {})
                    feat['te_omrade_pris'] = te_pris.get(omrade, pkg.get('te_global_pris', 0))
                    te_kvm = pkg.get('te_map_kvm', {})
                    feat['te_omrade_kvm'] = te_kvm.get(omrade, pkg.get('te_global_kvm', 0))
                    # Log-transforms
                    feat['log_boarea'] = float(np.log(max(boarea, 1)))
                    feat['log_tomtarea'] = float(np.log1p(tomtarea))
                    feat['log_driftkostnad'] = 0  # okänt vid manuell input
                    feat['byggdekad'] = (byggar // 10) * 10
                    feat['alder_ej_renoverad'] = 2026 - byggar
                    feat['driftkostnad_per_kvm'] = 0
                    # Interaktioner
                    feat['tomt_boarea_interact'] = (tomtarea * boarea) / 1e4
                    feat['boarea_log_tomt'] = feat['log_boarea'] * feat['log_tomtarea']
                    omr_hist = feat.get('te_omrade_kvm', feat.get('te_omrade_pris', 0))
                    feat['omrade_hist_pris_kvm'] = omr_hist
                    avst_c = df_typ['avstand_centrum_km'].median() if 'avstand_centrum_km' in df_typ.columns else 5
                    feat['avst_pris_interact'] = (avst_c * omr_hist) / 1000
                    feat['tomt_avst_interact'] = feat['log_tomtarea'] * avst_c
                    feat['forvantat_komps_pris'] = df_typ['comps_pris_kvm_90d'].median() * boarea if 'comps_pris_kvm_90d' in df_typ.columns else 0
                    # KMeans cluster
                    km = pkg.get('kmeans')
                    km_sc = pkg.get('kmeans_scaler')
                    km_fs = pkg.get('kmeans_feats', [])
                    c_map = pkg.get('cluster_te_map', {})
                    c_glob = pkg.get('cluster_global', 0)
                    if km and km_sc and km_fs:
                        X_c = np.array([[feat.get(f, df_typ[f].median() if f in df_typ.columns else 0) for f in km_fs]])
                        cid = int(km.predict(km_sc.transform(X_c))[0])
                        feat['cluster_te'] = c_map.get(cid, c_glob)
                    else:
                        feat['cluster_te'] = c_glob

                # Fyll saknade features med medianvärden från träningsdata
                for col in fnames:
                    if feat.get(col, 0) == 0 and col in df_typ.columns and not col.startswith('omrade_grupp_'):
                        med = df_typ[col].median()
                        if pd.notna(med) and col not in ['prisforandring_pct', 'budkrig', 'prissankt', 'renoverad']:
                            feat[col] = med

                # Area-dummy (v1–v4 bakåtkompatibilitet)
                ocol = f'omrade_grupp_{omrade}'
                if ocol in feat:
                    feat[ocol] = 1

                X = pd.DataFrame([feat])[fnames].fillna(0)
                if pkg.get('scaler'):
                    X = pkg['scaler'].transform(X)

                pred = pkg['model'].predict(X)[0]
                if pkg.get('log_transform', True):
                    est = int(np.expm1(pred))
                    if 'model_q10' in pkg and 'model_q90' in pkg:
                        ci_lo = int(np.expm1(pkg['model_q10'].predict(X)[0]))
                        ci_hi = int(np.expm1(pkg['model_q90'].predict(X)[0]))
                    else:
                        _c = pkg.get('confidence', {})
                        std = _c.get('residual_std_log', 0.15) if isinstance(_c, dict) else 0.15
                        ci_lo = int(np.expm1(pred - 1.96 * std))
                        ci_hi = int(np.expm1(pred + 1.96 * std))
                    _c = pkg.get('confidence', {})
                    ci_pct = _c.get('interval_pct', 15) if isinstance(_c, dict) else 15
                else:
                    est = int(pred)
                    ci_lo, ci_hi, ci_pct = int(est * 0.85), int(est * 1.15), 15

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("💰 Estimat", f"{est:,} kr")
                c2.metric("📐 kr/m²", f"{int(est/boarea):,}")
                c3.metric("📊 95% CI", f"{ci_lo:,} – {ci_hi:,} kr")
                c4.metric("🎯 Precision", f"±{ci_pct:.0f}%")

                st.caption(f"Modell: {pkg.get('model_name', '?')} · R² = {_model_r2(pkg):.3f}")

                # Liknande bostäder
                similar = df[(df['boarea_kvm'].between(boarea - 15, boarea + 15)) &
                             (df['bostadstyp'] == typ_key)]
                if len(similar) > 5:
                    st.markdown("---")
                    fig = px.histogram(similar, x="slutpris", nbins=30,
                                       title=f"Liknande {bostadstyp.lower()} ({boarea}±15 m²) — {len(similar)} st",
                                       color_discrete_sequence=['#10b981'])
                    fig.add_vline(x=est, line_dash="dash", line_color="#ef4444",
                                  annotation_text=f"Ditt estimat: {est:,}")
                    fig.add_vrect(x0=ci_lo, x1=ci_hi, fillcolor="#6366f1", opacity=0.08, line_width=0)
                    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Antal",
                                      xaxis_title="Slutpris (kr)")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Modell saknas.")


# ============================================================
# 3. KARTA
# ============================================================

elif page == "🗺️ Karta":
    st.markdown("# 🗺️ Karta")

    if 'latitude' not in df.columns:
        st.info("Kör geokodning först.")
    else:
        tab_sold, tab_live = st.tabs(["Sålda bostäder", "Aktiva annonser"])

        with tab_sold:
            typ_k = st.multiselect("Typ", list(TYP_LABELS.values()),
                                   default=list(TYP_LABELS.values()), key="k_typ")

            mdf = map_df_cached.copy()
            mdf['typ_label'] = mdf['bostadstyp'].map(TYP_LABELS)
            mdf = mdf[mdf['typ_label'].isin(typ_k)]
            mdf['pris_per_kvm'] = mdf['pris_per_kvm'].fillna(mdf['pris_per_kvm'].median())
            mdf = mdf.reset_index(drop=True)

            # Jitter
            np.random.seed(42)
            mdf['lat_j'] = mdf['latitude'] + np.random.uniform(-0.001, 0.001, len(mdf))
            mdf['lon_j'] = mdf['longitude'] + np.random.uniform(-0.001, 0.001, len(mdf))

            st.caption(f"{len(mdf):,} bostäder — klicka på en punkt för detaljer")

            # Lägg till index-kolumn för klick-identifiering
            mdf['_idx'] = mdf.index

            hover_cols = {c: True for c in ['slutpris', 'boarea_kvm', 'pris_per_kvm',
                                             'antal_rum', 'sald_datum', '_idx']
                         if c in mdf.columns}
            hover_cols['lat_j'] = False
            hover_cols['lon_j'] = False

            fig = px.scatter_mapbox(
                mdf, lat="lat_j", lon="lon_j",
                color="pris_per_kvm",
                color_continuous_scale=["#3b82f6", "#10b981", "#f59e0b", "#ef4444"],
                size_max=6, zoom=11,
                hover_name="omrade_clean" if 'omrade_clean' in mdf.columns else None,
                hover_data=hover_cols,
                mapbox_style="carto-positron",
                custom_data=[c for c in ['_idx', 'slutpris', 'boarea_kvm', 'antal_rum',
                                         'pris_per_kvm', 'sald_datum', 'omrade_clean',
                                         'bostadstyp'] if c in mdf.columns],
            )
            fig.update_traces(marker=dict(size=6, opacity=0.75))
            fig.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)',
                              margin=dict(l=0, r=0, t=0, b=0))

            sel = st.plotly_chart(fig, use_container_width=True,
                                  on_select="rerun", selection_mode="points",
                                  key=f"sold_map_{str(typ_k)}")

            # Klick → detaljpanel
            pts = sel.get("selection", {}).get("points", []) if sel else []
            if pts:
                pt = pts[0]
                cd = pt.get("customdata", [])
                # customdata ordning: _idx, slutpris, boarea_kvm, antal_rum, pris_per_kvm, sald_datum, omrade_clean, bostadstyp
                def _cd(i, fallback="—"):
                    try:
                        v = cd[i]
                        return v if v is not None else fallback
                    except Exception:
                        return fallback

                idx_val   = _cd(0)
                slutpris  = _cd(1)
                boarea    = _cd(2)
                rum       = _cd(3)
                kvm_pris  = _cd(4)
                datum     = _cd(5)
                omrade    = _cd(6)
                typ_val   = _cd(7)

                typ_label = TYP_LABELS.get(str(typ_val), str(typ_val))
                pris_fmt  = f"{int(float(slutpris)):,} kr" if slutpris != "—" else "—"
                kvm_fmt   = f"{int(float(kvm_pris)):,} kr/m²" if kvm_pris != "—" else "—"
                bo_fmt    = f"{float(boarea):.0f} m²" if boarea != "—" else "—"
                rum_fmt   = f"{float(rum):.0f} rum" if rum != "—" else "—"

                # Hämta prisförändring om möjligt
                extra_html = ""
                if idx_val != "—":
                    try:
                        row = mdf.loc[int(idx_val)]
                        pf_pct = row.get('prisforandring_pct', None)
                        budkrig = row.get('budkrig', None)
                        byggar = row.get('byggar', None)
                        avgift = row.get('avgift_kr', None)

                        if pf_pct is not None and pd.notna(pf_pct):
                            pf_color = '#10b981' if pf_pct > 0 else '#ef4444'
                            pf_label = f"Budkrig +{pf_pct:.1f}%" if pf_pct > 0 else f"Prissänkt {pf_pct:.1f}%"
                            extra_html += (f"<div style='display:flex;justify-content:space-between;"
                                           f"padding:6px 0;border-bottom:1px solid #374151;'>"
                                           f"<span style='color:#6b7280;font-size:13px;'>Prisskillnad</span>"
                                           f"<span style='color:{pf_color};font-weight:600;'>{pf_label}</span></div>")
                        if byggar is not None and pd.notna(byggar):
                            extra_html += (f"<div style='display:flex;justify-content:space-between;"
                                           f"padding:6px 0;border-bottom:1px solid #374151;'>"
                                           f"<span style='color:#6b7280;font-size:13px;'>Byggår</span>"
                                           f"<span>{int(byggar)}</span></div>")
                        if avgift is not None and pd.notna(avgift) and float(avgift) > 0:
                            extra_html += (f"<div style='display:flex;justify-content:space-between;"
                                           f"padding:6px 0;border-bottom:1px solid #374151;'>"
                                           f"<span style='color:#6b7280;font-size:13px;'>Avgift</span>"
                                           f"<span>{int(float(avgift)):,} kr/mån</span></div>")
                    except Exception:
                        pass

                st.markdown(
                    f'<div style="background:#1f2937;border:1px solid #374151;border-radius:14px;'
                    f'padding:20px;margin-top:12px;">'
                    f'<div style="font-size:18px;font-weight:700;margin-bottom:4px;">{omrade}</div>'
                    f'<div style="color:#9ca3af;font-size:13px;margin-bottom:16px;">'
                    f'{typ_label} · {datum}</div>'
                    f'<div style="display:flex;gap:32px;margin-bottom:16px;">'
                    f'<div><div style="color:#6b7280;font-size:11px;">SLUTPRIS</div>'
                    f'<div style="font-size:22px;font-weight:700;color:#10b981;">{pris_fmt}</div></div>'
                    f'<div><div style="color:#6b7280;font-size:11px;">KR/M²</div>'
                    f'<div style="font-size:22px;font-weight:700;">{kvm_fmt}</div></div>'
                    f'<div><div style="color:#6b7280;font-size:11px;">STORLEK</div>'
                    f'<div style="font-size:22px;font-weight:700;">{bo_fmt} · {rum_fmt}</div></div>'
                    f'</div>'
                    + extra_html +
                    f'</div>',
                    unsafe_allow_html=True
                )

        with tab_live:
            if df_active is not None and df_coords is not None:
                typ_tabs = st.tabs(["Alla", "Lägenheter", "Villor", "Radhus"])

                # Bygg en snabb coord-lookup med case-insensitive + prefix-strippning
                _coords_index = {row['omrade'].lower().strip(): (row['lat'], row['lon'])
                                 for _, row in df_coords.iterrows()}
                _PREFIXES = ['radhus ', 'lägenhet ', 'villa ', 'radhus', 'lägenhet', 'villa']

                def _lookup_coord(name):
                    if pd.isna(name):
                        return None, None
                    key = str(name).lower().strip()
                    if key in _coords_index:
                        return _coords_index[key]
                    for p in _PREFIXES:
                        if key.startswith(p):
                            stripped = key[len(p):].strip()
                            if stripped in _coords_index:
                                return _coords_index[stripped]
                    return None, None

                def show_live_map(data):
                    if len(data) == 0:
                        st.info("Inga annonser.")
                        return
                    am = data.copy()
                    lats, lons = [], []
                    for _, row in am.iterrows():
                        lat, lon = _lookup_coord(row.get('omrade', ''))
                        lats.append(lat)
                        lons.append(lon)
                    am['latitude'] = lats
                    am['longitude'] = lons

                    no_coords = am['latitude'].isna().sum()
                    am_ok = am[am['latitude'].notna()].copy()

                    if no_coords > 0:
                        st.caption(f"⚠️ {no_coords} annonser saknar koordinater och visas ej.")

                    if len(am_ok) == 0:
                        st.info("Inga annonser med känd position.")
                        return

                    np.random.seed(42)
                    am_ok['lat_j'] = am_ok['latitude'] + np.random.uniform(-0.002, 0.002, len(am_ok))
                    am_ok['lon_j'] = am_ok['longitude'] + np.random.uniform(-0.002, 0.002, len(am_ok))

                    hover = {'utgangspris': ':,.0f', 'boarea_kvm': ':.0f', 'lat_j': False, 'lon_j': False}
                    if 'estimerat_varde' in am_ok.columns:
                        hover['estimerat_varde'] = ':,.0f'
                    if 'deal_score' in am_ok.columns:
                        hover['deal_score'] = True

                    kw = dict(lat="lat_j", lon="lon_j", size_max=8, zoom=11,
                              hover_name="omrade", hover_data=hover,
                              mapbox_style="carto-positron")
                    if 'deal_kategori' in am_ok.columns:
                        kw['color'] = 'deal_kategori'
                        kw['color_discrete_map'] = DEAL_COLORS

                    fig = px.scatter_mapbox(am_ok, **kw)
                    fig.update_traces(marker=dict(size=8, opacity=0.9))
                    fig.update_layout(height=650, paper_bgcolor='rgba(0,0,0,0)',
                                      margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                with typ_tabs[0]:
                    show_live_map(df_active)
                with typ_tabs[1]:
                    show_live_map(df_active[df_active['bostadstyp'] == 'lagenheter'])
                with typ_tabs[2]:
                    show_live_map(df_active[df_active['bostadstyp'] == 'villor'])
                with typ_tabs[3]:
                    show_live_map(df_active[df_active['bostadstyp'] == 'radhus'])
            else:
                st.info("Kör daily_update.py och geokodning.")


# ============================================================
# 4. MARKNADSANALYS
# ============================================================

elif page == "📈 Marknadsanalys":
    st.markdown("# 📈 Marknadsanalys")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Pristrender", "Säsongsvariation", "Områdesjämförelse", "Budkrig", "Områdesguide"])

    with tab1:
        if 'sald_datum' in df.columns:
            tf = st.multiselect("Typ", list(TYP_LABELS.values()),
                                default=list(TYP_LABELS.values()), key="t_typ")
            dft = df.copy()
            dft['bostadstyp'] = dft['bostadstyp'].map(TYP_LABELS)
            dft = dft[dft['bostadstyp'].isin(tf)]
            mon = dft.groupby([pd.Grouper(key='sald_datum', freq='ME'), 'bostadstyp']
                              )['slutpris'].median().reset_index()
            fig = px.line(mon, x='sald_datum', y='slutpris', color='bostadstyp',
                          labels={'sald_datum': '', 'slutpris': 'Medianpris (kr)', 'bostadstyp': 'Typ'},
                          color_discrete_map=COLOR_MAP, markers=True)
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            if 'omrade_clean' in df.columns:
                st.markdown("#### Per område")
                ol = df['omrade_clean'].value_counts()[lambda x: x >= 15].index.tolist()
                vo = st.selectbox("Område", sorted(ol), key="t_omr")
                oq = df[df['omrade_clean'] == vo].groupby(
                    pd.Grouper(key='sald_datum', freq='QE'))['slutpris'].median().reset_index()
                fig2 = px.line(oq, x='sald_datum', y='slutpris', markers=True,
                               color_discrete_sequence=['#10b981'],
                               labels={'sald_datum': '', 'slutpris': 'Medianpris'})
                fig2.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                                   plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if 'sald_manad' in df.columns:
            sea = df.groupby('sald_manad').agg(
                medianpris=('slutpris', 'median'), antal=('slutpris', 'count')).reset_index()
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'Maj', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec']
            sea['m'] = sea['sald_manad'].map(dict(enumerate(months, 1)))
            fig = go.Figure()
            fig.add_bar(x=sea['m'], y=sea['antal'], name='Antal', marker_color='#10b981', opacity=0.4)
            fig.add_scatter(x=sea['m'], y=sea['medianpris'], name='Medianpris',
                            yaxis='y2', line=dict(color='#ef4444', width=3))
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              yaxis=dict(title='Antal'), yaxis2=dict(title='Medianpris', overlaying='y', side='right'))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if 'omrade_clean' in df.columns:
            mn = st.slider("Min antal", 5, 50, 20, key="a_mn")
            astats = df.groupby('omrade_clean').agg(
                medianpris=('slutpris', 'median'), antal=('slutpris', 'count'),
                kvm=('pris_per_kvm', 'median')).reset_index()
            astats = astats[astats['antal'] >= mn].sort_values('medianpris')
            fig = px.bar(astats, x='medianpris', y='omrade_clean', orientation='h',
                         color='kvm', color_continuous_scale=["#1e293b", "#10b981"],
                         labels={'medianpris': 'Medianpris (kr)', 'omrade_clean': '', 'kvm': 'kr/m²'})
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### ⚔️ Budkrig")
        st.markdown("Budkrig = slutpriset överstiger utgångspriset. Visar hur vanligt det är per bostadstyp och hur marknaden rört sig över tid.")
        if 'budkrig' in df.columns and 'prisforandring_pct' in df.columns:

            # KPI per bostadstyp
            typ_stats = []
            for bk_typ, bk_label in [('lagenheter', 'Lägenheter'), ('villor', 'Villor'), ('radhus', 'Radhus')]:
                sub = df[df['bostadstyp'] == bk_typ]
                if len(sub) > 0:
                    rate = sub['budkrig'].mean() * 100
                    avg_ov = sub[sub['budkrig'] == 1]['prisforandring_pct'].mean()
                    typ_stats.append({'typ': bk_label, 'rate': rate, 'avg_ov': avg_ov, 'antal': len(sub)})

            col1, col2, col3 = st.columns(3)
            colors_typ = {'Lägenheter': '#10b981', 'Villor': '#6366f1', 'Radhus': '#f43f5e'}
            for col_el, ts in zip([col1, col2, col3], typ_stats):
                c = colors_typ.get(ts['typ'], '#9ca3af')
                col_el.markdown(
                    f"<div style='background:#1f2937;border:1px solid #374151;border-radius:12px;"
                    f"padding:16px;border-top:3px solid {c};'>"
                    f"<div style='color:#9ca3af;font-size:12px;'>{ts['typ']}</div>"
                    f"<div style='font-size:28px;font-weight:700;color:{c};'>{ts['rate']:.0f}%</div>"
                    f"<div style='color:#9ca3af;font-size:12px;'>har budkrig</div>"
                    f"<div style='color:#6b7280;font-size:11px;margin-top:6px;'>"
                    f"Snitt +{ts['avg_ov']:.1f}% vid budkrig</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # En enkel linjegraf: budkrig-% per år, per bostadstyp
            if 'sald_ar' in df.columns:
                bk_ar_typ = df.groupby(['sald_ar', 'bostadstyp']).agg(
                    budkrig_pct=('budkrig', lambda x: x.mean() * 100)
                ).reset_index()
                bk_ar_typ['Typ'] = bk_ar_typ['bostadstyp'].map(TYP_LABELS)
                fig_bk = px.line(
                    bk_ar_typ, x='sald_ar', y='budkrig_pct', color='Typ',
                    color_discrete_map=colors_typ, markers=True,
                    labels={'sald_ar': 'År', 'budkrig_pct': 'Andel budkrig (%)'},
                )
                fig_bk.update_layout(
                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', height=340, hovermode='x unified',
                    title="Hur vanligt är budkrig per år?"
                )
                st.plotly_chart(fig_bk, use_container_width=True)

            # Enkel tolkning
            if typ_stats:
                max_typ = max(typ_stats, key=lambda x: x['rate'])
                min_typ = min(typ_stats, key=lambda x: x['rate'])
                st.info(
                    f"**{max_typ['typ']}** har flest budkrig ({max_typ['rate']:.0f}%) — "
                    f"konkurrensen är hårdast här. **{min_typ['typ']}** har minst budkrig "
                    f"({min_typ['rate']:.0f}%) — mer förhandlingsutrymme."
                )
        else:
            st.info("Budkrig-data saknas i datasetet.")

    with tab5:
        st.markdown("### 🗺️ Områdesguide — jämför alla områden")
        if 'omrade_clean' in df.columns:
            mn2 = st.slider("Min antal försäljningar", 10, 50, 15, key="og_mn")
            og = df.groupby('omrade_clean').agg(
                medianpris=('slutpris', 'median'),
                kvm_pris=('pris_per_kvm', 'median'),
                antal=('slutpris', 'count'),
                budkrig_pct=('budkrig', lambda x: x.mean() * 100 if 'budkrig' in df.columns else 0),
                median_diff=('prisforandring_pct', 'median'),
                senaste_ar=('sald_ar', 'max'),
            ).reset_index()
            og = og[og['antal'] >= mn2].sort_values('medianpris', ascending=False).reset_index(drop=True)

            # Prisindex: normalisera medianpris mot genomsnittet
            snitt = og['medianpris'].mean()
            og['prisindex'] = (og['medianpris'] / snitt * 100).round(1)

            # Trend: jämför sista 2 år mot de föregående 2
            if 'sald_ar' in df.columns:
                max_ar = df['sald_ar'].max()
                recent = df[df['sald_ar'] >= max_ar - 1].groupby('omrade_clean')['slutpris'].median()
                older = df[(df['sald_ar'] >= max_ar - 3) & (df['sald_ar'] < max_ar - 1)].groupby('omrade_clean')['slutpris'].median()
                trend = ((recent / older - 1) * 100).round(1).rename('trend_2ar')
                og = og.merge(trend, left_on='omrade_clean', right_index=True, how='left')
            else:
                og['trend_2ar'] = None

            tbl_cols = ['omrade_clean', 'medianpris', 'kvm_pris', 'antal',
                        'budkrig_pct', 'median_diff', 'prisindex', 'trend_2ar']
            tbl_cols = [c for c in tbl_cols if c in og.columns]
            st.dataframe(og[tbl_cols], use_container_width=True, hide_index=True,
                         height=450,
                         column_config={
                             'omrade_clean': st.column_config.TextColumn('Område'),
                             'medianpris': st.column_config.NumberColumn('Medianpris', format='%d kr'),
                             'kvm_pris': st.column_config.NumberColumn('kr/m²', format='%d kr'),
                             'antal': st.column_config.NumberColumn('Antal sålda', format='%d'),
                             'budkrig_pct': st.column_config.NumberColumn('Budkrig %', format='%.1f%%'),
                             'median_diff': st.column_config.NumberColumn('Median prisskillnad', format='%.1f%%'),
                             'prisindex': st.column_config.ProgressColumn('Prisindex', min_value=0, max_value=200, format='%.0f'),
                             'trend_2ar': st.column_config.NumberColumn('Trend 2 år', format='%.1f%%'),
                         })

            # Scatter: kvm-pris vs budkrig-andel
            st.markdown("#### Presnivå vs budkrig-andel per område")
            fig_sc = px.scatter(og, x='kvm_pris', y='budkrig_pct', size='antal',
                                text='omrade_clean', color='trend_2ar',
                                color_continuous_scale=["#ef4444", "#6b7280", "#10b981"],
                                labels={'kvm_pris': 'kr/m²', 'budkrig_pct': 'Andel budkrig (%)',
                                        'trend_2ar': 'Trend 2 år (%)'},
                                hover_data={'medianpris': ':,.0f'})
            fig_sc.update_traces(textposition='top center', textfont_size=9)
            fig_sc.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                                 plot_bgcolor='rgba(0,0,0,0)', height=500)
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Områdesdata saknas.")


# ============================================================
# 5. SCENARIOANALYS
# ============================================================

elif page == "🔮 Scenarioanalys":
    st.markdown("# 🔮 Prisprognos")
    st.markdown("Tre realistiska scenarier baserade på Örebros historiska prisutveckling 2013–2026.")

    c1, c2 = st.columns(2)
    with c1:
        pris_nu = st.number_input("Nuvarande värde (kr)", value=2500000, step=100000, format="%d")
    with c2:
        ar = st.slider("År framåt", 1, 15, 10)

    if 'sald_ar' in df.columns:
        yrl = df.groupby('sald_ar')['slutpris'].median().reset_index()
        yrl = yrl[yrl['sald_ar'] >= 2015]
        if len(yrl) > 2:
            yrl['ch'] = yrl['slutpris'].pct_change()
            ag = float(yrl['ch'].median())
            bg = float(yrl['ch'].quantile(0.8))
            sg = float(yrl['ch'].quantile(0.2))
        else:
            ag, bg, sg = 0.03, 0.07, -0.03
    else:
        ag, bg, sg = 0.03, 0.07, -0.03

    bg = max(bg, 0.05)
    ag = max(ag, 0.02)
    sg = min(sg, -0.01)

    scenarios = {
        '🟢 Optimistiskt — Stark marknad': {
            'rate': bg, 'color': '#10b981',
            'forklaring': (
                f"Tillväxt på **{bg*100:.1f}% per år** — i linje med Örebros starkaste perioder. "
                "Förutsätter låg räntenivå, stark arbetsmarknad och fortsatt inflyttning till Örebro. "
                "Sannolikt om Riksbanken sänker räntan ytterligare och bostadsbyggandet bromsar."
            ),
        },
        '🔵 Basscenario — Marknaden stabiliseras': {
            'rate': ag, 'color': '#6366f1',
            'forklaring': (
                f"Tillväxt på **{ag*100:.1f}% per år** — median för de senaste 10 åren. "
                "Förutsätter att räntan håller sig på nuvarande nivå och att utbudet är balanserat. "
                "Det historiskt vanligaste utfallet för en medelstor svensk stad."
            ),
        },
        '🔴 Pessimistiskt — Prisjustering': {
            'rate': sg, 'color': '#ef4444',
            'forklaring': (
                f"Prisförändring på **{sg*100:.1f}% per år** — bottenkvartal de senaste 10 åren. "
                "Förutsätter höga räntor, ökad arbetslöshet eller kraftigt ökat utbud. "
                "Modellen visar att nedgångar historiskt tenderar att plana ut efter 2–3 år."
            ),
        },
    }

    yrs = list(range(2026, 2026 + ar + 1))
    fig = go.Figure()
    for name, s in scenarios.items():
        prices = [pris_nu]
        for y in range(ar):
            rate = s['rate'] if '🔴' not in name or y < 2 else abs(s['rate']) * 0.4
            prices.append(prices[-1] * (1 + rate))
        fig.add_trace(go.Scatter(
            x=yrs, y=prices, name=name.split(' — ')[1],
            line=dict(color=s['color'], width=3),
            hovertemplate='%{y:,.0f} kr<extra>' + name.split(' — ')[1] + '</extra>'
        ))

    fig.add_hline(y=pris_nu, line_color='#9ca3af', line_dash='dot',
                  annotation_text="Nuvarande värde", annotation_position="bottom right")
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', height=420, hovermode='x unified',
                      yaxis_title="Värde (kr)", legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig, use_container_width=True)

    # Slutvärden
    st.markdown("#### Prognosticerat värde om " + str(ar) + " år")
    res_cols = st.columns(3)
    for col_el, (name, s) in zip(res_cols, scenarios.items()):
        prices = [pris_nu]
        for y in range(ar):
            rate = s['rate'] if '🔴' not in name or y < 2 else abs(s['rate']) * 0.4
            prices.append(prices[-1] * (1 + rate))
        slutval = prices[-1]
        diff = slutval - pris_nu
        col_el.markdown(f"""
        <div class="deal-card" style="border-left:3px solid {s['color']};text-align:center;">
            <div style="font-size:13px;color:#9ca3af;">{name.split(' — ')[0]}</div>
            <div style="font-size:11px;color:#6b7280;margin-bottom:8px;">{name.split(' — ')[1]}</div>
            <div style="font-size:22px;font-weight:700;color:{s['color']};">{int(slutval):,} kr</div>
            <div style="font-size:12px;color:#9ca3af;">{'+'if diff>=0 else ''}{int(diff):,} kr</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Varför tänker vi så?")
    for name, s in scenarios.items():
        with st.expander(name):
            st.markdown(s['forklaring'])

    st.caption("⚠️ Baserat på historiska trender i Örebro 2013–2026. Inte finansiell rådgivning.")


# ============================================================
# 6. KÖPKALKYL
# ============================================================

elif page == "🏦 Köpkalkyl":
    st.markdown("# 🏦 Köpkalkyl")
    st.markdown("Räkna ut vad bostaden faktiskt kostar — månadsvis, vid köp och över tid.")

    # ── Indata ──────────────────────────────────────────────
    st.markdown("#### Bostad & finansiering")
    c1, c2, c3 = st.columns(3)
    with c1:
        kp  = st.number_input("Köpeskilling (kr)", value=2_500_000, step=50_000, format="%d", key="kp")
        ki  = st.slider("Kontantinsats (%)", 10, 50, 15, key="ki",
                        help="Minst 15% krävs för bolån i Sverige")
        lt  = st.slider("Löptid (år)", 10, 50, 30, key="lt")
    with c2:
        ra  = st.slider("Bolåneränta (%)", 1.0, 10.0, 4.5, step=0.1, key="ra")
        am  = st.slider("Amortering (%/år av lån)", 0.0, 3.0, 1.0, step=0.5, key="am",
                        help="Lagkrav: ≥1% om lån >50% av värdet, ≥2% om lån >70%")
        av  = st.number_input("Månadsavgift/drift (kr)", value=3_500, step=100, format="%d", key="av")
    with c3:
        hush = st.number_input("Hushållsinkomst brutto/mån (kr)", value=60_000, step=1_000,
                               format="%d", key="hush",
                               help="Används för kvar-att-leva-på beräkning")
        ovr  = st.number_input("Övriga fasta kostnader/mån (kr)", value=15_000, step=500,
                               format="%d", key="ovr",
                               help="Mat, bil, försäkringar, etc.")
        skatt_pct = st.slider("Kommunalskatt (%)", 28.0, 35.0, 32.0, step=0.5, key="skatt")

    # ── Beräkningar ─────────────────────────────────────────
    kontant      = int(kp * ki / 100)
    lan          = kp - kontant
    r_man        = ra / 100 / 12
    n_man        = lt * 12
    ranta_man    = lan * r_man * (1 + r_man)**n_man / ((1 + r_man)**n_man - 1) if r_man > 0 else lan / n_man
    amortering_man = lan * am / 100 / 12
    ranta_ar     = ranta_man * 12

    # Ränteavdrag 30% (skattereduktion)
    ranteavdrag_man = ranta_man * 0.30

    # Transaktionskostnader vid köp
    stampelskatt = kp * 0.015
    pantbrev     = lan * 0.02
    pantbrev_exp = 825
    tot_trans    = stampelskatt + pantbrev + pantbrev_exp

    # Nettolön (schablonmässig)
    netto_lon = hush * (1 - skatt_pct / 100)

    # Boendekostnad per månad (efter ränteavdrag)
    boende_man   = ranta_man - ranteavdrag_man + amortering_man + av
    kvar_att_leva = netto_lon - boende_man - ovr

    # Belåningsgrad
    belaning_pct = lan / kp * 100

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Nyckeltal ────────────────────────────────────────────
    st.markdown("#### Nyckeltal")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Kontantinsats", f"{kontant:,} kr")
    m2.metric("Lån", f"{lan:,} kr",
              delta=f"{belaning_pct:.0f}% belåning",
              delta_color="inverse" if belaning_pct > 70 else "normal")
    m3.metric("Ränta/mån", f"{int(ranta_man):,} kr",
              delta=f"−{int(ranteavdrag_man):,} kr avdrag",
              delta_color="normal")
    m4.metric("Total boendekostnad/mån", f"{int(boende_man):,} kr")
    m5.metric("Kvar att leva på", f"{int(kvar_att_leva):,} kr/mån",
              delta="OK" if kvar_att_leva > 10_000 else "Tight",
              delta_color="normal" if kvar_att_leva > 10_000 else "inverse")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Engångskostnader ────────────────────────────────────
    with st.expander("📋 Vad kostar det att köpa? (engångsutgifter)", expanded=True):
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Kontantinsats", f"{kontant:,} kr")
        ec2.metric("Stämpelskatt (1,5%)", f"{int(stampelskatt):,} kr",
                   help="Lagfartsstämpel till Lantmäteriet")
        ec3.metric("Pantbrev (2% av lån)", f"{int(pantbrev + pantbrev_exp):,} kr",
                   help="Kostnad för nya pantbrevsinteckningar")
        ec4.metric("Totalt kapital vid köp", f"{int(kontant + tot_trans):,} kr",
                   help="Summa du behöver ha likvid på kontot")
        st.info(f"💡 Utöver köpeskillingen behöver du **{int(tot_trans):,} kr** i engångskostnader "
                f"({tot_trans/kp*100:.1f}% av köpeskillingen).")

    # ── Månadsbudget ────────────────────────────────────────
    st.markdown("#### Månadsbudget")
    mb1, mb2 = st.columns(2)
    with mb1:
        st.markdown("**Inkomst**")
        rows_in = [
            ("Bruttolön/mån", f"{hush:,} kr"),
            ("− Kommunalskatt", f"−{int(hush * skatt_pct/100):,} kr"),
            ("= Nettolön/mån", f"**{int(netto_lon):,} kr**"),
        ]
        for label, val in rows_in:
            st.markdown(f"<div style='display:flex;justify-content:space-between;padding:4px 0;"
                        f"border-bottom:1px solid #1f2937;'><span style='color:#9ca3af;'>{label}</span>"
                        f"<span>{val}</span></div>", unsafe_allow_html=True)
    with mb2:
        st.markdown("**Boendekostnad**")
        rows_bo = [
            ("Ränta", f"{int(ranta_man):,} kr"),
            ("− Ränteavdrag (30%)", f"−{int(ranteavdrag_man):,} kr"),
            ("Amortering", f"{int(amortering_man):,} kr"),
            ("Avgift/drift", f"{int(av):,} kr"),
            ("= Boende totalt/mån", f"**{int(boende_man):,} kr**"),
        ]
        for label, val in rows_bo:
            st.markdown(f"<div style='display:flex;justify-content:space-between;padding:4px 0;"
                        f"border-bottom:1px solid #1f2937;'><span style='color:#9ca3af;'>{label}</span>"
                        f"<span>{val}</span></div>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Grafer ──────────────────────────────────────────────
    g1, g2 = st.columns(2)

    with g1:
        st.markdown("##### Amorteringsplan")
        saldo = [lan]
        ranta_kum = [0]
        for y in range(lt):
            ny_saldo = max(0, saldo[-1] - amortering_man * 12)
            saldo.append(ny_saldo)
            ranta_kum.append(ranta_kum[-1] + saldo[-2] * ra / 100)
        yrs = list(range(2026, 2026 + lt + 1))
        fig_am = go.Figure()
        fig_am.add_scatter(x=yrs, y=saldo, name="Kvarvarande lån",
                           fill='tozeroy', line=dict(color='#6366f1', width=2))
        fig_am.add_scatter(x=yrs, y=ranta_kum, name="Ackumulerad ränta",
                           line=dict(color='#ef4444', width=2, dash='dot'))
        fig_am.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                             plot_bgcolor='rgba(0,0,0,0)', height=300,
                             yaxis_title="kr", hovermode='x unified',
                             legend=dict(x=0, y=1))
        st.plotly_chart(fig_am, use_container_width=True)

    with g2:
        st.markdown("##### Månadsbudget — fördelning")
        labels = ["Ränta (netto)", "Amortering", "Avgift/drift", "Övriga kostnader", "Kvar att leva"]
        values = [
            max(0, int(ranta_man - ranteavdrag_man)),
            int(amortering_man),
            int(av),
            int(ovr),
            max(0, int(kvar_att_leva)),
        ]
        colors = ['#6366f1', '#10b981', '#f59e0b', '#9ca3af', '#1f2937']
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.5,
            marker=dict(colors=colors),
            textinfo='label+percent',
            textfont_size=11,
        ))
        fig_pie.add_annotation(text=f"{int(netto_lon):,}<br>kr/mån",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=13, color='#f9fafb'))
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300,
                              showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Känslighetsanalys ───────────────────────────────────
    st.markdown("#### Vad händer om räntan ändras?")
    rantor = [r/10 for r in range(10, 101, 5)]
    boende_per_ranta = []
    for r_test in rantor:
        rm = lan * (r_test/100/12) * (1 + r_test/100/12)**n_man / ((1 + r_test/100/12)**n_man - 1)
        boende_per_ranta.append(int(rm - rm*0.30 + amortering_man + av))
    fig_sens = go.Figure()
    fig_sens.add_scatter(x=rantor, y=boende_per_ranta, mode='lines+markers',
                         line=dict(color='#10b981', width=2),
                         hovertemplate='%{x:.1f}% ränta → %{y:,} kr/mån<extra></extra>')
    fig_sens.add_vline(x=ra, line_color='#fbbf24', line_dash='dash',
                       annotation_text=f"Din ränta {ra}%", annotation_position="top right")
    fig_sens.add_hline(y=int(netto_lon * 0.35), line_color='#ef4444', line_dash='dot',
                       annotation_text="35% av nettolön (rekommenderat tak)",
                       annotation_position="bottom right")
    fig_sens.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)', height=280,
                           xaxis_title="Ränta (%)", yaxis_title="Boendekostnad/mån (kr)")
    st.plotly_chart(fig_sens, use_container_width=True)

    st.caption("⚠️ Ränteavdrag är 30% upp till 100 000 kr ränta/år, 21% däröver. Beräkning förenklad.")


# ============================================================
# 7. INVESTERINGSKALKYL
# ============================================================

elif page == "💼 Investeringskalkyl":
    st.markdown("# 💼 Investeringskalkyl")
    st.markdown("Beräkna lönsamhet, cashflow och avkastning för en investeringsfastighet i Örebro.")

    tab_kalkyl, tab_exit = st.tabs(["📊 Löpande kalkyl", "🚪 Exit-analys"])

    with tab_kalkyl:
        st.markdown("#### Köp & finansiering")
        c1, c2, c3 = st.columns(3)
        with c1:
            ip = st.number_input("Köpeskilling (kr)", value=2_000_000, step=50_000, format="%d", key="ip")
            ik = st.slider("Kontantinsats (%)", 10, 50, 25, key="ik")
            ir = st.slider("Bolåneränta (%)", 1.0, 10.0, 4.5, step=0.1, key="ir")
        with c2:
            hy = st.number_input("Hyra/mån (kr)", value=9_000, step=500, format="%d")
            ia = st.number_input("Avgift/mån (kr)", value=3_000, step=100, format="%d", key="ia")
            dr = st.number_input("Underhåll & drift/år (kr)", value=15_000, step=1_000, format="%d")
        with c3:
            vk = st.slider("Vakans (%)", 0, 20, 5, help="Andel av året fastigheten står tom")
            am_pct = st.slider("Amortering (%/år av lån)", 0.0, 3.0, 1.0, step=0.5)
            hz = st.slider("Horisont (år)", 1, 20, 10)

        # --- Beräkningar ---
        il = int(ip * (1 - ik / 100))
        ikr = ip - il   # kontantinsats kr

        # Transaktionskostnader
        stampelskatt = ip * 0.015
        pantbrev = il * 0.02
        pantbrev_exp = 825
        tot_trans = stampelskatt + pantbrev + pantbrev_exp

        # Månadsränta
        ri = ir / 100 / 12
        n = 30 * 12
        ml = il * ri * (1 + ri)**n / ((1 + ri)**n - 1) if ri > 0 else il / n
        amortering_manad = il * am_pct / 100 / 12

        # Intäkter
        eff_hyra = hy * 12 * (1 - vk / 100)

        # Skatt på hyresintäkter (schablon: 40 000 kr avdrag, 30% på överskott)
        hyra_skatteplikt = max(0, eff_hyra - 40_000)
        hyra_skatt = hyra_skatteplikt * 0.30

        # Ränteavdrag 30%
        ranteavdrag = ml * 12 * 0.30

        # Kostnader
        kost_lopande = ia * 12 + dr + ml * 12
        kost_efter_skatt = kost_lopande + hyra_skatt - ranteavdrag

        # Netto
        netto_fore_skatt = eff_hyra - kost_lopande
        netto_efter_skatt = eff_hyra - kost_efter_skatt

        # Avkastning
        by = eff_hyra / ip * 100
        ny = netto_efter_skatt / ip * 100
        cash_manad = netto_efter_skatt / 12

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Transaktionskostnader
        with st.expander("📋 Engångskostnader vid köp", expanded=False):
            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.metric("Kontantinsats", f"{int(ikr):,} kr")
            ec2.metric("Stämpelskatt (1,5%)", f"{int(stampelskatt):,} kr")
            ec3.metric("Pantbrevskostnad (2%)", f"{int(pantbrev):,} kr")
            ec4.metric("Totalt kapital vid köp", f"{int(ikr + tot_trans):,} kr",
                       help="Kontantinsats + alla engångskostnader")

        # Nyckeltal
        st.markdown("#### Nyckeltal")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Bruttoavkastning", f"{by:.1f}%", help="Hyresintäkter / köpeskilling")
        m2.metric("Nettoavkastning", f"{ny:.1f}%",
                  delta="Positivt" if ny > 0 else "Negativt",
                  delta_color="normal" if ny > 0 else "inverse",
                  help="Netto efter skatter och kostnader / köpeskilling")
        m3.metric("Cashflow/mån", f"{int(cash_manad):,} kr",
                  delta_color="normal" if cash_manad > 0 else "inverse")
        m4.metric("Ränteavdrag/år", f"{int(ranteavdrag):,} kr",
                  help="30% skattereduktion på räntekostnader")
        m5.metric("Hyresskatt/år", f"{int(hyra_skatt):,} kr",
                  help="30% skatt på hyresöverskott efter 40 000 kr schablonavdrag")

        if netto_efter_skatt > 0:
            aterbetalningstid = (ikr + tot_trans) / netto_efter_skatt
            st.success(f"✅ Återbetalningstid på investerat kapital: **{aterbetalningstid:.1f} år**")
        else:
            st.error("❌ Negativ avkastning — kostnader överstiger intäkter efter skatt.")

        # Månadsbudget
        st.markdown("#### Månadsbudget")
        mb1, mb2, mb3 = st.columns(3)
        with mb1:
            st.markdown("**Intäkter**")
            st.markdown(f"Hyra (efter vakans): **{int(eff_hyra/12):,} kr/mån**")
        with mb2:
            st.markdown("**Kostnader**")
            st.markdown(f"Ränta: {int(ml):,} kr  \nAmortering: {int(amortering_manad):,} kr  \n"
                        f"Avgift: {int(ia):,} kr  \nDrift: {int(dr/12):,} kr")
        with mb3:
            st.markdown("**Skatter**")
            st.markdown(f"Hyresskatt: {int(hyra_skatt/12):,} kr/mån  \n"
                        f"Ränteavdrag: -{int(ranteavdrag/12):,} kr/mån")

        # Graf: värde + ackumulerat cashflow
        vo = st.slider("Antagen värdeökning (%/år)", 0.0, 8.0, 3.0, step=0.5)
        yrs = list(range(2026, 2026 + hz + 1))
        vals = [ip * (1 + vo / 100)**i for i in range(hz + 1)]
        cf_acc = [max(0, netto_efter_skatt * i) for i in range(hz + 1)]
        lan_kvar = [max(0, il * (1 - am_pct/100)**i) for i in range(hz + 1)]

        fig = go.Figure()
        fig.add_scatter(x=yrs, y=vals, name="Fastighetsvärde", line=dict(color='#10b981', width=3))
        fig.add_scatter(x=yrs, y=lan_kvar, name="Kvarvarande lån", line=dict(color='#ef4444', width=2, dash='dot'))
        if netto_efter_skatt > 0:
            fig.add_bar(x=yrs, y=cf_acc, name="Ack. cashflow (efter skatt)", marker_color='rgba(99,102,241,0.4)')
        fig.add_hline(y=ip + tot_trans, line_color='#f59e0b', line_dash='dot',
                      annotation_text="Totalt investerat kapital", annotation_position="bottom right")
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', height=420, hovermode='x unified',
                          yaxis_title="Kronor")
        st.plotly_chart(fig, use_container_width=True)

    with tab_exit:
        st.markdown("#### Vad händer om du säljer?")
        st.markdown("Beräknar din vinst och skattekostnad vid försäljning efter X år.")

        ex_ar = st.slider("Sälj efter (år)", 1, 20, 5, key="ex_ar")
        ex_vo = st.slider("Värdeökning (%/år)", 0.0, 8.0, 3.0, step=0.5, key="ex_vo")
        ip_ex = st.number_input("Köpeskilling (kr)", value=2_000_000, step=50_000, format="%d", key="ip_ex")
        ik_ex = st.slider("Kontantinsats (%)", 10, 50, 25, key="ik_ex")
        am_ex = st.slider("Amortering (%/år)", 0.0, 3.0, 1.0, step=0.5, key="am_ex")

        il_ex = int(ip_ex * (1 - ik_ex / 100))
        ikr_ex = ip_ex - il_ex
        tot_trans_ex = ip_ex * 0.015 + il_ex * 0.02 + 825

        forsaljningspris = ip_ex * (1 + ex_vo / 100)**ex_ar
        kvar_lan = il_ex * (1 - am_ex / 100)**ex_ar
        maklararvode = forsaljningspris * 0.025   # ~2,5%

        # Kapitalvinst och skatt
        inköpspris_inkl = ip_ex + tot_trans_ex
        kapitalvinst = forsaljningspris - inköpspris_inkl - maklararvode
        kapitalvinstskatt = max(0, kapitalvinst * 0.22)  # 22% reavinstskatt

        netto_fran_forsaljning = forsaljningspris - kvar_lan - maklararvode - kapitalvinstskatt
        total_avkastning = netto_fran_forsaljning - ikr_ex - tot_trans_ex

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        ex1, ex2, ex3, ex4 = st.columns(4)
        ex1.metric("Försäljningspris", f"{int(forsaljningspris):,} kr")
        ex2.metric("Kapitalvinst", f"{int(kapitalvinst):,} kr")
        ex3.metric("Reavinstskatt (22%)", f"{int(kapitalvinstskatt):,} kr")
        ex4.metric("Netto i handen", f"{int(netto_fran_forsaljning):,} kr",
                   delta=f"{'+' if total_avkastning >= 0 else ''}{int(total_avkastning):,} kr vs investerat",
                   delta_color="normal" if total_avkastning >= 0 else "inverse")

        st.markdown(f"""
        | Post | Belopp |
        |---|---|
        | Försäljningspris | {int(forsaljningspris):,} kr |
        | − Kvarvarande lån | {int(kvar_lan):,} kr |
        | − Mäklararvode (~2,5%) | {int(maklararvode):,} kr |
        | − Reavinstskatt (22%) | {int(kapitalvinstskatt):,} kr |
        | **= Netto efter försäljning** | **{int(netto_fran_forsaljning):,} kr** |
        | − Ursprunglig investering | {int(ikr_ex + tot_trans_ex):,} kr |
        | **= Total avkastning** | **{int(total_avkastning):,} kr** |
        """)

        st.caption("⚠️ Inte finansiell rådgivning. Skatteberäkningarna är förenklade — rådgör med skatterådgivare.")


# ============================================================
# 8. OM MODELLEN
# ============================================================

elif page == "ℹ️ Om modellen":
    st.markdown("# ℹ️ Om ValuEstate")

    # Hero-sektion
    h1, h2, h3, h4 = st.columns(4)
    h1.markdown(f"""<div style='background:#1f2937;border:1px solid #374151;border-radius:12px;
        padding:20px;text-align:center;border-top:3px solid #10b981;'>
        <div style='font-size:32px;font-weight:700;color:#10b981;'>{len(df):,}</div>
        <div style='color:#9ca3af;font-size:13px;'>Sålda bostäder</div>
        <div style='color:#6b7280;font-size:11px;'>2013–2026</div></div>""", unsafe_allow_html=True)
    h2.markdown(f"""<div style='background:#1f2937;border:1px solid #374151;border-radius:12px;
        padding:20px;text-align:center;border-top:3px solid #6366f1;'>
        <div style='font-size:32px;font-weight:700;color:#6366f1;'>3</div>
        <div style='color:#9ca3af;font-size:13px;'>ML-modeller</div>
        <div style='color:#6b7280;font-size:11px;'>Lägenhet · Villa · Radhus</div></div>""", unsafe_allow_html=True)

    best_r2 = max((_model_r2(m) for m in v2_models.values()), default=0) if v2_models else 0
    h3.markdown(f"""<div style='background:#1f2937;border:1px solid #374151;border-radius:12px;
        padding:20px;text-align:center;border-top:3px solid #fbbf24;'>
        <div style='font-size:32px;font-weight:700;color:#fbbf24;'>{best_r2:.3f}</div>
        <div style='color:#9ca3af;font-size:13px;'>Bästa R²</div>
        <div style='color:#6b7280;font-size:11px;'>Förklaringsgrad</div></div>""", unsafe_allow_html=True)
    live_count = len(df_active) if df_active is not None else 0
    h4.markdown(f"""<div style='background:#1f2937;border:1px solid #374151;border-radius:12px;
        padding:20px;text-align:center;border-top:3px solid #f43f5e;'>
        <div style='font-size:32px;font-weight:700;color:#f43f5e;'>{live_count}</div>
        <div style='color:#9ca3af;font-size:13px;'>Aktiva annonser</div>
        <div style='color:#6b7280;font-size:11px;'>Analyserade idag</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Hur det fungerar
    st.markdown("### Hur fungerar det?")
    s1, s2, s3 = st.columns(3)
    for col_el, num, title, desc, col in zip(
        [s1, s2, s3],
        ["1", "2", "3"],
        ["Scrapar Hemnet", "ML analyserar", "Deal Score"],
        [
            "Varje dag hämtar vi alla aktiva annonser i Örebro kommun direkt från Hemnet — pris, storlek, område och mer.",
            "Tre separata maskininlärningsmodeller (en per bostadstyp) estimerar marknadsvärdet baserat på 30+ egenskaper.",
            "Varje annons får ett poäng 0–100 baserat på undervärdering, område, jämförbara försäljningar och modellkonfidans."
        ],
        ["#10b981", "#6366f1", "#fbbf24"]
    ):
        col_el.markdown(
            f"<div style='background:#1f2937;border:1px solid #374151;border-radius:12px;padding:20px;height:160px;'>"
            f"<div style='font-size:28px;font-weight:700;color:{col};margin-bottom:8px;'>{num}</div>"
            f"<div style='font-size:15px;font-weight:600;color:#f9fafb;margin-bottom:8px;'>{title}</div>"
            f"<div style='font-size:13px;color:#9ca3af;line-height:1.5;'>{desc}</div></div>",
            unsafe_allow_html=True
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Modellkort
    if v2_models:
        st.markdown("### Modellprestanda")
        mc = st.columns(len(v2_models))
        colors_m = {'lagenheter': '#10b981', 'villor': '#6366f1', 'radhus': '#f43f5e'}
        for col_el, (typ, m) in zip(mc, v2_models.items()):
            c = colors_m.get(typ, '#6b7280')
            r2 = _model_r2(m)
            mae = m['metrics'].get('MAE') or m['metrics'].get('lgbm_test', {}).get('MAE', 0)
            _conf = m.get('confidence', {})
            ci = _conf.get('interval_pct', '?') if isinstance(_conf, dict) else '?'
            _default_names = {'lagenheter': 'Random Forest', 'villor': 'LightGBM + CatBoost', 'radhus': 'CatBoost'}
            name = m.get('model_name') or _default_names.get(typ, '?')
            nfeat = len(m.get('feature_names', []))
            ci_color = '#ef4444' if isinstance(ci, (int, float)) and ci > 20 else '#f9fafb'
            col_el.markdown(
                f"<div style='background:#1f2937;border:1px solid #374151;border-radius:12px;"
                f"padding:20px;border-top:3px solid {c};'>"
                f"<div style='font-size:16px;font-weight:700;color:{c};'>{TYP_LABELS[typ]}</div>"
                f"<div style='color:#6b7280;font-size:12px;margin-bottom:12px;'>{name}</div>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;'>"
                f"<span style='color:#9ca3af;font-size:12px;'>R²</span>"
                f"<span style='font-weight:600;color:#f9fafb;'>{r2:.4f}</span></div>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;'>"
                f"<span style='color:#9ca3af;font-size:12px;'>MAE</span>"
                f"<span style='font-weight:600;color:#f9fafb;'>{mae:,} kr</span></div>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;'>"
                f"<span style='color:#9ca3af;font-size:12px;'>Osäkerhet</span>"
                f"<span style='font-weight:600;color:{ci_color};'>±{ci}%</span></div>"
                f"<div style='display:flex;justify-content:space-between;'>"
                f"<span style='color:#9ca3af;font-size:12px;'>Features</span>"
                f"<span style='font-weight:600;color:#f9fafb;'>{nfeat} st</span></div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Begränsningar
    st.markdown("### Viktigt att veta")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("""
**Vad modellen ser:**
- Boarea, rum, avgift, byggår, våning
- Område och avstånd till centrum/station
- Historiska jämförbara försäljningar
- Säsong och marknadstrender
""")
    with b2:
        st.markdown("""
**Vad modellen inte ser:**
- Skick, renovering och utsikt
- Solläge och bullernivå
- Planlösning och material
- Budgivarnas psykologi
""")

    st.warning("⚠️ ValuEstate är ett beslutsstöd — inte finansiell rådgivning. Alltid konsultera mäklare.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("*Byggd av **Loran Ali** · Statistik, Data & AI · Örebro 2026*")
