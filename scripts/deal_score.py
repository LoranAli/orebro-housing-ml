"""
Örebro Housing Intelligence — Deal Score Engine v4.1
=====================================================
Trösklar kalibrerade mot faktisk poängdistribution (2026-03-24, 720 annonser).

Modellkvalitet per typ:
  Lägenheter: R²=0.947, CI ±10% → hög träffsäkerhet, kategorier är tillförlitliga
  Radhus:     R²=0.905, CI ±18% → god träffsäkerhet, kategorier är rimliga
  Villor:     R²=0.725, CI ±50% → svag träffsäkerhet, kategorier är INDIKATIVA

Kategorier är typ-relativa och kalibrerade mot ~0.5% Exceptionellt / ~2% Bra / ~5% Potentiellt:
  Lägenheter: 🔥 58+ | 🟢 48–57 | 🟡 38–47 | ⚪ 0–37
  Radhus:     🔥 46+ | 🟢 34–45 | 🟡 30–33 | ⚪ 0–29
  Villor:     🔥 50+ | 🟢 46–49 | 🟡 38–45 | ⚪ 0–37
  (Villor-scores påverkas av hög CI-rabatt; visa alltid med varning)

TOTAL_ADJUSTMENT (empiriskt kalibrerade):
  Lägenheter: -2% | Radhus: +3% | Villor: -16%

"Överprissatt" visas BARA som informationstext, aldrig som kategori.
"""

import numpy as np
import pandas as pd


# ============================================================
# DYNAMISKA KATEGORIER PER BOSTADSTYP
# ============================================================

# Trösklar kalibrerade mot faktisk poängdistribution (2026-03-24: 720 annonser).
# Mål: ~0.5% Exceptionellt, ~1.5-2% Bra, ~4-5% Potentiellt av annonser per körning.
# Radhus-trösklar baserade på simulerad distribution efter TOTAL_ADJUSTMENT +3%.
DEAL_CATEGORIES_BY_TYPE = {
    'lagenheter': [
        {'min': 58, 'key': 'exceptional', 'label': 'Exceptionellt fynd', 'icon': '🔥', 'color': '#FFD700'},
        {'min': 48, 'key': 'good',        'label': 'Bra fynd',           'icon': '🟢', 'color': '#00D4AA'},
        {'min': 38, 'key': 'interesting', 'label': 'Potentiellt intressant', 'icon': '🟡', 'color': '#feca57'},
        {'min': 0,  'key': 'fair',        'label': 'Rimligt pris',       'icon': '⚪', 'color': '#888888'},
    ],
    'villor': [
        # Villor: modell R²=0.725, CI±50% — trösklar avspeglar lägre förtroende
        {'min': 50, 'key': 'exceptional', 'label': 'Exceptionellt fynd', 'icon': '🔥', 'color': '#FFD700'},
        {'min': 46, 'key': 'good',        'label': 'Bra fynd',           'icon': '🟢', 'color': '#00D4AA'},
        {'min': 38, 'key': 'interesting', 'label': 'Potentiellt intressant', 'icon': '🟡', 'color': '#feca57'},
        {'min': 0,  'key': 'fair',        'label': 'Rimligt pris',       'icon': '⚪', 'color': '#888888'},
    ],
    'radhus': [
        # Radhus: trösklar baserade på simulerad distribution (TOTAL_ADJUSTMENT +3%)
        # ~1% Exceptionellt / ~9% Bra / ~10% Potentiellt av aktiva radhus-annonser
        {'min': 46, 'key': 'exceptional', 'label': 'Exceptionellt fynd', 'icon': '🔥', 'color': '#FFD700'},
        {'min': 34, 'key': 'good',        'label': 'Bra fynd',           'icon': '🟢', 'color': '#00D4AA'},
        {'min': 30, 'key': 'interesting', 'label': 'Potentiellt intressant', 'icon': '🟡', 'color': '#feca57'},
        {'min': 0,  'key': 'fair',        'label': 'Rimligt pris',       'icon': '⚪', 'color': '#888888'},
    ],
}

# Fallback (same as lägenheter)
DEAL_CATEGORIES_BY_TYPE['default'] = DEAL_CATEGORIES_BY_TYPE['lagenheter']


def get_deal_category(score, bostadstyp='lagenheter'):
    cats = DEAL_CATEGORIES_BY_TYPE.get(bostadstyp, DEAL_CATEGORIES_BY_TYPE['lagenheter'])
    for cat in cats:
        if score >= cat['min']:
            return cat
    return cats[-1]


# ============================================================
# SANITY CHECKS
# ============================================================

PRICE_RANGES = {
    'lagenheter': {'min': 300_000,   'max': 8_000_000},
    'villor':     {'min': 800_000,   'max': 15_000_000},
    'radhus':     {'min': 500_000,   'max': 8_000_000},
}

BOAREA_RANGES = {
    'lagenheter': {'min': 15,  'max': 200},
    'villor':     {'min': 40,  'max': 400},
    'radhus':     {'min': 30,  'max': 250},
}


def is_estimate_sane(estimat, utgangspris, bostadstyp):
    typ = bostadstyp if bostadstyp in PRICE_RANGES else 'lagenheter'
    pr = PRICE_RANGES[typ]
    if estimat < pr['min'] or estimat > pr['max']:
        return False
    if utgangspris > 0:
        ratio = estimat / utgangspris
        if ratio < 0.4 or ratio > 2.5:
            return False
    return True


def is_listing_valid(listing):
    typ = listing.get('bostadstyp', 'lagenheter')
    bo = listing.get('boarea_kvm', 0)
    pris = listing.get('utgangspris', 0)
    rum = listing.get('antal_rum', 0)
    br = BOAREA_RANGES.get(typ, BOAREA_RANGES['lagenheter'])
    pr = PRICE_RANGES.get(typ, PRICE_RANGES['lagenheter'])
    if bo < br['min'] or bo > br['max']:
        return False
    if pris < pr['min'] or pris > pr['max']:
        return False
    if rum > 0 and bo > 0 and (bo / rum < 5 or bo / rum > 80):
        return False
    return True


# ============================================================
# KORRIGERING: BUDGIVNING + MODELL-BIAS (empiriskt kalibrerade värden)
#
# Lägenheter: slutpris ~-3% under utgångspris, modell bias ~+1% → justering -2%
# Villor:     slutpris ~+5% över utgångspris, modell bias ~-21% → justering -16%
# Radhus:     slutpris ~+5% över utgångspris, modell bias ~-2%  → justering +3%
#   (observerad median skillnad_pct = +6.7% ≈ budpremien; bias-del är minimal)
#
# OBS: Dessa värden är kalibrerade mot faktiska utfall. Uppdatera vid modelomträning.
# ============================================================

TOTAL_ADJUSTMENT = {
    'lagenheter': -2.0,
    'villor':     -16.0,
    'radhus':      3.0,
}

# Statisk fallback CI (används om ci_low/ci_high saknas i annonsen)
MODEL_CI_FALLBACK = {
    'lagenheter': 5.4,
    'villor':     27.7,
    'radhus':     9.5,
}


def _get_listing_ci_pct(listing):
    """
    Beräkna faktisk CI% per annons från ci_low/ci_high.
    Faller tillbaka på statisk MODEL_CI_FALLBACK om värdena saknas.
    """
    ci_low  = listing.get('ci_low',  None)
    ci_high = listing.get('ci_high', None)
    estimat = listing.get('estimerat_varde', 0)
    bostadstyp = listing.get('bostadstyp', 'lagenheter')

    if (ci_low is not None and ci_high is not None and
            estimat > 0 and
            not (isinstance(ci_low, float) and np.isnan(ci_low)) and
            not (isinstance(ci_high, float) and np.isnan(ci_high))):
        half_range = (float(ci_high) - float(ci_low)) / 2
        ci_pct = half_range / estimat * 100
        # Klipp till rimligt intervall
        return max(2.0, min(60.0, ci_pct))

    return MODEL_CI_FALLBACK.get(bostadstyp, 15.0)


# ============================================================
# KOMPONENT 1: UNDERVÄRDERING (0–50 poäng)
# ============================================================

def score_undervaluation(listing, estimat, area_stats):
    """
    Rättad formel: (estimat - adjusted_pris) / adjusted_pris * 100
    Valideringsbonus via comps_pris_kvm_90d.
    """
    utgangspris = listing.get('utgangspris', 0)
    bostadstyp  = listing.get('bostadstyp', 'lagenheter')
    boarea      = listing.get('boarea_kvm', 0)

    if estimat <= 0 or utgangspris <= 0:
        return 0, 0

    if not is_estimate_sane(estimat, utgangspris, bostadstyp):
        return 0, 0

    # Justera utgångspris med kombinerad korrigering
    adj = TOTAL_ADJUSTMENT.get(bostadstyp, 0)
    adjusted_pris = utgangspris * (1 + adj / 100)

    # Formel: (estimat - adjusted_pris) / estimat * 100
    # Tolkning: hur många % av estimatet sparar du vs justerat pris
    underval_pct = (estimat - adjusted_pris) / estimat * 100

    # Beräkna rå score
    if underval_pct >= 25:
        score = 45 + min(5, (underval_pct - 25) / 15 * 5)
    elif underval_pct >= 15:
        score = 30 + (underval_pct - 15) / 10 * 15
    elif underval_pct >= 10:
        score = 18 + (underval_pct - 10) / 5 * 12
    elif underval_pct >= 5:
        score = 5 + (underval_pct - 5) / 5 * 13
    elif underval_pct >= 0:
        score = underval_pct
    else:
        score = max(-10, underval_pct * 0.5)

    # KONFIDENSRABATT baserad på faktisk CI per annons
    ci_pct = _get_listing_ci_pct(listing)
    if ci_pct <= 8:
        discount = 1.0
    elif ci_pct <= 12:
        discount = 0.85
    elif ci_pct <= 20:
        discount = 0.65
    else:
        discount = 0.45

    if score > 0:
        score = score * discount

    # VALIDERINGSBONUS: comps_pris_kvm_90d
    # Om marknadens medianpris/kvm stödjer estimatet → liten bonus
    comps_kvm = listing.get('comps_pris_kvm_90d', None)
    if (comps_kvm is not None and boarea > 0 and
            not (isinstance(comps_kvm, float) and np.isnan(comps_kvm))):
        market_estimat = float(comps_kvm) * boarea
        if market_estimat > 0:
            comps_underval = (market_estimat - adjusted_pris) / adjusted_pris * 100
            # Bonus om comps bekräftar undervärdering (max ±3 poäng)
            if comps_underval > 5 and underval_pct > 5:
                score = min(50, score + 2)
            elif comps_underval < -5 and underval_pct > 5:
                # comps säger tvärtemot → sänk lite
                score = max(0, score - 3)

    return round(min(50, score), 1), round(underval_pct, 1)


# ============================================================
# KOMPONENT 2: OMRÅDE (0–15 poäng)
# ============================================================

def score_area(omrade, area_stats):
    """
    Inkluderar nu budkrig_rate: högt budkrig → konkurrens → svårare hitta fynd.
    """
    if omrade == 'övrigt' or omrade not in area_stats:
        return 3

    stats  = area_stats[omrade]
    trend  = stats.get('trend', 0)
    antal  = stats.get('antal', 0)
    budkrig_rate = stats.get('budkrig_rate', 0.5)  # 0–1

    score = 0

    # Pristrend (0–8)
    if trend > 5:
        score += 8
    elif trend > 2:
        score += 6
    elif trend > 0:
        score += 4
    elif trend > -3:
        score += 2

    # Antal sålda senaste 12 mån (0–7)
    if antal >= 30:
        score += 7
    elif antal >= 15:
        score += 5
    elif antal >= 5:
        score += 3
    else:
        score += 1

    # Budkrig-penalty: hög budkrigsfrekvens = svårt att köpa billigt
    # 0% budkrig → +0, 50% budkrig → -2, 80%+ budkrig → -4
    if budkrig_rate >= 0.8:
        score -= 4
    elif budkrig_rate >= 0.5:
        score -= 2
    elif budkrig_rate >= 0.3:
        score -= 1

    return max(0, min(15, score))


# ============================================================
# KOMPONENT 3: COMPS-STÖD (0–15 poäng)
# ============================================================

def score_comps(listing, omrade, bostadstyp, df_train):
    """
    Använder comps_antal_90d från annonsen som primär signal (senaste 90 dagar).
    Faller tillbaka på historisk df_train om comps_antal_90d saknas.
    """
    # Primär: comps_antal_90d direkt i annonsen
    comps_antal = listing.get('comps_antal_90d', None) if listing is not None else None

    if (comps_antal is not None and
            not (isinstance(comps_antal, float) and np.isnan(comps_antal))):
        n = int(comps_antal)
        if n >= 30:
            return 15
        elif n >= 15:
            return 12
        elif n >= 8:
            return 9
        elif n >= 3:
            return 6
        else:
            return 3

    # Fallback: historisk df_train — senaste 365 dagar för relevans
    if df_train is None or 'omrade_clean' not in df_train.columns:
        return 5
    if omrade == 'övrigt':
        return 3

    recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
    df_recent = df_train
    if 'sald_datum' in df_train.columns:
        dates = pd.to_datetime(df_train['sald_datum'], errors='coerce')
        df_recent = df_train[dates >= recent_cutoff]

    n = len(df_recent[
        (df_recent['bostadstyp'] == bostadstyp) &
        (df_recent['omrade_clean'] == omrade)
    ])
    if n >= 50:
        return 15
    elif n >= 20:
        return 11
    elif n >= 10:
        return 8
    elif n >= 5:
        return 5
    else:
        return 2


# ============================================================
# KOMPONENT 4: MARKNADSPOSITION (0–10 poäng)
# Ersätter den gamla score_data_quality som alltid gav 10
# ============================================================

def score_market_position(listing):
    """
    Värderar hur länge bostaden legat ute + om priset sänkts.
    Lång tid ute = potentiellt förhandlingsutrymme = bra.
    Prissänkning = tydlig signal = bra.
    Men validerar också att kritiska fält finns.
    """
    if not is_listing_valid(listing):
        return 0

    dagar = listing.get('dagar_pa_marknaden', None)
    pris_sankts = listing.get('pris_sankts', False)
    pris_sank_kr = listing.get('pris_sank_kr', 0) or 0
    utgangspris = listing.get('utgangspris', 1)

    score = 5  # Baspoäng om annonsen är giltig

    # Dagar på marknaden (lång tid = mer förhandlingsutrymme)
    if dagar is not None and not (isinstance(dagar, float) and np.isnan(dagar)):
        d = int(dagar)
        if d >= 60:
            score += 5
        elif d >= 30:
            score += 3
        elif d >= 14:
            score += 1

    # Prissänkning (tydlig signal om säljaren vill sälja)
    if pris_sankts:
        sank_pct = abs(pris_sank_kr) / utgangspris * 100 if utgangspris > 0 else 0
        if sank_pct >= 5:
            score += 3
        elif sank_pct >= 2:
            score += 2
        else:
            score += 1

    return min(10, score)


# ============================================================
# KOMPONENT 5: MODELLKONFIDANS (0–10 poäng)
# ============================================================

def score_confidence(interval_pct):
    if interval_pct <= 8:
        return 10
    elif interval_pct <= 12:
        return 8
    elif interval_pct <= 18:
        return 6
    elif interval_pct <= 25:
        return 4
    else:
        return 2


# ============================================================
# HUVUDFUNKTION
# ============================================================

def compute_deal_score(listing, estimat, confidence_pct, area_stats,
                       df_train=None):
    utgangspris = listing.get('utgangspris', 0)
    omrade      = listing.get('omrade', 'övrigt')
    bostadstyp  = listing.get('bostadstyp', 'lagenheter')

    # Sätt estimat på listing-objektet så _get_listing_ci_pct kan använda det
    if isinstance(listing, dict):
        listing_with_est = {**listing, 'estimerat_varde': estimat}
    else:
        listing_with_est = listing.copy()
        listing_with_est['estimerat_varde'] = estimat

    if not is_estimate_sane(estimat, utgangspris, bostadstyp):
        return {
            'deal_score': 20, 'category_label': 'Rimligt pris',
            'category_icon': '⚪', 'category_color': '#888888',
            'underval_pct': 0, 'underval_kr': 0,
            'reasons': ['Estimat utanför rimligt intervall'],
            'sane_estimate': False,
        }

    underval_score, underval_pct = score_undervaluation(listing_with_est, estimat, area_stats)
    market_score    = score_market_position(listing)
    area_score      = score_area(omrade, area_stats)
    comps_score     = score_comps(listing, omrade, bostadstyp, df_train)

    # Faktisk CI per annons
    ci_pct = _get_listing_ci_pct(listing_with_est)
    confidence_score = score_confidence(ci_pct)

    total = underval_score + market_score + area_score + comps_score + confidence_score
    total = round(min(100, max(0, total)), 0)

    category = get_deal_category(total, bostadstyp)

    # Bygg förklaring (cappa bara visad text, inte score)
    reasons = []
    if underval_pct >= 40:
        reasons.append('Tydligt under marknadspris')
    elif underval_pct > 15:
        reasons.append(f'Undervärderad med ~{underval_pct:.0f}%')
    elif underval_pct > 5:
        reasons.append(f'Något undervärderad (~{underval_pct:.0f}%)')
    elif underval_pct > -5:
        reasons.append('Nära marknadspris')
    else:
        reasons.append(f'Något över marknadspris (~{min(abs(underval_pct), 40):.0f}%)')

    if area_score >= 12:
        reasons.append('Starkt område')
    elif area_score <= 4:
        reasons.append('Svagt/okänt område')

    budkrig_rate = area_stats.get(omrade, {}).get('budkrig_rate', 0)
    if budkrig_rate >= 0.6:
        reasons.append(f'Hög budkrigsfrekvens ({budkrig_rate:.0%})')

    if comps_score >= 11:
        reasons.append('Många jämförbara i området')
    elif comps_score <= 3:
        reasons.append('Få jämförbara')


    dagar = listing.get('dagar_pa_marknaden', None)
    if dagar is not None and not (isinstance(dagar, float) and np.isnan(dagar)) and int(dagar) >= 30:
        reasons.append(f'{int(dagar)} dagar på marknaden')

    if listing.get('pris_sankts'):
        reasons.append('Priset har sänkts')

    return {
        'deal_score': int(total),
        'category_label': category['label'],
        'category_icon': category['icon'],
        'category_color': category['color'],
        'underval_pct': underval_pct,
        'underval_kr': int(estimat - utgangspris),
        'reasons': reasons,
        'sane_estimate': True,
    }


# ============================================================
# BATCH
# ============================================================

def compute_deal_scores_batch(df_listings, df_train, confidence_data):
    area_stats = {}
    if df_train is not None and 'omrade_clean' in df_train.columns:
        df_train = df_train.copy()
        df_train['sald_datum'] = pd.to_datetime(df_train.get('sald_datum'), errors='coerce')
        latest = df_train['sald_datum'].max()
        if pd.notna(latest):
            rc = latest - pd.Timedelta(days=365)
            oc = rc - pd.Timedelta(days=365)
            recent = df_train[df_train['sald_datum'] >= rc]
            older  = df_train[(df_train['sald_datum'] >= oc) & (df_train['sald_datum'] < rc)]
            for omrade in df_train['omrade_clean'].unique():
                rm = recent[recent['omrade_clean'] == omrade]['slutpris'].median()
                om = older[older['omrade_clean'] == omrade]['slutpris'].median()
                trend = round((rm / om - 1) * 100, 1) if pd.notna(rm) and pd.notna(om) and om > 0 else 0
                antal = len(recent[recent['omrade_clean'] == omrade])

                # Budkrig-rate per område (om kolumnen finns)
                budkrig_rate = 0.0
                if 'budkrig' in df_train.columns:
                    omr_recent = recent[recent['omrade_clean'] == omrade]
                    if len(omr_recent) > 0:
                        budkrig_rate = float(omr_recent['budkrig'].mean())

                area_stats[omrade] = {
                    'trend': trend,
                    'antal': antal,
                    'budkrig_rate': budkrig_rate,
                }

    results = []
    for _, row in df_listings.iterrows():
        typ = row.get('bostadstyp', 'lagenheter')
        conf_pct = confidence_data.get(typ, {}).get('interval_pct', 20)
        results.append(compute_deal_score(
            listing=row, estimat=row.get('estimerat_varde', 0),
            confidence_pct=conf_pct, area_stats=area_stats, df_train=df_train,
        ))

    new_cols = {
        'deal_score':    [r['deal_score']      for r in results],
        'deal_kategori': [r['category_label']  for r in results],
        'deal_ikon':     [r['category_icon']   for r in results],
        'deal_color':    [r['category_color']  for r in results],
        'underval_pct':  [r['underval_pct']    for r in results],
        'underval_kr':   [r['underval_kr']     for r in results],
        'deal_reasons':  [' | '.join(r['reasons']) for r in results],
        'sane_estimate': [r['sane_estimate']   for r in results],
    }
    df_listings = pd.concat(
        [df_listings, pd.DataFrame(new_cols, index=df_listings.index)],
        axis=1,
    )

    # ================================================================
    # PERCENTILNORMALISERING — deal_score_pct (0–100)
    #
    # Industristandardmetod (Redfin, Investorlift): konvertera råpoäng
    # till percentilrang inom samma bostadstyp och aktiv batch.
    # Score 78 = "bättre än 78% av aktiva [lägenheter/villor/radhus] idag".
    #
    # Kategorier (Exceptionellt/Bra/etc.) baseras fortfarande på absoluta
    # deal_score-trösklar (kvalitetsmärkning, inte relativ).
    # deal_score_pct är det VISADE talet i dashboarden (använder hela 0–100).
    # ================================================================
    # Percentilrang per bostadstyp — vektoriserad utan loop (undviker PerformanceWarning)
    df_listings['deal_score_pct'] = (
        df_listings.groupby('bostadstyp')['deal_score']
        .transform(lambda x: x.rank(pct=True, method='average') * 100 if len(x) > 1 else 50)
        .round(0).astype(int)
    )

    return df_listings
