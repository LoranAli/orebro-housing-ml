"""
ValuEstate — Email-notiser för bevakningsfilter
================================================
Körs av daily_update.py efter scoring.

Kräver env-variabler (eller .env):
    EMAIL_SENDER    ex: dinemail@gmail.com
    EMAIL_PASSWORD  Gmail App Password (16 tecken)

Sparar filter i data/processed/email_alerts.json.
"""

import json
import os
import smtplib
import sys
import uuid
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_SCRIPTS)

ALERTS_PATH = os.path.join(_PROJECT, 'data', 'processed', 'email_alerts.json')

# ─────────────────────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────────────────────

def load_alerts() -> list[dict]:
    if os.path.exists(ALERTS_PATH):
        try:
            with open(ALERTS_PATH, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_alerts(alerts: list[dict]) -> None:
    os.makedirs(os.path.dirname(ALERTS_PATH), exist_ok=True)
    with open(ALERTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(alerts, f, ensure_ascii=False, indent=2)


def add_alert(email: str, label: str, bostadstyp: str,
              omraden: list[str], max_pris: int,
              min_rum: float, min_deal_score: int) -> dict:
    alerts = load_alerts()
    entry = {
        'id': str(uuid.uuid4())[:8],
        'email': email.strip().lower(),
        'label': label,
        'bostadstyp': bostadstyp,
        'omraden': omraden,
        'max_pris': max_pris,
        'min_rum': min_rum,
        'min_deal_score': min_deal_score,
        'created': datetime.now().strftime('%Y-%m-%d'),
        'last_notified_urls': [],
    }
    alerts.append(entry)
    save_alerts(alerts)
    return entry


def delete_alert(alert_id: str) -> None:
    alerts = [a for a in load_alerts() if a['id'] != alert_id]
    save_alerts(alerts)


# ─────────────────────────────────────────────────────────────
# MATCHNING
# ─────────────────────────────────────────────────────────────

def match_listings(df: pd.DataFrame, alert: dict) -> pd.DataFrame:
    """Returnerar rader i df som matchar alertens filter."""
    mask = pd.Series(True, index=df.index)

    if alert.get('bostadstyp') and alert['bostadstyp'] != 'alla':
        mask &= df['bostadstyp'] == alert['bostadstyp']

    if alert.get('omraden'):
        omr = [o.lower() for o in alert['omraden']]
        mask &= df['omrade_clean'].str.lower().isin(omr)

    if alert.get('max_pris') and 'utgangspris' in df.columns:
        mask &= df['utgangspris'] <= alert['max_pris']

    if alert.get('min_rum') and 'antal_rum' in df.columns:
        mask &= df['antal_rum'] >= alert['min_rum']

    if alert.get('min_deal_score') and 'deal_score' in df.columns:
        mask &= df['deal_score'] >= alert['min_deal_score']

    return df[mask]


# ─────────────────────────────────────────────────────────────
# EMAIL
# ─────────────────────────────────────────────────────────────

def _build_html(matches: pd.DataFrame, alert: dict) -> str:
    rows = ""
    for _, r in matches.iterrows():
        score    = int(r.get('deal_score', 0))
        kat      = r.get('deal_kategori', '')
        pris     = int(r.get('utgangspris', 0))
        estimat  = int(r.get('estimerat_varde', 0))
        avv      = r.get('skillnad_pct', 0)
        omrade   = r.get('omrade_clean', '')
        typ      = r.get('bostadstyp', '')
        rum      = r.get('antal_rum', '')
        boarea   = r.get('boarea_kvm', '')
        url      = r.get('url', '#')

        avv_color = '#10b981' if avv >= 0 else '#ef4444'
        rows += f"""
        <tr>
          <td style="padding:12px;border-bottom:1px solid #374151;">
            <a href="{url}" style="color:#10b981;font-weight:700;text-decoration:none;">
              {omrade} · {typ} · {rum} rum · {boarea} m²
            </a>
          </td>
          <td style="padding:12px;border-bottom:1px solid #374151;text-align:right;">
            {pris:,} kr
          </td>
          <td style="padding:12px;border-bottom:1px solid #374151;text-align:right;">
            {estimat:,} kr
          </td>
          <td style="padding:12px;border-bottom:1px solid #374151;text-align:right;color:{avv_color};">
            {avv:+.1f}%
          </td>
          <td style="padding:12px;border-bottom:1px solid #374151;text-align:center;">
            <strong>{score}/100</strong><br>
            <span style="font-size:11px;color:#9ca3af;">{kat}</span>
          </td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<body style="background:#0d1117;color:#f9fafb;font-family:sans-serif;padding:24px;">
  <h2 style="color:#10b981;">ValuEstate — {len(matches)} ny/nya match(ar)</h2>
  <p style="color:#9ca3af;">Bevakning: <strong style="color:#f9fafb;">{alert['label']}</strong></p>
  <table style="width:100%;border-collapse:collapse;background:#1f2937;border-radius:8px;">
    <thead>
      <tr style="color:#9ca3af;font-size:12px;">
        <th style="padding:10px;text-align:left;">Bostad</th>
        <th style="padding:10px;text-align:right;">Utgångspris</th>
        <th style="padding:10px;text-align:right;">ML-estimat</th>
        <th style="padding:10px;text-align:right;">Avvikelse</th>
        <th style="padding:10px;text-align:center;">Deal Score</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <p style="color:#6b7280;font-size:11px;margin-top:16px;">
    Skickat av ValuEstate · {datetime.now().strftime('%Y-%m-%d %H:%M')}
  </p>
</body>
</html>"""


def send_email(to_email: str, subject: str, html: str,
               sender: str, password: str) -> None:
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = f'ValuEstate <{sender}>'
    msg['To']      = to_email
    msg.attach(MIMEText(html, 'html', 'utf-8'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as srv:
        srv.login(sender, password)
        srv.sendmail(sender, to_email, msg.as_string())


# ─────────────────────────────────────────────────────────────
# HUVUDFUNKTION — anropas av daily_update.py
# ─────────────────────────────────────────────────────────────

def run_alerts(df_active: pd.DataFrame) -> None:
    """Kör alla sparade bevakningar mot aktiva annonser och skicka email vid match."""
    # Ladda dotenv om det finns (lokal körning)
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_PROJECT, '.env'))
    except ImportError:
        pass

    sender   = os.environ.get('EMAIL_SENDER', '')
    password = os.environ.get('EMAIL_PASSWORD', '')
    # Streamlit Cloud secrets (om körs via streamlit)
    if not sender:
        try:
            import streamlit as _st
            sender   = _st.secrets.get('EMAIL_SENDER', '')
            password = _st.secrets.get('EMAIL_PASSWORD', '')
        except Exception:
            pass
    if not sender or not password:
        print('[email_alerts] EMAIL_SENDER/EMAIL_PASSWORD ej satta — hoppar över notiser.')
        return

    alerts = load_alerts()
    if not alerts:
        return

    updated = False
    for alert in alerts:
        matches = match_listings(df_active, alert)
        if matches.empty:
            continue

        # Filtrera bort redan notifierade
        prev = set(alert.get('last_notified_urls', []))
        new_matches = matches[~matches['url'].isin(prev)]
        if new_matches.empty:
            continue

        try:
            html    = _build_html(new_matches, alert)
            subject = f"ValuEstate: {len(new_matches)} ny annons matchar '{alert['label']}'"
            send_email(alert['email'], subject, html, sender, password)
            print(f"[email_alerts] Skickade {len(new_matches)} match(ar) till {alert['email']}")
            alert['last_notified_urls'] = list(prev | set(new_matches['url'].tolist()))
            updated = True
        except Exception as e:
            print(f"[email_alerts] Email misslyckades för {alert['email']}: {e}")

    if updated:
        save_alerts(alerts)
