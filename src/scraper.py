"""
Hemnet Slutpriser Scraper — Örebro kommun
==========================================
Scraper för att hämta slutpriser från Hemnet för Örebro kommun.
Använder requests + BeautifulSoup med respektfull rate limiting.

Användning:
    python src/scraper.py

OBS: Denna scraper är för utbildnings-/portfoliosyfte.
     Respektera Hemnets villkor och använd rate limiting.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import json
import os
from datetime import datetime
from tqdm import tqdm

# ============================================================
# KONFIGURATION
# ============================================================

# Hemnet slutpriser-URL:er för Örebro kommun
# Du kan hitta location_id genom att söka på Hemnet och kolla URL:en
BASE_URLS = {
    "lagenheter": "https://www.hemnet.se/salda/bostader?location_ids%5B%5D=17849&item_types%5B%5D=bostadsratt",
    "villor": "https://www.hemnet.se/salda/bostader?location_ids%5B%5D=17849&item_types%5B%5D=villa",
    "radhus": "https://www.hemnet.se/salda/bostader?location_ids%5B%5D=17849&item_types%5B%5D=radhus",
}

# 17849 = Örebro kommun (kan behöva verifieras)
# Tips: Gå till hemnet.se/salda/orebro-kommun och inspektera URL:en

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "sv-SE,sv;q=0.9,en-US;q=0.8,en;q=0.7",
}

# Rate limiting — var snäll mot Hemnet
MIN_DELAY = 2.0  # sekunder
MAX_DELAY = 4.0  # sekunder
MAX_PAGES = 50   # max antal sidor per bostadstyp

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


# ============================================================
# SCRAPER-FUNKTIONER
# ============================================================

def get_page(url: str, page: int = 1) -> requests.Response | None:
    """Hämta en sida med rate limiting."""
    params = {"page": page}
    
    try:
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        print(f"  ⚠ Fel vid hämtning av sida {page}: {e}")
        return None


def parse_listing_card(card) -> dict | None:
    """
    Parsa ett enskilt slutpris-kort från Hemnet.
    
    Hemnet's HTML-struktur ändras ibland, så du kan behöva
    uppdatera selektorer. Inspektera sidan i din webbläsare
    för att verifiera aktuella CSS-klasser.
    """
    try:
        listing = {}
        
        # === Adress ===
        address_el = card.select_one("h2 a, .sold-property-listing__heading a, "
                                      "[class*='address'] a")
        if address_el:
            listing["adress"] = address_el.get_text(strip=True)
            listing["url"] = "https://www.hemnet.se" + address_el.get("href", "")
        
        # === Område ===
        location_el = card.select_one(".sold-property-listing__location, "
                                       "[class*='location']")
        if location_el:
            location_text = location_el.get_text(strip=True)
            # Ofta format: "Område, Örebro kommun"
            listing["omrade"] = location_text.split(",")[0].strip() if location_text else ""
            listing["kommun"] = "Örebro kommun"
        
        # === Slutpris ===
        price_el = card.select_one(".sold-property-listing__price, "
                                    "[class*='sold-price'], [class*='price']")
        if price_el:
            price_text = price_el.get_text(strip=True)
            listing["slutpris_raw"] = price_text
            listing["slutpris"] = parse_price(price_text)
        
        # === Boarea ===
        size_el = card.select_one(".sold-property-listing__size, "
                                   "[class*='living-area'], [class*='size']")
        if size_el:
            size_text = size_el.get_text(strip=True)
            listing["boarea_raw"] = size_text
            listing["boarea_kvm"] = parse_area(size_text)
        
        # === Antal rum ===
        rooms_el = card.select_one("[class*='rooms']")
        if rooms_el:
            rooms_text = rooms_el.get_text(strip=True)
            listing["antal_rum"] = parse_rooms(rooms_text)
        
        # === Avgift (bostadsrätt) ===
        fee_el = card.select_one("[class*='fee']")
        if fee_el:
            fee_text = fee_el.get_text(strip=True)
            listing["avgift_kr"] = parse_price(fee_text)
        
        # === Sålddatum ===
        date_el = card.select_one(".sold-property-listing__sold-date, "
                                   "[class*='sold-date']")
        if date_el:
            date_text = date_el.get_text(strip=True)
            listing["sald_datum_raw"] = date_text
            listing["sald_datum"] = parse_date(date_text)
        
        # === Prisförändring ===
        change_el = card.select_one("[class*='price-change'], "
                                     "[class*='price-development']")
        if change_el:
            listing["prisforandring_raw"] = change_el.get_text(strip=True)
        
        # === KVM-pris ===
        if listing.get("slutpris") and listing.get("boarea_kvm"):
            if listing["boarea_kvm"] > 0:
                listing["pris_per_kvm"] = round(
                    listing["slutpris"] / listing["boarea_kvm"]
                )
        
        # Kolla att vi har minst pris och adress
        if listing.get("slutpris") and listing.get("adress"):
            return listing
        
        return None
        
    except Exception as e:
        print(f"  ⚠ Fel vid parsing av kort: {e}")
        return None


def parse_listing_detail(url: str) -> dict:
    """
    Hämta extra detaljer från en enskild listningssida.
    Använd detta för att berika datan med fler features.
    
    OBS: Mycket långsammare — gör bara för ett urval.
    """
    details = {}
    response = get_page(url)
    if not response:
        return details
    
    soup = BeautifulSoup(response.text, "lxml")
    
    # Attribut-tabell (byggår, tomtarea, etc.)
    attribute_rows = soup.select(".sold-property__attribute, "
                                  "[class*='attribute'] dl")
    for row in attribute_rows:
        key_el = row.select_one("dt")
        val_el = row.select_one("dd")
        if key_el and val_el:
            key = key_el.get_text(strip=True).lower()
            val = val_el.get_text(strip=True)
            
            if "byggår" in key:
                details["byggar"] = val
            elif "tomtarea" in key:
                details["tomtarea_kvm"] = parse_area(val)
            elif "våning" in key:
                details["vaning"] = val
            elif "driftkostnad" in key:
                details["driftkostnad_kr"] = parse_price(val)
            elif "förening" in key:
                details["forening"] = val
    
    return details


# ============================================================
# PARSERS (hjälpfunktioner)
# ============================================================

def parse_price(text: str) -> int | None:
    """Konvertera pristext till heltal. '2 500 000 kr' → 2500000"""
    if not text:
        return None
    # Ta bort allt utom siffror
    digits = "".join(c for c in text if c.isdigit())
    return int(digits) if digits else None


def parse_area(text: str) -> float | None:
    """Konvertera areatext till float. '83,5 m²' → 83.5"""
    if not text:
        return None
    # Hantera komma som decimaltecken
    text = text.replace(",", ".").replace("m²", "").replace("m2", "").strip()
    # Ta bort extra area (t.ex. '106+131 m²' → ta första)
    if "+" in text:
        text = text.split("+")[0]
    try:
        return float("".join(c for c in text if c.isdigit() or c == "."))
    except ValueError:
        return None


def parse_rooms(text: str) -> float | None:
    """Konvertera rumtext till float. '3 rum' → 3.0, '1,5 rum' → 1.5"""
    if not text:
        return None
    text = text.replace(",", ".").replace("rum", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_date(text: str) -> str | None:
    """Konvertera datumtext. 'Såld 15 nov. 2025' → '2025-11-15'"""
    if not text:
        return None
    
    months = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "maj": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "okt": "10", "nov": "11", "dec": "12",
    }
    
    text = text.lower().replace("såld", "").replace(".", "").strip()
    parts = text.split()
    
    if len(parts) >= 3:
        day = parts[0].zfill(2)
        month = months.get(parts[1][:3], "00")
        year = parts[2]
        return f"{year}-{month}-{day}"
    
    return None


# ============================================================
# HUVUD-SCRAPING
# ============================================================

def scrape_hemnet(
    url: str,
    bostadstyp: str,
    max_pages: int = MAX_PAGES,
) -> list[dict]:
    """
    Scrapa alla slutpriser för en given bostadstyp.
    """
    all_listings = []
    
    print(f"\n{'='*60}")
    print(f"  Scrapar: {bostadstyp}")
    print(f"  URL: {url}")
    print(f"{'='*60}")
    
    for page in tqdm(range(1, max_pages + 1), desc=f"  {bostadstyp}"):
        response = get_page(url, page=page)
        
        if not response:
            print(f"  ✗ Kunde inte hämta sida {page}, hoppar över")
            continue
        
        soup = BeautifulSoup(response.text, "lxml")
        
        # Hitta alla listningskort
        # OBS: Hemnet ändrar ibland sina CSS-klasser
        cards = soup.select(
            ".sold-property-listing, "
            "[class*='sold-results'] li, "
            "[data-testid='search-result']"
        )
        
        if not cards:
            print(f"\n  ℹ Inga fler resultat på sida {page}. Klart!")
            break
        
        page_count = 0
        for card in cards:
            listing = parse_listing_card(card)
            if listing:
                listing["bostadstyp"] = bostadstyp
                listing["scrapad_datum"] = datetime.now().isoformat()
                all_listings.append(listing)
                page_count += 1
        
        if page_count == 0:
            print(f"\n  ℹ Inga parseable resultat på sida {page}. Klart!")
            break
    
    print(f"  ✓ Totalt {len(all_listings)} listningar scrapade för {bostadstyp}")
    return all_listings


def main():
    """Kör hela scraping-pipelinen."""
    
    print("=" * 60)
    print("  HEMNET SLUTPRISER SCRAPER — ÖREBRO KOMMUN")
    print(f"  Startar: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # Skapa output-mapp
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_data = []
    
    for bostadstyp, url in BASE_URLS.items():
        listings = scrape_hemnet(url, bostadstyp)
        all_data.extend(listings)
    
    if not all_data:
        print("\n⚠ Ingen data hämtad! Möjliga orsaker:")
        print("  1. Hemnet har ändrat sin HTML-struktur")
        print("  2. Du är blockerad (för snabb scraping)")
        print("  3. location_id är fel — verifiera i webbläsaren")
        print("\nTips: Öppna hemnet.se/salda/orebro-kommun i webbläsaren")
        print("      och inspektera HTML:en för att uppdatera selektorer.")
        return
    
    # Spara som CSV och JSON
    df = pd.DataFrame(all_data)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    csv_path = os.path.join(OUTPUT_DIR, f"hemnet_orebro_{timestamp}.csv")
    json_path = os.path.join(OUTPUT_DIR, f"hemnet_orebro_{timestamp}.json")
    
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    # Sammanfattning
    print(f"\n{'='*60}")
    print(f"  SAMMANFATTNING")
    print(f"{'='*60}")
    print(f"  Totalt antal listningar: {len(df)}")
    print(f"  Bostadstyper: {df['bostadstyp'].value_counts().to_dict()}")
    
    if "slutpris" in df.columns and df["slutpris"].notna().any():
        print(f"  Medianpris: {df['slutpris'].median():,.0f} kr")
        print(f"  Prisspann: {df['slutpris'].min():,.0f} - {df['slutpris'].max():,.0f} kr")
    
    if "boarea_kvm" in df.columns and df["boarea_kvm"].notna().any():
        print(f"  Median boarea: {df['boarea_kvm'].median():.1f} m²")
    
    print(f"\n  Sparat till:")
    print(f"    CSV:  {csv_path}")
    print(f"    JSON: {json_path}")
    print(f"\n  Nästa steg: Kör src/scb_fetcher.py för att hämta SCB-data")


if __name__ == "__main__":
    main()
