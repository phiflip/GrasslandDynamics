"""MeteoSchweiz-Tagesdaten für GrasslandGrowthLab herunterladen.

2025: vollständiges Jahr -> data/input/weather/historical_data/<site>/
2026: bis zum letzten verfügbaren Tag -> data/input/weather/recent_data/<site>/

Aufruf aus dem Repository-Hauptordner:
    python scripts/SwissMeteoGrid_parser_GrasslandGrowthLab.py

Benötigt: pandas, numpy.
Datenquelle: https://opendatadocs.meteoswiss.ch/a-data-groundbased/a1-automatic-weather-stations
"""

from datetime import date
from pathlib import Path
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # Parser liegt in scripts/
WEATHER_DIR = ROOT / "data" / "input" / "weather"
YEARS = (2025, 2026)

# MeteoSchweiz-Stationskürzel und geographische Breite [° N]
STATIONS = {
    "ber": 46.9907,  # Bern/Zollikofen
    "dav": 46.8167,  # Davos
    "cdf": 47.0833,  # La Chaux-de-Fonds
    "lug": 46.0042,  # Lugano
}

BASE_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
REQUIRED = ("tre200d0", "tre200dn", "tre200dx", "gre000d0", "rre150d0")


def download(site: str, period: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{site}/ogd-smn_{site}_d_{period}.csv"
    try:
        return pd.read_csv(url, sep=";", encoding="cp1252", low_memory=False)
    except Exception as exc:
        print(f"[WARN] {site.upper()} {period}: Download fehlgeschlagen: {exc}")
        return pd.DataFrame()


def ra_mm(latitude_deg: float, doy: int) -> float:
    """Extraterrestrische Strahlung als Verdunstungsäquivalent [mm/d], FAO-56."""
    phi = math.radians(latitude_deg)
    dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
    delta = 0.409 * math.sin(2 * math.pi * doy / 365 - 1.39)
    ws = math.acos(-math.tan(phi) * math.tan(delta))
    ra_mj = (24 * 60 / math.pi) * 0.0820 * (
        dr * (ws * math.sin(phi) * math.sin(delta)
              + math.cos(phi) * math.cos(delta) * math.sin(ws))
    )
    return ra_mj / 2.45


def prepare_year(raw: pd.DataFrame, year: int, latitude_deg: float) -> pd.DataFrame:
    """ModVege-Format: Year, DOY, Ta, PAR, Precip, ETc.

    Fehlende Tage werden innerhalb des beobachteten Zeitraums ergänzt.
    Temperatur, PAR und ETc werden interpoliert; fehlender Niederschlag = 0 mm.
    Keine Extrapolation zukünftiger Tage im laufenden Jahr.
    """
    if raw.empty:
        return pd.DataFrame()

    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        print(f"[SKIP] {year}: Spalten fehlen: {sorted(missing)}")
        return pd.DataFrame()

    df = raw.copy()
    dates = pd.to_datetime(df["reference_timestamp"], format="%d.%m.%Y %H:%M", errors="coerce")
    if dates.isna().any():
        dates = dates.fillna(pd.to_datetime(df["reference_timestamp"], dayfirst=True, errors="coerce"))
    df["date"] = dates.dt.normalize()
    df = df.loc[df["date"].dt.year == year].copy()
    if df.empty:
        return pd.DataFrame()

    for col in REQUIRED:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["DOY"] = df["date"].dt.dayofyear
    df = df.drop_duplicates(subset="DOY", keep="last").sort_values("DOY")

    # Keine scheinbar vollständigen 2026-Daten erzeugen.
    last_day = int(df["DOY"].max())
    n_days = 365 if year < date.today().year else last_day
    if year < date.today().year and last_day < 340:
        print(f"[SKIP] {year}: Reihe endet bereits bei DOY {last_day}.")
        return pd.DataFrame()

    cov = df[list(REQUIRED)].notna().sum().min() / n_days
    if cov < 0.80:
        print(f"[SKIP] {year}: Abdeckung nur {cov:.0%}, mindestens 80 % erforderlich.")
        return pd.DataFrame()

    out = pd.DataFrame({"DOY": range(1, n_days + 1)})
    df = out.merge(df[["DOY", *REQUIRED]], on="DOY", how="left")

    ra = np.array([ra_mm(latitude_deg, doy) for doy in df["DOY"]])
    t_mean = df["tre200d0"]
    t_min = df["tre200dn"]
    t_max = df["tre200dx"]
    eto = 0.0023 * (t_mean + 17.8) * np.sqrt((t_max - t_min).clip(lower=0)) * ra

    result = pd.DataFrame({
        "Year": year,
        "DOY": df["DOY"],
        "Ta": t_mean,
        "PAR": df["gre000d0"] * 86400 * 1e-6 * 0.47,
        "Precip": df["rre150d0"],
        "ETc": 1.15 * eto,
    })

    for col in ("Ta", "PAR", "ETc"):
        result[col] = result[col].interpolate(limit_direction="both")
    result["Precip"] = result["Precip"].fillna(0.0)
    if result[["Ta", "PAR", "Precip", "ETc"]].isna().any().any():
        raise ValueError(f"{year}: unvollständige Wetterreihe nach Interpolation")
    return result


def main() -> None:
    for site, latitude in STATIONS.items():
        print(f"\n{site.upper()}: historische und aktuelle Tagesdaten laden ...")
        hist = download(site, "historical")
        recent = download(site, "recent")
        raw = pd.concat([hist, recent], ignore_index=True)
        if raw.empty:
            continue

        for year in YEARS:
            result = prepare_year(raw, year, latitude)
            if result.empty:
                print(f"[SKIP] {site.upper()} {year}: keine verwertbaren Daten")
                continue

            # Unvollständiges aktuelles Jahr nicht in die App-Jahresauswahl aufnehmen.
            period = "recent_data" if year == date.today().year else "historical_data"
            dest = WEATHER_DIR / period / site
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / f"{site}INT_{year}_DailyWeatherMeteoStation.dat"
            result.to_csv(target, index=False, sep="\t", float_format="%.5f")
            status = "laufendes Jahr, unvollständig" if period == "recent_data" else "vollständig"
            print(f"[OK] {site.upper()} {year}: {len(result)} Tage ({status}) -> {target}")


if __name__ == "__main__":
    main()
