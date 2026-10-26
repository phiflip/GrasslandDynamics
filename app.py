"""Interaktive Wachstumsanalyse, inklusive laufendem Wetterjahr.

Start: streamlit run app.py
Historische Daten: data/input/weather/historical_data/<station>/.
Laufendes Jahr: data/input/weather/recent_data/<station>/.
"""

from datetime import date, timedelta
from pathlib import Path
from textwrap import dedent
import calendar
import math
import random

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from scripts.modvege import load_parameters, load_weather, simulate

ROOT = Path(__file__).resolve().parent
WEATHER_ROOT = ROOT / "data/input/weather/historical_data"
RECENT_ROOT = ROOT / "data/input/weather/recent_data"
STATIONS = {
    "Bern": "ber",
    "Davos": "dav",
    "La Chaux-de-Fonds": "cdf",
    "Lugano (Tessin)": "lug",
}
NI_FIXED = 1.2
WHC_FIXED = 80.0
DEFAULT_CUTS = [(5, 25), (7, 20), (9, 10), (10, 5), (11, 1)]
GRASS_TYPES = {
    "Intensive Wiese (50 % A / 50 % B)": (0.5, 0.5, 0.0, 0.0),
    "Grasgruppe A": (1.0, 0.0, 0.0, 0.0),
    "Grasgruppe B": (0.0, 1.0, 0.0, 0.0),
    "Grasgruppe C": (0.0, 0.0, 1.0, 0.0),
    "Grasgruppe D": (0.0, 0.0, 0.0, 1.0),
}
GREEN, BLUE, AMBER = "#19865c", "#5886b1", "#d89139"

st.set_page_config(page_title="GrasslandDynamics", page_icon="🌱", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1370px; padding-top: 1.45rem;}
.kicker {color: #22885d; text-transform: uppercase; letter-spacing: .1em; font-size: .78rem; font-weight: 750;}
.caption-muted {color: #61736c; font-size: .92rem;}
.growth-metrics {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .55rem; width: 100%;}
.growth-metric {background: rgba(131, 151, 139, .08); border: 1px solid rgba(131, 151, 139, .19); border-radius: .65rem; padding: .65rem .7rem; min-width: 0;}
.growth-metric-label {font-size: .78rem; opacity: .73; margin-bottom: .3rem; line-height: 1.25;}
.growth-metric-value {font-size: 1.5rem; font-weight: 650; line-height: 1.15; font-variant-numeric: tabular-nums;}
.growth-metric-unit {font-size: .75rem; opacity: .73; margin-top: .2rem; line-height: 1.3; overflow-wrap: anywhere;}
</style>
""", unsafe_allow_html=True)


def weather_path(station_code: str, year: int) -> Path:
    """Bei aktuellen Daten recent_data bevorzugen, sonst historical_data."""
    filename = f"{station_code}INT_{year}_DailyWeatherMeteoStation.dat"
    for folder in (RECENT_ROOT, WEATHER_ROOT):
        path = folder / station_code / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"Meteodaten fehlen: {filename}")


def years_available(station_code: str) -> list[int]:
    years = set()
    for folder in (RECENT_ROOT, WEATHER_ROOT):
        for file in (folder / station_code).glob(f"{station_code}INT_*_DailyWeatherMeteoStation.dat"):
            try:
                years.add(int(file.name.split("_")[1]))
            except (ValueError, IndexError):
                continue
    return sorted(years, reverse=True)


def weather_version(station_code: str, year: int) -> tuple[int, int]:
    """Cache aktualisieren, wenn der Parser neue Wettertage schreibt."""
    stat = weather_path(station_code, year).stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_data(show_spinner=False, max_entries=40)
def _read_weather(path: str, version: tuple[int, int]) -> pd.DataFrame:
    return load_weather(Path(path))


def get_weather(station_code: str, year: int) -> pd.DataFrame:
    path = weather_path(station_code, year)
    return _read_weather(str(path), weather_version(station_code, year))


@st.cache_data(show_spinner=False)
def get_model_inputs():
    return load_parameters(ROOT)


@st.cache_data(show_spinner=False, max_entries=120)
def run_model(station_code: str, year: int, cut_days: tuple[int, ...],
              cut_height: float, grass_weights: tuple[float, ...],
              temperature: float = 0.0,
              weather_stamp: tuple[int, int] = (0, 0)) -> pd.DataFrame:
    parameters, initial = get_model_inputs()
    weights = dict(zip(("FG_A", "FG_B", "FG_C", "FG_D"), grass_weights))
    output = simulate(
        weather=get_weather(station_code, year), year=year, cut_days=list(cut_days),
        model_parameters=parameters, initial_conditions=initial,
        cut_height_cm=cut_height, functional_group_weights=weights,
        ni=NI_FIXED, whc=WHC_FIXED, temp_offset=temperature,
    )
    return output


def meadow_html(lai: float, f_abs: float, snow: bool, cut: bool, seed: int = 22) -> str:
    """Pädagogische SVG-Illustration. Visuelle Höhe ist keine Modellgrösse."""
    rng = random.Random(seed)
    lai = max(0.0, float(lai))
    f_abs = min(1.0, max(0.0, float(f_abs)))
    count = min(175, max(7, int(9 + 29 * lai)))
    # Nur grafische Blattlänge, nicht die reale Aufwuchshöhe.
    canopy = 15 + 143 * (1 - math.exp(-lai / 2.3))
    y = 263
    leaves = []
    for i in range(count):
        x = (i + .5) * 800 / count + rng.uniform(-7, 7)
        h = canopy * rng.uniform(.5, 1.08)
        xt = x + rng.uniform(-17, 17)
        xm = (x + xt) / 2 + rng.uniform(-13, 13)
        color = rng.choice(("#1f7447", "#318d52", "#469d58", "#6db66c", "#91c57a"))
        leaves.append(f'<path d="M{x:.1f},{y} Q{xm:.1f},{y-h*.63:.1f} {xt:.1f},{y-h:.1f}" '
                      f'stroke="{color}" stroke-width="{rng.uniform(2,4.2):.1f}" fill="none" '
                      f'stroke-linecap="round"/>')
    beams = "".join(
        f'<path d="M{x},45 L{x-42},248" stroke="#f4c644" stroke-width="9" '
        f'opacity="{.13+.62*(1-f_abs):.2f}"/>' for x in (170, 245, 320, 395)
    )
    snow_svg = ('<path d="M0 250 Q145 242 300 251 T600 251 T800 250 V276 H0Z" '
                'fill="#f6fbff" opacity="0.94"/>' if snow else "")
    cut_svg = ('<rect x="580" y="24" rx="9" width="188" height="36" fill="#fff4df"/>'
               '<text x="596" y="48" font-size="19" fill="#9a5a18">✂ Schnitt heute</text>' if cut else "")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<style>html, body {{margin:0;background:transparent}} svg {{width:100%;height:auto;display:block}}
svg .grass {{animation: appear .45s ease-out}} @keyframes appear {{from {{opacity:.15}} to {{opacity:1}}}}
</style></head><body><svg viewBox="0 0 800 320" xmlns="http://www.w3.org/2000/svg" role="img"
aria-label="Schematische Grasnarbe: Dichte und grafische Höhe aus LAI abgeleitet">
<defs><linearGradient id="sky" x1="0" x2="0" y1="0" y2="1">
<stop stop-color="#addced"/><stop offset="1" stop-color="#eef8e4"/></linearGradient></defs>
<rect width="800" height="320" rx="14" fill="url(#sky)"/>
<circle cx="90" cy="64" r="32" fill="#f7d15e"/>
<g>{beams}</g>
<path d="M0,231 Q130,186 280,226 T590,209 T800,234 V279 H0Z" fill="#b4c5a3" opacity=".35"/>
<rect y="263" width="800" height="57" fill="#8a684c"/>
<path d="M0,265 Q170,259 325,266 T800,265" fill="none" stroke="#5d7e49" stroke-width="5"/>
<g class="grass">{''.join(leaves)}</g>{snow_svg}{cut_svg}
<rect x="555" y="{'69' if cut else '24'}" rx="9" width="214" height="36" fill="#fff" opacity=".94"/>
<text x="570" y="{'94' if cut else '49'}" fill="#1e5138" font-size="19" font-weight="bold">LAI = {lai:.2f}</text>
<rect x="17" y="19" rx="9" width="295" height="32" fill="#fff" opacity=".92"/>
<text x="30" y="41" fill="#41554a" font-size="16">Absorbierte Strahlung: {f_abs*100:.0f} %</text>
</svg></body></html>"""



def water_bucket_html(wr: float, whc: float, precip: float,
                      influx: float, aet: float, fw: float) -> str:
    """Schematischer Bodenwasserspeicher; keine reale Bodentiefe."""
    fraction = min(1.0, max(0.0, float(wr) / whc))
    fw = min(1.0, max(0.0, float(fw)))
    water_h = 134 * fraction
    water_y = 194 - water_h
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<style>html, body {{margin:0;background:transparent; font-family:Arial,sans-serif}}
svg {{display:block;width:100%;height:auto}}</style></head><body>
<svg viewBox="0 0 334 282" role="img" xmlns="http://www.w3.org/2000/svg"
aria-label="Bodenwasserspeicher {wr:.0f} von {whc:.0f} Millimeter, Wasserfaktor {fw:.2f}">
<rect x="1" y="1" width="332" height="280" rx="15" fill="#f2f8fb" stroke="#d5e7ed"/>
<text x="17" y="28" font-size="16" font-weight="bold" fill="#25516b">Bodenwasserspeicher</text>
<defs><clipPath id="tank"><rect x="22" y="60" width="89" height="134" rx="6"/></clipPath></defs>
<rect x="18" y="56" width="97" height="143" rx="10" fill="white" stroke="#6d9cb5" stroke-width="3"/>
<rect x="22" y="{water_y:.1f}" width="89" height="{water_h:.1f}"
fill="#61bee4" clip-path="url(#tank)"/>
<line x1="18" y1="91" x2="29" y2="91" stroke="#699ab4"/>
<line x1="18" y1="126" x2="29" y2="126" stroke="#699ab4"/>
<line x1="18" y1="161" x2="29" y2="161" stroke="#699ab4"/>
<text x="66" y="125" text-anchor="middle" font-size="23" font-weight="bold" fill="#173f54">{fraction*100:.0f}%</text>
<text x="66" y="215" text-anchor="middle" font-size="12" fill="#376078">WHC {whc:.0f} mm</text>
<text x="131" y="78" font-size="27" font-weight="bold" fill="#174866">{wr:.1f} mm</text>
<text x="132" y="98" font-size="12" fill="#527487">Bodenwasser (WR)</text>
<path d="M132 118 l8 10 l8 -10" fill="none" stroke="#2890bd" stroke-width="3"/>
<text x="158" y="129" font-size="13" fill="#24556e">Zufluss: {influx:.1f} mm/d</text>
<path d="M132 147 l8 -10 l8 10" fill="none" stroke="#c47a32" stroke-width="3"/>
<text x="158" y="149" font-size="13" fill="#795126">AET: {aet:.1f} mm/d</text>
<text x="132" y="172" font-size="12" fill="#527487">Niederschlag: {precip:.1f} mm/d</text>
<text x="17" y="234" font-size="13" font-weight="bold" fill="#245a43">Wachstumsfaktor f(W): {fw:.2f}</text>
<rect x="17" y="250" width="296" height="12" rx="5" fill="#d5e5de"/>
<rect x="17" y="250" width="{296*fw:.1f}" height="12" rx="5" fill="#31966b"/>
</svg></body></html>"""


def plot_water_balance(df: pd.DataFrame, day: int) -> go.Figure:
    """Niederschlag und tatsächliche Evapotranspiration, plus WR als Zustand."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df["date"], y=df["Precip"], name="Niederschlag",
        marker_color="#63b4de", opacity=0.65,
        hovertemplate="%{x|%d.%m.}: %{y:.1f} mm<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["AET"], name="Tatsächliche ET",
        mode="lines", line={"color": AMBER, "width": 2},
        hovertemplate="%{x|%d.%m.}: %{y:.1f} mm<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["WR"], name="Bodenwasser (WR)",
        mode="lines", line={"color": GREEN, "width": 2.6},
        hovertemplate="%{x|%d.%m.}: %{y:.1f} mm<extra></extra>",
    ), secondary_y=True)
    x = df.iloc[day - 1]["date"].timestamp() * 1000
    fig.add_vline(x=x, line_color="#374151", opacity=.72)
    fig.update_layout(
        height=340, margin={"l": 10, "r": 15, "t": 45, "b": 25},
        hovermode="x unified", barmode="overlay",
        legend={"orientation": "h", "y": 1.19},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title_text="Tageswert [mm/d]", secondary_y=False, rangemode="tozero",
                     gridcolor="rgba(155,165,157,.2)")
    fig.update_yaxes(title_text="WR [mm]", secondary_y=True, range=[0, WHC_FIXED], showgrid=False)
    fig.update_xaxes(tickformat="%b", dtick="M1", range=[df["date"].iloc[0], df["date"].iloc[-1]])
    return fig


def growth_start(df: pd.DataFrame):
    """Datum des modellierten Vegetationsstarts (SGS aus der GTS-Berechnung)."""
    doy = df["SGS"].iloc[0]
    if pd.isna(doy):
        return None
    return df.iloc[int(doy) - 1]["date"]


def plot_series(series: list[tuple[str, pd.DataFrame, str]], col: str, label: str,
                unit: str, day: int, cuts: tuple[int, ...], factor: float = 1.0,
                height: int = 290, show_growth_start: bool = True,
                rolling_days: int = 0) -> go.Figure:
    fig = go.Figure()
    for name, df, color in series:
        values = df[col] * factor
        if rolling_days:
            fig.add_trace(go.Scatter(
                x=df["date"], y=values, name=f"{name} · täglich",
                mode="lines", line={"color": color, "width": 1}, opacity=0.4,
                hovertemplate=f"{name} täglich<br>%{{x|%d.%m.}}: %{{y:.2f}} {unit}<extra></extra>",
            ))
            values = values.rolling(window=rolling_days, min_periods=1).mean()
        trace_name = f"{name} · {rolling_days}-Tage-Mittel" if rolling_days else name
        fig.add_trace(go.Scatter(
            x=df["date"], y=values, name=trace_name,
            mode="lines", line={"color": color, "width": 3 if name != "Referenz" else 2,
                               "dash": "dash" if name == "Referenz" else "solid"},
            hovertemplate=f"{trace_name}<br>%{{x|%d.%m.}}: %{{y:.2f}} {unit}<extra></extra>",
        ))
    # GTS-Wachstumsstart nur in LAI- und Biomassegrafik markieren.
    if show_growth_start:
        starts = {}
        for name, sim, color in series:
            start = growth_start(sim)
            if start is not None:
                starts.setdefault(start, color)
        for start, color in starts.items():
            fig.add_vline(x=start.timestamp() * 1000, line_color=color,
                          line_dash="longdash", opacity=.62)
    df = series[0][1]
    for cut_day in cuts:
        if 1 <= cut_day <= len(df):
            x = df.iloc[cut_day - 1]["date"].timestamp() * 1000
            fig.add_vline(x=x, line_color=AMBER, line_dash="dot", opacity=.55)
    current_x = df.iloc[day - 1]["date"].timestamp() * 1000
    fig.add_vline(x=current_x, line_color="#374151", opacity=.75)
    fig.update_layout(
        title={"text": label, "font": {"size": 17}},
        height=height, margin={"l": 10, "r": 12, "t": 60, "b": 12},
        hovermode="x unified", xaxis_title=None, yaxis_title=unit,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.18},
    )
    fig.update_yaxes(gridcolor="rgba(155,165,157,.22)", rangemode="tozero")
    fig.update_xaxes(gridcolor="rgba(155,165,157,.14)")
    return fig



def plot_growth_overview(series: list[tuple[str, pd.DataFrame, str]],
                         day: int, cuts: tuple[int, ...]) -> go.Figure:
    """Drei übereinanderliegende Diagramme mit exakt gemeinsamer Zeitachse."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.085,
        subplot_titles=("Blattflächenindex (LAI)", "Oberirdische Biomasse",
                        "Täglicher Biomassezuwachs (dBM)"),
        row_heights=[0.30, 0.35, 0.35],
    )

    for name, df, color in series:
        line = {"color": color, "width": 2.6,
                "dash": "dash" if name == "Referenz" else "solid"}
        for row_no, variable, factor, unit in (
            (1, "LAI", 1.0, "m²/m²"),
            (2, "BM", 0.001, "t TM/ha"),
        ):
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[variable] * factor,
                name=name, legendgroup=name, showlegend=row_no == 1,
                mode="lines", line=line,
                hovertemplate=f"{name}<br>%{{x|%d.%m.%Y}}: %{{y:.2f}} {unit}<extra></extra>",
            ), row=row_no, col=1)

        daily = df["dBM"]
        moving_mean = daily.rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["date"], y=daily,
            name=f"{name} · Tageswert", legendgroup=name, showlegend=False,
            mode="lines", opacity=0.30,
            line={"color": color, "width": 1},
            hovertemplate=f"{name} täglich<br>%{{x|%d.%m.%Y}}: %{{y:.1f}} kg TM/(ha·d)<extra></extra>",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df["date"], y=moving_mean,
            name=f"{name} · 7-Tage-Mittel", legendgroup=name, showlegend=False,
            mode="lines", line={**line, "width": 3},
            hovertemplate=f"{name} · 7-Tage-Mittel<br>%{{x|%d.%m.%Y}}: %{{y:.1f}} kg TM/(ha·d)<extra></extra>",
        ), row=3, col=1)

    # Wachstumsbeginn aus GTS nur in LAI und Biomasse markieren.
    starts = {}
    for name, sim, color in series:
        start = growth_start(sim)
        if start is not None:
            starts.setdefault(start, color)
    for start, color in starts.items():
        x = start.timestamp() * 1000
        for row_no in (1, 2):
            fig.add_vline(x=x, row=row_no, col=1,
                          line_color=color, line_dash="longdash", opacity=0.62)

    dates = series[0][1]["date"]
    for cut_day in cuts:
        if 1 <= cut_day <= len(dates):
            x = dates.iloc[cut_day - 1].timestamp() * 1000
            for row_no in (1, 2, 3):
                fig.add_vline(x=x, row=row_no, col=1,
                              line_color=AMBER, line_dash="dot", opacity=0.55)

    current_x = dates.iloc[day - 1].timestamp() * 1000
    for row_no in (1, 2, 3):
        fig.add_vline(x=current_x, row=row_no, col=1,
                      line_color="#374151", opacity=0.72)

    fig.update_layout(
        height=870,
        margin={"l": 15, "r": 20, "t": 80, "b": 30},
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "x": 0, "y": 1.08},
    )
    fig.update_annotations(font={"size": 17})
    fig.update_yaxes(gridcolor="rgba(155,165,157,.22)", rangemode="tozero")
    fig.update_yaxes(title_text="m²/m²", row=1, col=1)
    fig.update_yaxes(title_text="t TM/ha", row=2, col=1)
    fig.update_yaxes(title_text="kg TM/(ha·d)", row=3, col=1)
    fig.update_xaxes(gridcolor="rgba(155,165,157,.14)",
                     range=[dates.iloc[0], dates.iloc[-1]])
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(tickformat="%b", dtick="M1", title_text="Datum", row=3, col=1)
    return fig



def harvest_table(df: pd.DataFrame) -> pd.DataFrame:
    harvests = df.loc[df["CUT"] == 1, ["date", "cut_yield", "LAI", "ST"]].copy()
    harvests["Datum"] = harvests["date"].dt.strftime("%d.%m.%Y")
    harvests["Ertrag [t TM/ha]"] = (harvests["cut_yield"] / 1000).round(2)
    harvests["LAI nach Schnitt"] = harvests["LAI"].round(2)
    harvests["Temperatursumme [°C·d]"] = harvests["ST"].round(0)
    return harvests[["Datum", "Ertrag [t TM/ha]", "LAI nach Schnitt", "Temperatursumme [°C·d]"]]


st.markdown('<div class="kicker">BFH · Wachstumsanalyse · Interaktives Experiment</div>', unsafe_allow_html=True)
st.title("GrasslandDynamics")
st.markdown('<div class="caption-muted">Wachstum, Schnittmanagement und Klima am gewählten Standort untersuchen.</div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("Versuch")
    station_name = st.selectbox("Standort", list(STATIONS), index=0)
    station_code = STATIONS[station_name]
    years = years_available(station_code)
    if not years:
        st.error(f"Keine Meteodaten für {station_name} gefunden.")
        st.code(f"data/input/weather/recent_data/{station_code}/{station_code}INT_2026_DailyWeatherMeteoStation.dat")
        st.stop()
    year = st.selectbox("Wetterjahr", years, index=0)
    weather = get_weather(station_code, year)
    n_available = len(weather)
    partial_year = n_available < 365
    start_date = date(year, 1, 1)
    max_date = start_date + timedelta(days=n_available - 1)
    stamp = weather_version(station_code, year)

    if partial_year:
        st.info(f"{year}: Wetterdaten bis {max_date:%d.%m.%Y}. Keine Prognose für den Rest des Jahres.")
    mode = st.radio("Lernmodus", ["1 · Wachstum verstehen", "2 · Schnitte ausprobieren", "3 · Temperatur verändern"])
    grass_name = st.selectbox("Vegetation", list(GRASS_TYPES))
    weights = GRASS_TYPES[grass_name]

    # Referenzregime wird auf bereits verfügbare Tage begrenzt.
    default_days = tuple(
        (date(year, m, d) - start_date).days + 1
        for m, d in DEFAULT_CUTS[:3]
        if date(year, m, d) <= max_date
    )
    st.divider()
    delta_t = 0.0
    scenario_title = ""
    guided = False
    if mode.startswith("1"):
        st.caption("Keine Schnitte. Blattfläche und Biomasse im Jahresverlauf beobachten.")
        cuts = ()
        height_cm = 5.0
    elif mode.startswith("2"):
        exercise = st.radio("Versuchsaufbau", ["Geführte Aufgabe", "Freies Schnittmanagement"])
        guided = exercise == "Geführte Aufgabe"
        if guided:
            st.caption("Nur den ersten Schnitt verändern. Übrige Schnitte und Schnitthöhe bleiben gleich.")
            if partial_year:
                st.caption("Für den Vergleich vollständiger Jahreserträge eignet sich 2025 besser.")
            shift = st.slider("Erster Schnitt gegenüber 25. Mai [Tage]", -21, 21, 0, 7)
            first = date(year, 5, 25) + timedelta(days=shift)
            all_dates = [first, date(year, 7, 20), date(year, 9, 10)]
            cuts = tuple((d - start_date).days + 1 for d in all_dates if start_date <= d <= max_date)
            height_cm = 5.0
            st.caption(f"Erster Schnitt: {first:%d.%m.%Y}; weitere Termine: 20.07. und 10.09.")
        else:
            n_cuts = st.slider("Anzahl Schnitte", 0, 5, 3)
            height_cm = st.slider("Schnitthöhe [cm]", 2.0, 12.0, 5.0, .5)
            picked_dates = []
            for i in range(n_cuts):
                m, d = DEFAULT_CUTS[i]
                default = min(date(year, m, d), max_date)
                chosen = st.date_input(
                    f"Schnitt {i + 1}", value=default, min_value=start_date,
                    max_value=max_date, key=f"cut_{station_code}_{year}_{i}",
                )
                picked_dates.append(chosen)
            cuts = tuple(sorted({(d - start_date).days + 1 for d in picked_dates}))
            if len(cuts) != len(picked_dates):
                st.warning("Mehrere Schnitte fallen auf denselben Tag.")
    else:
        cuts = default_days
        height_cm = 5.0
        st.caption("Schnitte fest: 25.05., 20.07., 10.09., soweit bereits vergangen.")
        delta_t = st.slider("Temperaturänderung [°C]", -3.0, 3.0, 2.0, .5)
        scenario_title = f"ΔT = {delta_t:+.1f} °C"
        st.caption("Nur die Tagesmitteltemperatur wird verändert. ETc bleibt gleich.")
    st.divider()
    st.caption(f"Feste Modellannahmen: NI = {NI_FIXED:.1f}, WHC = {WHC_FIXED:.0f} mm.")
    if calendar.isleap(year) and n_available == 365:
        st.caption("365 Modell-Tage im Schaltjahr: letzter dargestellter Tag ist der 30.12.")

try:
    active = run_model(station_code, year, cuts, height_cm, weights,
                       temperature=delta_t, weather_stamp=stamp)
    reference = None
    if not mode.startswith("1"):
        ref_cuts = default_days if mode.startswith("2") else cuts
        reference = run_model(station_code, year, ref_cuts,
                              5.0 if mode.startswith("2") else height_cm,
                              weights, weather_stamp=stamp)
except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"Simulation konnte nicht ausgeführt werden: {error}")
    st.stop()

# Das Modell bestimmt den Vegetationsstart bereits aus der GTS (GTS > 200 °C·d).
active_start = growth_start(active)
reference_start = growth_start(reference) if reference is not None else None

if partial_year:
    st.warning(f"Laufendes Wetterjahr {year}: Simulation bis {max_date:%d.%m.%Y} "
               f"({n_available:d} Tage). Angezeigte Schnitterträge sind Zwischenstände, "
               "keine Jahresprognosen. Geplante Schnitte nach diesem Datum bleiben unberücksichtigt.")

st.markdown("#### Modellierter Wachstumsstart")
if reference is None:
    st.metric("Grünlandtemperatursumme > 200 °C·d", active_start.strftime("%d.%m.%Y") if active_start is not None else "Nicht erreicht")
else:
    m1, m2, m3 = st.columns(3)
    m1.metric("Experiment", active_start.strftime("%d.%m.%Y") if active_start is not None else "Nicht erreicht")
    m2.metric("Referenz", reference_start.strftime("%d.%m.%Y") if reference_start is not None else "Nicht erreicht")
    if active_start is not None and reference_start is not None:
        diff = (active_start - reference_start).days
        m3.metric("Startverschiebung", f"{diff:+d} Tage" if diff else "0 Tage")
    else:
        m3.metric("Startverschiebung", "Nicht berechenbar")
st.caption("ModVege-Definition: gewichtete positive Tagesmitteltemperatur (Jan × 0.5, Feb × 0.75, ab März × 1), ohne Tage mit Schnee. Start bei GTS > 200 °C·d.")

max_day = len(active)
slider_key = f"simulation_day_{station_code}_{year}"
if slider_key not in st.session_state or st.session_state[slider_key] > max_day:
    st.session_state[slider_key] = max_day if partial_year else min(185, max_day)

st.markdown(f"### Zeitreise durch das {'bisherige ' if partial_year else ''}Wachstumsjahr · {station_name}, {year}")
l, mid, r = st.columns([1, 8, 1], vertical_alignment="bottom")
with l:
    if st.button("◀ 7 Tage", use_container_width=True):
        st.session_state[slider_key] = max(1, st.session_state[slider_key] - 7)
with r:
    if st.button("7 Tage ▶", use_container_width=True):
        st.session_state[slider_key] = min(max_day, st.session_state[slider_key] + 7)
with mid:
    day = st.slider("Tag im Jahr", 1, max_day, key=slider_key)

row = active.iloc[day - 1]
today = row["date"].strftime("%d.%m.%Y")
st.caption(f"**{today}** · Die Visualisierung zeigt den Bestand nach allen Modellprozessen dieses Tages.")
if active_start is not None:
    if row["date"] < active_start:
        st.info("Vor dem modellierten Wachstumsstart: Die dargestellte Wiese enthält überwinterte Biomasse, aber noch keinen modellierten Zuwachs.")
    elif row["date"] == active_start:
        st.success("Heute beginnt laut GTS-Kriterium das modellierte Wachstum.")

meadow_col, water_col, metrics_col = st.columns([1.7, 0.85, 1.0], gap="medium")
with meadow_col:
    components.html(meadow_html(row["LAI"], row["f_abs"], bool(row["snow"]), bool(row["CUT"])),
                    height=310, scrolling=False)
    st.caption("Schematische Grasnarbe auf Basis des LAI. Die gezeichnete Grashöhe ist **keine** simulierte Bestandeshöhe.")
    st.markdown(f"**Absorbierte Strahlung:** {row['f_abs']*100:.0f} %")
    st.progress(min(1.0, max(0.0, float(row["f_abs"]))))
    st.caption(r"$f_{abs}=1-e^{-0.6\,LAI}$")
with water_col:
    components.html(water_bucket_html(
        row["WR"], WHC_FIXED, row["Precip"], row["water_input"], row["AET"], row["fW"]
    ), height=300, scrolling=False)
    st.caption("WR ist der modellierte Wasservorrat, nicht die Bodentiefe. "
               "Bei Schnee wird Niederschlag erst nach der Schmelze als Zufluss wirksam. "
               "f(W) hängt auch vom Verdunstungsbedarf ab.")
with metrics_col:
    st.markdown(dedent(f"""
    <div class="growth-metrics">
      <div class="growth-metric">
        <div class="growth-metric-label">LAI</div>
        <div class="growth-metric-value">{row['LAI']:.2f}</div>
        <div class="growth-metric-unit">m²/m²</div>
      </div>
      <div class="growth-metric">
        <div class="growth-metric-label">Biomasse</div>
        <div class="growth-metric-value">{row['BM']/1000:.2f}</div>
        <div class="growth-metric-unit">t TM/ha</div>
      </div>
      <div class="growth-metric">
        <div class="growth-metric-label">Wachstum heute (GRO)</div>
        <div class="growth-metric-value">{row['GRO']:.0f}</div>
        <div class="growth-metric-unit">kg TM/(ha·d)</div>
      </div>
      <div class="growth-metric">
        <div class="growth-metric-label">ST</div>
        <div class="growth-metric-value">{row['ST']:.0f}</div>
        <div class="growth-metric-unit">°C·d</div>
      </div>
    </div>
    """).strip(), unsafe_allow_html=True)
    st.caption(f"GTS bis heute: {row['GTS']:.0f} °C·d · Wachstumsstart bei 200 °C·d")
    if row["CUT"] == 1:
        st.success(f"Heute geerntet: {row['cut_yield']/1000:.2f} t TM/ha")
    if row["snow"] == 1:
        st.info("Schneedecke: Wachstum und Schnitt sind im Modell ausgesetzt.")

with st.expander("Niederschlag, Evapotranspiration und Bodenwasser im Jahresverlauf"):
    st.plotly_chart(plot_water_balance(active, day), use_container_width=True)
    st.caption("Niederschlag und tatsächliche Evapotranspiration (AET) sind Tagesflüsse. "
               "WR ist der gespeicherte Wasservorrat. Der effektive Bodenzufluss "
               "kann wegen Schneespeicherung und Schneeschmelze vom Niederschlag abweichen. "
               "Überschüssiges Wasser verlässt den Speicher durch Drainage.")

if mode.startswith("1"):
    st.plotly_chart(
        plot_growth_overview([(station_name, active, GREEN)], day, ()),
        use_container_width=True,
    )
    st.caption("Gemeinsame Zeitachse für LAI, Biomasse und dBM. "
               "Lange gestrichelte Linie: Wachstumsstart (GTS > 200 °C·d). "
               "Bei dBM: dünne Linie = Tageswert, kräftige Linie = gleitender 7-Tage-Mittelwert. "
               "dBM zeigt nur positive Nettozuwächse nach Seneszenz und Abszission; "
               "negative Änderungen und Schnitttage werden als 0 dargestellt.")
    with st.expander("Was steuert das Wachstum?"):
        st.write(f"Temperaturfaktor fT: **{row['fT']:.2f}** · Wasserfaktor fW: **{row['fW']:.2f}** · "
                 f"Strahlungsfaktor fPAR: **{row['fPAR']:.2f}** · Saisonfaktor SEA: **{row['SEA']:.2f}**")
        st.latex(r"PGRO = PAR_i \cdot RUE_{\max} \cdot (1-e^{-0.6\,LAI}) \cdot 10")
        st.latex(r"GRO = PGRO \cdot ENV \cdot SEA")
        st.caption("PGRO ist das strahlungsabhängige Wachstumspotenzial. "
                   "GRO enthält zusätzlich Umwelt- und Saisonfaktoren. "
                   "Die stehende Biomasse wird auch durch Verluste und Ernte beeinflusst.")
        st.latex(r"ENV = NI \cdot f(PAR) \cdot f(T) \cdot f(W)")
        st.caption("Die ungeschnittene Wiese ist ein vereinfachtes Lernexperiment, kein realistisches Bewirtschaftungsregime.")
else:
    assert reference is not None
    baseline_name = "Referenz"  # Fester Referenzschnitt im Schnittmodus; gleiche Schnitte im Szenariomodus.
    series = [("Experiment", active, GREEN), (baseline_name, reference, BLUE)]
    active_y = active["hvBM"].iloc[-1] / 1000
    ref_y = reference["hvBM"].iloc[-1] / 1000
    m1, m2, m3 = st.columns(3)
    m1.metric("Ertrag Experiment" + (" bisher" if partial_year else ""), f"{active_y:.2f} t TM/ha")
    m2.metric("Ertrag Referenz" + (" bisher" if partial_year else ""), f"{ref_y:.2f} t TM/ha")
    m3.metric("Differenz", f"{active_y-ref_y:+.2f} t TM/ha")
    if mode.startswith("2"):
        st.caption("Referenz: 25.05., 20.07., 10.09., Schnitthöhe 5 cm. "
                   "Nur bereits vergangene Schnitttermine werden berücksichtigt. "
                   "Das Regime ist nicht standortspezifisch optimiert.")
    else:
        st.caption("Referenz und Experiment verwenden dieselben Schnittdaten, Schnitthöhe und Grasgruppe.")

    if mode.startswith("2") and len(cuts) == 0:
        st.info("Ohne Schnitt zeigt das Experiment die ungestörte Bestandesentwicklung. Die Referenz wird weiterhin dreimal geschnitten. Der Schnittertrag des Experiments ist dann 0, obwohl stehende Biomasse vorhanden ist. Seneszenz und Abszission bleiben wirksam.")
    cuts_for_plot = tuple(sorted(set(cuts).union(default_days)))
    st.plotly_chart(
        plot_growth_overview(series, day, cuts_for_plot),
        use_container_width=True,
    )
    st.caption("Gemeinsame Zeitachse für LAI, Biomasse und dBM. "
               "Lange gestrichelte Linien: Wachstumsstart (GTS > 200 °C·d), "
               "gepunktete Linien: Schnitttermine. "
               "Bei dBM: dünne Linie = Tageswert, kräftige Linie = 7-Tage-Mittel. "
               "dBM zeigt positive Nettozuwächse nach Seneszenz und Abszission; "
               "negative Änderungen und Schnitttage werden als 0 dargestellt.")
    st.plotly_chart(plot_series(series, "hvBM", "Kumulierter Schnittertrag", "t TM/ha", day,
                                cuts_for_plot, .001, 290, show_growth_start=False),
                    use_container_width=True)
    skipped = [d for d in cuts if int(active.iloc[d-1]["CUT"]) == 0]
    if skipped:
        st.warning("Nicht ausgeführte Schnitte (z. B. wegen Schnee): " +
                   ", ".join(active.iloc[d-1]["date"].strftime("%d.%m.") for d in skipped))
    tab_a, tab_b = st.tabs(["Schnitterträge", "Modellprozesse"])
    with tab_a:
        t = harvest_table(active)
        if t.empty:
            st.info("Es wurden bisher keine Schnitte ausgeführt.")
        else:
            st.dataframe(t, hide_index=True, use_container_width=True)
        if guided:
            st.caption("Schnitterträge der Referenz")
            st.dataframe(harvest_table(reference), hide_index=True, use_container_width=True)
    with tab_b:
        a, b, c = st.columns(3)
        a.metric("Temperaturfaktor fT", f"{row['fT']:.2f}")
        b.metric("Wasserfaktor fW", f"{row['fW']:.2f}")
        c.metric("Saisonfaktor SEA", f"{row['SEA']:.2f}")
        st.caption("Umwelteinflüsse und Seneszenz verändern den Biomassebestand; Wachstum und Ertrag sind nicht identisch.")

with st.expander("Modellannahmen und Grenzen"):
    st.markdown(f"""
- **Standort:** {station_name}. Es werden ausschliesslich die Meteodaten des gewählten Standorts ausgetauscht.
- **Modellparameter:** NI = {NI_FIXED:.1f}, WHC = {WHC_FIXED:.0f} mm für alle Standorte.
- **Modell:** ModVege mit täglichem Wachstum, Seneszenz, Wasserhaushalt und vereinfachtem Schneemodell.
- **Bodenwasser:** WR ist der simulierte Speicherinhalt (0 bis WHC), nicht Bodenfeuchte in Volumenprozent. Zufluss umfasst Regen und Schneeschmelze. AET ist die tatsächliche Evapotranspiration. fW ist die daraus abgeleitete Wirkung auf das Wachstum.
- **Schnitte:** Feste Kalendertage. Die Schnittbiomasse wird aus einer stoppelhöhenabhängigen Restbiomasse ermittelt.
- **Biomasse:** kg Trockenmasse pro Hektare (Anzeige in t TM/ha). LAI in m²/m².
- **Visualisierung:** Die Wiesenhöhe ist ausschliesslich schematisch. ModVege simuliert keine reale Höhe.
- **Modellgrenzen:** Keine explizite Wurzel- oder Reservestoffsimulation. Die Parameter sind nicht standortspezifisch kalibriert. Die Ergebnisse sind Modellszenarien, keine lokalen Ertragsprognosen.
- **Wachstumsstart (SGS):** Erster Tag mit Grünlandtemperatursumme > 200 °C·d. GTS berücksichtigt Jan × 0.5, Feb × 0.75, März bis Juni × 1.0, ausschliesslich positive Temperaturen und Tage ohne Schneedecke. Das ist die schneeangepasste Berechnung aus dem ursprünglichen Skript.
- **GTS vs. ST:** GTS bestimmt den Wachstumsstart; ST ist die ModVege-Temperatursumme für den Saisonfaktor SEA.
- **Meteodaten:** Historische und aktuelle Tageswerte aus deinen Dateien. Laufende Jahre enden am letzten verfügbaren Tag; es erfolgt keine Prognose.
- **Referenz:** Drei feste, nicht standortspezifisch optimierte Schnitte. Im Temperaturszenario gleiches Management ohne Temperaturänderung. ETc bleibt bei ΔT konstant.
""")
