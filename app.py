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
GRASS_WEIGHTS = (0.5, 0.5, 0.0, 0.0)
GREEN, BLUE, AMBER = "#19865c", "#5886b1", "#d89139"

st.set_page_config(page_title="GrasslandDynamics", page_icon="🌱", layout="wide")
# Sprachauswahl wird im Seitenkopf angezeigt. Keine Übersetzung der Modellberechnung.
header_left, header_right = st.columns([7, 1.7], vertical_alignment="center")
with header_right:
    lang = st.radio(
        "Sprache / Langue", ("DE", "FR"), horizontal=True,
        label_visibility="collapsed", key="ui_lang",
    )


def L(de: str, fr: str) -> str:
    """Aktuelle UI-Sprache; die Modell- und Datenspaltennamen bleiben unverändert."""
    return de if lang == "DE" else fr


UNIT_YIELD = L("t TM/ha", "t MS/ha")
UNIT_GROWTH = L("kg TM/(ha·d)", "kg MS/(ha·j)")
MONTHS = (
    ("Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez")
    if lang == "DE" else
    ("Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Aoû", "Sep", "Oct", "Nov", "Déc")
)


def set_month_ticks(fig: go.Figure, dates: pd.Series, row: int | None = None):
    """Unabhängig von Plotly-Browserlocale Monatsnamen in DE/FR anzeigen."""
    first = dates.iloc[0]
    last = dates.iloc[-1]
    ticks = [pd.Timestamp(first.year, month, 1) for month in range(1, 13)
             if pd.Timestamp(first.year, month, 1) >= first
             and pd.Timestamp(first.year, month, 1) <= last]
    labels = [MONTHS[t.month - 1] for t in ticks]
    if row is None:
        fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=labels)
    else:
        fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=labels, row=row, col=1)


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
               f'<text x="596" y="48" font-size="19" fill="#9a5a18">✂ {L("Schnitt heute", "Coupe aujourd’hui")}</text>' if cut else "")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<style>html, body {{margin:0;background:transparent}} svg {{width:100%;height:auto;display:block}}
svg .grass {{animation: appear .45s ease-out}} @keyframes appear {{from {{opacity:.15}} to {{opacity:1}}}}
</style></head><body><svg viewBox="0 0 800 320" xmlns="http://www.w3.org/2000/svg" role="img"
aria-label="{L("Schematische Grasnarbe: Dichte und Höhe aus LAI abgeleitet", "Prairie schématique : densité et hauteur déduites du LAI")}">
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
<text x="30" y="41" fill="#41554a" font-size="16">{L("Absorbierte Strahlung", "Rayonnement absorbé")}: {f_abs*100:.0f} %</text>
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
aria-label="{L("Bodenwasserspeicher", "Réserve en eau du sol")} {wr:.0f} / {whc:.0f} mm; f(W) {fw:.2f}">
<rect x="1" y="1" width="332" height="280" rx="15" fill="#f2f8fb" stroke="#d5e7ed"/>
<text x="17" y="28" font-size="16" font-weight="bold" fill="#25516b">{L("Bodenwasserspeicher", "Réserve en eau du sol")}</text>
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
<text x="132" y="98" font-size="12" fill="#527487">{L("Bodenwasser (WR)", "Eau du sol (WR)")}</text>
<path d="M132 118 l8 10 l8 -10" fill="none" stroke="#2890bd" stroke-width="3"/>
<text x="158" y="129" font-size="13" fill="#24556e">{L("Zufluss", "Apport")}: {influx:.1f} {L("mm/d", "mm/j")}</text>
<path d="M132 147 l8 -10 l8 10" fill="none" stroke="#c47a32" stroke-width="3"/>
<text x="158" y="149" font-size="13" fill="#795126">AET: {aet:.1f} {L("mm/d", "mm/j")}</text>
<text x="132" y="172" font-size="12" fill="#527487">{L("Niederschlag", "Précipitations")}: {precip:.1f} {L("mm/d", "mm/j")}</text>
<text x="17" y="234" font-size="13" font-weight="bold" fill="#245a43">{L("Wachstumsfaktor", "Facteur de croissance")} f(W): {fw:.2f}</text>
<rect x="17" y="250" width="296" height="12" rx="5" fill="#d5e5de"/>
<rect x="17" y="250" width="{296*fw:.1f}" height="12" rx="5" fill="#31966b"/>
</svg></body></html>"""


def plot_water_balance(df: pd.DataFrame, day: int) -> go.Figure:
    """Niederschlag und tatsächliche Evapotranspiration, plus WR als Zustand."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df["date"], y=df["Precip"], name=L("Niederschlag", "Précipitations"),
        marker_color="#63b4de", opacity=0.65,
        hovertemplate="%{x|%d.%m.}: %{y:.1f} mm<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["AET"], name=L("Tatsächliche ET", "Évapotranspiration réelle"),
        mode="lines", line={"color": AMBER, "width": 2},
        hovertemplate="%{x|%d.%m.}: %{y:.1f} mm<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["WR"], name=L("Bodenwasser (WR)", "Eau du sol (WR)"),
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
    fig.update_yaxes(title_text=L("Tageswert [mm/d]", "Valeur journalière [mm/j]"), secondary_y=False, rangemode="tozero",
                     gridcolor="rgba(155,165,157,.2)")
    fig.update_yaxes(title_text="WR [mm]", secondary_y=True, range=[0, WHC_FIXED], showgrid=False)
    fig.update_xaxes(range=[df["date"].iloc[0], df["date"].iloc[-1]])
    set_month_ticks(fig, df["date"])
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
                x=df["date"], y=values, name=f"{name} · {L('täglich', 'quotidien')}",
                mode="lines", line={"color": color, "width": 1}, opacity=0.4,
                hovertemplate=f"{name} {L('täglich', 'quotidien')}<br>%{{x|%d.%m.}}: %{{y:.2f}} {unit}<extra></extra>",
            ))
            values = values.rolling(window=rolling_days, min_periods=1).mean()
        trace_name = f"{name} · {L(f'{rolling_days}-Tage-Mittel', f'moyenne sur {rolling_days} jours')}" if rolling_days else name
        fig.add_trace(go.Scatter(
            x=df["date"], y=values, name=trace_name,
            mode="lines", line={"color": color, "width": 3 if name != L("Referenz", "Référence") else 2,
                               "dash": "dash" if name == L("Referenz", "Référence") else "solid"},
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
        subplot_titles=(L("Blattflächenindex (LAI)", "Indice de surface foliaire (LAI)"),
                        L("Oberirdische Biomasse", "Biomasse aérienne"),
                        L("Täglicher Biomassezuwachs (dBM)", "Accroissement journalier de biomasse (dBM)")),
        row_heights=[0.30, 0.35, 0.35],
    )

    for name, df, color in series:
        line = {"color": color, "width": 2.6,
                "dash": "dash" if name == L("Referenz", "Référence") else "solid"}
        for row_no, variable, factor, unit in (
            (1, "LAI", 1.0, "m²/m²"),
            (2, "BM", 0.001, UNIT_YIELD),
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
            name=f"{name} · {L('Tageswert', 'valeur journalière')}", legendgroup=name, showlegend=False,
            mode="lines", opacity=0.30,
            line={"color": color, "width": 1},
            hovertemplate=f"{name} {L('täglich', 'quotidien')}<br>%{{x|%d.%m.%Y}}: %{{y:.1f}} {UNIT_GROWTH}<extra></extra>",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df["date"], y=moving_mean,
            name=f"{name} · {L('7-Tage-Mittel', 'moyenne sur 7 jours')}", legendgroup=name, showlegend=False,
            mode="lines", line={**line, "width": 3},
            hovertemplate=f"{name} · {L('7-Tage-Mittel', 'moyenne sur 7 jours')}<br>%{{x|%d.%m.%Y}}: %{{y:.1f}} {UNIT_GROWTH}<extra></extra>",
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
    fig.update_yaxes(title_text=UNIT_YIELD, row=2, col=1)
    fig.update_yaxes(title_text=UNIT_GROWTH, row=3, col=1)
    fig.update_xaxes(gridcolor="rgba(155,165,157,.14)",
                     range=[dates.iloc[0], dates.iloc[-1]])
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(title_text=L("Datum", "Date"), row=3, col=1)
    set_month_ticks(fig, dates, row=3)
    return fig



def harvest_table(df: pd.DataFrame) -> pd.DataFrame:
    harvests = df.loc[df["CUT"] == 1, ["date", "cut_yield", "LAI", "ST"]].copy()
    names = [L("Datum", "Date"), L("Ertrag", "Rendement") + f" [{UNIT_YIELD}]",
             L("LAI nach Schnitt", "LAI après la coupe"),
             L("Temperatursumme [°C·d]", "Somme des températures [°C·j]")]
    harvests[names[0]] = harvests["date"].dt.strftime("%d.%m.%Y")
    harvests[names[1]] = (harvests["cut_yield"] / 1000).round(2)
    harvests[names[2]] = harvests["LAI"].round(2)
    harvests[names[3]] = harvests["ST"].round(0)
    return harvests[names]



with header_left:
    st.markdown(
        f'<div class="kicker">{L("BFH · Wachstumsanalyse · Interaktives Experiment", "BFH · Analyse de croissance · Expérience interactive")}</div>',
        unsafe_allow_html=True,
    )
    st.title("GrasslandDynamics")
    st.markdown(
        f'<div class="caption-muted">{L("Wachstum, Schnittmanagement und Klima am gewählten Standort untersuchen.", "Explorer la croissance, la gestion des coupes et le climat du site sélectionné.")}</div>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header(L("Versuch", "Expérience"))
    station_name = st.selectbox(L("Standort", "Site"), list(STATIONS), index=0)
    station_code = STATIONS[station_name]
    years = years_available(station_code)
    if not years:
        st.error(L(f"Keine Meteodaten für {station_name} gefunden.",
                   f"Aucune donnée météorologique pour {station_name}."))
        st.code(f"data/input/weather/recent_data/{station_code}/{station_code}INT_2026_DailyWeatherMeteoStation.dat")
        st.stop()
    year = st.selectbox(L("Wetterjahr", "Année météorologique"), years, index=0)
    weather = get_weather(station_code, year)
    n_available = len(weather)
    partial_year = n_available < 365
    start_date = date(year, 1, 1)
    max_date = start_date + timedelta(days=n_available - 1)
    stamp = weather_version(station_code, year)

    if partial_year:
        st.info(L(f"{year}: Wetterdaten bis {max_date:%d.%m.%Y}. Keine Prognose für den Rest des Jahres.",
                  f"{year} : données météorologiques jusqu’au {max_date:%d.%m.%Y}. Pas de prévision pour la fin de l’année."))

    modes = [
        L("1 · Wachstum verstehen", "1 · Comprendre la croissance"),
        L("2 · Schnitte ausprobieren", "2 · Tester les coupes"),
        L("3 · Temperatur verändern", "3 · Modifier la température"),
    ]
    mode = st.radio(L("Lernmodus", "Mode d’apprentissage"), modes)
    weights = GRASS_WEIGHTS
    st.caption(L("Vegetation: 50 % Gruppe A + 50 % Gruppe B (fest).",
                 "Végétation : 50 % groupe A + 50 % groupe B (fixe)."))

    default_days = tuple(
        (date(year, m, d) - start_date).days + 1
        for m, d in DEFAULT_CUTS[:3]
        if date(year, m, d) <= max_date
    )
    st.divider()
    delta_t = 0.0
    guided = False
    if mode.startswith("1"):
        st.caption(L("Keine Schnitte. Blattfläche und Biomasse im Jahresverlauf beobachten.",
                     "Sans coupe. Observer l’évolution annuelle de la surface foliaire et de la biomasse."))
        cuts = ()
        height_cm = 5.0
    elif mode.startswith("2"):
        exercises = [
            L("Geführte Aufgabe", "Exercice guidé"),
            L("Freies Schnittmanagement", "Gestion libre des coupes"),
        ]
        exercise = st.radio(L("Versuchsaufbau", "Configuration de l’expérience"), exercises)
        guided = exercise == exercises[0]
        if guided:
            st.caption(L("Nur den ersten Schnitt verändern. Übrige Schnitte und Schnitthöhe bleiben gleich.",
                         "Modifier uniquement la première coupe. Les autres dates et la hauteur de coupe restent inchangées."))
            if partial_year:
                st.caption(L("Für den Vergleich vollständiger Jahreserträge eignet sich 2025 besser.",
                             "Pour comparer les rendements annuels complets, privilégier 2025."))
            shift = st.slider(L("Erster Schnitt gegenüber 25. Mai [Tage]",
                                "Décalage de la première coupe par rapport au 25 mai [jours]"),
                              -21, 21, 0, 7)
            first = date(year, 5, 25) + timedelta(days=shift)
            all_dates = [first, date(year, 7, 20), date(year, 9, 10)]
            cuts = tuple((d - start_date).days + 1 for d in all_dates if start_date <= d <= max_date)
            height_cm = 5.0
            st.caption(L(f"Erster Schnitt: {first:%d.%m.%Y}; weitere Termine: 20.07. und 10.09.",
                         f"Première coupe : {first:%d.%m.%Y} ; autres dates : 20.07. et 10.09."))
        else:
            n_cuts = st.slider(L("Anzahl Schnitte", "Nombre de coupes"), 0, 5, 3)
            height_cm = st.slider(L("Schnitthöhe [cm]", "Hauteur de coupe [cm]"), 2.0, 12.0, 5.0, .5)
            picked_dates = []
            for i in range(n_cuts):
                m, d = DEFAULT_CUTS[i]
                default = min(date(year, m, d), max_date)
                chosen = st.date_input(
                    L(f"Schnitt {i + 1}", f"Coupe {i + 1}"),
                    value=default, min_value=start_date,
                    max_value=max_date, key=f"cut_{station_code}_{year}_{i}",
                    format="DD.MM.YYYY",
                )
                picked_dates.append(chosen)
            cuts = tuple(sorted({(d - start_date).days + 1 for d in picked_dates}))
            if len(cuts) != len(picked_dates):
                st.warning(L("Mehrere Schnitte fallen auf denselben Tag.",
                             "Plusieurs coupes sont prévues le même jour."))
    else:
        cuts = default_days
        height_cm = 5.0
        st.caption(L("Schnitte fest: 25.05., 20.07., 10.09., soweit bereits vergangen.",
                     "Dates de coupe fixes : 25.05., 20.07. et 10.09., si elles sont déjà passées."))
        delta_t = st.slider(L("Temperaturänderung [°C]", "Variation de température [°C]"),
                            -3.0, 3.0, 2.0, .5)
        st.caption(L("Nur die Tagesmitteltemperatur wird verändert. ETc bleibt gleich.",
                     "Seule la température moyenne journalière est modifiée. L’ETc reste constante."))
    st.divider()
    st.caption(L(f"Feste Modellannahmen: NI = {NI_FIXED:.1f}, WHC = {WHC_FIXED:.0f} mm.",
                 f"Hypothèses fixes du modèle : NI = {NI_FIXED:.1f}, WHC = {WHC_FIXED:.0f} mm."))
    if calendar.isleap(year) and n_available == 365:
        st.caption(L("365 Modell-Tage im Schaltjahr: letzter dargestellter Tag ist der 30.12.",
                     "365 jours modélisés pour une année bissextile : dernier jour affiché le 30.12."))

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
    st.error(L("Simulation konnte nicht ausgeführt werden", "Échec de la simulation") + f": {error}")
    st.stop()

active_start = growth_start(active)
reference_start = growth_start(reference) if reference is not None else None

if partial_year:
    st.warning(L(
        f"Laufendes Wetterjahr {year}: Simulation bis {max_date:%d.%m.%Y} ({n_available:d} Tage). "
        "Angezeigte Schnitterträge sind Zwischenstände, keine Jahresprognosen. "
        "Geplante Schnitte nach diesem Datum bleiben unberücksichtigt.",
        f"Année météorologique en cours {year} : simulation jusqu’au {max_date:%d.%m.%Y} ({n_available:d} jours). "
        "Les rendements affichés sont provisoires et non des prévisions annuelles. "
        "Les coupes ultérieures ne sont pas prises en compte.",
    ))

st.markdown(L("#### Modellierter Wachstumsstart", "#### Début de croissance modélisé"))
no_start = L("Nicht erreicht", "Non atteint")
if reference is None:
    st.metric(L("Grünlandtemperatursumme > 200 °C·d", "Somme des températures de croissance > 200 °C·j"),
              active_start.strftime("%d.%m.%Y") if active_start is not None else no_start)
else:
    m1, m2, m3 = st.columns(3)
    m1.metric(L("Experiment", "Expérience"), active_start.strftime("%d.%m.%Y") if active_start is not None else no_start)
    m2.metric(L("Referenz", "Référence"), reference_start.strftime("%d.%m.%Y") if reference_start is not None else no_start)
    if active_start is not None and reference_start is not None:
        diff = (active_start - reference_start).days
        m3.metric(L("Startverschiebung", "Décalage du démarrage"),
                  L(f"{diff:+d} Tage" if diff else "0 Tage", f"{diff:+d} jours" if diff else "0 jour"))
    else:
        m3.metric(L("Startverschiebung", "Décalage du démarrage"), L("Nicht berechenbar", "Non calculable"))
st.caption(L(
    "ModVege-Definition: positive Tagesmitteltemperaturen gewichtet (Jan × 0,5; Feb × 0,75; ab März × 1), ohne Tage mit Schnee. Start bei GTS > 200 °C·d.",
    "Définition ModVege : températures moyennes journalières positives pondérées (janvier × 0,5 ; février × 0,75 ; dès mars × 1), hors jours avec neige. Début lorsque la GTS dépasse 200 °C·j.",
))

max_day = len(active)
slider_key = f"simulation_day_{station_code}_{year}"
if slider_key not in st.session_state or st.session_state[slider_key] > max_day:
    st.session_state[slider_key] = max_day if partial_year else min(185, max_day)

st.markdown(L(
    f"### Zeitreise durch das {'bisherige ' if partial_year else ''}Wachstumsjahr · {station_name}, {year}",
    f"### Explorer {'la partie disponible de ' if partial_year else ''}l’année de croissance · {station_name}, {year}",
))
l, mid, r = st.columns([1, 8, 1], vertical_alignment="bottom")
with l:
    if st.button(L("◀ 7 Tage", "◀ 7 jours"), use_container_width=True):
        st.session_state[slider_key] = max(1, st.session_state[slider_key] - 7)
with r:
    if st.button(L("7 Tage ▶", "7 jours ▶"), use_container_width=True):
        st.session_state[slider_key] = min(max_day, st.session_state[slider_key] + 7)
with mid:
    day = st.slider(L("Tag im Jahr", "Jour de l’année"), 1, max_day, key=slider_key)

row = active.iloc[day - 1]
today = row["date"].strftime("%d.%m.%Y")
st.caption(L(f"**{today}** · Die Visualisierung zeigt den Bestand nach allen Modellprozessen dieses Tages.",
             f"**{today}** · La visualisation représente la prairie à la fin de la journée simulée."))
if active_start is not None:
    if row["date"] < active_start:
        st.info(L("Vor dem modellierten Wachstumsstart: Der Bestand enthält überwinterte Biomasse, aber noch keinen modellierten Zuwachs.",
                  "Avant le début de croissance modélisé : la prairie conserve de la biomasse hivernale, sans nouvel accroissement simulé."))
    elif row["date"] == active_start:
        st.success(L("Heute beginnt laut GTS-Kriterium das modellierte Wachstum.",
                     "Le modèle indique aujourd’hui le début de la croissance selon le critère GTS."))

meadow_col, water_col, metrics_col = st.columns([1.7, 0.85, 1.0], gap="medium")
with meadow_col:
    components.html(meadow_html(row["LAI"], row["f_abs"], bool(row["snow"]), bool(row["CUT"])),
                    height=310, scrolling=False)
    st.caption(L("Schematische Grasnarbe auf Basis des LAI. Die gezeichnete Grashöhe ist **keine** simulierte Bestandeshöhe.",
                 "Prairie schématique basée sur le LAI. La hauteur dessinée **ne correspond pas** à une hauteur d’herbe simulée."))
    st.markdown(L(f"**Absorbierte Strahlung :** {row['f_abs']*100:.0f} %",
                  f"**Rayonnement absorbé :** {row['f_abs']*100:.0f} %"))
    st.progress(min(1.0, max(0.0, float(row["f_abs"]))))
    st.caption(r"$f_{abs}=1-e^{-0.6\,LAI}$")
with water_col:
    components.html(water_bucket_html(
        row["WR"], WHC_FIXED, row["Precip"], row["water_input"], row["AET"], row["fW"]
    ), height=300, scrolling=False)
    st.caption(L(
        "WR ist der modellierte Wasservorrat, nicht die Bodentiefe. Bei Schnee wird Niederschlag erst nach der Schmelze als Zufluss wirksam. f(W) hängt auch vom Verdunstungsbedarf ab.",
        "WR représente la réserve en eau simulée, pas la profondeur du sol. La neige alimente la réserve après sa fonte. f(W) dépend aussi de la demande évaporative.",
    ))
with metrics_col:
    st.markdown(dedent(f"""
    <div class="growth-metrics">
      <div class="growth-metric">
        <div class="growth-metric-label">LAI</div>
        <div class="growth-metric-value">{row['LAI']:.2f}</div>
        <div class="growth-metric-unit">m²/m²</div>
      </div>
      <div class="growth-metric">
        <div class="growth-metric-label">{L('Biomasse', 'Biomasse')}</div>
        <div class="growth-metric-value">{row['BM']/1000:.2f}</div>
        <div class="growth-metric-unit">{UNIT_YIELD}</div>
      </div>
      <div class="growth-metric">
        <div class="growth-metric-label">{L('Wachstum heute (GRO)', 'Croissance du jour (GRO)')}</div>
        <div class="growth-metric-value">{row['GRO']:.0f}</div>
        <div class="growth-metric-unit">{UNIT_GROWTH}</div>
      </div>
      <div class="growth-metric">
        <div class="growth-metric-label">ST</div>
        <div class="growth-metric-value">{row['ST']:.0f}</div>
        <div class="growth-metric-unit">{L('°C·d', '°C·j')}</div>
      </div>
    </div>
    """).strip(), unsafe_allow_html=True)
    st.caption(L(f"GTS bis heute: {row['GTS']:.0f} °C·d · Wachstumsstart bei 200 °C·d",
                 f"GTS cumulée : {row['GTS']:.0f} °C·j · Début de croissance à 200 °C·j"))
    if row["CUT"] == 1:
        st.success(L(f"Heute geerntet: {row['cut_yield']/1000:.2f} {UNIT_YIELD}",
                     f"Récolte du jour : {row['cut_yield']/1000:.2f} {UNIT_YIELD}"))
    if row["snow"] == 1:
        st.info(L("Schneedecke: Wachstum und Schnitt sind im Modell ausgesetzt.",
                  "Couverture neigeuse : la croissance et les coupes sont suspendues dans le modèle."))

with st.expander(L("Niederschlag, Evapotranspiration und Bodenwasser im Jahresverlauf",
                   "Précipitations, évapotranspiration et eau du sol au fil de l’année")):
    st.plotly_chart(plot_water_balance(active, day), use_container_width=True)
    st.caption(L(
        "Niederschlag und tatsächliche Evapotranspiration (AET) sind Tagesflüsse. WR ist der gespeicherte Wasservorrat. Durch Schnee kann sich der Bodenzufluss vom Niederschlag unterscheiden. Überschüssiges Wasser verlässt den Speicher durch Drainage.",
        "Les précipitations et l’évapotranspiration réelle (AET) sont des flux journaliers. WR est la réserve d’eau stockée. La neige peut décaler l’apport d’eau au sol. Le surplus quitte la réserve par drainage.",
    ))

caption_growth = L(
    "Gemeinsame Zeitachse für LAI, Biomasse und dBM. Gestrichelte Linie: Wachstumsstart (GTS > 200 °C·d); gepunktete Linien: Schnitte. Beim dBM: dünn = Tageswert, kräftig = gleitender 7-Tage-Mittelwert. Negative Änderungen und Schnitttage werden beim dBM als 0 dargestellt.",
    "Même axe temporel pour le LAI, la biomasse et le dBM. Trait discontinu : début de croissance (GTS > 200 °C·j) ; traits pointillés : coupes. Pour le dBM : trait fin = valeur journalière, trait épais = moyenne mobile sur 7 jours. Les variations négatives et les jours de coupe sont affichés à 0 pour le dBM.",
)

if mode.startswith("1"):
    st.plotly_chart(plot_growth_overview([(station_name, active, GREEN)], day, ()), use_container_width=True)
    st.caption(caption_growth)
    with st.expander(L("Was steuert das Wachstum?", "Quels facteurs déterminent la croissance ?")):
        st.write(L(
            f"Temperaturfaktor fT : **{row['fT']:.2f}** · Wasserfaktor fW : **{row['fW']:.2f}** · Strahlungsfaktor fPAR : **{row['fPAR']:.2f}** · Saisonfaktor SEA : **{row['SEA']:.2f}**",
            f"Facteur température fT : **{row['fT']:.2f}** · Facteur eau fW : **{row['fW']:.2f}** · Facteur rayonnement fPAR : **{row['fPAR']:.2f}** · Facteur saisonnier SEA : **{row['SEA']:.2f}**",
        ))
        st.latex(r"PGRO = PAR_i \cdot RUE_{\max} \cdot (1-e^{-0.6\,LAI}) \cdot 10")
        st.latex(r"GRO = PGRO \cdot ENV \cdot SEA")
        st.caption(L(
            "PGRO beschreibt das strahlungsabhängige Wachstumspotenzial. GRO enthält zusätzlich Umwelt- und Saisonfaktoren. Die stehende Biomasse wird auch durch Verluste und Ernte beeinflusst.",
            "PGRO représente le potentiel de croissance lié au rayonnement. GRO intègre aussi les facteurs environnementaux et saisonniers. La biomasse sur pied dépend également des pertes et des récoltes.",
        ))
        st.latex(r"ENV = NI \cdot f(PAR) \cdot f(T) \cdot f(W)")
        st.caption(L("Die ungeschnittene Wiese ist ein vereinfachtes Lernexperiment, kein realistisches Bewirtschaftungsregime.",
                     "La prairie non fauchée constitue une expérience didactique simplifiée et non un mode d’exploitation réaliste."))
else:
    assert reference is not None
    baseline_name = L("Referenz", "Référence")
    series = [(L("Experiment", "Expérience"), active, GREEN), (baseline_name, reference, BLUE)]
    active_y = active["hvBM"].iloc[-1] / 1000
    ref_y = reference["hvBM"].iloc[-1] / 1000
    m1, m2, m3 = st.columns(3)
    m1.metric(L("Ertrag Experiment", "Rendement de l’expérience") + L(" bisher", " provisoire") * partial_year,
              f"{active_y:.2f} {UNIT_YIELD}")
    m2.metric(L("Ertrag Referenz", "Rendement de référence") + L(" bisher", " provisoire") * partial_year,
              f"{ref_y:.2f} {UNIT_YIELD}")
    m3.metric(L("Differenz", "Différence"), f"{active_y-ref_y:+.2f} {UNIT_YIELD}")
    if mode.startswith("2"):
        st.caption(L(
            "Referenz: 25.05., 20.07., 10.09., Schnitthöhe 5 cm. Nur bereits vergangene Schnitttermine werden berücksichtigt. Das Regime ist nicht standortspezifisch optimiert.",
            "Référence : 25.05., 20.07. et 10.09., hauteur de coupe 5 cm. Seules les coupes déjà possibles sont prises en compte. Le régime n’est pas optimisé pour le site.",
        ))
    else:
        st.caption(L("Referenz und Experiment verwenden dieselben Schnittdaten, Schnitthöhe und Vegetation.",
                     "La référence et l’expérience utilisent les mêmes dates de coupe, la même hauteur et la même végétation."))

    if mode.startswith("2") and len(cuts) == 0:
        st.info(L(
            "Ohne Schnitt zeigt das Experiment die ungestörte Bestandesentwicklung. Die Referenz wird dreimal geschnitten. Der Ertrag des Experiments ist dann 0, obwohl Biomasse vorhanden ist. Seneszenz und Abszission bleiben wirksam.",
            "Sans coupe, l’expérience montre le développement non perturbé de la prairie. La référence est fauchée trois fois. Le rendement récolté de l’expérience vaut alors 0 malgré la biomasse sur pied. Sénescence et abscission restent actives.",
        ))
    cuts_for_plot = tuple(sorted(set(cuts).union(default_days)))
    st.plotly_chart(plot_growth_overview(series, day, cuts_for_plot), use_container_width=True)
    st.caption(caption_growth)
    st.plotly_chart(plot_series(series, "hvBM", L("Kumulierter Schnittertrag", "Rendement cumulé des coupes"),
                                UNIT_YIELD, day, cuts_for_plot, .001, 290, show_growth_start=False),
                    use_container_width=True)
    skipped = [d for d in cuts if int(active.iloc[d-1]["CUT"]) == 0]
    if skipped:
        st.warning(L("Nicht ausgeführte Schnitte (z. B. wegen Schnee): ",
                     "Coupes non réalisées (p. ex. en raison de la neige) : ") +
                   ", ".join(active.iloc[d-1]["date"].strftime("%d.%m.") for d in skipped))
    tab_a, tab_b = st.tabs([L("Schnitterträge", "Rendements des coupes"),
                             L("Modellprozesse", "Processus du modèle")])
    with tab_a:
        table = harvest_table(active)
        if table.empty:
            st.info(L("Es wurden bisher keine Schnitte ausgeführt.", "Aucune coupe n’a encore été effectuée."))
        else:
            st.dataframe(table, hide_index=True, use_container_width=True)
        if guided:
            st.caption(L("Schnitterträge der Referenz", "Rendements des coupes de référence"))
            st.dataframe(harvest_table(reference), hide_index=True, use_container_width=True)
    with tab_b:
        a, b, c = st.columns(3)
        a.metric(L("Temperaturfaktor fT", "Facteur température fT"), f"{row['fT']:.2f}")
        b.metric(L("Wasserfaktor fW", "Facteur eau fW"), f"{row['fW']:.2f}")
        c.metric(L("Saisonfaktor SEA", "Facteur saisonnier SEA"), f"{row['SEA']:.2f}")
        st.caption(L("Umweltbedingungen und Seneszenz verändern die Biomasse. Wachstum und Ertrag sind nicht identisch.",
                     "Les conditions environnementales et la sénescence modifient la biomasse. Croissance et rendement ne sont pas identiques."))

with st.expander(L("Modellannahmen und Grenzen", "Hypothèses et limites du modèle")):
    st.markdown(L(f"""
- **Standort :** {station_name}. Nur die Meteodaten des Standorts werden ausgetauscht.
- **Vegetation :** 50 % funktionelle Gruppe A, 50 % Gruppe B (intensiver Grasbestand).
- **Parameter :** NI = {NI_FIXED:.1f}, WHC = {WHC_FIXED:.0f} mm an allen Standorten.
- **Modell :** ModVege mit täglichem Wachstum, Seneszenz, Wasserhaushalt und vereinfachtem Schneemodell.
- **Bodenwasser :** WR ist der modellierte Speicher [mm], keine volumetrische Bodenfeuchte. Zufluss umfasst Regen und Schneeschmelze; Überschüsse werden drainiert.
- **Schnitte :** Feste Kalendertage; die Entnahme hängt von der Restbiomasse und Schnitthöhe ab.
- **Biomasse :** Angezeigt in {UNIT_YIELD}; LAI in m²/m². Die dargestellte Wiesenhöhe ist schematisch.
- **Modellgrenzen :** Keine explizite Wurzel- oder Reservestoffsimulation. Keine standortspezifische Kalibrierung und keine lokalen Ertragsprognosen.
- **Wachstumsstart :** GTS > 200 °C·d mit gewichteten positiven Temperaturen und Ausschluss von Tagen mit Schnee.
- **GTS / ST :** GTS bestimmt den Wachstumsbeginn, ST beeinflusst über SEA die Saisonentwicklung.
- **Meteodaten :** Historische und laufende Jahre. Das laufende Jahr wird nur bis zum letzten vorhandenen Tag simuliert.
- **Referenz :** Drei feste Schnittdaten, nicht standortspezifisch optimiert. Im Temperaturszenario bleibt ETc unverändert.
""", f"""
- **Site :** {station_name}. Seules les données météorologiques du site sélectionné changent.
- **Végétation :** 50 % groupe fonctionnel A et 50 % groupe B (prairie intensive).
- **Paramètres :** NI = {NI_FIXED:.1f}, WHC = {WHC_FIXED:.0f} mm pour tous les sites.
- **Modèle :** ModVege avec croissance journalière, sénescence, bilan hydrique et modèle simplifié de neige.
- **Eau du sol :** WR est la réserve en eau simulée [mm], pas la teneur en eau volumique. L’apport comprend la pluie et la fonte des neiges ; l’excédent est drainé.
- **Coupes :** Dates calendaires fixes ; la biomasse récoltée dépend de la biomasse résiduelle et de la hauteur de coupe.
- **Biomasse :** Exprimée en {UNIT_YIELD} ; LAI en m²/m². La hauteur de la prairie illustrée est schématique.
- **Limites :** Pas de simulation explicite des racines ni des réserves. Pas d’étalonnage spécifique aux sites ni de prévision locale des rendements.
- **Début de croissance :** GTS > 200 °C·j, sur la base des températures positives pondérées, hors jours avec neige.
- **GTS / ST :** GTS fixe le début de croissance ; ST agit sur le développement saisonnier par SEA.
- **Données météo :** Années historiques et en cours. Une année en cours n’est simulée que jusqu’au dernier jour disponible.
- **Référence :** Trois coupes fixes non optimisées pour le site. Dans le scénario de température, l’ETc reste inchangée.
"""))
