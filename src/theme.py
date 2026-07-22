"""
Design system for the Punjab Development dashboard.

Identity: "development report / data journalism" — warm paper surfaces, serif
display type, restrained earth-tone palette. Inspired by FT visual journalism,
Our World in Data, and UNDP Human Development Reports rather than SaaS defaults.

Single source of truth for every color, font, CSS rule, and chart template.
All modules import from here; no hex codes should live anywhere else.
"""

import streamlit as st
import matplotlib as mpl
import plotly.graph_objects as go
import plotly.io as pio

# ─── Palette ────────────────────────────────────────────────────────────────
PAPER = "#F6F1E7"       # app background (warm paper)
CARD = "#FDFBF6"        # card surface
LINE = "#E4DBC8"        # hairline borders
INK = "#211D16"         # primary text
INK2 = "#5D5546"        # secondary text
MUTED = "#93897A"       # captions, axis labels

SOUTH = "#BC4B26"       # Southern Punjab accent — terracotta, used EVERYWHERE
SOUTH_TINT = "#F4E0D5"
TEAL = "#175F6B"        # Rest of Punjab / default data series — deep teal
TEAL_TINT = "#DBE9E7"
OCHRE = "#C89A3D"       # tertiary accent
GREEN = "#4C7A3F"       # positive deltas
CLAY = "#A85E3B"        # quaternary
SLATE = "#44524E"       # quinary

# Categorical order (fixed; never cycled)
CATEGORICAL = [TEAL, SOUTH, OCHRE, SLATE, CLAY, GREEN, "#7A6A94", "#3E7C8F"]

# Sequential ramp (teal, light -> dark) for choropleths / magnitude
SEQ_TEAL = ["#EAF0ED", "#CBDDD8", "#A6C8C1", "#7EB0A8", "#579990", "#328179",
            "#1D6A70", "#175F6B", "#104A54", "#0B3840"]

# Earth-tone division colors (identity, fixed assignment)
DIVISION_COLORS = {
    "Multan": SOUTH, "Bahawalpur": OCHRE, "DG Khan": CLAY,
    "Lahore": TEAL, "Faisalabad": "#3E7C8F", "Rawalpindi": SLATE,
    "Gujranwala": "#6C8A4A", "Sargodha": "#A76B2E", "Sahiwal": "#67693E",
}

SIDEBAR_BG = "#211B12"


# ─── CSS ────────────────────────────────────────────────────────────────────
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&display=swap');

html, body, p, span, div, label {{ font-family: 'Inter', sans-serif; }}
.material-symbols-rounded, .material-symbols-outlined {{
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', sans-serif !important;
}}

.stApp {{ background: {PAPER}; }}
.main .block-container {{ padding: 1.2rem 2.2rem 3rem; max-width: 1280px; }}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ background: transparent !important; }}

/* ── Typography ─────────────────────────────────────── */
h1, h2, h3, h4 {{ font-family: 'Source Serif 4', Georgia, serif !important; }}
h1 {{ color: {INK} !important; font-weight: 700 !important; letter-spacing: -0.01em; font-size: 2rem !important; }}
h2 {{ color: {INK} !important; font-weight: 700 !important; }}
h3 {{ color: {INK} !important; font-weight: 600 !important; font-size: 1.15rem !important; }}
.main p, .main li {{ color: {INK2}; }}

/* ── Sidebar ────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG};
    border-right: 1px solid rgba(255,255,255,0.06);
}}
section[data-testid="stSidebar"] * {{ color: #D9D2C3 !important; }}
section[data-testid="stSidebar"] .stRadio label {{
    padding: 0.45rem 0.75rem;
    border-left: 3px solid transparent;
    border-radius: 0;
    transition: all 0.15s ease;
    margin-bottom: 1px;
    font-size: 0.92rem;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(246,241,231,0.06);
}}
section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[aria-checked="true"] {{
    border-left: 3px solid {SOUTH};
    background: rgba(188,75,38,0.14);
    font-weight: 600;
}}

/* ── Masthead (page header) ─────────────────────────── */
.masthead {{
    background: transparent;
    border-top: 3px solid {INK};
    border-bottom: 1px solid {LINE};
    padding: 1.1rem 0 1.3rem;
    margin-bottom: 1.6rem;
    position: relative;
}}
.masthead::before {{
    content: '';
    position: absolute;
    top: 4px; left: 0; right: 0;
    height: 1px;
    background: {INK};
}}
.masthead .kicker {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: {SOUTH}; margin-bottom: 0.5rem;
}}
.masthead h1 {{
    font-size: 1.9rem !important; line-height: 1.15; margin: 0 0 0.5rem;
}}
.masthead .standfirst {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.02rem; color: {INK2}; max-width: 720px; line-height: 1.5;
}}
.masthead .badge {{
    display: inline-block;
    border: 1px solid {LINE};
    background: {CARD};
    color: {INK2};
    padding: 2px 10px;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-right: 6px; margin-top: 10px;
}}

/* ── Stat cards ─────────────────────────────────────── */
.metric-card {{
    background: {CARD};
    border: 1px solid {LINE};
    border-left: 3px solid {TEAL};
    padding: 0.9rem 1.1rem 0.85rem;
    transition: border-color 0.15s ease;
    height: 100%;
}}
.metric-card:hover {{ border-left-width: 5px; }}
.metric-card.blue {{ border-left-color: {TEAL}; }}
.metric-card.teal {{ border-left-color: {TEAL}; }}
.metric-card.red {{ border-left-color: {SOUTH}; }}
.metric-card.green {{ border-left-color: {GREEN}; }}
.metric-card.amber {{ border-left-color: {OCHRE}; }}
.metric-card.purple {{ border-left-color: {SLATE}; }}

.metric-label {{
    font-size: 0.7rem; font-weight: 600; color: {MUTED};
    text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 0.35rem;
}}
.metric-value {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.75rem; font-weight: 700; color: {INK};
    line-height: 1.05; margin-bottom: 0.3rem;
}}
.metric-delta {{
    font-size: 0.76rem; font-weight: 600; display: inline-block;
}}
.metric-delta.negative {{ color: {SOUTH}; }}
.metric-delta.positive {{ color: {GREEN}; }}
.metric-delta.neutral  {{ color: {OCHRE}; }}

/* ── Section cards ──────────────────────────────────── */
/* Streamlit auto-closes unwrapped divs, so opener/closer markdown calls
   yield empty divs — collapse those instead of showing hollow boxes. */
.section-card {{
    background: {CARD};
    border: 1px solid {LINE};
    padding: 1.4rem 1.5rem;
    margin-bottom: 1.1rem;
}}
.section-card:empty {{ display: none; }}
.section-card h3 {{
    margin-top: 0 !important;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid {LINE};
    margin-bottom: 1rem;
}}

/* ── Tabs: editorial underline style ────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 1.4rem;
    background: transparent;
    border-bottom: 1px solid {LINE};
    border-radius: 0;
    padding: 0;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 0;
    padding: 8px 2px 10px;
    font-weight: 600;
    font-size: 0.9rem;
    background: transparent;
    border: none;
    color: {MUTED};
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: transparent !important;
    color: {INK} !important;
    box-shadow: inset 0 -2px 0 {SOUTH};
}}
.stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* ── Data frames / inputs ───────────────────────────── */
.stDataFrame {{ border: 1px solid {LINE}; }}
.stSelectbox > div > div, .stSlider > div {{ border-radius: 2px; }}

hr {{
    border: none; height: 1px;
    background: {LINE};
    margin: 1.4rem 0;
}}

/* ── Margin note (insight) ──────────────────────────── */
.insight-box {{
    background: {CARD};
    border-left: 3px solid {SOUTH};
    border-top: 1px solid {LINE};
    border-right: 1px solid {LINE};
    border-bottom: 1px solid {LINE};
    padding: 0.85rem 1.1rem;
    margin: 0.9rem 0;
    font-size: 0.92rem;
    color: {INK2};
}}
.insight-box strong {{ color: {INK}; }}

/* ── About cards ────────────────────────────────────── */
.about-card {{
    background: {CARD};
    border: 1px solid {LINE};
    padding: 1.2rem;
    text-align: center;
}}
.about-card .icon {{ font-size: 1.6rem; margin-bottom: 0.4rem; }}
.about-card h4 {{ color: {INK}; margin: 0.4rem 0 0.2rem; }}
.about-card p {{ color: {MUTED}; font-size: 0.82rem; }}

/* ── Region tag ─────────────────────────────────────── */
.region-tag {{
    padding: 3px 12px; font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.05em; text-transform: uppercase;
}}
.region-tag.south {{ background: {SOUTH_TINT}; color: {SOUTH}; }}
.region-tag.rest  {{ background: {TEAL_TINT};  color: {TEAL}; }}
</style>
"""


def inject_theme():
    """Inject the global CSS. Call once, immediately after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def masthead(kicker, title, standfirst="", badges=None):
    """Editorial page header: kicker, serif headline, standfirst, source badges."""
    badge_html = "".join(f'<span class="badge">{b}</span>' for b in (badges or []))
    st.markdown(f"""
    <div class="masthead">
        <div class="kicker">{kicker}</div>
        <h1>{title}</h1>
        <div class="standfirst">{standfirst}</div>
        <div>{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)


def register_plotly_template():
    """Register and set the editorial plotly template as default."""
    pio.templates["punjab_report"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Inter, sans-serif", color=INK2, size=13),
            title_font=dict(family="Source Serif 4, Georgia, serif", color=INK, size=17),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=CATEGORICAL,
            xaxis=dict(gridcolor=LINE, linecolor=LINE, zerolinecolor=LINE,
                       tickcolor=MUTED, tickfont=dict(color=MUTED)),
            yaxis=dict(gridcolor=LINE, linecolor=LINE, zerolinecolor=LINE,
                       tickcolor=MUTED, tickfont=dict(color=MUTED)),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)),
            hoverlabel=dict(bgcolor=CARD, bordercolor=LINE,
                            font=dict(family="Inter, sans-serif", color=INK)),
            margin=dict(t=48, r=16, b=48, l=56),
        )
    )
    pio.templates.default = "punjab_report"


def apply_matplotlib_theme():
    """Set matplotlib rcParams to match the editorial identity."""
    mpl.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": CARD,
        "savefig.facecolor": "none",
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "grid.color": LINE,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK2,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Arial", "DejaVu Sans"],
        "legend.frameon": False,
    })
