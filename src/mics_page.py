"""
MICS Longitudinal Analysis page.

Household-microdata results from three MICS Punjab rounds (2011, 2014, 2017-18):
a household-level HDI externally validated against Naveed & Gordon (2024),
convergence econometrics, within-district inequality, and SHAP deprivation
drivers from the corrected (exogenous-features-only) ML analysis.

Data: data/mics/*.csv
Boundaries: data/punjab_districts.geojson (UN OCHA COD-AB admin2, 36 districts).
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.theme import (
    masthead, SOUTH, TEAL, OCHRE, MUTED, LINE, INK, INK2, CARD, SEQ_TEAL,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MICS_DIR = os.path.join(DATA_DIR, "mics")

# MICS project names -> COD-AB GeoJSON adm2_name
MICS_TO_GEO = {
    "DG Khan": "Dera Ghazi Khan",
    "RY Khan": "Rahim Yar Khan",
    "TT Singh": "Toba Tek Singh",
    "Layyah": "Leiah",
}

SOUTHERN_MICS = ["Rajanpur", "DG Khan", "Muzaffargarh", "Layyah", "Multan",
                 "Lodhran", "Khanewal", "Vehari", "Bahawalpur", "Bahawalnagar",
                 "RY Khan"]

ROUNDS = ["2011", "2014", "2017-18"]
T_YEARS = 6.5  # 2011 -> mid-2017-18

FEATURE_LABELS = {
    "improved_floor": "Improved floor material",
    "has_bank_account": "Bank account ownership",
    "open_defecation": "Open defecation (no toilet)",
    "solid_fuel": "Solid cooking fuel",
    "owns_agri_land": "Agricultural land ownership",
    "child_dependency": "Under-5 dependency ratio",
    "improved_wall": "Improved wall material",
    "is_southern_punjab": "Southern Punjab (region)",
    "improved_sanitation": "Improved sanitation (JMP)",
    "hh_size_val": "Household size",
    "has_electricity": "Electricity access",
    "persons_per_room": "Persons per room (crowding)",
    "improved_roof": "Improved roof material",
    "survey_round": "Survey round (time)",
    "rooms_sleeping": "Rooms used for sleeping",
    "owns_dwelling": "Owns dwelling",
    "treats_water": "Treats drinking water",
    "owns_animals": "Owns livestock",
    "water_time_min": "Water fetching time (minutes)",
    "is_urban": "Urban household",
    "water_on_premises": "Water source on premises",
    "female_hh_head": "Female-headed household",
    "improved_water": "Improved water source (JMP)",
}


def _label(feature):
    if feature.startswith("div_"):
        return f"Division: {feature[4:]}"
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


# ─── Data loading ───────────────────────────────────────────────────────────

@st.cache_data
def _load_hdi():
    df = pd.read_csv(os.path.join(MICS_DIR, "hdi_district_all_rounds.csv"))
    df["year"] = df["year"].astype(str)
    df["region"] = np.where(df["district_name"].isin(SOUTHERN_MICS),
                            "Southern Punjab", "Rest of Punjab")
    return df


@st.cache_data
def _load_gini():
    df = pd.read_csv(os.path.join(MICS_DIR,
                                  "disparity_within_vs_between_districts.csv"))
    df["year"] = df["year"].astype(str)
    return df


@st.cache_data
def _load_beta():
    return pd.read_csv(os.path.join(MICS_DIR, "beta_convergence_results.csv"))


@st.cache_data
def _load_shap():
    return pd.read_csv(os.path.join(
        MICS_DIR, "ml_shap_regional_comparison_corrected.csv"))


@st.cache_data
def _load_geojson():
    path = os.path.join(DATA_DIR, "punjab_districts.geojson")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _weighted_gini(values, weights):
    """Weighted Gini coefficient across district means."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    n = cw[-1]
    num = np.sum(w * v * (cw - w / 2.0))
    mu = np.sum(w * v) / n
    return 2.0 * num / (n * n * mu) - 1.0


def _card(label, value, note="", accent="teal"):
    note_html = f'<div class="metric-delta neutral">{note}</div>' if note else ""
    st.markdown(f"""
    <div class="metric-card {accent}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {note_html}
    </div>
    """, unsafe_allow_html=True)


# ─── Page ───────────────────────────────────────────────────────────────────

def render():
    masthead(
        "Longitudinal Study · MICS Microdata",
        "A Decade of Human Development, Household by Household",
        "Three MICS Punjab rounds — 185,303 households — distilled into a "
        "district HDI, validated against published research, and tracked from "
        "2011 to 2018. The finding: the province converged, and the South did "
        "the catching up.",
        badges=["MICS 2011", "MICS 2014", "MICS 2017-18",
                "185,303 Households", "Validated · Spearman 0.98"],
    )

    hdi = _load_hdi()
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "The Map", "Convergence", "Inequality", "Deprivation Drivers",
        "Validation & Rigor",
    ])

    # ── Tab 1: choropleth + trajectories ────────────────────────────────
    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### District HDI across three survey rounds")
        year = st.radio("Survey round", ROUNDS, index=2, horizontal=True,
                        key="mics_year")
        sub = hdi[hdi["year"] == year].copy()
        sub["geo_name"] = sub["district_name"].map(
            lambda d: MICS_TO_GEO.get(d, d))

        geo = _load_geojson()
        fig = px.choropleth_mapbox(
            sub, geojson=geo, locations="geo_name",
            featureidkey="properties.adm2_name",
            color="hdi_child_hh",
            color_continuous_scale=SEQ_TEAL,
            range_color=(0.20, 0.60),
            hover_name="district_name",
            hover_data={"hdi_child_hh": ":.3f", "geo_name": False},
            labels={"hdi_child_hh": "HDI"},
            mapbox_style="carto-positron",
            zoom=5.3, center={"lat": 30.5, "lon": 71.0}, opacity=0.85,
        )
        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=520,
            coloraxis_colorbar=dict(title="District HDI", thickness=14, len=0.6),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Household HDI (child-bearing households, primary metric), shared "
            "0.20–0.60 scale across rounds so improvement is visible. "
            "Source: MICS Punjab microdata, author's calculations."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### District trajectories")
        picks = st.multiselect(
            "Compare districts",
            sorted(hdi["district_name"].unique()),
            default=["Rajanpur", "DG Khan", "Muzaffargarh", "Lahore", "Rawalpindi"],
            key="mics_traj",
        )
        traj = go.Figure()
        prov = hdi.groupby("year")["hdi_child_hh"].mean().reindex(ROUNDS)
        traj.add_trace(go.Scatter(
            x=ROUNDS, y=prov.values, name="Punjab mean (36 districts)",
            mode="lines", line=dict(color=MUTED, width=2, dash="dash"),
        ))
        for d in picks:
            dd = hdi[hdi["district_name"] == d].set_index("year")\
                    .reindex(ROUNDS)["hdi_child_hh"]
            is_south = d in SOUTHERN_MICS
            traj.add_trace(go.Scatter(
                x=ROUNDS, y=dd.values, name=d, mode="lines+markers",
                line=dict(color=SOUTH if is_south else TEAL, width=2.5),
                marker=dict(size=9, line=dict(color="#FFFFFF", width=1)),
                opacity=0.9,
            ))
        traj.update_layout(
            height=430, yaxis_title="District HDI (Child-HH)",
            xaxis=dict(title="MICS round", type="category"),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(traj, use_container_width=True)
        st.markdown(
            '<div class="insight-box">Terracotta lines are <strong>Southern '
            'Punjab</strong> districts; teal lines the rest of the province. '
            'The southern trajectories are steeper — the statistical case for '
            'that is in the Convergence tab.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 2: β-convergence ────────────────────────────────────────────
    with tab2:
        beta = _load_beta()
        row = beta[(beta["target"] == "hdi_child_hh") &
                   (beta["spec"] == "OLS_HC1")].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _card("Convergence coefficient β", f"{row['beta']:.3f}",
                  f"HC1 SE {row['se_beta']:.4f} · p = {row['p_beta']:.0e}",
                  accent="red")
        with c2:
            _card("R² (initial level alone)", f"{row['r2']:.2f}",
                  "OLS, 36 districts", accent="teal")
        with c3:
            _card("Convergence speed λ", f"{row['lambda_pct_per_yr']:.1f}%/yr",
                  "Barro–Sala-i-Martin", accent="teal")
        with c4:
            _card("Gap half-life", f"{row['half_life_years']:.0f} years",
                  "at observed speed", accent="amber")

        st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Poorer districts grew faster, 2011 → 2017-18")

        wide = hdi.pivot(index="district_name", columns="year",
                         values="hdi_child_hh")
        growth = np.log(wide["2017-18"] / wide["2011"]) / T_YEARS * 100
        base = wide["2011"]
        south_mask = base.index.isin(SOUTHERN_MICS)

        scat = go.Figure()
        xs = np.linspace(np.log(base.min()) - 0.03, np.log(base.max()) + 0.03, 60)
        b1, b0 = np.polyfit(np.log(base), growth / 100, 1)
        scat.add_trace(go.Scatter(
            x=np.exp(xs), y=(b0 + b1 * xs) * 100, mode="lines",
            name=f"OLS fit (β = {row['beta']:.3f})",
            line=dict(color=INK, width=2, dash="dash"),
        ))
        for mask, name, color in [(~south_mask, "Rest of Punjab (25)", TEAL),
                                  (south_mask, "Southern Punjab (11)", SOUTH)]:
            scat.add_trace(go.Scatter(
                x=base[mask], y=growth[mask], mode="markers", name=name,
                text=base.index[mask], hovertemplate="<b>%{text}</b><br>"
                "2011 HDI %{x:.3f}<br>Growth %{y:.2f}%/yr<extra></extra>",
                marker=dict(size=11, color=color,
                            line=dict(color="#FFFFFF", width=1.5)),
            ))
        scat.update_layout(
            height=480,
            xaxis=dict(title="District HDI in 2011 (log scale)", type="log"),
            yaxis_title="Annualized HDI growth (% per year)",
            legend=dict(orientation="h", y=-0.22),
        )
        st.plotly_chart(scat, use_container_width=True)
        st.markdown(
            f'<div class="insight-box"><strong>Unconditional β-convergence.</strong> '
            f'A district\'s 2011 starting level alone explains '
            f'<strong>{row["r2"]:.0%}</strong> of its subsequent growth. The poorest '
            f'districts (Rajanpur, DG Khan, Muzaffargarh) grew 4.5–5.6% per year '
            f'while Lahore and Rawalpindi grew about 1%. σ-convergence agrees: '
            f'the spread of district HDI fell by roughly a third — so this is '
            f'genuine catching-up, not a statistical artifact.</div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 3: inequality ───────────────────────────────────────────────
    with tab3:
        gini = _load_gini()
        g0 = gini[gini["year"] == "2011"].set_index("district_name")[
            "within_gini_child_hh"]
        g1 = gini[gini["year"] == "2017-18"].set_index("district_name")[
            "within_gini_child_hh"]
        order = g0.sort_values(ascending=True).index
        g0o, g1o = g0.reindex(order), g1.reindex(order)
        fell = int((g1 - g0 < 0).sum())

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Within-district inequality, 2011 vs 2017-18")
        st.caption(
            f"Household-level HDI Gini inside each district. Inequality fell in "
            f"{fell} of 36 districts — fastest where it was highest."
        )

        dumb = go.Figure()
        for d in order:
            dumb.add_trace(go.Scatter(
                x=[g0o[d], g1o[d]], y=[d, d], mode="lines",
                line=dict(color=LINE, width=2), showlegend=False,
                hoverinfo="skip",
            ))
        dumb.add_trace(go.Scatter(
            x=g0o.values, y=list(order), mode="markers", name="2011",
            marker=dict(size=9, color=MUTED),
            hovertemplate="<b>%{y}</b> 2011: %{x:.3f}<extra></extra>",
        ))
        dumb.add_trace(go.Scatter(
            x=g1o.values, y=list(order), mode="markers", name="2017-18",
            marker=dict(size=9, color=TEAL),
            hovertemplate="<b>%{y}</b> 2017-18: %{x:.3f}<extra></extra>",
        ))
        south_rows = [d for d in order if d in SOUTHERN_MICS]
        dumb.add_trace(go.Scatter(
            x=[float(min(g0.min(), g1.min())) - 0.02] * len(south_rows),
            y=south_rows, mode="markers", name="Southern Punjab",
            marker=dict(symbol="square", size=7, color=SOUTH),
            hoverinfo="skip",
        ))
        dumb.update_layout(
            height=820, xaxis_title="Within-district Gini of household HDI",
            legend=dict(orientation="h", y=1.04, x=0),
            yaxis=dict(tickfont=dict(size=11)),
            margin=dict(l=140),
        )
        st.plotly_chart(dumb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Between-district inequality is falling too")
        rows = []
        for yr in ROUNDS:
            sub = hdi[hdi["year"] == yr]
            for col, label in [("hdi_child_hh", "Child-HH HDI (primary)"),
                               ("hdi_all_hh", "All-HH HDI (secondary)")]:
                rows.append(dict(
                    year=yr, metric=label,
                    gini=_weighted_gini(sub[col],
                                        sub["n_households_matched"]),
                ))
        bt = pd.DataFrame(rows)
        btfig = px.line(
            bt, x="year", y="gini", color="metric", markers=True,
            color_discrete_sequence=[TEAL, OCHRE],
            labels={"gini": "Between-district Gini", "year": "MICS round",
                    "metric": ""},
        )
        btfig.update_traces(line=dict(width=2.5), marker=dict(size=10))
        btfig.update_xaxes(type="category")
        btfig.update_layout(height=380, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(btfig, use_container_width=True)
        st.markdown(
            '<div class="insight-box">Both layers of inequality moved the same '
            'way: the <strong>spatial Gini across districts fell by roughly a '
            'third</strong> between 2011 and 2017-18, while within-district '
            'inequality fell in nearly every district. Punjab\'s development '
            'story over this decade is one of narrowing gaps at every level.'
            '</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 4: SHAP drivers ─────────────────────────────────────────────
    with tab4:
        shap_df = _load_shap()
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### What predicts household deprivation?")
        scope = st.radio(
            "Scope", ["Punjab (global)", "Southern Punjab", "Rest of Punjab"],
            horizontal=True, key="mics_shap_scope",
        )
        col = {"Punjab (global)": "mean_abs_shap",
               "Southern Punjab": "shap_southern",
               "Rest of Punjab": "shap_rest"}[scope]
        bar_color = SOUTH if scope == "Southern Punjab" else TEAL

        top = shap_df.nlargest(15, col).copy()
        top["label"] = top["feature"].map(_label)
        top = top.sort_values(col)

        sf = go.Figure(go.Bar(
            x=top[col], y=top["label"], orientation="h",
            marker=dict(color=bar_color),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        sf.update_layout(
            height=520, xaxis_title="Mean |SHAP| (impact on household HDI)",
            margin=dict(l=230),
        )
        st.plotly_chart(sf, use_container_width=True)
        st.markdown(
            '<div class="insight-box"><strong>Corrected, exogenous-features-only '
            'analysis</strong> (XGBoost on 185,303 households, R² ≈ 0.43–0.52). '
            'Features are strictly outside the HDI formula — housing quality, '
            'financial inclusion, WASH, energy. The top five drivers — improved '
            'flooring, bank account ownership, open defecation, solid cooking '
            'fuel, and agricultural land — are identical in Southern Punjab and '
            'the rest of the province; what differs is emphasis (electricity '
            'access matters far more in the South, ranking 8th vs 16th).</div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 5: validation ───────────────────────────────────────────────
    with tab5:
        st.markdown("### Why these numbers can be trusted")
        st.markdown(
            "Every headline finding on this page survived an explicit "
            "validation or robustness check before being reported."
        )
        st.markdown("<div style='height: 0.4rem'></div>", unsafe_allow_html=True)

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            _card("External validation", "ρ = 0.982",
                  "Spearman vs Naveed & Gordon (2024), 36 districts",
                  accent="teal")
        with r1c2:
            _card("Rank concordance", "τ = 0.911",
                  "Kendall; bottom-5 districts match exactly", accent="teal")
        with r1c3:
            _card("Immunization cross-check", "78% vs ~80%",
                  "Our 12–23-month estimate vs PDHS 2017-18", accent="green")

        st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            _card("Honest model fit", "R² 0.43–0.52",
                  "After removing circular features (was 0.97)", accent="red")
        with r2c2:
            _card("Convergence, two ways", "β and σ agree",
                  "Dispersion fell 32% — immune to mean-reversion artifacts",
                  accent="amber")
        with r2c3:
            _card("Robustness", "Top-5 drivers stable",
                  "Survive feature-drop and reweighting checks", accent="purple")

        st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### The checks, in brief")
        st.markdown("""
- **External validation.** District HDI ranks were compared against the
  independently published district HDI of Naveed & Gordon (2024): Spearman
  ρ = 0.982, Kendall τ = 0.911, and the five most-deprived districts
  (Rajanpur, DG Khan, Muzaffargarh, RY Khan, Lodhran) match rank-for-rank.
- **Immunization sanity check.** The health index deliberately evaluates a
  wide 12–59-month window, which yields lower levels than official reports.
  Re-running the identical algorithm on the official 12–23-month band
  reproduces published coverage (78.0% vs ~80% in PDHS 2017-18) — the gap is
  the window, not measurement error.
- **Circularity correction.** An early model version scored R² ≈ 0.97 by
  accidentally using components of the HDI as predictors. It was rebuilt with
  strictly exogenous features; the honest R² of 0.43–0.52 is reported instead.
  The SHAP rankings shown here come exclusively from the corrected model.
- **Convergence robustness.** β-convergence can be inflated by measurement
  error (the Friedman–Quah critique), so σ-convergence is reported alongside:
  the standard deviation of log district HDI fell 32%, confirming genuine
  narrowing. Estimates are robust to weighting districts by sample size.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(
            "Methodology: MICS Punjab household microdata (2011, 2014, 2017-18); "
            "HDI = mean(asset, education, child-health indices) per household; "
            "36-district panel. Full pipeline documentation in the research "
            "repository."
        )
