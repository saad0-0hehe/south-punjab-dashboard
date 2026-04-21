"""
Choropleth Map Module
South Punjab Development Dashboard

Renders interactive Plotly choropleth maps of Punjab districts.
"""

import plotly.express as px
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Mapping: CSV district name -> GeoJSON district name
CSV_TO_GEO_NAME = {
    "DG Khan": "Dera Ghazi Khan",
    "Layyah": "Leiah",
}

# Reverse mapping
GEO_TO_CSV_NAME = {v: k for k, v in CSV_TO_GEO_NAME.items()}

# Indicators where higher = worse (need reversed color scale)
DEPRIVATION_INDICATORS = {"poverty_headcount", "out_of_school_rate", "unemployment_rate", "mpi_score"}

INDICATOR_LABELS = {
    "poverty_headcount": "Poverty Headcount (%)",
    "literacy_rate": "Literacy Rate (%)",
    "out_of_school_rate": "Out of School Rate (%)",
    "internet_access": "Internet Access (%)",
    "immunization_coverage": "Immunization Coverage (%)",
}


def load_punjab_geojson():
    """Load the Punjab-only GeoJSON file."""
    path = os.path.join(DATA_DIR, "punjab_districts.geojson")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_choropleth(df, indicator="poverty_headcount"):
    """
    Render a Plotly choropleth_mapbox map of all 36 Punjab districts.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataset with a 'district' column and the indicator column.
    indicator : str
        Column name to use for the color scale.

    Returns
    -------
    plotly.graph_objects.Figure or None
    """
    geo = load_punjab_geojson()
    if geo is None:
        return None

    # Create a copy with GeoJSON-compatible names
    plot_df = df[["district", indicator]].copy()
    plot_df["geo_name"] = plot_df["district"].map(
        lambda x: CSV_TO_GEO_NAME.get(x, x)
    )

    # Choose color scale direction
    is_deprivation = indicator in DEPRIVATION_INDICATORS
    color_scale = "RdYlGn" if not is_deprivation else "RdYlGn_r"

    label = INDICATOR_LABELS.get(indicator, indicator.replace("_", " ").title())

    fig = px.choropleth_mapbox(
        plot_df,
        geojson=geo,
        locations="geo_name",
        featureidkey="properties.adm2_name",
        color=indicator,
        color_continuous_scale=color_scale,
        hover_name="district",
        hover_data={indicator: ":.1f", "geo_name": False},
        labels={indicator: label},
        mapbox_style="carto-positron",
        zoom=5.3,
        center={"lat": 30.5, "lon": 71.0},
        opacity=0.75,
    )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=500,
        coloraxis_colorbar=dict(
            title=label,
            thickness=15,
            len=0.6,
        ),
    )

    return fig
