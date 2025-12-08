import streamlit as st
from streamlit_float import *
from PIL import Image

import geopandas as gpd
import pandas as pd
from shapely import wkt

from datetime import datetime, timedelta

# import folium # REMOVED: Not needed for line graph
# import leafmap.foliumap as leafmap # REMOVED: Not needed for line graph
# from branca.element import Template, MacroElement, Element # REMOVED: Not needed for line graph
import numpy as np
import json
import plotly.express as px # ADDED: For line graph

# PAGE CONFIG
icon = Image.open("icon.png")
st.set_page_config(page_title="Denvue Dashboard", layout="wide", page_icon=icon)
st.logo(image="logo.png", size="large")
float_init()

# [STREAMLIT] ADJUST PADDING
padding = """
<style>
.block-container {
    padding-top: 0rem;
    padding-bottom: 2.5rem;
}
[class="stVerticalBlock st-emotion-cache-tn0cau e1wguzas3"] {
    gap: 0.5rem;
}
</style>
"""
st.markdown(padding, unsafe_allow_html=True)

# [STREAMLIT] TOOLBAR BACKGROUND
toolbar_bg = """
<style>
div[data-testid="stToolbar"] {
    background-color: #698C6E;
}
</style>
"""
st.markdown(toolbar_bg, unsafe_allow_html=True)

# [STREAMLIT] HIDE MENU
hide_menu = """
<style>
div[data-testid="stToolbarActions"] {
    display: none;
}
span[data-testid="stMainMenu"] {
    display: none;
}
div[data-testid="stDecoration"] {
    visibility: hidden;
    height: 0%;
    position: fixed;
}
div[data-testid="stStatusWidget"] {
    visibility: hidden;
    height: 0%;
    position: fixed;
}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

# [STREAMLIT] ADJUST HEADER
header = """
<style>
[data-testid="stHeader"] {
    z-index: 1;
}
</style>
    """
st.markdown(header, unsafe_allow_html=True)

# [STREAMLIT] HEADER COLOR
header_color = """
<style>
div[data-testid="stHeadingWithActionElements"] {
    color: #234528;
}
</style>
"""
st.markdown(header_color, unsafe_allow_html=True)

# [STREAMLIT] REMOVE HEADER ACTION ELEMENT
header_action = """
<style>
[data-testid="stHeaderActionElements"] {
    display: none;
}
</style>
"""
st.markdown(header_action, unsafe_allow_html=True)

# [STREAMLIT] BOTTOM ALIGN CONTENT
bottom_align = """
<style>
.stColumn.st-emotion-cache-1wpb1x8.e1wguzas2 > .stVerticalBlock.st-emotion-cache-wfksaw.e1wguzas3 {
    display: flex;
    justify-content: flex-end;
    align-items: flex-end;
}
</style>
"""
st.markdown(bottom_align, unsafe_allow_html=True)

# [LEAFMAP] ADD MAP BORDER - CSS still included, but map is removed
map_border_style = """
<style>
iframe {
    border: 1px solid #E0E0E0 !important;
    box-sizing: border-box;
    border-radius: 0.5rem;
}
</style>
"""
st.markdown(map_border_style, unsafe_allow_html=True)

# [STREAMLIT] METRIC VALUE SIZE
metric_value = """
<style>
div[data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 800;
}
</style>
"""
st.markdown(metric_value, unsafe_allow_html=True)

# [STREAMLIT] METRIC STYLE
metric_style = """
<style>
div[data-testid="stMetric"] {
    color: #234528,
    background: white;
    border: 1px solid #E0E0E0;
    border-radius: 0.5rem;
    padding-top: 0.6rem;
    padding-bottom: 0.6rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
</style>
"""
st.markdown(metric_style, unsafe_allow_html=True)

# LOAD DATA
@st.cache_data
def load_data():
    cdo_barangays = pd.read_csv("cdo_barangays.csv")
    cdo_barangays["Geometry"] = cdo_barangays["Geometry"].apply(wkt.loads)
    gdf_barangays = gpd.GeoDataFrame(cdo_barangays, geometry="Geometry", crs="EPSG:4326")

    forecasts = pd.read_csv("all_models_forecasts.csv")
    forecasts["Date"] = pd.to_datetime(forecasts["Date"])
    merged = forecasts.merge(gdf_barangays.drop(columns=['Geometry']), on="Barangay", how="left") # Dropping geometry for merge
    merged_all = merged # No longer a GeoDataFrame, just a DataFrame
    
    return gdf_barangays, merged_all

gdf_barangays, merged_all = load_data()

# SESSION STATE
available_years = sorted(merged_all["Year"].unique()) if "Year" in merged_all.columns else sorted(merged_all["Date"].dt.year.unique())
if "Year" not in merged_all.columns:
    merged_all["Year"] = merged_all["Date"].dt.year
merged_all["Week"] = merged_all["Date"].dt.isocalendar().week.astype(int)

model_list = merged_all["Model"].unique()
default_model = "random_forest" if "random_forest" in model_list else model_list[0]
default_year = 2025 if 2025 in available_years else available_years[-1]
barangay_list = sorted(merged_all["Barangay"].unique())
default_barangay = barangay_list[0] if barangay_list else ""

st.session_state.setdefault("selected_model", default_model)
st.session_state.setdefault("selected_year", default_year)
st.session_state.setdefault("selected_barangay", default_barangay) # ADDED: New session state for barangay

year_weeks = sorted(merged_all[merged_all["Year"] == st.session_state.selected_year]["Week"].unique())
if "selected_week" not in st.session_state:
    st.session_state.selected_week = min(year_weeks) if year_weeks else 1

# FILTER DATA
# UPDATED: Filtered data now only contains one barangay for a given year/model
filtered_data_all_weeks = merged_all[
    (merged_all["Model"] == st.session_state.selected_model)
    & (merged_all["Year"] == st.session_state.selected_year)
    & (merged_all["Barangay"] == st.session_state.selected_barangay) # ADDED: Barangay filter
].copy()

filtered_data_all_weeks["Forecast_Cases"] = pd.to_numeric(filtered_data_all_weeks["Forecast_Cases"], errors="coerce").fillna(0)
filtered_data_all_weeks["Confidence"] = (filtered_data_all_weeks["Confidence"] * 100).round(1).astype(str) + "%"

# Data for the specific selected week (for metrics/table in col2)
filtered_data_current_week = filtered_data_all_weeks[
    (filtered_data_all_weeks["Week"] == st.session_state.selected_week)
].copy()

# DASHBOARD LAYOUT
col1, col2 = st.columns(2)

# LEFT COLUMN
with col1:
    with st.container():
        # GET THE ACTUAL DATE RANGE
        # UPDATED: Date range now reflects the full selected year for the chart, or just the current week for the title
        start_date = datetime.fromisocalendar(st.session_state.selected_year, st.session_state.selected_week, 1)  # Monday
        end_date = datetime.fromisocalendar(st.session_state.selected_year, st.session_state.selected_week, 7)    # Sunday
        date_range_str = f"Current Week: {start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')}"

        # CHART SECTION
        title, date = st.columns(2)
        with title:
            # UPDATED: Title to reflect time series
            st.write(f"#### **{st.session_state.selected_barangay} Dengue Forecast Time Series**")
        with date:
            st.markdown(
                f"""
                <div style='display: flex; height: 100%; align-items: flex-end; justify-content: flex-end;'>
                    <h6 style='margin: 0;'>{date_range_str}</h6>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Line Chart using Plotly
        if not filtered_data_all_weeks.empty:
            fig = px.line(
                filtered_data_all_weeks,
                x='Week',
                y='Forecast_Cases',
                title=f'Weekly Forecast Cases for {st.session_state.selected_barangay} ({st.session_state.selected_year})',
                labels={'Forecast_Cases': 'Forecast Cases', 'Week': 'ISO Week Number'},
                markers=True,
                height=450
            )
            
            # Highlight the currently selected week
            fig.add_vline(
                x=st.session_state.selected_week, 
                line_width=2, 
                line_dash="dash", 
                line_color="red", 
                annotation_text=f"Selected Week {st.session_state.selected_week}",
                annotation_position="bottom right"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No forecast data available for {st.session_state.selected_barangay} in {st.session_state.selected_year} using the {st.session_state.selected_model} model.")
            st.markdown(f'<div style="height: 450px;"></div>', unsafe_allow_html=True) # Maintain height

        # FILTERS CONTROLS
        filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2]) # UPDATED: Added a third column for Barangay
        
        with filter_col1:
            selected_year = st.selectbox(
                "Select Year",
                available_years,
                index=available_years.index(st.session_state.selected_year)
            )

        with filter_col2:
            selected_barangay = st.selectbox( # ADDED: Barangay selection
                "Select Barangay",
                barangay_list,
                index=barangay_list.index(st.session_state.selected_barangay)
            )
        
        with filter_col3:
            available_weeks = sorted(merged_all[merged_all["Year"] == selected_year]["Week"].unique())
            
            default_week = st.session_state.selected_week
            if default_week not in available_weeks:
                default_week = available_weeks[0] if available_weeks else 1
            
            selected_week = st.select_slider(
                "Select Week",
                available_weeks,
                value=default_week
            )
        
        # UPDATE SESSION STATE ON CHANGE
        if selected_year != st.session_state.selected_year or selected_week != st.session_state.selected_week or selected_barangay != st.session_state.selected_barangay:
            st.session_state.selected_year = selected_year
            st.session_state.selected_week = selected_week
            st.session_state.selected_barangay = selected_barangay # UPDATED: Update session state for barangay
            st.rerun()

        # MAP DESCRIPTION
        map_description = st.container(border=True)
        # UPDATED: Description reflects the line graph
        map_description.write("""
        The chart visualizes the **forecasted dengue cases** over the selected year for the **selected barangay** using the chosen prediction model.
        This time series analysis allows for the detection of temporal patterns, such as seasonal peaks or unusual case surges for a specific location.
        The red dashed line marks the currently selected week, which determines the summary metrics and risk level displayed in the right column.
        """)
            
# RIGHT COLUMN
with col2:
    # METRICS SECTION
    st.write("#### **Summary Metrics**")
    
    # UPDATED: Metrics now use the filtered data for the current week AND barangay
    if not filtered_data_current_week.empty:
        current_case_data = filtered_data_current_week.iloc[0]
        current_cases = current_case_data['Forecast_Cases']
        current_confidence = current_case_data['Confidence']
        current_risk = current_case_data['Risk_Level']
    else:
        # Fallback if no data for the specific week/barangay
        current_cases = 0
        current_confidence = "N/A"
        current_risk = "N/A"

    m1, m2, m3 = st.columns(3)
    m1.metric("Forecasted Cases (This Week)", f"{current_cases}")
    m2.metric("Confidence Level", current_confidence)
    m3.metric("Risk Level", current_risk) # UPDATED: Changed the metric to "Risk Level"
    
    # TABLE SECTION
    st.write("#### **Risk Ranking by Barangay**")
    
    # Table logic now uses all barangays for the selected year/week
    # Note: I need to re-filter the data for ALL BARANGAYS for the selected week to show a *ranking*
    filtered_data_ranking = merged_all[
        (merged_all["Model"] == st.session_state.selected_model)
        & (merged_all["Year"] == st.session_state.selected_year)
        & (merged_all["Week"] == st.session_state.selected_week)
    ].copy()
    filtered_data_ranking["Forecast_Cases"] = pd.to_numeric(filtered_data_ranking["Forecast_Cases"], errors="coerce").fillna(0)
    filtered_data_ranking["Confidence"] = (filtered_data_ranking["Confidence"] * 100).round(1).astype(str) + "%"

    risk_colors = { # Re-defined here since it was removed with map code
        "Low": "#E9F3F2",
        "Moderate": "#F3B705",
        "High": "#F17404",
        "Critical": "#D9042C"
    }
    
    table_df = filtered_data_ranking[['Barangay', 'Forecast_Cases', 'Confidence', 'Risk_Level']].copy()
    table_df = table_df.rename(columns={"Forecast_Cases": "Forecast Cases", "Confidence": "Confidence", "Risk_Level": "Risk Level"})
    table_df["Forecast Cases"] = table_df["Forecast Cases"].astype(str)
    
    risk_order = ["Low", "Moderate", "High", "Critical"]
    table_df["Risk Level"] = pd.Categorical(table_df["Risk Level"], categories=risk_order, ordered=True)
    table_df = table_df.sort_values(by=['Risk Level', 'Forecast Cases'], ascending=[False, False]).reset_index(drop=True)

    def color_forecast(val):
        if pd.isna(val):
            return 'background-color: #E9F3F2; color: black'
        color = risk_colors.get(val, "#E9F3F2")
        if color == "#E9F3F2" or color == "#F3B705":
            text_color = "black"
        else:
            text_color = "white"
        return f'background-color: {color}; color: {text_color}; font-weight: bold'
    
    styled_table = table_df.style.applymap(color_forecast, subset=['Risk Level'])
    st.dataframe(styled_table, width='stretch', height=380)

    # RISK LEVEL DESCRIPTION
    risk_description = st.container(border=True)
    risk_description.write("""
    Each barangay’s dengue forecast is evaluated relative to its own historical trends, providing a context-aware risk measure.
    “Low Risk” indicates forecasts within the bottom 25th percentile of past values, while “Moderate Risk” covers the 25th–50th percentile range.
    “High Risk” spans the 50th–75th percentile, and “Critical Risk” exceeds the 75th percentile, signaling an unusual spike.
    This adaptive scale ensures that even subtle local increases are detected early for timely interventions.
    """)

# MODEL OPTIONS
@st.dialog("Model Options")
def open_model_options():
    model_name_map = {
        "linear_regression": "Linear Regression",
        "lstm": "LSTM",
        "random_forest": "Random Forest (Recommended)",
        "xgboost": "XGBoost",
    }
    model_display = [model_name_map[m] for m in merged_all["Model"].unique() if m in model_name_map]
    model_to_key = {v: k for k, v in model_name_map.items()}

    selected_model_display = st.selectbox(
        "Select Model",
        model_display,
        index=model_display.index(model_name_map[st.session_state.selected_model]),
    )
    selected_model = model_to_key[selected_model_display]

    # UPDATE SESSION STATE ON CHANGE
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.rerun()

button_container = st.container()
with button_container:
    if st.button("⚙️"):
        open_model_options()
    
button_css = float_css_helper(width="3rem", height="3rem", right="0.8rem", top="0.6rem", transition=0)
button_container.float(button_css)
