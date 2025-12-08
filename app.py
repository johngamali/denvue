import streamlit as st
from streamlit_float import *
from PIL import Image

import geopandas as gpd
import pandas as pd
from shapely import wkt

from datetime import datetime, timedelta

import numpy as np
import json
# Removed plotly.express

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
    merged = forecasts.merge(gdf_barangays.drop(columns=['Geometry']), on="Barangay", how="left")
    merged_all = merged
    
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
st.session_state.setdefault("selected_barangay", default_barangay)

# Filter for the line chart (all weeks of the selected year/barangay)
filtered_data_line_chart = merged_all[
    (merged_all["Model"] == st.session_state.selected_model)
    & (merged_all["Year"] == st.session_state.selected_year)
    & (merged_all["Barangay"] == st.session_state.selected_barangay)
].copy()

# Ensure Forecast_Cases is numeric at the DataFrame level for the chart
filtered_data_line_chart["Forecast_Cases"] = pd.to_numeric(filtered_data_line_chart["Forecast_Cases"], errors="coerce").fillna(0)
filtered_data_line_chart["Confidence"] = (filtered_data_line_chart["Confidence"] * 100).round(1).astype(str) + "%"

# Determine the last available week's data for metrics/ranking
if not filtered_data_line_chart.empty:
    last_week_data = filtered_data_line_chart.sort_values("Week", ascending=False).iloc[0]
    last_week = last_week_data["Week"]
else:
    last_week = None


# DASHBOARD LAYOUT
col1, col2 = st.columns(2)

# LEFT COLUMN
with col1:
    with st.container():
        # GET THE ACTUAL DATE RANGE for the LAST WEEK
        if last_week is not None:
            # Safely calculate start and end dates for the last week
            try:
                start_date = datetime.fromisocalendar(st.session_state.selected_year, last_week, 1)  # Monday
                end_date = datetime.fromisocalendar(st.session_state.selected_year, last_week, 7)    # Sunday
                date_range_str = f"Latest Forecast: {start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')}"
            except ValueError:
                date_range_str = "Latest Forecast: Invalid Date Range"
        else:
            date_range_str = "Latest Forecast: No Data Available"


        # CHART SECTION
        st.write(f"#### **{st.session_state.selected_barangay} Dengue Forecast**")
        
        # Line Chart using st.line_chart
        if not filtered_data_line_chart.empty:
            # Prepare data for st.line_chart (requires index or column for X-axis)
            chart_data = filtered_data_line_chart[['Week', 'Forecast_Cases']].set_index('Week')
            st.line_chart(chart_data, use_container_width=True, height=450)
        else:
            st.info(f"No forecast data available for {st.session_state.selected_barangay} in {st.session_state.selected_year} using the {st.session_state.selected_model} model.")
            st.markdown(f'<div style="height: 450px;"></div>', unsafe_allow_html=True) # Maintain height

        # FILTERS CONTROLS
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            selected_year = st.selectbox(
                "Select Year",
                available_years,
                index=available_years.index(st.session_state.selected_year)
            )

        with filter_col2:
            selected_barangay = st.selectbox(
                "Select Barangay",
                barangay_list,
                index=barangay_list.index(st.session_state.selected_barangay)
            )
        
        # UPDATE SESSION STATE ON CHANGE
        if selected_year != st.session_state.selected_year or selected_barangay != st.session_state.selected_barangay:
            st.session_state.selected_year = selected_year
            st.session_state.selected_barangay = selected_barangay
            st.rerun()

        # MAP DESCRIPTION
        map_description = st.container(border=True)
        map_description.write("""
        The chart visualizes the **forecasted dengue cases** over the selected year for the **selected barangay** using the chosen prediction model.
        This time series analysis allows for the detection of temporal patterns, such as seasonal peaks or unusual case surges for a specific location.
        The summary metrics and risk ranking in the right column are based on the **last available forecast week** shown in the chart.
        """)
            

# RIGHT COLUMN
with col2:
    # METRICS SECTION
    st.write("#### **Summary Metrics**")
    
    # Filter data for the current barangay and the last available week for metrics
    if last_week is not None:
        filtered_data_current_point = merged_all[
            (merged_all["Model"] == st.session_state.selected_model)
            & (merged_all["Year"] == st.session_state.selected_year)
            & (merged_all["Barangay"] == st.session_state.selected_barangay)
            & (merged_all["Week"] == last_week)
        ].copy()
        
        if not filtered_data_current_point.empty:
            current_case_data = filtered_data_current_point.iloc[0]
            
            # --- FIX APPLIED HERE ---
            # Check if the single value is NaN before trying to convert to int
            forecast_value = pd.to_numeric(current_case_data['Forecast_Cases'], errors="coerce")
            
            if pd.isna(forecast_value):
                current_cases = 0
            else:
                current_cases = int(forecast_value)
            # --- END FIX ---
            
            # Confidence was already formatted as string in filtered_data_line_chart but here we use merged_all
            current_confidence_raw = current_case_data['Confidence']
            current_confidence = (current_confidence_raw * 100).round(1).astype(str) + "%"
            current_risk = current_case_data['Risk_Level']
        else:
            current_cases = 0
            current_confidence = "N/A"
            current_risk = "N/A"
    else:
        current_cases = 0
        current_confidence = "N/A"
        current_risk = "N/A"

    # Total Forecasted Cases across ALL barangays for the latest week
    total_cases_latest_week = 0
    if last_week is not None:
        total_cases_df = merged_all[
            (merged_all["Model"] == st.session_state.selected_model)
            & (merged_all["Year"] == st.session_state.selected_year)
            & (merged_all["Week"] == last_week)
        ].copy()
        total_cases_latest_week = pd.to_numeric(total_cases_df['Forecast_Cases'], errors="coerce").fillna(0).sum().astype(int)

    m1, m2, m3 = st.columns(3)
    # Metric 1: Total cases across all barangays for the latest week
    m1.metric("Total Cases (Latest Week)", f"{total_cases_latest_week}")
    # Metric 2: Selected Barangay's cases for the latest week
    m2.metric(f"{st.session_state.selected_barangay} Cases", f"{current_cases}")
    # Metric 3: Selected Barangay's Risk Level for the latest week
    m3.metric(f"{st.session_state.selected_barangay} Risk", current_risk)
    
    # TABLE SECTION
    st.write("#### **Top 10 Risk Ranking by Barangay (Latest Week)**")
    
    # Table logic uses all barangays for the LAST available week of the selected year
    if last_week is not None:
        filtered_data_ranking = merged_all[
            (merged_all["Model"] == st.session_state.selected_model)
            & (merged_all["Year"] == st.session_state.selected_year)
            & (merged_all["Week"] == last_week)
        ].copy()
    else:
        filtered_data_ranking = pd.DataFrame()
        
    if not filtered_data_ranking.empty:
        filtered_data_ranking["Forecast_Cases"] = pd.to_numeric(filtered_data_ranking["Forecast_Cases"], errors="coerce").fillna(0)
        filtered_data_ranking["Confidence"] = (filtered_data_ranking["Confidence"] * 100).round(1).astype(str) + "%"

        risk_colors = { 
            "Low": "#E9F3F2",
            "Moderate": "#F3B705",
            "High": "#F17404",
            "Critical": "#D9042C"
        }
        
        table_df = filtered_data_ranking[['Barangay', 'Forecast_Cases', 'Confidence', 'Risk_Level']].copy()
        table_df = table_df.rename(columns={"Forecast_Cases": "Cases", "Confidence": "Confidence", "Risk_Level": "Risk Level"})
        table_df["Cases"] = table_df["Cases"].astype(str)
        
        # Sort data for Top 10
        risk_order = ["Low", "Moderate", "High", "Critical"]
        risk_order_map = {level: i for i, level in enumerate(risk_order)}
        table_df["Risk_Sort"] = table_df["Risk Level"].map(risk_order_map)
        
        # Sort by Risk Level (descending), then by Cases (descending)
        table_df = table_df.sort_values(
            by=['Risk_Sort', 'Cases'], 
            ascending=[False, False]
        ).reset_index(drop=True)
        
        table_df = table_df.drop(columns=['Risk_Sort'])
        
        # Limit to Top 10
        table_df_top10 = table_df.head(10)

        def color_forecast(val):
            if pd.isna(val):
                return 'background-color: #E9F3F2; color: black'
            color = risk_colors.get(val, "#E9F3F2")
            if color == "#E9F3F2" or color == "#F3B705":
                text_color = "black"
            else:
                text_color = "white"
            return f'background-color: {color}; color: {text_color}; font-weight: bold'
        
        # Apply style to the Top 10 table
        styled_table = table_df_top10.style.applymap(color_forecast, subset=['Risk Level'])
        
        # Adjust height for 10 rows + header
        st.dataframe(styled_table, width='stretch', height=380) 
    else:
        st.info("No ranking data available for the latest week.")
        st.markdown(f'<div style="height: 380px;"></div>', unsafe_allow_html=True) # Maintain height


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


