import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Scientific Visualisation Assignment", layout="wide")

# 1. Load Cleaned Datasets from Repository
df_income = pd.read_csv("hh_income_state_CLEANED.csv")
df_amenities = pd.read_csv("hh_access_amenities_cleaned.csv")
df_marriages = pd.read_csv("marriages_state_CLEANED.csv")

st.title("Malaysia Socioeconomic, Infrastructure & Demographic Analysis")
st.markdown("Individual Assignment — Visualizations 2, 4, and 5 (2019–2024)")

# =========================================================
# VISUALIZATION 2: Choropleth Geospatial Map (Objective 1)
# =========================================================
st.header("1. Visualization 2: Median Household Income by State (2024)")
st.markdown("**Objective 1:** *To analyze economic inequality across states and assess household wealth distribution across Malaysia.*")

df_inc_2024 = df_income[df_income["date"].astype(str).str.contains("2024")].copy()

# Summary Metrics
top_row = df_inc_2024.sort_values(by="income_median", ascending=False).iloc[0]
low_row = df_inc_2024.sort_values(by="income_median", ascending=True).iloc[0]
nat_avg = df_inc_2024["income_median"].mean()

c1, c2, c3 = st.columns(3)
c1.metric("Highest Median Income State", f"{top_row['state']}", f"RM {top_row['income_median']:,}")
c2.metric("National Mean of State Medians", f"RM {nat_avg:,.2f}")
c3.metric("Lowest Median Income State", f"{low_row['state']}", f"RM {low_row['income_median']:,}")

# Load Malaysia GeoJSON
urls = [
    "https://raw.githubusercontent.com/atifmustaffa/malaysia-geojson/master/malaysia.state.min.geojson",
    "https://raw.githubusercontent.com/mptwaktusolat/jakim.geojson/master/malaysia.state.geojson"
]
malaysia_geojson = None
for url in urls:
    resp = requests.get(url)
    if resp.status_code == 200 and resp.text.strip().startswith("{"):
        malaysia_geojson = resp.json()
        break

props = malaysia_geojson["features"][0]["properties"]
prop_key = next((k for k in ["name", "state", "negeri", "NAME_1", "Name"] if k in props), list(props.keys())[0])
geo_states = [f["properties"][prop_key] for f in malaysia_geojson["features"]]

def match_state(csv_name):
    clean = csv_name.replace("W.P. ", "").strip()
    aliases = {
        "Pulau Pinang": ["Pulau Pinang", "Penang"],
        "Melaka": ["Melaka", "Malacca"],
        "Kuala Lumpur": ["Kuala Lumpur", "Wilayah Persekutuan Kuala Lumpur", "W.P. Kuala Lumpur"],
        "Putrajaya": ["Putrajaya", "Wilayah Persekutuan Putrajaya", "W.P. Putrajaya"],
        "Labuan": ["Labuan", "Wilayah Persekutuan Labuan", "W.P. Labuan"],
    }
    for candidate in aliases.get(clean, [clean, csv_name]):
        for g in geo_states:
            if candidate.lower() == str(g).lower():
                return g
    return clean

df_inc_2024["geo_state"] = df_inc_2024["state"].apply(match_state)

fig2 = px.choropleth_map(
    df_inc_2024,
    geojson=malaysia_geojson,
    locations="geo_state",
    featureidkey=f"properties.{prop_key}",
    color="income_median",
    hover_name="state",
    hover_data={"geo_state": False, "income_median": ":,.0f"},
    color_continuous_scale="YlGnBu",
    map_style="carto-positron",
    zoom=4.8,
    center={"lat": 4.2, "lon": 109.5},
    opacity=0.85,
    labels={"income_median": "Median Income (RM)"},
    title="Geospatial Distribution of Median Household Income Across Malaysia (2024)"
)
fig2.update_layout(height=550, margin={"r": 0, "t": 50, "l": 0, "b": 0})
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# =========================================================
# VISUALIZATION 4: Grouped Bar Chart (Objective 2)
# =========================================================
st.header("2. Visualization 4: Infrastructure Completion Comparison (2024)")
st.markdown("**Objective 2:** *To evaluate regional accessibility of essential public utilities across states to guide targeted federal budget allocation.*")

df_amen_2024 = df_amenities[
    (df_amenities["district"] == "All Districts") & 
    (df_amenities["date"].astype(str).str.contains("2024"))
].copy()

lowest_water = df_amen_2024.sort_values(by="piped_water").iloc[0]
avg_elec = df_amen_2024["electricity"].mean()

m1, m2 = st.columns(2)
m1.metric("Lowest Piped Water Coverage (2024)", f"{lowest_water['state']}", f"{lowest_water['piped_water']}%")
m2.metric("Average State Electricity Access (2024)", f"{avg_elec:.2f}%")

df_amen_melted = df_amen_2024.melt(
    id_vars=["state"],
    value_vars=["piped_water", "sanitation", "electricity"],
    var_name="Utility_Pillar",
    value_name="Coverage_Percentage"
).replace({
    "piped_water": "Piped Water (%)",
    "sanitation": "Sanitation (%)",
    "electricity": "Electricity (%)"
})

fig4 = px.bar(
    df_amen_melted,
    x="state",
    y="Coverage_Percentage",
    color="Utility_Pillar",
    barmode="group",
    labels={"state": "State / Federal Territory", "Coverage_Percentage": "Access Rate (%)", "Utility_Pillar": "Infrastructure Pillar"},
    title="Comparison of Piped Water, Sanitation & Electricity Coverage by State (2024)"
)
fig4.update_layout(yaxis_range=[0, 105], xaxis_tickangle=-45, height=550)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# =========================================================
# VISUALIZATION 5: Scatter Plot with Regression (Objective 3)
# =========================================================
st.header("3. Visualization 5: Marriage Rate vs. Median Household Income (2019–2024)")
st.markdown("**Objective 3:** *To explore the correlation between macroeconomic conditions (median income) and demographic behaviors (marriage rates) over time.*")

df_marr_grouped = df_marriages.groupby(["state", "date"], as_index=False)["rate"].mean()
df_merged = pd.merge(df_income, df_marr_grouped, on=["state", "date"])
df_merged["Year"] = pd.to_datetime(df_merged["date"]).dt.year.astype(str)

corr_val = df_merged["income_median"].corr(df_merged["rate"])
st.metric("Pearson Correlation Coefficient (r)", f"{corr_val:.3f}", "Negative / Inverse Relationship")

fig5 = px.scatter(
    df_merged,
    x="income_median",
    y="rate",
    color="state",
    symbol="Year",
    trendline="ols",
    trendline_scope="overall",
    hover_name="state",
    hover_data={"Year": True, "income_median": ":,.0f", "rate": ":.2f"},
    labels={
        "income_median": "Median Household Income (RM)",
        "rate": "Average Marriage Rate (per 1,000 unmarried population)",
        "state": "State"
    },
    title="Correlation Between Median Household Income and Marriage Rates Across States (2019, 2022, 2024)"
)
fig5.update_layout(height=600)
st.plotly_chart(fig5, use_container_width=True)
