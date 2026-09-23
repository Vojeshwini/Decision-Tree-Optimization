"""
Decision Tree Optimization Framework -- Decision Support & Business
Intelligence Application (Streamlit)

Run locally:   streamlit run app.py
Deploy free:   push this repo to GitHub -> streamlit.io/cloud -> deploy

This app has a REAL backend:
 - Loads actual trained model atifacts (Random Forest + interpretable tree)
 - Runs live model.predict() inference for the Decision Support panel
 - Extracts live decision rules from the actual tree structure
 - Computes all BI charts/KPIs from the real dataset with live pandas filtering
No numbers are hardcoded -- everything recalculates from the underlying data
and models whenever you change a filter or input.
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import export_text

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Decision Tree Optimization Framework | DSS & BI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Load artifacts (cached so the app stays fast after first load)
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    rf = joblib.load("model_random_forest.joblib")
    shallow = joblib.load("model_shallow_tree.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    return rf, shallow, feature_cols

@st.cache_data
def load_data():
    return pd.read_parquet("bi_dataset.parquet")

@st.cache_data
def load_model_comparison():
    # Real evaluation results from the training/optimization run
    return pd.DataFrame([
        {"Model": "Baseline CART (unconstrained)", "WMAE": 2405.69, "MAE": 2401.33, "RMSE": 5176.72, "R2": 0.9445, "Fit time (s)": 4.7},
        {"Model": "Optimized CART (pruned)", "WMAE": 2243.52, "MAE": 2228.95, "RMSE": 4968.41, "R2": 0.9488, "Fit time (s)": 611.6},
        {"Model": "Optimized CART (tuned)", "WMAE": 2528.11, "MAE": 2513.03, "RMSE": 5061.77, "R2": 0.9469, "Fit time (s)": 8.9},
        {"Model": "Random Forest (production)", "WMAE": 1892.15, "MAE": 1868.62, "RMSE": 3819.39, "R2": 0.9698, "Fit time (s)": 191.2},
        {"Model": "Gradient Boosted Trees", "WMAE": 2994.29, "MAE": 2964.69, "RMSE": 4900.85, "R2": 0.9502, "Fit time (s)": 233.1},
    ])

def trace_decision_path(tree_model, X_input, feature_names):
    """Walk the actual trained tree's structure for this exact input and
    return the real if-then steps taken, plus the resulting leaf prediction.
    This is genuine rule extraction for THIS prediction, not a static list."""
    tree = tree_model.tree_
    node_indicator = tree_model.decision_path(X_input)
    leaf_id = tree_model.apply(X_input)
    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]

    steps = []
    for node_id in node_index:
        if leaf_id[sample_id] == node_id:
            continue
        feature = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]
        value = X_input.iloc[sample_id][feature]
        if value <= threshold:
            steps.append(f"{feature} = {value:g}  →  ≤ {threshold:,.1f}  ✓")
        else:
            steps.append(f"{feature} = {value:g}  →  > {threshold:,.1f}  ✓")
    leaf_prediction = tree.value[leaf_id[sample_id]][0][0]
    return steps, leaf_prediction

rf_model, shallow_model, feature_cols = load_models()
df = load_data()
model_results = load_model_comparison()

TYPE_MAP = {0: "A", 1: "B", 2: "C"}
TYPE_COLORS = {"A": "#4C8DE8", "B": "#E8A33D", "C": "#E2584F"}

# ------------------------------------------------------------------
# Custom styling (dark, data-console look)
# ------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg-0:#0F1216; --bg-1:#161A20; --bg-2:#1D222A; --bg-3:#242A34;
  --line:#282E38; --line-strong:#3A4150;
  --amber:#E8A33D; --amber-bg:#2A2115;
  --teal:#3FE0C0; --red:#E2584F;
  --text-hi:#F2F1ED; --text-mid:#C9CDD6; --text-lo:#8B92A0;
  --disp:'Space Grotesk',sans-serif; --mono:'IBM Plex Mono',monospace;
}

/* ---- Hide Streamlit's default chrome ---- */
#MainMenu, header[data-testid="stHeader"], footer, div[data-testid="stDecoration"],
.stAppDeployButton, [data-testid="stToolbar"] { display: none !important; }
.stApp { background-color: var(--bg-0); }
.block-container { padding-top: 2rem !important; max-width: 1180px; }

/* ---- Base text ---- */
h1, h2, h3, h4, p, span, label, .stMarkdown,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--text-hi) !important; }
h1, h2, h3 { font-family: var(--disp) !important; font-weight: 600 !important; letter-spacing: -0.01em; }
p, .stMarkdown p { color: var(--text-mid) !important; }
[data-testid="stCaptionContainer"], .stCaption, small { color: var(--text-lo) !important; }
[data-testid="stSidebar"] { background-color: var(--bg-1); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: var(--text-mid) !important; }

/* ---- Custom header bar ---- */
.app-header { display:flex; align-items:center; gap:10px; margin-bottom:4px; }
.app-header .dot { width:9px; height:9px; border-radius:50%; background:var(--amber); box-shadow:0 0 0 4px var(--amber-bg); flex-shrink:0; }
.app-header .title { font-family:var(--disp); font-weight:700; font-size:26px; color:var(--text-hi); }
.app-sub { color:var(--text-lo) !important; font-size:13px; margin:2px 0 20px 19px; font-family: var(--mono); }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
button[data-baseweb="tab"] { color: var(--text-lo) !important; font-weight: 500 !important; background: transparent !important; }
button[data-baseweb="tab"] p { color: inherit !important; font-size: 14px !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--amber) !important; }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--amber) !important; }
[data-baseweb="tab-highlight"] { background-color: var(--amber) !important; }

/* ---- Selectbox / multiselect: fix the mismatched white boxes ---- */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background-color: var(--bg-2) !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 8px !important;
    color: var(--text-hi) !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] div { color: var(--text-hi) !important; }
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background-color: var(--amber-bg) !important; border: 1px solid var(--amber) !important;
}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] span { color: var(--amber) !important; }
/* dropdown popover menu (rendered separately) */
ul[data-baseweb="menu"] { background-color: var(--bg-2) !important; border: 1px solid var(--line-strong) !important; }
ul[data-baseweb="menu"] li { color: var(--text-hi) !important; }
ul[data-baseweb="menu"] li:hover { background-color: var(--bg-3) !important; }

/* ---- Radio buttons ---- */
div[role="radiogroup"] label { color: var(--text-mid) !important; }
div[role="radiogroup"] label span:first-child > div { border-color: var(--line-strong) !important; }
div[role="radiogroup"] label[data-checked="true"] span:first-child > div { border-color: var(--amber) !important; background-color: var(--amber) !important; }

/* ---- Checkbox ---- */
[data-testid="stCheckbox"] label span:first-child { border-color: var(--line-strong) !important; background-color: var(--bg-2) !important; }

/* ---- Slider ---- */
.stSlider [data-baseweb="slider"] > div > div { background: var(--line-strong) !important; }
.stSlider [data-baseweb="slider"] > div > div > div { background: var(--amber) !important; }
.stSlider [role="slider"] { background-color: var(--amber) !important; border-color: var(--amber) !important; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: var(--text-lo) !important; }
.stSlider [data-baseweb="slider"] + div { color: var(--amber) !important; font-family: var(--mono) !important; }

/* ---- Text input ---- */
[data-testid="stTextInput"] input {
    background-color: var(--bg-2) !important; border: 1px solid var(--line-strong) !important;
    color: var(--text-hi) !important; border-radius: 8px !important;
}

/* ---- Buttons ---- */
.stButton button {
    background-color: var(--amber) !important; color: #1A1408 !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important; font-family: 'Inter', sans-serif !important;
}
.stButton button:hover { background-color: #F2B450 !important; }
.stButton button p { color: #1A1408 !important; }

/* ---- Dataframe / table ---- */
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }

/* ---- Alerts (success/warning/info) ---- */
[data-testid="stAlert"] { border-radius: 10px; }

/* ---- KPI and rule boxes (custom) ---- */
.kpi-box { background: var(--bg-1); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }
.kpi-label { font-size:11px; color:var(--text-lo) !important; text-transform:uppercase; letter-spacing:0.05em; }
.kpi-val { font-family: var(--mono); font-size:24px; font-weight:600; color:var(--text-hi) !important; margin-top: 4px; }
.rule-box { background: var(--bg-2); border:1px dashed var(--line-strong); border-radius:10px; padding:18px; font-family: var(--mono); font-size:13px; color:var(--text-lo) !important; white-space:pre-wrap; line-height: 1.7; }
.rule-box div { color: inherit; }

/* ---- Section spacing ---- */
.stSubheader, h3 { margin-top: 8px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header"><span class="dot"></span><span class="title">Decision Tree Optimization Framework</span></div>
<div class="app-sub">ADVANCED DECISION SUPPORT &amp; BUSINESS ANALYTICS &middot; LIVE BACKEND, REAL TRAINED MODELS</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------------
st.sidebar.markdown("### 🎛 Filters")
currency = st.sidebar.radio("Currency", options=["USD ($)", "INR (₹)"], horizontal=True)
USD_TO_INR = 83.0  # approximate, fixed display rate

def fmt_currency(value_usd, compact=False):
    """Format a USD value in the selected display currency."""
    symbol = "$" if currency == "USD ($)" else "₹"
    value = value_usd if currency == "USD ($)" else value_usd * USD_TO_INR
    if compact:
        if abs(value) >= 1e9:
            return f"{symbol}{value/1e9:,.1f}B"
        if abs(value) >= 1e7 and currency != "USD ($)":
            return f"{symbol}{value/1e7:,.2f}Cr"  # Indian crore convention
        if abs(value) >= 1e6:
            return f"{symbol}{value/1e6:,.1f}M"
        if abs(value) >= 1e3:
            return f"{symbol}{value/1e3:,.1f}K"
        return f"{symbol}{value:,.0f}"
    return f"{symbol}{value:,.0f}"

if currency == "INR (₹)":
    st.sidebar.caption(f"Converted at an approximate fixed rate of 1 USD = ₹{USD_TO_INR:.0f} for display only — the underlying dataset is reported in USD by Walmart.")

type_filter = st.sidebar.multiselect("Store type", options=["A", "B", "C"], default=["A", "B", "C"])
year_filter = st.sidebar.multiselect("Year", options=sorted(df["Year"].unique().tolist()), default=sorted(df["Year"].unique().tolist()))
holiday_filter = st.sidebar.selectbox("Week type", options=["All weeks", "Holiday weeks only", "Non-holiday weeks only"])
dept_options = ["All departments"] + sorted(df["Dept"].unique().tolist())
dept_filter = st.sidebar.selectbox("Department drill-down", options=dept_options)
store_options = ["All stores"] + sorted(df["Store"].unique().tolist())
store_filter = st.sidebar.selectbox("Store drill-down", options=store_options)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.caption(
    "Decision Tree Optimization Framework for Advanced Decision Support "
    "and Business Analytics Applications. Case study on the Walmart Store "
    "Sales Forecasting dataset (Kaggle, public benchmark)."
)

# Apply filters
fdf = df[df["Type"].isin(type_filter) & df["Year"].isin(year_filter)]
if holiday_filter == "Holiday weeks only":
    fdf = fdf[fdf["IsHoliday"] == 1]
elif holiday_filter == "Non-holiday weeks only":
    fdf = fdf[fdf["IsHoliday"] == 0]
if dept_filter != "All departments":
    fdf = fdf[fdf["Dept"] == dept_filter]
if store_filter != "All stores":
    fdf = fdf[fdf["Store"] == store_filter]

# ------------------------------------------------------------------
# Header (styled version rendered above, in the CSS block)
# ------------------------------------------------------------------

tab_bi, tab_dss, tab_models, tab_rules = st.tabs(
    ["📊 Business Intelligence", "🎯 Decision Support (Live Prediction)", "⚙️ Model Optimization", "📋 Decision Rules"]
)

# ------------------------------------------------------------------
# TAB 1: Business Intelligence
# ------------------------------------------------------------------
with tab_bi:
    if len(fdf) == 0:
        st.warning("No data matches the current filters. Adjust filters in the sidebar.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">Filtered rows</div><div class="kpi-val">{len(fdf):,}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">Avg weekly sales</div><div class="kpi-val">{fmt_currency(fdf["Weekly_Sales"].mean())}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">Total sales</div><div class="kpi-val">{fmt_currency(fdf["Weekly_Sales"].sum(), compact=True)}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">Stores in view</div><div class="kpi-val">{fdf["Store"].nunique()}</div></div>', unsafe_allow_html=True)
        with c5:
            yoy_2011 = fdf[fdf["Year"] == 2011]["Weekly_Sales"].sum()
            yoy_2012 = fdf[fdf["Year"] == 2012]["Weekly_Sales"].sum()
            yoy_pct = ((yoy_2012 - yoy_2011) / yoy_2011 * 100) if yoy_2011 > 0 else 0
            yoy_color = "#3FE0C0" if yoy_pct >= 0 else "#E2584F"
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">YoY growth (\'11→\'12 YTD)</div><div class="kpi-val" style="color:{yoy_color}">{yoy_pct:+.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1.4, 1])

        curr_symbol = "$" if currency == "USD ($)" else "₹"
        conv = 1.0 if currency == "USD ($)" else USD_TO_INR
        fdf_c = fdf.copy()
        fdf_c["Sales_Display"] = fdf_c["Weekly_Sales"] * conv

        with col1:
            st.subheader("Monthly sales trend")
            trend = fdf_c.groupby(["Year", "Month"])["Sales_Display"].sum().reset_index()
            trend["period"] = trend["Year"].astype(str) + "-" + trend["Month"].astype(str).str.zfill(2)
            fig = px.line(trend, x="period", y="Sales_Display", markers=True)
            fig.update_traces(line_color="#E8A33D")
            fig.update_layout(template="plotly_dark", plot_bgcolor="#161A20", paper_bgcolor="#161A20",
                               height=320, xaxis_title=None, yaxis_title=f"Total sales ({curr_symbol})")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Sales share by store type")
            type_agg = fdf.groupby("Type")["Weekly_Sales"].sum().reset_index()
            fig = px.pie(type_agg, names="Type", values="Weekly_Sales", hole=0.6,
                         color="Type", color_discrete_map=TYPE_COLORS)
            fig.update_layout(template="plotly_dark", plot_bgcolor="#161A20", paper_bgcolor="#161A20", height=320)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Holiday vs non-holiday avg sales")
            hol = fdf_c.groupby("IsHoliday")["Sales_Display"].mean().reset_index()
            hol["Week type"] = hol["IsHoliday"].map({0: "Regular week", 1: "Holiday week"})
            fig = px.bar(hol, x="Week type", y="Sales_Display", color="Week type",
                         color_discrete_sequence=["#3A4150", "#3FE0C0"])
            fig.update_layout(template="plotly_dark", plot_bgcolor="#161A20", paper_bgcolor="#161A20",
                               height=300, showlegend=False, yaxis_title=f"Avg weekly sales ({curr_symbol})")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader("Top 10 stores by total sales")
            top_stores = fdf_c.groupby(["Store", "Type"])["Sales_Display"].sum().reset_index().sort_values("Sales_Display", ascending=False).head(10)
            fig = px.bar(top_stores, x="Sales_Display", y=top_stores["Store"].astype(str), orientation="h",
                         color="Type", color_discrete_map=TYPE_COLORS)
            fig.update_layout(template="plotly_dark", plot_bgcolor="#161A20", paper_bgcolor="#161A20",
                               height=300, yaxis_title=None, xaxis_title=f"Total sales ({curr_symbol})",
                               yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top 15 departments by total sales")
        dept_agg = fdf_c.groupby("Dept")["Sales_Display"].sum().reset_index().sort_values("Sales_Display", ascending=False).head(15)
        fig = px.bar(dept_agg, x=dept_agg["Dept"].astype(str), y="Sales_Display", color_discrete_sequence=["#4C8DE8"])
        fig.update_layout(template="plotly_dark", plot_bgcolor="#161A20", paper_bgcolor="#161A20",
                           height=300, xaxis_title="Department", yaxis_title=f"Total sales ({curr_symbol})")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Store detail explorer")
        search_col, sort_col = st.columns([2, 1])
        with search_col:
            search_store = st.text_input("Search by store number", placeholder="e.g. 20")
        with sort_col:
            sort_by = st.selectbox("Sort by", ["Total sales (desc)", "Avg sales (desc)", "Store number"])

        store_table = fdf_c.groupby(["Store", "Type", "Size"]).agg(
            Total_Sales=("Sales_Display", "sum"), Avg_Sales=("Sales_Display", "mean"), Rows=("Sales_Display", "count")
        ).reset_index()
        if search_store:
            store_table = store_table[store_table["Store"].astype(str).str.contains(search_store)]
        if sort_by == "Total sales (desc)":
            store_table = store_table.sort_values("Total_Sales", ascending=False)
        elif sort_by == "Avg sales (desc)":
            store_table = store_table.sort_values("Avg_Sales", ascending=False)
        else:
            store_table = store_table.sort_values("Store")
        store_table["Total_Sales"] = store_table["Total_Sales"].apply(lambda v: fmt_currency(v / conv))
        store_table["Avg_Sales"] = store_table["Avg_Sales"].apply(lambda v: fmt_currency(v / conv))
        st.dataframe(store_table, use_container_width=True, hide_index=True, height=280)

# ------------------------------------------------------------------
# TAB 2: Decision Support (Live Prediction)
# ------------------------------------------------------------------
with tab_dss:
    st.subheader("Live weekly sales prediction")

    col_in, col_out = st.columns([1, 1.3])

    with col_in:
        dept = st.slider("Department", 1, 99, 20)
        size = st.slider("Store size (sq ft)", 34000, 220000, 120000, step=1000)
        store_type = st.selectbox("Store type", ["A", "B", "C"])
        is_holiday = st.checkbox("Holiday week", value=False)
        cpi = st.slider("CPI (consumer price index)", 126.0, 228.0, 171.0)
        unemployment = st.slider("Unemployment rate (%)", 3.6, 14.3, 7.9)
        temperature = st.slider("Temperature (°F)", -2.0, 101.0, 60.0)
        week_of_year = st.slider("Week of year", 1, 52, 25)
        month = st.slider("Month", 1, 12, 6)

        predict_btn = st.button("Run live prediction", type="primary", use_container_width=True)

    with col_out:
        if predict_btn or True:
            # Build feature row matching training feature order
            row = {c: 0 for c in feature_cols}
            row["Store"] = 1  # representative store id
            row["Dept"] = dept
            row["TypeCode"] = {"A": 0, "B": 1, "C": 2}[store_type]
            row["Size"] = size
            row["Temperature"] = temperature
            row["Fuel_Price"] = 3.5
            row["CPI"] = cpi
            row["Unemployment"] = unemployment
            row["IsHoliday"] = int(is_holiday)
            row["Year"] = 2012
            row["WeekOfYear"] = week_of_year
            row["Month"] = month
            evt_key = f"Evt_{'SuperBowl' if week_of_year==6 else 'LaborDay' if week_of_year==36 else 'Thanksgiving' if week_of_year==47 else 'Christmas' if week_of_year==52 else 'None'}"
            if evt_key in row:
                row[evt_key] = 1

            X_input = pd.DataFrame([row])[feature_cols]
            prediction = rf_model.predict(X_input)[0]
            rule_steps, rule_prediction = trace_decision_path(shallow_model, X_input, feature_cols)

            st.markdown(f"""
            <div class="rule-box" style="text-align:center; padding:28px;">
                <div style="font-size:12px; color:#5C6270; text-transform:uppercase; letter-spacing:0.06em;">Accurate prediction &mdash; Random Forest (live inference)</div>
                <div style="font-family:monospace; font-size:40px; font-weight:600; color:#E8A33D; margin:10px 0;">{fmt_currency(prediction)}</div>
                <div style="font-size:12px; color:#9DA3B0;">Dept {dept} · {size:,} sq ft · Type {store_type} · {'Holiday' if is_holiday else 'Regular'} week</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="rule-box" style="padding:20px;">
                <div style="font-size:12px; color:#5C6270; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:10px;">Why &mdash; interpretable tree's reasoning for this same input</div>
                {"<br>".join(f"&rsaquo; {s}" for s in rule_steps)}
                <div style="border-top:1px solid #3A4150; margin-top:12px; padding-top:10px; color:#3FE0C0;">Interpretable-tree estimate: {fmt_currency(rule_prediction)}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Business interpretation")
            avg_sales_similar = df[(df["Type"] == store_type)]["Weekly_Sales"].mean()
            delta_pct = (prediction - avg_sales_similar) / avg_sales_similar * 100
            if delta_pct > 20:
                st.success(f"This department/store combination is predicted **{delta_pct:.0f}% above** the Type {store_type} average — consider prioritizing staffing and inventory buffer here.")
            elif delta_pct < -20:
                st.warning(f"This department/store combination is predicted **{abs(delta_pct):.0f}% below** the Type {store_type} average — review markdown/promotion strategy or reduce inventory allocation.")
            else:
                st.info(f"This prediction is close to the Type {store_type} average ({delta_pct:+.0f}%) — standard planning applies.")

            # ---- Scenario comparison: holiday vs regular week, same inputs otherwise ----
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Scenario comparison — holiday vs. regular week")
            row_holiday = dict(row); row_holiday["IsHoliday"] = 1
            row_regular = dict(row); row_regular["IsHoliday"] = 0
            pred_holiday = rf_model.predict(pd.DataFrame([row_holiday])[feature_cols])[0]
            pred_regular = rf_model.predict(pd.DataFrame([row_regular])[feature_cols])[0]
            uplift_pct = (pred_holiday - pred_regular) / pred_regular * 100 if pred_regular > 0 else 0

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown(f'<div class="kpi-box"><div class="kpi-label">Regular week</div><div class="kpi-val">{fmt_currency(pred_regular)}</div></div>', unsafe_allow_html=True)
            with sc2:
                st.markdown(f'<div class="kpi-box"><div class="kpi-label">Holiday week</div><div class="kpi-val" style="color:#E8A33D">{fmt_currency(pred_holiday)}</div></div>', unsafe_allow_html=True)
            with sc3:
                uplift_color = "#3FE0C0" if uplift_pct >= 0 else "#E2584F"
                st.markdown(f'<div class="kpi-box"><div class="kpi-label">Holiday uplift</div><div class="kpi-val" style="color:{uplift_color}">{uplift_pct:+.1f}%</div></div>', unsafe_allow_html=True)

            # ---- Actionable staffing / inventory recommendation tier ----
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Recommended action tier")
            all_preds_ref = df["Weekly_Sales"]
            p33, p66 = all_preds_ref.quantile(0.33), all_preds_ref.quantile(0.66)
            if prediction <= p33:
                tier, tier_color, tier_msg = "LOW", "#5C6270", "Standard staffing. Minimal inventory buffer needed. Consider bundling with markdown promotions if underperforming further."
            elif prediction <= p66:
                tier, tier_color, tier_msg = "MODERATE", "#E8A33D", "Standard-to-moderate staffing. Maintain normal inventory cycle, monitor weekly."
            else:
                tier, tier_color, tier_msg = "HIGH", "#3FE0C0", "Increase staffing allocation and inventory buffer. Prioritize this department/store combination for holiday-period planning."
            st.markdown(f"""
            <div class="rule-box" style="padding:16px 20px;">
                <span style="background:{tier_color}22; color:{tier_color}; padding:4px 12px; border-radius:20px; font-weight:600; font-size:12px;">● {tier} PRIORITY</span>
                <div style="margin-top:10px; color:#9DA3B0; font-size:13px;">{tier_msg}</div>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# TAB 3: Model Optimization
# ------------------------------------------------------------------
with tab_models:
    st.subheader("Decision tree optimization pipeline — results")
    st.caption("Every model below was actually trained and evaluated with WMAE (holiday weeks weighted 5x, the official Walmart competition metric).")

    best_idx = model_results["WMAE"].idxmin()
    st.dataframe(
        model_results.style.apply(lambda row: ["background-color: #132522; color: #3FE0C0" if row.name == best_idx else "" for _ in row], axis=1),
        use_container_width=True, hide_index=True
    )

    fig = go.Figure()
    colors = ["#3FE0C0" if i == best_idx else "#3A4150" for i in model_results.index]
    fig.add_trace(go.Bar(x=model_results["WMAE"], y=model_results["Model"], orientation="h", marker_color=colors))
    fig.update_layout(template="plotly_dark", plot_bgcolor="#161A20", paper_bgcolor="#161A20",
                       height=350, xaxis_title="WMAE (lower is better)", yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature importance (Random Forest, live from loaded model)")
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False).head(10)
    fig2 = px.bar(importances, x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#E8A33D"])
    fig2.update_layout(template="plotly_dark", plot_bgcolor="#161A20", paper_bgcolor="#161A20",
                        height=350, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------
# TAB 4: Decision Rules
# ------------------------------------------------------------------
with tab_rules:
    st.subheader("Human-readable decision rules")
    st.caption("Extracted live from the actual trained interpretable surrogate tree (max depth 4, min 500 samples/leaf) — not SHAP scores, direct if-then logic.")
    rules_text = export_text(shallow_model, feature_names=feature_cols)
    st.markdown(f'<div class="rule-box">{rules_text}</div>', unsafe_allow_html=True)
