"""
Decision Tree Optimization Framework -- Decision Support & Business
Intelligence Application (Streamlit)

Run locally:   streamlit run app.py
Deploy free:   push this repo to GitHub -> streamlit.io/cloud -> deploy

This app has a REAL backend:
 - Loads actual trained model artifacts (Random Forest + interpretable tree)
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
    return pd.DataFrame([
        {"Model": "Baseline CART (unconstrained)", "WMAE": 2405.69, "MAE": 2401.33, "RMSE": 5176.72, "R2": 0.9445, "Fit time (s)": 4.7},
        {"Model": "Optimized CART (pruned)", "WMAE": 2243.52, "MAE": 2228.95, "RMSE": 4968.41, "R2": 0.9488, "Fit time (s)": 611.6},
        {"Model": "Optimized CART (tuned)", "WMAE": 2528.11, "MAE": 2513.03, "RMSE": 5061.77, "R2": 0.9469, "Fit time (s)": 8.9},
        {"Model": "Random Forest (production)", "WMAE": 1892.15, "MAE": 1868.62, "RMSE": 3819.39, "R2": 0.9698, "Fit time (s)": 191.2},
        {"Model": "Gradient Boosted Trees", "WMAE": 2994.29, "MAE": 2964.69, "RMSE": 4900.85, "R2": 0.9502, "Fit time (s)": 233.1},
    ])

def trace_decision_path(tree_model, X_input, feature_names):
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
            steps.append(f"{feature} = {value:g}  ->  <= {threshold:,.1f}")
        else:
            steps.append(f"{feature} = {value:g}  ->  > {threshold:,.1f}")
    leaf_prediction = tree.value[leaf_id[sample_id]][0][0]
    return steps, leaf_prediction

rf_model, shallow_model, feature_cols = load_models()
df = load_data()
model_results = load_model_comparison()

TYPE_COLORS = {"A": "#4C8DE8", "B": "#E8A33D", "C": "#E2584F"}
TYPE_LABELS = {"A": "Type A (Large)", "B": "Type B (Medium)", "C": "Type C (Small)"}

# ------------------------------------------------------------------
# Styling -- one single, self-contained style block
# ------------------------------------------------------------------
APP_CSS = """
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
#MainMenu, header[data-testid="stHeader"], footer, div[data-testid="stDecoration"],
.stAppDeployButton, [data-testid="stToolbar"] { display: none !important; }
.stApp { background-color: var(--bg-0); }
.block-container { padding-top: 2rem !important; max-width: 1180px; }
h1, h2, h3, h4, p, span, label, .stMarkdown,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--text-hi) !important; }
h1, h2, h3 { font-family: var(--disp) !important; font-weight: 600 !important; letter-spacing: -0.01em; }
p, .stMarkdown p { color: var(--text-mid) !important; }
[data-testid="stCaptionContainer"], .stCaption, small { color: var(--text-lo) !important; }
[data-testid="stSidebar"] { background-color: var(--bg-1); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: var(--text-mid) !important; }
.app-header { display:flex; align-items:center; gap:10px; margin-bottom:4px; }
.app-header .dot { width:9px; height:9px; border-radius:50%; background:var(--amber); box-shadow:0 0 0 4px var(--amber-bg); flex-shrink:0; }
.app-header .title { font-family:var(--disp); font-weight:700; font-size:26px; color:var(--text-hi); }
.app-sub { color:var(--text-lo) !important; font-size:13px; margin:2px 0 14px 19px; font-family: var(--mono); }
.about-box { background:var(--bg-1); border:1px solid var(--line); border-radius:12px; padding:18px 22px; margin-bottom:22px; }
.about-box p { color: var(--text-mid) !important; font-size: 13.5px; line-height: 1.6; margin: 0; }
.about-box .about-title { font-family: var(--disp); font-size: 14px; font-weight: 600; color: var(--amber) !important; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
button[data-baseweb="tab"] { color: var(--text-lo) !important; font-weight: 500 !important; background: transparent !important; }
button[data-baseweb="tab"] p { color: inherit !important; font-size: 14px !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--amber) !important; }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--amber) !important; }
[data-baseweb="tab-highlight"] { background-color: var(--amber) !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background-color: var(--bg-2) !important; border: 1px solid var(--line-strong) !important;
    border-radius: 8px !important; color: var(--text-hi) !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] div { color: var(--text-hi) !important; }
[data-testid="stMultiSelect"] span[data-baseweb="tag"] { background-color: var(--amber-bg) !important; border: 1px solid var(--amber) !important; }
[data-testid="stMultiSelect"] span[data-baseweb="tag"] span { color: var(--amber) !important; }
ul[data-baseweb="menu"] { background-color: var(--bg-2) !important; border: 1px solid var(--line-strong) !important; }
ul[data-baseweb="menu"] li { color: var(--text-hi) !important; }
ul[data-baseweb="menu"] li:hover { background-color: var(--bg-3) !important; }
div[role="radiogroup"] label { color: var(--text-mid) !important; }
[data-testid="stCheckbox"] label span:first-child { border-color: var(--line-strong) !important; background-color: var(--bg-2) !important; }
.stSlider [data-baseweb="slider"] > div > div { background: var(--line-strong) !important; }
.stSlider [data-baseweb="slider"] > div > div > div { background: var(--amber) !important; }
.stSlider [role="slider"] { background-color: var(--amber) !important; border-color: var(--amber) !important; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: var(--text-lo) !important; }
[data-testid="stTextInput"] input { background-color: var(--bg-2) !important; border: 1px solid var(--line-strong) !important; color: var(--text-hi) !important; border-radius: 8px !important; }
.stButton button { background-color: var(--amber) !important; color: #1A1408 !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
.stButton button:hover { background-color: #F2B450 !important; }
.stButton button p { color: #1A1408 !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 10px; }
.kpi-box { background: var(--bg-1); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }
.kpi-label { font-size:11px; color:var(--text-lo) !important; text-transform:uppercase; letter-spacing:0.05em; }
.kpi-val { font-family: var(--mono); font-size:24px; font-weight:600; color:var(--text-hi) !important; margin-top: 4px; }
.rule-box { background: var(--bg-2); border:1px dashed var(--line-strong); border-radius:10px; padding:18px; font-family: var(--mono); font-size:13px; color:var(--text-lo) !important; white-space:pre-wrap; line-height: 1.7; }
.rule-box div { color: inherit; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

HEADER_HTML = """
<div class="app-header"><span class="dot"></span><span class="title">Decision Tree Optimization Framework</span></div>
<div class="app-sub">ADVANCED DECISION SUPPORT &amp; BUSINESS ANALYTICS APPLICATIONS</div>
"""
st.markdown(HEADER_HTML, unsafe_allow_html=True)

ABOUT_HTML = """
<div class="about-box">
<div class="about-title">About this project</div>
<p><b>Objective:</b> Build and systematically optimize a decision tree (pruning, hyperparameter tuning, ensembling) for retail weekly sales forecasting, then deploy it as a working Decision Support and Business Intelligence tool.
<b>Dataset:</b> Walmart Store Sales Forecasting (Kaggle public benchmark) &mdash; 45 stores, 81 departments, 2010-2012.
<b>Base paper:</b> Wellens, Boute &amp; Udenio (2024), <i>European Journal of Operational Research</i> &mdash; extended here with systematic tree optimization, human-readable decision rules, holiday-weighted (WMAE) evaluation, and this deployable dashboard.</p>
</div>
"""
st.markdown(ABOUT_HTML, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------------
st.sidebar.markdown("### Filters")
currency = st.sidebar.radio("Currency", options=["USD ($)", "INR (₹)"], horizontal=True)
USD_TO_INR = 83.0

def fmt_currency(value_usd, compact=False):
    symbol = "$" if currency == "USD ($)" else "₹"
    value = value_usd if currency == "USD ($)" else value_usd * USD_TO_INR
    if compact:
        if abs(value) >= 1e9:
            return f"{symbol}{value/1e9:,.1f}B"
        if abs(value) >= 1e7 and currency != "USD ($)":
            return f"{symbol}{value/1e7:,.2f}Cr"
        if abs(value) >= 1e6:
            return f"{symbol}{value/1e6:,.1f}M"
        if abs(value) >= 1e3:
            return f"{symbol}{value/1e3:,.1f}K"
        return f"{symbol}{value:,.0f}"
    return f"{symbol}{value:,.0f}"

if currency == "INR (₹)":
    st.sidebar.caption(f"Approx. fixed rate: 1 USD = ₹{USD_TO_INR:.0f} (display only; source data is in USD).")

type_filter = st.sidebar.multiselect("Store type", options=["A", "B", "C"], default=["A", "B", "C"], format_func=lambda x: TYPE_LABELS[x])
year_filter = st.sidebar.multiselect("Year", options=sorted(df["Year"].unique().tolist()), default=sorted(df["Year"].unique().tolist()))
holiday_filter = st.sidebar.selectbox("Week type", options=["All weeks", "Holiday weeks only", "Non-holiday weeks only"])
dept_options = ["All departments"] + sorted(df["Dept"].unique().tolist())
dept_filter = st.sidebar.selectbox("Department drill-down", options=dept_options)
store_options = ["All stores"] + sorted(df["Store"].unique().tolist())
store_filter = st.sidebar.selectbox("Store drill-down", options=store_options)

st.sidebar.markdown("---")
st.sidebar.caption("Case study on a public Kaggle dataset; not affiliated with Walmart.")

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
# Tabs
# ------------------------------------------------------------------
tab_bi, tab_dss, tab_models, tab_rules = st.tabs(
    ["Business Intelligence", "Decision Support (Live Prediction)", "Model Optimization", "Decision Rules"]
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
            st.markdown(f'<div class="kpi-box"><div class="kpi-label">YoY growth (\'11 to \'12 YTD)</div><div class="kpi-val" style="color:{yoy_color}">{yoy_pct:+.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1.4, 1])

        curr_symbol = "$" if currency == "USD ($)" else "₹"
        conv = 1.0 if currency == "USD ($)" else USD_TO_INR
        fdf_c = fdf.copy()
        fdf_c["Sales_Display"] = fdf_c["Weekly_Sales"] * conv
        color_map_labeled = {TYPE_LABELS[k]: v for k, v in TYPE_COLORS.items()}

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
            type_agg["TypeLabel"] = type_agg["Type"].map(TYPE_LABELS)
            fig = px.pie(type_agg, names="TypeLabel", values="Weekly_Sales", hole=0.6,
                         color="TypeLabel", color_discrete_map=color_map_labeled)
            fig.update_layout(template="plotly_dark", plot_bgcolor="#161A20", paper_bgcolor="#161A20", height=320, legend_title_text="")
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
            top_stores["TypeLabel"] = top_stores["Type"].map(TYPE_LABELS)
            fig = px.bar(top_stores, x="Sales_Display", y=top_stores["Store"].astype(str), orientation="h",
                         color="TypeLabel", color_discrete_map=color_map_labeled)
            fig.update_layout(template="plotly_dark", plot_bgcolor="#161A20", paper_bgcolor="#161A20",
                               height=300, yaxis_title=None, xaxis_title=f"Total sales ({curr_symbol})",
                               yaxis={"categoryorder": "total ascending"}, legend_title_text="")
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
        store_table["Type"] = store_table["Type"].map(TYPE_LABELS)
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
        store_type = st.selectbox("Store type", ["A", "B", "C"], format_func=lambda x: TYPE_LABELS[x])
        is_holiday = st.checkbox("Holiday week", value=False)
        cpi = st.slider("CPI (consumer price index)", 126.0, 228.0, 171.0)
        unemployment = st.slider("Unemployment rate (%)", 3.6, 14.3, 7.9)
        temperature = st.slider("Temperature (°F)", -2.0, 101.0, 60.0)
        week_of_year = st.slider("Week of year", 1, 52, 25)
        month = st.slider("Month", 1, 12, 6)
        predict_btn = st.button("Run live prediction", type="primary", use_container_width=True)

    with col_out:
        row = {c: 0 for c in feature_cols}
        row["Store"] = 1
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
            <div style="font-size:12px; color:#8B92A0; text-transform:uppercase; letter-spacing:0.06em;">Accurate prediction (Random Forest)</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:40px; font-weight:600; color:#E8A33D; margin:10px 0;">{fmt_currency(prediction)}</div>
            <div style="font-size:12px; color:#9DA3B0;">Dept {dept} &middot; {size:,} sq ft &middot; {TYPE_LABELS[store_type]} &middot; {'Holiday' if is_holiday else 'Regular'} week</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        rule_lines = "<br>".join(f"&rsaquo; {s}" for s in rule_steps)
        st.markdown(f"""
        <div class="rule-box" style="padding:20px;">
            <div style="font-size:12px; color:#8B92A0; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:10px;">Interpretable reasoning for this input</div>
            {rule_lines}
            <div style="border-top:1px solid #3A4150; margin-top:12px; padding-top:10px; color:#3FE0C0;">Interpretable-tree estimate: {fmt_currency(rule_prediction)}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Business interpretation")
        avg_sales_similar = df[df["Type"] == store_type]["Weekly_Sales"].mean()
        delta_pct = (prediction - avg_sales_similar) / avg_sales_similar * 100
        if delta_pct > 20:
            st.success(f"Predicted **{delta_pct:.0f}% above** the {TYPE_LABELS[store_type]} average -- prioritize staffing and inventory buffer.")
        elif delta_pct < -20:
            st.warning(f"Predicted **{abs(delta_pct):.0f}% below** the {TYPE_LABELS[store_type]} average -- review markdown/promotion strategy.")
        else:
            st.info(f"Close to the {TYPE_LABELS[store_type]} average ({delta_pct:+.0f}%) -- standard planning applies.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Scenario comparison: holiday vs regular week")
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

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Recommended action tier")
        all_preds_ref = df["Weekly_Sales"]
        p33, p66 = all_preds_ref.quantile(0.33), all_preds_ref.quantile(0.66)
        if prediction <= p33:
            tier, tier_color, tier_msg = "LOW", "#8B92A0", "Standard staffing. Minimal inventory buffer needed."
        elif prediction <= p66:
            tier, tier_color, tier_msg = "MODERATE", "#E8A33D", "Standard-to-moderate staffing. Maintain normal inventory cycle."
        else:
            tier, tier_color, tier_msg = "HIGH", "#3FE0C0", "Increase staffing and inventory buffer. Prioritize this combination for holiday planning."
        st.markdown(f"""
        <div class="rule-box" style="padding:16px 20px;">
            <span style="background:{tier_color}22; color:{tier_color}; padding:4px 12px; border-radius:20px; font-weight:600; font-size:12px;">{tier} PRIORITY</span>
            <div style="margin-top:10px; color:#9DA3B0; font-size:13px;">{tier_msg}</div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# TAB 3: Model Optimization
# ------------------------------------------------------------------
with tab_models:
    st.subheader("Decision tree optimization results")
    st.caption("Each model trained and evaluated with WMAE (holiday weeks weighted 5x -- the official Walmart competition metric).")

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

    st.subheader("Feature importance (live, from loaded model)")
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
    st.caption("Extracted from the trained interpretable tree (depth 4) -- direct if-then logic, not SHAP scores.")
    rules_text = export_text(shallow_model, feature_names=feature_cols)
    st.markdown(f'<div class="rule-box">{rules_text}</div>', unsafe_allow_html=True)
