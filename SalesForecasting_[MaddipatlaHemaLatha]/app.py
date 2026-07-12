"""
End-to-End Sales Forecasting & Demand Intelligence System
Streamlit Dashboard (Task 7)

Run locally with:   streamlit run app.py
Deploy on Streamlit Community Cloud by pushing this repo to GitHub and
pointing share.streamlit.io at this file.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence", layout="wide")

# ---------------------------------------------------------------------------
# NOTE ON "BEST MODEL": Task 3 in the notebook compares SARIMA, Prophet, and
# XGBoost and picks whichever has the lowest RMSE. This app uses SARIMA for
# the live Forecast Explorer, since it's dependency-light and fast enough to
# refit on every page interaction without slowing the app down. If XGBoost or
# Prophet won in your notebook, swap the model used in run_sarima_forecast()
# below for consistency with your written findings.
# ---------------------------------------------------------------------------

# Load data files relative to THIS SCRIPT's location, not the current working
# directory. This matters on Streamlit Community Cloud, which runs the app
# with the working directory set to the repo root -- if app.py and train.csv
# live inside a subfolder, a plain pd.read_csv("train.csv") won't find them.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "train.csv")


# ------------------------- Data loading & caching --------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(TRAIN_CSV_PATH, encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    return df


@st.cache_data
def get_monthly_series(_df, column=None, value=None):
    data = _df if column is None else _df[_df[column] == value]
    monthly = data.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum()
    return monthly


@st.cache_data
def get_weekly_series(_df):
    weekly = _df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"].sum()
    return weekly


def month_to_season(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    return "Autumn"


# ------------------------------ Model helpers -------------------------------

def run_sarima_forecast(series, test_months=3, future_months=3):
    """Fits SARIMA, returns (test_forecast, test_actual, mae, rmse, future_forecast)."""
    if len(series) < test_months + 6:
        return None  # not enough data for a meaningful split

    train, test = series.iloc[:-test_months], series.iloc[-test_months:]
    seasonal_period = 12 if len(train) >= 24 else 0

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, seasonal_period) if seasonal_period else (0, 0, 0, 0)

    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    test_forecast = model.forecast(steps=test_months)

    mae = mean_absolute_error(test, test_forecast)
    rmse = np.sqrt(mean_squared_error(test, test_forecast))

    # Refit on full series for the true future forecast
    final_model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    future_forecast = final_model.forecast(steps=future_months)

    return test_forecast, test, mae, rmse, future_forecast


@st.cache_data
def compute_anomalies(_df):
    weekly = get_weekly_series(_df)
    weekly_df = weekly.reset_index()
    weekly_df.columns = ["Date", "Sales"]

    # Isolation Forest
    X = weekly_df["Sales"].values.reshape(-1, 1)
    iso = IsolationForest(contamination=0.05, random_state=42).fit(X)
    weekly_df["iso_anomaly"] = iso.predict(X) == -1

    # Z-score (rolling mean/std of PRECEDING 4 weeks, excluding current week)
    weekly_df["rolling_mean"] = weekly_df["Sales"].shift(1).rolling(4).mean()
    weekly_df["rolling_std"] = weekly_df["Sales"].shift(1).rolling(4).std()
    weekly_df["z_score"] = (weekly_df["Sales"] - weekly_df["rolling_mean"]) / weekly_df["rolling_std"]
    weekly_df["zscore_anomaly"] = weekly_df["z_score"].abs() > 2

    weekly_df["any_anomaly"] = weekly_df["iso_anomaly"] | weekly_df["zscore_anomaly"]
    return weekly_df


@st.cache_data
def compute_clusters(_df, k=4):
    rows = []
    for subcat in _df["Sub-Category"].unique():
        sub = _df[_df["Sub-Category"] == subcat]
        total_sales = sub["Sales"].sum()

        yearly = sub.groupby("Year")["Sales"].sum().sort_index()
        if len(yearly) >= 2 and yearly.iloc[0] != 0:
            growth_rate = ((yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0]) * 100
        else:
            growth_rate = 0

        monthly = sub.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum()
        volatility = monthly.std()
        avg_order_value = sub["Sales"].mean()

        rows.append({
            "Sub-Category": subcat, "Total_Sales": total_sales,
            "Growth_Rate_Pct": growth_rate, "Volatility": volatility,
            "Avg_Order_Value": avg_order_value
        })

    feat_df = pd.DataFrame(rows).set_index("Sub-Category")
    feature_cols = ["Total_Sales", "Growth_Rate_Pct", "Volatility", "Avg_Order_Value"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df[feature_cols])

    k = min(k, len(feat_df) - 1) if len(feat_df) > 1 else 1
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    feat_df["Cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    feat_df["PC1"], feat_df["PC2"] = X_pca[:, 0], X_pca[:, 1]

    return feat_df


# ---------------------------------- App -------------------------------------

st.title("📊 Sales Forecasting & Demand Intelligence")

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"`train.csv` not found at `{TRAIN_CSV_PATH}`. Make sure train.csv sits in the "
             f"same folder as app.py in your repo.")
    st.stop()

page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"]
)

# ----------------------------- Page 1: Overview ------------------------------
if page == "Sales Overview":
    st.header("Sales Overview Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Sales by Year")
        yearly_sales = df.groupby("Year")["Sales"].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        yearly_sales.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_ylabel("Sales")
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Sales Trend")
        monthly = get_monthly_series(df)
        fig, ax = plt.subplots(figsize=(6, 4))
        monthly.plot(ax=ax, color="darkorange")
        ax.set_ylabel("Sales")
        st.pyplot(fig)

    st.subheader("Sales by Region & Category")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        regions = st.multiselect("Region", options=sorted(df["Region"].unique()),
                                  default=sorted(df["Region"].unique()))
    with fcol2:
        categories = st.multiselect("Category", options=sorted(df["Category"].unique()),
                                     default=sorted(df["Category"].unique()))

    filtered = df[df["Region"].isin(regions) & df["Category"].isin(categories)]
    pivot = filtered.groupby(["Region", "Category"])["Sales"].sum().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Sales")
    ax.set_title("Sales by Region & Category (filtered)")
    st.pyplot(fig)

    st.dataframe(pivot)

# ------------------------- Page 2: Forecast Explorer -------------------------
elif page == "Forecast Explorer":
    st.header("Forecast Explorer")
    st.caption("Model: SARIMA — see the note at the top of app.py if a different "
               "model won in your Task 3 comparison.")

    seg_type = st.selectbox("Select segment type", ["Category", "Region"])
    seg_value = st.selectbox(f"Select {seg_type}", sorted(df[seg_type].unique()))
    horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=3, value=3)

    series = get_monthly_series(df, column=seg_type, value=seg_value)

    result = run_sarima_forecast(series, test_months=3, future_months=3)

    if result is None:
        st.warning("Not enough historical data for this segment to fit a reliable model.")
    else:
        test_forecast, test_actual, mae, rmse, future_forecast = result
        future_forecast = future_forecast.iloc[:horizon]

        fig, ax = plt.subplots(figsize=(10, 5))
        series.plot(ax=ax, label="Historical", color="steelblue")
        future_forecast.plot(ax=ax, label=f"{horizon}-Month Forecast", color="red", marker="o")
        ax.set_title(f"{seg_value} ({seg_type}) — Sales Forecast")
        ax.set_ylabel("Sales")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Forecasted Values")
        st.dataframe(future_forecast.rename("Forecast").to_frame())

        st.subheader("Model Accuracy (on held-out test months)")
        m1, m2 = st.columns(2)
        m1.metric("MAE", f"{mae:,.2f}")
        m2.metric("RMSE", f"{rmse:,.2f}")

# --------------------------- Page 3: Anomaly Report --------------------------
elif page == "Anomaly Report":
    st.header("Anomaly Report")
    st.caption("Combines Isolation Forest and Z-score methods from Task 5.")

    weekly_df = compute_anomalies(df)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(weekly_df["Date"], weekly_df["Sales"], label="Weekly Sales", color="steelblue", alpha=0.7)

    both = weekly_df[weekly_df["iso_anomaly"] & weekly_df["zscore_anomaly"]]
    only_iso = weekly_df[weekly_df["iso_anomaly"] & ~weekly_df["zscore_anomaly"]]
    only_z = weekly_df[weekly_df["zscore_anomaly"] & ~weekly_df["iso_anomaly"]]

    ax.scatter(both["Date"], both["Sales"], color="purple", marker="*", s=180, label="Both methods", zorder=6)
    ax.scatter(only_iso["Date"], only_iso["Sales"], color="red", marker="x", s=100, label="Isolation Forest only", zorder=5)
    ax.scatter(only_z["Date"], only_z["Sales"], color="darkorange", marker="^", s=100, label="Z-score only", zorder=5)
    ax.set_title("Weekly Sales — Anomalies Highlighted")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Detected Anomaly Weeks")
    anomaly_table = weekly_df[weekly_df["any_anomaly"]][
        ["Date", "Sales", "iso_anomaly", "zscore_anomaly"]
    ].sort_values("Date")
    st.dataframe(anomaly_table, use_container_width=True)

# ---------------------- Page 4: Product Demand Segments ----------------------
elif page == "Product Demand Segments":
    st.header("Product Demand Segments")
    st.caption("K-Means clustering of product sub-categories (Task 6).")

    k = st.slider("Number of clusters (k)", min_value=2, max_value=6, value=4)
    clusters = compute_clusters(df, k=k)

    fig, ax = plt.subplots(figsize=(9, 7))
    for cluster_id in sorted(clusters["Cluster"].unique()):
        subset = clusters[clusters["Cluster"] == cluster_id]
        ax.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster_id}", s=100)
        for subcat, row in subset.iterrows():
            ax.annotate(subcat, (row["PC1"], row["PC2"]), fontsize=8, alpha=0.7,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_title("Sub-Category Clusters (PCA-reduced)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Sub-Category → Cluster Mapping")
    st.caption("Cluster NUMBERS are arbitrary — assign your own business labels "
               "(e.g. 'High Volume, Stable Demand') based on the averages below.")

    display_cols = ["Cluster", "Total_Sales", "Growth_Rate_Pct", "Volatility", "Avg_Order_Value"]
    st.dataframe(clusters[display_cols].sort_values("Cluster"), use_container_width=True)

    st.subheader("Cluster Averages (use this to assign meaningful labels)")
    st.dataframe(clusters.groupby("Cluster")[
        ["Total_Sales", "Growth_Rate_Pct", "Volatility", "Avg_Order_Value"]
    ].mean())
