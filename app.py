import streamlit as st
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# =============================
# Helpers
# =============================
def _normalize_cols(cols):
    import re
    return [re.sub(r"_+", "_", re.sub(r"[^\w]+", "_", c.strip().lower())).strip("_") for c in cols]

@st.cache_data
def load_data():
    df = pd.read_csv("Book.csv")
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all").reset_index(drop=True)

    df.columns = _normalize_cols(df.columns)

    area_wholly_irrigated = "area_of_wholly_irrigated_holdings_uom_ha_hectare_scaling_factor_1"
    area_wholly_unirrigated = "area_of_wholly_unirrigated_holdings_uom_ha_hectare_scaling_factor_1"
    area_partially_irrigated = "area_of_partially_irrigated_holdings_uom_ha_hectare_scaling_factor_1"
    net_irrigated_col = "net_irrigated_area_of_holdings_receiving_irrigation_uom_ha_hectare_scaling_factor_1"

    numeric_cols = [area_wholly_irrigated, area_wholly_unirrigated, area_partially_irrigated, net_irrigated_col]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0.0
            st.warning(f"Missing expected column in data: {c}. Filled with zeros.")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    area_cols = [area_wholly_irrigated, area_wholly_unirrigated, area_partially_irrigated]
    df["total_area"] = df[area_cols].sum(axis=1)

    df["irrigation_coverage_pct"] = np.where(
        df["total_area"] > 0,
        (df[net_irrigated_col] / df["total_area"]) * 100.0,
        0.0
    )

    def assign_priority(pct: float) -> str:
        if pd.isna(pct):
            return "Medium"
        if pct < 33:
            return "High"
        elif pct < 66:
            return "Medium"
        else:
            return "Low"

    df["subsidy_priority"] = df["irrigation_coverage_pct"].apply(assign_priority)

    required_cats = ["social_group_type", "land_area_size", "category_of_holdings"]
    for cat in required_cats:
        if cat not in df.columns:
            raise KeyError(
                f"Expected column '{cat}' not found after normalization.\n"
                f"Available columns: {list(df.columns)}"
            )
        df[cat] = df[cat].astype(str)

    df = df[df["subsidy_priority"].isin(["High", "Medium", "Low"])].copy()
    return df

@st.cache_resource
def train_model(df: pd.DataFrame):
    feature_cols = ["social_group_type", "land_area_size", "category_of_holdings"]

    # Down-sample each class to the minority count for balanced training
    y = df["subsidy_priority"]
    min_count = y.value_counts().min()
    parts = []
    for cls in ["High", "Medium", "Low"]:
        d = df[df["subsidy_priority"] == cls]
        if not d.empty:
            parts.append(d.sample(min(len(d), min_count), random_state=42))
    df_bal = pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

    X_bal = df_bal[feature_cols].astype(str)
    y_bal = df_bal["subsidy_priority"].astype(str)

    pipe = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ("clf", DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=6,
            min_samples_leaf=50
        ))
    ])
    pipe.fit(X_bal, y_bal)
    return pipe

# =============================
# App
# =============================
st.title("🚜 Irrigation Subsidy Priority Predictor")
st.write("Predicts whether a farmer should get **High**, **Medium**, or **Low** priority for irrigation subsidies.")

df = load_data()
model = train_model(df)

# Inputs
social_group_type = st.selectbox("Social Group Type", sorted(df["social_group_type"].dropna().unique()))
land_area_size = st.selectbox("Land Area Size", sorted(df["land_area_size"].dropna().unique()))
category_of_holdings = st.selectbox("Category of Holdings", sorted(df["category_of_holdings"].dropna().unique()))

if st.button("Predict Priority"):
    new_data = pd.DataFrame({
        "social_group_type": [social_group_type],
        "land_area_size": [land_area_size],
        "category_of_holdings": [category_of_holdings],
    })
    prediction = model.predict(new_data)[0]
    st.success(f"🎯 Predicted Subsidy Priority: **{prediction}**")
