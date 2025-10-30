# src/plot_logic_data.py

from __future__ import annotations
from typing import List
import pandas as pd
import streamlit as st

# !!! CORRECTED IMPORT: Get helpers from the new utility file
from .plot_utils import _ensure_time_column_after_reset, _truncate_label, DEVICE_ID_COLUMN

# ==================== Core Data Prep (Cached) ====================
@st.cache_data(show_spinner=False)
def prepare_long_data(
    df: pd.DataFrame,
    selected_features: List[str],
    selected_devices: List[str],
    compact_labels: bool,
    label_max_len: int,
) -> pd.DataFrame:
    """
    Filters data, coerces features, melts to long format, and creates SeriesKey/LegendLabel.
    CACHED to prevent re-running when only plot *style* changes.
    """
    # 1) Filter by devices
    df_filtered = df[df[DEVICE_ID_COLUMN].astype(str).isin([str(x) for x in selected_devices])].copy()

    # 2) Reset index -> Time
    df_long = _ensure_time_column_after_reset(df_filtered)

    # 3) Filter and coerce features
    selected_features = [c for c in selected_features if c in df_long.columns]
    for col in selected_features:
        if col in df_long.columns:
            # Use errors='coerce' to turn non-numeric strings into NaN
            df_long[col] = pd.to_numeric(df_long[col], errors="coerce")

    # 4) Melt to long format
    if not selected_features:
        return pd.DataFrame(columns=["Time", DEVICE_ID_COLUMN, "Feature", "Value", "SeriesKey", "LegendLabel"])
    df_melted = pd.melt(
        df_long,
        id_vars=["Time", DEVICE_ID_COLUMN],
        value_vars=selected_features,
        var_name="Feature",
        value_name="Value",
    )

    # 5) Create unique identity (SeriesKey) and display label (LegendLabel)
    df_melted.dropna(subset=["Time"], inplace=True)
    df_melted["SeriesKey"] = df_melted[DEVICE_ID_COLUMN].astype(str) + " - " + df_melted["Feature"].astype(str)

    base_labels = df_melted["SeriesKey"].copy()
    if compact_labels:
        base_labels = (df_melted[DEVICE_ID_COLUMN].astype(str) + " - " + df_melted["Feature"].astype(str)).apply(
            lambda x: _truncate_label(x, label_max_len)
        )
    df_melted["LegendLabel"] = base_labels

    return df_melted.sort_values("Time")