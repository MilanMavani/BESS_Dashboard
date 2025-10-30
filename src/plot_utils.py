# src/plot_utils.py

from __future__ import annotations
import re
from typing import List, Tuple, Dict
import pandas as pd

# Global constant (can be shared via a separate config file if desired)
DEVICE_ID_COLUMN = "device-address:uid"

# ==================== Helpers ====================

def _infer_numeric_like_columns(
    df: pd.DataFrame, exclude_cols: list[str] | set[str] | None = None, sample_n: int = 200, min_success: float = 0.6
) -> List[str]:
    """Infers columns that are numeric or can be coerced to numeric."""
    if exclude_cols is None: exclude_cols = set()
    else: exclude_cols = set(exclude_cols)
    numeric_cols = []
    for col in df.columns:
        if col in exclude_cols: continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif df[col].dtype == "object":
            s = df[col].dropna()
            if not s.empty:
                coerced = pd.to_numeric(s.head(sample_n), errors="coerce")
                success = coerced.notna().mean()
                if success >= min_success:
                    numeric_cols.append(col)
    ordered = [col for col in df.columns if col in numeric_cols]
    return ordered

def _ensure_time_column_after_reset(df_filtered: pd.DataFrame) -> pd.DataFrame:
    # Ensures 'Time' column exists and is datetime type
    df_long = df_filtered.reset_index()
    time_col = getattr(df_filtered.index, "name", None) or "index"
    if time_col in df_long.columns and time_col != "Time":
        df_long.rename(columns={time_col: "Time"}, inplace=True)
    elif "index" in df_long.columns:
        df_long.rename(columns={"index": "Time"}, inplace=True)
    if not pd.api.types.is_datetime64_any_dtype(df_long["Time"]):
        df_long["Time"] = pd.to_datetime(df_long["Time"], errors="coerce")
    return df_long

def _truncate_label(s: str, max_len: int = 26) -> str:
    # Truncates a string for compact legends
    s = str(s)
    return s if len(s) <= max_len else (s[: max_len - 1] + "…")

def _build_unique_legend_labels(series_keys_in_order: List[str], base_labels: Dict[str, str]) -> Dict[str, str]:
    """Ensures legend labels are unique by appending a count if necessary (e.g., 'Label (2)')."""
    counts: Dict[str, int] = {}
    out: Dict[str, str] = {}
    for k in series_keys_in_order:
        base = base_labels[k]
        n = counts.get(base, 0) + 1
        counts[base] = n
        if n == 1:
            out[k] = base
        else:
            out[k] = f"{base} ({n})"
    return out

def diagnose_missing_series(
    df_long: pd.DataFrame,
    selected_features: List[str],
    selected_devices: List[str],
) -> Tuple[int, int, Dict[str, str]]:
    """Returns diagnostics on expected vs. actual plotted series."""
    devices_present = df_long[DEVICE_ID_COLUMN].astype(str).unique().tolist() if DEVICE_ID_COLUMN in df_long.columns else []
    n_devices = len(devices_present)

    expected_series = len(selected_features) * n_devices
    actual_series = len(df_long["SeriesKey"].unique()) if "SeriesKey" in df_long.columns else 0

    reasons = {}
    plotted_features = set(df_long["Feature"].unique().tolist()) if "Feature" in df_long.columns else set()

    for f in selected_features:
        if f not in plotted_features:
            reasons[f] = "No data points (likely all NA) or column not in filtered data."
    return expected_series, actual_series, reasons