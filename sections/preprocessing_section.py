# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from typing import Optional, Tuple, List
DEVICE_ID_COLUMN = "device-address:uid"

# ---------- Profile detection ----------
def _detect_profile(df: pd.DataFrame, fallback_profile: Optional[str]) -> str:

    if fallback_profile in {"Hymon", "Sc_Com / HyCon", "Cell Data"}:
        return fallback_profile
    cols = set(df.columns.astype(str))
    if {"TT.MM.JJJJ", "hh:mm:ss:fff"}.issubset(cols):
        return "Cell Data"
    if {"Date", "Time"}.issubset(cols):
        return "Hymon"
    if "Timestamp" in cols:
        return "Sc_Com / HyCon"
    return "Unknown"


# ---------- Datetime index creation (automatic, no UI) ----------
def _set_datetime_index_auto(df_in: pd.DataFrame, profile: str) -> Tuple[pd.DataFrame, str]:

    df = df_in.copy()

    # If already DatetimeIndex: just sort (no dedup)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
        return df, "Datetime index already present"

    cols = df.columns.astype(str)
    try:
        # Sc_Com / HyCon: single combined "Timestamp"
        if profile == "Sc_Com / HyCon" and "Timestamp" in cols:
            ts = pd.to_datetime(df["Timestamp"], errors="coerce", infer_datetime_format=True)
            df = df.set_index(ts.rename("Timestamp")).sort_index()
            # Keep original rows; optionally drop the source column for cleanliness
            if "Timestamp" in df.columns:
                df.drop(columns=["Timestamp"], inplace=True, errors="ignore")
            return df, "Index Column created 'Timestamp'"

        # Hymon: separate Date + Time (drop TZ if present)
        if profile == "Hymon" and {"Date", "Time"}.issubset(cols):
            combo = df["Date"].astype(str) + " " + df["Time"].astype(str)
            ts = pd.to_datetime(combo, errors="coerce", infer_datetime_format=True)
            df = df.set_index(ts.rename("Timestamp"))
            df.drop(columns=[c for c in ["Date", "Time", "TZ"] if c in df.columns], inplace=True, errors="ignore")
            df = df.sort_index()
            return df, "Index Column created 'Date' + 'Time'"

        # Cell Data: explicit known format DD.MM.YYYY HH:MM:SS:ms
        if profile == "Cell Data" and {"TT.MM.JJJJ", "hh:mm:ss:fff"}.issubset(cols):
            combo = df["TT.MM.JJJJ"].astype(str) + " " + df["hh:mm:ss:fff"].astype(str)
            # NOTE: pandas can parse %f with 1–6 digits; if your data holds 3 digits (ms),
            # pandas will interpret appropriately. If needed later, we can pad to 6 digits.
            ts = pd.to_datetime(combo, errors="coerce", format="%d.%m.%Y %H:%M:%S:%f")
            df = df.set_index(ts.rename("Timestamp"))
            df.drop(columns=[c for c in ["TT.MM.JJJJ", "hh:mm:ss:fff"] if c in df.columns], inplace=True, errors="ignore")
            df = df.sort_index()
            return df, "Index Column created 'TT.MM.JJJJ' + 'hh:mm:ss:fff'"

        # Fallbacks
        if "Timestamp" in cols:
            ts = pd.to_datetime(df["Timestamp"], errors="coerce", infer_datetime_format=True)
            df = df.set_index(ts.rename("Timestamp")).sort_index()
            if "Timestamp" in df.columns:
                df.drop(columns=["Timestamp"], inplace=True, errors="ignore")
            return df, "Index Column created "

        # Try first column that parses well
        best_col = None
        best_null_ratio = 1.0
        for c in cols:
            if c == DEVICE_ID_COLUMN:
                continue
            parsed = pd.to_datetime(df[c].astype(str), errors="coerce", infer_datetime_format=True)
            null_ratio = parsed.isna().mean()
            if null_ratio < 0.2 and null_ratio < best_null_ratio:
                best_null_ratio = null_ratio
                best_col = c

        if best_col:
            ts = pd.to_datetime(df[best_col].astype(str), errors="coerce", infer_datetime_format=True)
            df = df.set_index(ts.rename("Timestamp")).sort_index()
            df.drop(columns=[best_col], inplace=True, errors="ignore")
            return df, f"Datetime index created from '{best_col}' (heuristic); duplicates preserved."

        return df, "Datetime index could not be inferred; data left unchanged."

    except Exception as e:
        return df_in.copy(), f"Datetime step skipped due to error: {e}"


# ---------- Missing values (optional, user-triggered) ----------
def _missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isna().sum()
    pct = (counts / max(1, len(df))) * 100.0
    out = pd.DataFrame({"missing": counts, "missing_%": pct.round(2)})
    return out.sort_values(by="missing", ascending=False)


def _apply_imputation(
    df_in: pd.DataFrame,
    target_cols: List[str],
    strategy: str,
    *,
    per_device: bool,
    fill_value: Optional[str] = None,
    interp_method: str = "time",
) -> pd.DataFrame:
    # Imputation methods
    df = df_in.copy()
    if not target_cols:
        return df

    # Never touch device id column
    target_cols = [c for c in target_cols if c != DEVICE_ID_COLUMN and c in df.columns]
    if not target_cols:
        return df

    # Helpers
    def _ffill(g: pd.DataFrame) -> pd.DataFrame:
        gg = g.copy()
        gg[target_cols] = gg[target_cols].ffill()
        return gg

    def _bfill(g: pd.DataFrame) -> pd.DataFrame:
        gg = g.copy()
        gg[target_cols] = gg[target_cols].bfill()
        return gg

    def _interpolate(g: pd.DataFrame) -> pd.DataFrame:
        gg = g.copy()
        # interpolate only numeric columns to avoid coercion
        num_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(gg[c])]
        if not num_cols:
            return gg
        method = interp_method
        # 'time' requires DatetimeIndex; fallback to 'linear' if not available
        if method == "time" and not isinstance(gg.index, pd.DatetimeIndex):
            method = "linear"
        gg[num_cols] = gg[num_cols].interpolate(method=method, limit_direction="both")
        return gg

    def _fill_constant(g: pd.DataFrame) -> pd.DataFrame:
        gg = g.copy()
        gg[target_cols] = gg[target_cols].fillna(fill_value)
        return gg

    def _fill_stat(g: pd.DataFrame, stat: str) -> pd.DataFrame:
        gg = g.copy()
        num_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(gg[c])]
        if not num_cols:
            return gg
        if stat == "mean":
            vals = gg[num_cols].mean()
        elif stat == "median":
            vals = gg[num_cols].median()
        elif stat == "mode":
            vals = gg[num_cols].mode().iloc[0] if not gg[num_cols].mode().empty else gg[num_cols].median()
        else:
            return gg
        for c in num_cols:
            gg[c] = gg[c].fillna(vals[c])
        return gg

    # Row drops
    def _drop_any(g: pd.DataFrame) -> pd.DataFrame:
        return g.dropna(subset=target_cols, how="any")

    def _drop_all(g: pd.DataFrame) -> pd.DataFrame:
        return g.dropna(subset=target_cols, how="all")

    # Choose operator
    def _apply_group(g: pd.DataFrame) -> pd.DataFrame:
        if strategy == "Forward Fill (ffill)":
            return _ffill(g)
        if strategy == "Backward Fill (bfill)":
            return _bfill(g)
        if strategy == "Interpolate":
            return _interpolate(g)
        if strategy == "Fill with constant":
            return _fill_constant(g)
        if strategy == "Fill with mean":
            return _fill_stat(g, "mean")
        if strategy == "Fill with median":
            return _fill_stat(g, "median")
        if strategy == "Fill with mode":
            return _fill_stat(g, "mode")
        if strategy == "Drop rows (if any selected is NaN)":
            return _drop_any(g)
        if strategy == "Drop rows (if all selected are NaN)":
            return _drop_all(g)
        return g

    # Apply
    if per_device and DEVICE_ID_COLUMN in df.columns:
        df = (
            df.sort_index() if isinstance(df.index, pd.DatetimeIndex) else df
        ).groupby(DEVICE_ID_COLUMN, group_keys=False).apply(_apply_group)
    else:
        if strategy in {"Drop rows (if any selected is NaN)", "Drop rows (if all selected are NaN)"}:
            # For drops, do not copy twice
            if strategy == "Drop rows (if any selected is NaN)":
                df = df.dropna(subset=target_cols, how="any")
            else:
                df = df.dropna(subset=target_cols, how="all")
        else:
            df = _apply_group(df)

    return df


# ---------- Main (minimal UI + optional imputation) ----------
def preprocessing_section(df_original):

    if df_original is None or (isinstance(df_original, pd.DataFrame) and df_original.empty):
        st.warning("No data loaded. Please upload a file and click 'Load Data' first.")
        st.session_state.processed_data = None
        return

    st.header("🔬 Data Preprocessing")
    st.caption(
        "A Datetime index is created automatically based on the detected data profile. "
        "Missing values are shown below; you can optionally apply an imputation strategy."
    )

    # Always start from the provided data for datetime step
    df_in = df_original.copy()

    # Detect profile (use the one chosen in Load tab if present)
    selected_profile = st.session_state.get("csv_profile_selector")
    profile = _detect_profile(df_in, selected_profile)

    # Apply automatic datetime index (only change we make automatically)
    df_dt, dt_info = _set_datetime_index_auto(df_in, profile)

    # Persist after datetime step (before any optional imputation)
    st.session_state.processed_data = df_dt

    # ---------- Summary ----------
    rows, cols = df_dt.shape
    total_missing = int(df_dt.isna().sum().sum())
    st.subheader("Current Data Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", rows)
    c2.metric("Columns", cols)
    c3.metric("Missing values (total)", total_missing)

    with st.expander("ℹ️ Automatic datetime step (details)", expanded=False):
        st.markdown(f"- **Profile detected**: `{profile}`")
        st.markdown(f"- **Result**: {dt_info}")
        if not isinstance(df_dt.index, pd.DatetimeIndex):
            st.warning("Datetime index is still not set (data kept unchanged). Plotting may require a datetime index.")

    st.caption("First 5 rows (Processed):")
    st.dataframe(df_dt.head())

    st.caption("Column data types:")
    st.dataframe(pd.DataFrame(df_dt.dtypes, columns=["dtype"]))

    # ---------- Missing values (read-only) ----------
    missing_table = _missing_summary(df_dt)
    if missing_table["missing"].sum() == 0:
        st.success("No missing values found in the current dataset.")
        return

    with st.expander("🧩 Missing Values (optional imputation)", expanded=False):
        st.markdown("Below are the columns with missing values:")
        st.dataframe(missing_table[missing_table["missing"] > 0])

        cols_with_na = missing_table[missing_table["missing"] > 0].index.tolist()
        target_cols = st.multiselect(
            "Select column(s) to handle missing values:",
            options=cols_with_na,
            default=cols_with_na[:1] if cols_with_na else [],
        )

        strategy = st.selectbox(
            "Imputation strategy:",
            options=[
                "Forward Fill (ffill)",
                "Backward Fill (bfill)",
                "Interpolate",
                "Fill with constant",
                "Fill with mean",
                "Fill with median",
                "Fill with mode",
                "Drop rows (if any selected is NaN)",
                "Drop rows (if all selected are NaN)",
            ],
            index=0,
            help="Choose how to handle missing values in the selected column(s).",
        )

        # Strategy-specific inputs
        fill_value = None
        interp_method = "time"
        col1, col2 = st.columns(2)
        if strategy == "Fill with constant":
            with col1:
                fill_value = st.text_input("Constant value to fill (auto-cast for numeric columns)", value="0")
        if strategy == "Interpolate":
            with col1:
                interp_method = st.selectbox(
                    "Interpolation method",
                    options=["time", "linear"],
                    index=0,
                    help="‘time’ requires a DatetimeIndex; falls back to ‘linear’ if index is not datetime.",
                )

        per_device_default = DEVICE_ID_COLUMN in df_dt.columns
        per_device = st.checkbox(
            f"Apply per device (group by '{DEVICE_ID_COLUMN}')",
            value=per_device_default,
            disabled=not per_device_default,
        )

        # Preview counts
        before_missing = int(df_dt[target_cols].isna().sum().sum()) if target_cols else 0
        st.caption(f"Missing in selected columns (before): **{before_missing}**")

        if st.button("Apply imputation to selected column(s)"):
            if not target_cols:
                st.warning("Please select at least one column.")
            else:
                try:
                    df_new = _apply_imputation(
                        df_dt,
                        target_cols=target_cols,
                        strategy=strategy,
                        per_device=per_device,
                        fill_value=fill_value,
                        interp_method=interp_method,
                    )
                    after_missing = int(df_new[target_cols].isna().sum().sum())
                    delta = before_missing - after_missing
                    st.success(
                        f"Imputation applied: {strategy}."
                        f" Missing in selected columns: {before_missing} → {after_missing} (Δ {delta})."
                    )
                    # Persist the change
                    st.session_state.processed_data = df_new
                    # Refresh summary snippet
                    st.caption("Updated head (Processed):")
                    st.dataframe(df_new.head())
                except Exception as e:
                    st.error(f"Error applying imputation: {e}")
