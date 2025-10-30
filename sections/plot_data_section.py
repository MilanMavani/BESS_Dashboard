# sections/plot_data_section.py

from typing import Optional, List
from datetime import datetime
import pandas as pd
import streamlit as st

# Get the CACHED function
from src.plot_logic_data import prepare_long_data 
# Get the plotting functions
from src.plot_logic_chart import (
    plot_plotly_dual_y,
    plot_plotly_single_axis,
    plot_seaborn_dual_y,
    plot_seaborn_single_axis,
)
# Get the necessary utility helpers/constants
from src.plot_utils import (
    diagnose_missing_series,
    _infer_numeric_like_columns,
    DEVICE_ID_COLUMN
)


def plot_data_section():
    st.header("📊 Time Series Plotting")
    st.markdown(
        "Compare how different **devices** perform for the same **feature**, "
        "or how different **features** compare on the same **device** over time."
    )

    df = st.session_state.get("processed_data")
    if df is None or df.empty:
        st.warning("No data available. Please load data and apply preprocessing first.")
        return

    # Basic data checks
    if not isinstance(df.index, pd.DatetimeIndex):
        st.error("Datetime index not found. Please apply datetime preprocessing.")
        return

    if DEVICE_ID_COLUMN not in df.columns:
        st.error(f"Required column **'{DEVICE_ID_COLUMN}'** not found.")
        st.caption(f"Please ensure the column name is exactly '{DEVICE_ID_COLUMN}'.")
        return

    st.success("Data ready for multi-device plotting.")

    # --- Plotting Options (HARDCODED) ---
    legend_placement = "Right"
    compact_labels = True
    label_max_len = 26

    col_lib, col_mode = st.columns(2)
    with col_lib:
        plot_library = st.selectbox(
            "Select Plotting Library:",
            options=["Interactive", "Static"], # Renamed from 'Plotly (Interactive)' and 'Matplotlib/Seaborn (Static)'
            key="plot_lib_select"
        )
    with col_mode:
        plot_mode = st.selectbox(
            "Select Plot Mode:",
            options=["Single Axis", "Multi Axis"], # Renamed from 'Overlay Plot (All Features on Single Axis)' and 'Subplots (Separate Axis for Each Feature)'
            key="plot_mode_select"
        )
    
    # --- Feature and Device Selection ---
    feature_options = _infer_numeric_like_columns(df, exclude_cols=[DEVICE_ID_COLUMN])
    unique_devices = df[DEVICE_ID_COLUMN].astype(str).unique().tolist()
    
    # Initialize the selection in session state only if it doesn't exist
    if "features_selection" not in st.session_state:
        st.session_state["features_selection"] = feature_options[: min(3, len(feature_options))]
    
    # RENDER multiselect using the session state key
    selected_features = st.multiselect(
        "Select **one or more** primary features (left Y-axis):",
        options=feature_options,
        # The key handles both the default and the persistent value
        key="features_selection" 
    )

    if not selected_features:
        st.info("Select at least one primary feature to plot.")
        return

    # Initialize the device selection in session state only if it doesn't exist
    if "devices_selection" not in st.session_state:
        st.session_state["devices_selection"] = unique_devices[: min(3, len(unique_devices))]
        
    # RENDER multiselect using the session state key
    selected_devices = st.multiselect(
        f"Select **{DEVICE_ID_COLUMN}** values to include in the plot:",
        options=unique_devices,
        # The key handles both the default and the persistent value
        key="devices_selection" 
    )

    if not selected_devices:
        st.info("Select at least one device identifier to continue.")
        return

    # Secondary Y (Overlay only)
    enable_secondary = False
    secondary_feature: Optional[str] = None
    with st.expander("➕ Secondary Y-axis (optional)", expanded=False):
        st.caption("Add one feature on a separate right-side Y-axis (Single Axis mode only).")
        enable_secondary = st.checkbox("Enable secondary Y-axis", value=False, key="secY_checkbox")

        # Update check for the new mode name
        is_dual_y_attempt = enable_secondary and plot_mode != "Single Axis"
        if is_dual_y_attempt:
            st.warning("Secondary Y-axis is only available in **Single Axis** mode. Disable or switch to Single Axis.")

        # Update check for the new mode name
        if enable_secondary and plot_mode == "Single Axis":
            sec_candidates = [c for c in feature_options if c not in selected_features]
            if sec_candidates:
                secondary_feature = st.selectbox(
                    "Select secondary feature (right Y-axis):",
                    options=sec_candidates,
                    index=0,
                    key="secY_feature_select"
                )
            else:
                 st.info("No unselected features available for the secondary axis.")
                 secondary_feature = None

    # --- Time Range Filter ---
    st.subheader("Time Range Filter")
    min_time = df.index.min()
    max_time = df.index.max()
    col_start, col_end = st.columns(2)
    
    # Initialize Time/Date selections for persistence
    if "start_date" not in st.session_state:
        st.session_state["start_date"] = min_time.date()
        st.session_state["start_time"] = min_time.time()
        st.session_state["end_date"] = max_time.date()
        st.session_state["end_time"] = max_time.time()
        
    with col_start:
        start_date = st.date_input("Start Date", value=st.session_state["start_date"], min_value=min_time.date(), max_value=max_time.date(), key="start_date")
        start_time_val = st.time_input("Start Time", value=st.session_state["start_time"], key="start_time")
    with col_end:
        end_date = st.date_input("End Date", value=st.session_state["end_date"], min_value=min_time.date(), max_value=max_time.date(), key="end_date")
        end_time_val = st.time_input("End Time", value=st.session_state["end_time"], key="end_time")

    start_dt = datetime.combine(start_date, start_time_val)
    end_dt = datetime.combine(end_date, end_time_val)

    # Handle timezone awareness
    tz = df.index.tz
    start_ts = pd.Timestamp(start_dt, tz=tz) if tz is not None else pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt, tz=tz) if tz is not None else pd.Timestamp(end_dt)

    df_filtered = pd.DataFrame()
    if start_ts >= end_ts:
        st.error("Start time must be before end time.")
    else:
        # 1. Filter original DataFrame by device and time range
        df_devices = df[df[DEVICE_ID_COLUMN].astype(str).isin([str(x) for x in selected_devices])]
        df_filtered = df_devices[(df_devices.index >= start_ts) & (df_devices.index <= end_ts)].copy()

    # --- Plot Generation Logic (only runs when button is pressed) ---
    if st.button("Generate Time Series Plot", type="primary", key="generate_plot_btn"):
        if df_filtered.empty:
            st.warning("No data found matching the selected filters. Please adjust your selections.")
            return

        # 2. Prepare Primary Data (Calls cached function in src/plot_logic_data.py)
        df_long_primary = prepare_long_data(
            df_filtered, selected_features, selected_devices, compact_labels=compact_labels, label_max_len=label_max_len
        )

        # 3. Determine if Dual Y is active (using new mode name)
        is_dual_y_mode = (enable_secondary and plot_mode == "Single Axis" and secondary_feature)

        # 4. Prepare Secondary Data (Calls cached function in src/plot_logic_data.py)
        df_long_secondary = None
        if is_dual_y_mode:
            df_long_secondary = prepare_long_data(
                df_filtered, [secondary_feature], selected_devices, compact_labels=compact_labels, label_max_len=label_max_len
            )

        # --- Diagnostics ---
        expected_series, actual_series, missing_reasons = diagnose_missing_series(
            df_long_primary, selected_features, selected_devices
        )
        st.caption(f"Expected series (features × devices): {expected_series} | Actual series plotted: {actual_series}")
        if missing_reasons:
            with st.expander("Why some selected features did not plot", expanded=False):
                for f, reason in missing_reasons.items():
                    st.write(f"- **{f}** → {reason}")

        # --- Plotting (Calls functions in src/plot_logic_chart.py) ---
        st.subheader(f"Plotting {plot_library} - {plot_mode}")
        sec_note = " + Secondary Y" if is_dual_y_mode else ""
        
        # Map modes back to internal names for logic
        internal_plot_mode = "Overlay Plot (All Features on Single Axis)" if plot_mode == "Single Axis" else "Subplots (Separate Axis for Each Feature)"
        internal_plot_library = "Plotly (Interactive)" if plot_library == "Interactive" else "Matplotlib/Seaborn (Static)"

        try:
            if internal_plot_library == "Plotly (Interactive)":
                if is_dual_y_mode:
                    plot_plotly_dual_y(
                        df_long_primary, df_long_secondary, secondary_feature,
                        title=f"Plotly Overlay{sec_note}: Primary left, Secondary right",
                        legend_placement=legend_placement
                    )
                else:
                    plot_plotly_single_axis(df_long_primary, internal_plot_mode, legend_placement)
            else:
                if is_dual_y_mode:
                    plot_seaborn_dual_y(
                        df_long_primary, df_long_secondary, secondary_feature,
                        title=f"Seaborn Overlay{sec_note}: Primary left, Secondary right"
                    )
                else:
                    plot_seaborn_single_axis(df_long_primary, internal_plot_mode)
        except Exception as e:
            st.error("An error occurred while generating the chart.")
            st.exception(e)