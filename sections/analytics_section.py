# sections/analytics_section.py
import pandas as pd
import streamlit as st

# Import the new efficiency function
from src.analytics import compute_cumulative_deltas, compute_efficiency_metrics

def analytics_section():
    st.header("Energy Analysis")
    cumulative_counter_section()


# -------------------- Sc_Com Counters Section --------------------

def cumulative_counter_section():

    df = st.session_state.get("processed_data")
    if df is None or df.empty:
        st.info("No data available. Please load & preprocess data first.")
        return

    st.subheader("Sc_Com Energy Analysis (Cumulative Counters)")
    st.markdown(
        """
        Calculates **Energy Deltas (MWh)** and **System Efficiency** metrics (RTE, Loss, etc.).
        """
    )

    # --- 1. Column Selection ---
    cols = df.columns.astype(str).tolist()
    
    def pick_col(label, default_name, key_suffix):
        idx = 0
        if default_name in cols:
            idx = cols.index(default_name)
        opts = ["(None)"] + cols
        def_idx = idx + 1 if default_name in cols else 0
        return st.selectbox(f"{label}", opts, index=def_idx, key=f"sel_sc_{key_suffix}")

    with st.expander("⚙️ Column Configuration", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.caption("DC Side")
            col_dc_in = pick_col("DC Energy IN", "Cnt.TotDcWhIn MWh", "dc_in")
            col_dc_out = pick_col("DC Energy OUT", "Cnt.TotDcWhOut MWh", "dc_out")
        with c2:
            st.caption("AC Side")
            col_ac_in = pick_col("AC Energy IN", "Cnt.TotAcWhIn MWh", "ac_in")
            col_ac_out = pick_col("AC Energy OUT", "Cnt.TotAcWhOut MWh", "ac_out")

    # Collect valid selections to pass to calculation
    # We need to know WHICH column maps to WHICH role for the efficiency calc later
    col_map = {
        "AC_In": col_ac_in,
        "AC_Out": col_ac_out,
        "DC_In": col_dc_in,
        "DC_Out": col_dc_out
    }
    
    # Filter out "(None)"
    selected_counters = [c for c in col_map.values() if c != "(None)"]

    if not selected_counters:
        st.warning("⚠️ Please select at least one column above to proceed.")
        return

    # --- 2. Time Selection ---
    min_t, max_t = df.index.min(), df.index.max()
    
    c1, c2 = st.columns(2)
    with c1:
        sd = st.date_input("Start Date", value=min_t.date(), min_value=min_t.date(), max_value=max_t.date(), key="sc_sd")
        st_time = st.time_input("Start Time", value=min_t.time(), key="sc_st")
    with c2:
        ed = st.date_input("End Date", value=max_t.date(), min_value=min_t.date(), max_value=max_t.date(), key="sc_ed")
        et_time = st.time_input("End Time", value=max_t.time(), key="sc_et")

    start_ts = pd.Timestamp.combine(sd, st_time)
    end_ts = pd.Timestamp.combine(ed, et_time)
    if df.index.tz is not None:
        start_ts = start_ts.tz_localize(df.index.tz)
        end_ts = end_ts.tz_localize(df.index.tz)

    # --- 3. Calculate ---
    mask = (df.index >= start_ts) & (df.index <= end_ts)
    df_slice = df[mask]

    if df_slice.empty:
        st.warning("No data found in the selected time range.")
        return
    
    st.caption(f"Analyzing window: {start_ts} to {end_ts} ({len(df_slice)} samples)")

    if st.button("Calculate Energy & Efficiency", key="btn_calc_sc"):
        
        # A. Calculate Deltas (Energy MWh)
        result_df = compute_cumulative_deltas(df_slice, selected_counters)
        
        if not result_df.empty:
            st.subheader("1. Energy Results (MWh)")
            st.dataframe(
                result_df.style.format({
                    "Start (MWh)": "{:,.4f}",
                    "End (MWh)": "{:,.4f}",
                    "Total Energy (MWh)": "{:,.4f}"
                }), 
                use_container_width=True
            )
            
            # B. Extract values for Efficiency Calculation
            # Helper to safely get value from result_df
            def get_val(col_name):
                if col_name == "(None)": return 0.0
                row = result_df[result_df["Metric"] == col_name]
                if not row.empty:
                    return float(row["Total Energy (MWh)"].iloc[0])
                return 0.0

            e_ac_in = get_val(col_map["AC_In"])
            e_ac_out = get_val(col_map["AC_Out"])
            e_dc_in = get_val(col_map["DC_In"])
            e_dc_out = get_val(col_map["DC_Out"])

            # C. Calculate Efficiency
            eff_df = compute_efficiency_metrics(e_ac_in, e_ac_out, e_dc_in, e_dc_out)

            st.subheader("2. Efficiency & Loss Analysis")
            
            # Format for display
            st.dataframe(
                eff_df.style.format({"Value": "{:.4f}"}),
                use_container_width=True
            )

           
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Energy Results", data=csv, file_name="sc_com_energy_analysis.csv", mime="text/csv")
        else:
            st.warning("Could not calculate deltas (columns might be empty).")