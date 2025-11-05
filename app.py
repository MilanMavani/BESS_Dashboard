import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
import re
from typing import Optional, Dict, List, Tuple, Set
from pathlib import Path
import io

# =======================================================================
# SECTION 1: DC DATA LOADING FUNCTION
# =======================================================================

# @st.cache_data tells Streamlit to only run this once per file
@st.cache_data
def load_and_prep_dc_data(uploaded_file, sep=';', dayfirst=False) -> pd.DataFrame:
    """
    Loads and prepares the DC-side CSV file from an uploaded file object.
    """
    df = pd.read_csv(uploaded_file, sep=sep, dtype=str, engine='python')

    df['Date'] = df['Date'].str.strip()
    df['Time'] = df['Time'].str.strip()
    df['TZ'] = df['TZ'].astype(str).str.strip()

    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                    errors='coerce',
                                    dayfirst=dayfirst)

    def extract_tz_hours(tz_str):
        if pd.isna(tz_str): return pd.NA
        m = re.search(r'([+-]?\d{1,3})', tz_str)
        if not m: return pd.NA
        try: return int(m.group(1))
        except Exception: return pd.NA

    df['TZ_hours'] = df['TZ'].apply(extract_tz_hours)

    mask = df['TZ_hours'].notna() & df['Datetime'].notna()
    df.loc[mask, 'Datetime'] = df.loc[mask, 'Datetime'] + pd.to_timedelta(df.loc[mask, 'TZ_hours'], unit='h')

    df = df.drop(columns=['Date', 'Time', 'TZ', 'TZ_hours'])
    df = df.set_index('Datetime').sort_index()
    return df

# =======================================================================
# SECTION 2: CONSOLIDATED HELPER FUNCTIONS
# =======================================================================

def _check_cadence(dt_s: pd.Series,
                   expected_seconds: Optional[float],
                   rtol: float = 0.02,
                   atol: float = 0.5) -> dict:
    """Consolidated cadence checker for DC analysis."""
    x = dt_s.dropna().to_numpy(dtype=float)
    x = x[x > 0]
    if x.size == 0:
        return dict(is_regular=False, dt_median=np.nan, dt_p95=np.nan, frac_off=1.0)
    
    dt_median = float(np.median(x))
    dt_p95 = float(np.quantile(x, 0.95))
    
    if expected_seconds is None or np.isnan(expected_seconds):
        tol = max(abs(dt_median) * rtol, atol)
        frac_off = float((np.abs(x - dt_median) > tol).mean())
        is_regular = frac_off <= 0.05
    else:
        tol = max(abs(expected_seconds) * rtol, atol)
        frac_off = float((np.abs(x - expected_seconds) > tol).mean())
        is_regular = (frac_off <= 0.05) and (abs(dt_median - expected_seconds) <= tol)
        
    return dict(is_regular=is_regular, dt_median=dt_median, dt_p95=dt_p95, frac_off=frac_off)

def _strip_spaces(s: str) -> str:
    if not isinstance(s, str): return s
    return s.replace('\u00A0', '').replace('\u202F', '').replace(' ', '').strip()

def _classify_value(s: str):
    if s is None or s == '': return 'other'
    has_comma = ',' in s
    has_dot = '.' in s
    if has_comma and has_dot:
        return 'EU' if s.rfind(',') > s.rfind('.') else 'US'
    if has_comma: return 'comma_only'
    if has_dot: return 'dot_only'
    if re.fullmatch(r'[+-]?\d+', s): return 'int'
    return 'other'

def _convert_value(s: str, preference: str):
    if s is None or (isinstance(s, float) and pd.isna(s)): return np.nan
    if not isinstance(s, str): return s
    s0 = _strip_spaces(s)
    if s0 == '' or s0.lower() in ('nan', 'none', 'null'): return np.nan
    kind = _classify_value(s0)
    if kind == 'EU': # e.g., 1.234,56
        try: return float(s0.replace('.', '').replace(',', '.'))
        except Exception: return np.nan
    if kind == 'US': # e.g., 1,234.56
        try: return float(s0.replace(',', ''))
        except Exception: return np.nan
    if kind == 'comma_only':
        if preference == 'EU': # comma as decimal
            try: return float(s0.replace(',', '.'))
            except: return np.nan
        if preference == 'US': # comma as thousands
            try: return float(s0.replace(',', ''))
            except: return np.nan
        last_grp = s0.split(',')[-1]
        if last_grp.isdigit() and len(last_grp) == 3 and len(s0.split(',')) >= 2:
            try: return float(s0.replace(',', ''))
            except: return np.nan
        try: return float(s0.replace(',', '.'))
        except: return np.nan
    if kind == 'dot_only':
        if preference == 'US': # dot as decimal
            try: return float(s0)
            except: return np.nan
        if preference == 'EU': # dot as thousands
            try: return float(s0.replace('.', ''))
            except: return np.nan
        last_grp = s0.split('.')[-1]
        if last_grp.isdigit() and len(last_grp) == 3 and len(s0.split('.')) >= 2:
            try: return float(s0.replace('.', ''))
            except: return np.nan
        try: return float(s0)
        except: return np.nan
    if kind == 'int':
        try: return float(s0)
        except Exception: return np.nan
    return np.nan

@st.cache_data
def convert_mixed_numeric_columns(df_in: pd.DataFrame, exclude: set = None, verbose: bool = True) -> pd.DataFrame:
    """Robustly converts string columns to numeric, handling EU/US formats."""
    df_out = df_in.copy()
    exclude = set() if exclude is None else set(exclude)
    diagnostics = {}
    for col in df_out.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df_out[col]):
            continue
        s = df_out[col].astype(str)
        if not s.str.contains(r'\d', regex=True).any():
            continue
        s_clean = s.map(_strip_spaces)
        kinds = s_clean.map(_classify_value)
        eu_votes = int((kinds == 'EU').sum())
        us_votes = int((kinds == 'US').sum())
        preference = 'EU' if eu_votes > us_votes else ('US' if us_votes > eu_votes else None)
        converted = s_clean.map(lambda x: _convert_value(x, preference))
        valid_ratio = np.isfinite(converted).sum() / max(len(converted), 1)
        if valid_ratio < 0.1:
            diagnostics[col] = f"Skipped (valid_ratio={valid_ratio:.2f} < 0.1)"
            continue
        df_out[col] = pd.Series(converted, index=df_out.index, dtype="Float64")
        diagnostics[col] = f"Converted (pref={preference}, valid_ratio={valid_ratio:.2f})"
    
    # Use st.expander to hide the verbose output unless clicked
    if verbose and diagnostics:
        with st.expander("[DC Numeric Conversion Diagnostics]"):
            for c, info in diagnostics.items():
                st.text(f"- {c}: {info}")
    return df_out


# =======================================================================
# SECTION 3: THE INTEGRATED DC-ONLY ANALYZER CLASS
# =======================================================================

class DcCapacityTestAnalyzer:
    """
    Integrates all DC capacity test analyses into a single class.
    """
    def __init__(self, master_config: dict, df_dc: pd.DataFrame):
        self.config = master_config
        self.df_dc = df_dc.copy()
        
        # --- Result properties ---
        self.dfs_by_device = None
        self.dc_rte_summary = None
        self.dc_rte_system_totals = None
        self.dc_system_cumulative_energy = None
        self.dc_system_soc = None
        
        # --- Plot properties ---
        self.dc_cumulative_energy_plot = None
        self.dc_soc_plot = None
        
        st.info(f"DcCapacityTestAnalyzer initialized with {self.df_dc.shape[0]} rows.")

    def run_analysis(self):
        """Runs all DC analyses."""
        st.info("--- Starting Full DC Analysis ---")

        with st.spinner("Preparing DC data (cleaning and partitioning)..."):
            try:
                self._clean_and_partition_dc_df()
                st.success(f"DC data partitioned into {len(self.dfs_by_device)} devices.")
            except Exception as e:
                st.error(f"ERROR in DC prep: {e}")
                return

        with st.spinner("Running DC-side RTE analysis..."):
            try:
                self._run_dc_rte_analysis()
                st.success("DC-side RTE analysis complete.")
            except Exception as e:
                st.error(f"ERROR in DC RTE analysis: {e}")

        with st.spinner("Running DC-side Cumulative Energy analysis..."):
            try:
                self._run_dc_cumulative_energy_analysis()
                st.success("DC-side Cumulative Energy analysis complete.")
            except Exception as e:
                st.error(f"ERROR in DC Cumulative Energy analysis: {e}")

        with st.spinner("Running DC-side SOC analysis..."):
            try:
                self._run_dc_soc_analysis()
                st.success("DC-side SOC analysis complete.")
            except Exception as e:
                st.error(f"ERROR in DC SOC analysis: {e}")
                
        st.info("--- Full DC Analysis Complete ---")
        
    def _clean_and_partition_dc_df(self):
        dc_device_col = self.config['dc_device_col']
        exclude_cols = {dc_device_col}
        
        self.df_dc = convert_mixed_numeric_columns(self.df_dc, exclude=exclude_cols, verbose=True)
        
        dc_power_col = self.config['dc_power_col']
        dc_soc_col = self.config['dc_soc_col']
        wanted_cols = [dc_power_col, dc_soc_col]
        
        available_cols = [c for c in wanted_cols if c in self.df_dc.columns]
        
        if dc_device_col not in self.df_dc.columns:
                raise KeyError(f"Required column '{dc_device_col}' not found in DC df.")
        
        dfs_by_device = {}
        for dev, g in self.df_dc.groupby(dc_device_col, sort=False):
            g2 = g[available_cols].sort_index()
            g2 = g2.dropna(how='all', subset=available_cols)
            dfs_by_device[dev] = g2
            
        self.dfs_by_device = dfs_by_device

    def _run_dc_rte_analysis(self):
        per_device_rows = []
        
        CHARGE_START = pd.Timestamp(self.config['charge_start'])
        CHARGE_END = pd.Timestamp(self.config['charge_end'])
        DISCHARGE_START = pd.Timestamp(self.config['discharge_start'])
        DISCHARGE_END = pd.Timestamp(self.config['discharge_end'])
        POWER_COL = self.config['dc_power_col']
        DISCHARGE_POSITIVE = self.config.get('dc_discharge_positive', True)
        P_EPS_KW = self.config.get('dc_p_eps_kw', 0.0)
        SAMPLING_SECONDS = self.config.get('sampling_seconds')
        RTE_MIN_CHARGE_KWH = self.config.get('rte_min_charge_kwh', 0.01)
        
        def _prep_window(g: pd.DataFrame) -> pd.DataFrame:
            if g.empty: return g
            g["dt_s"] = g.index.to_series().diff().dt.total_seconds()
            if len(g) >= 2 and pd.isna(g.iloc[0]["dt_s"]):
                g.iloc[0, g.columns.get_loc("dt_s")] = (g.index[1] - g.index[0]).total_seconds()
            g = g.dropna(subset=["dt_s"])
            g = g[g["dt_s"] > 0]
            return g

        for dev, d in self.dfs_by_device.items():
            if d.empty or POWER_COL not in d.columns:
                continue
            
            dd = d.sort_index().copy()
            P = pd.to_numeric(dd[POWER_COL], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            if not DISCHARGE_POSITIVE:
                P = -P
            if P_EPS_KW > 0:
                P = np.where(np.abs(P) < P_EPS_KW, 0.0, P)
            dd["P"] = P

            d_charge = dd[(dd.index >= CHARGE_START) & (dd.index <= CHARGE_END)].copy()
            d_dis = dd[(dd.index >= DISCHARGE_START) & (dd.index <= DISCHARGE_END)].copy()

            d_charge = _prep_window(d_charge)
            d_dis = _prep_window(d_dis)

            reg_charge = _check_cadence(d_charge["dt_s"] if not d_charge.empty else pd.Series([], dtype=float), SAMPLING_SECONDS)
            reg_dis = _check_cadence(d_dis["dt_s"] if not d_dis.empty else pd.Series([], dtype=float), SAMPLING_SECONDS)

            P_charge_star = (-d_charge["P"]).clip(lower=0.0).to_numpy(dtype=float) if not d_charge.empty else np.array([], dtype=float)
            P_dis_star = (d_dis["P"]).clip(lower=0.0).to_numpy(dtype=float) if not d_dis.empty else np.array([], dtype=float)

            E_charge_kWh = 0.0
            E_discharge_kWh = 0.0
            
            if SAMPLING_SECONDS is not None and reg_charge["is_regular"] and reg_dis["is_regular"]:
                dt_h = float(SAMPLING_SECONDS) / 3600.0
                E_charge_kWh = float(P_charge_star.sum() * dt_h) if P_charge_star.size else 0.0
                E_discharge_kWh = float(P_dis_star.sum() * dt_h) if P_dis_star.size else 0.0
                rte_method = f"constant_dt({int(SAMPLING_SECONDS)}s)"
            else:
                rte_method = "trapezoid_dt"
                if not d_charge.empty and P_charge_star.size:
                    t_c = d_charge.index.view("int64").to_numpy() / 1e9
                    E_charge_kWh = float(np.trapz(P_charge_star, x=t_c) / 3600.0)
                if not d_dis.empty and P_dis_star.size:
                    t_d = d_dis.index.view("int64").to_numpy() / 1e9
                    E_discharge_kWh = float(np.trapz(P_dis_star, x=t_d) / 3600.0)

            eta_dc = np.nan
            if E_charge_kWh > float(RTE_MIN_CHARGE_KWH):
                eta_dc = E_discharge_kWh / E_charge_kWh
            
            per_device_rows.append({
                "Device": dev, "E_dc_in_kWh": E_charge_kWh, "E_dc_out_kWh": E_discharge_kWh,
                "eta_dc": eta_dc, "method": rte_method,
                "dt_med_charge": reg_charge["dt_median"], "dt_p95_charge": reg_charge["dt_p95"],
                "dt_med_dis": reg_dis["dt_median"], "dt_p95_dis": reg_dis["dt_p95"],
            })

        self.dc_rte_summary = pd.DataFrame(per_device_rows).sort_values("Device", kind="stable")
        
        system_totals = {}
        system_totals["Total_E_dc_in_kWh"] = float(self.dc_rte_summary["E_dc_in_kWh"].sum())
        system_totals["Total_E_dc_out_kWh"] = float(self.dc_rte_summary["E_dc_out_kWh"].sum())
        system_totals["eta_dc_system"] = (
            system_totals["Total_E_dc_out_kWh"] / system_totals["Total_E_dc_in_kWh"]
            if system_totals["Total_E_dc_in_kWh"] > 0 else np.nan
        )
        self.dc_rte_system_totals = system_totals

    def _run_dc_cumulative_energy_analysis(self):
        
        def _prep_device_power(df: pd.DataFrame, cfg: dict) -> pd.Series:
            POWER_COL = cfg['dc_power_col']
            if df is None or df.empty or (POWER_COL not in df.columns):
                return pd.Series(dtype=float)
            P = pd.to_numeric(df[POWER_COL], errors="coerce").fillna(0.0).astype(float)
            P_kW = P / 1000.0 if cfg.get('dc_is_power_in_watts', False) else P
            if not cfg.get('dc_discharge_positive', True):
                P_kW = -P_kW
            P_EPS_KW = cfg.get('dc_p_eps_kw', 0.0)
            if P_EPS_KW and P_EPS_KW > 0:
                P_kW = P_kW.where(P_kW.abs() >= P_EPS_KW, 0.0)
            P_kW.name = "P_kW"
            return P_kW.sort_index()

        def _compute_dt_seconds(idx: pd.DatetimeIndex) -> np.ndarray:
            if len(idx) == 0: return np.array([], dtype=float)
            dt_s = np.diff(idx.view("int64")) / 1e9
            if len(dt_s) == 0: return np.array([0.0], dtype=float)
            first = dt_s[0]
            return np.concatenate([[first], dt_s])

        def _cumulative_energy_from_power(P_kW: pd.Series, cfg: dict) -> pd.Series:
            if P_kW.empty:
                return pd.Series(index=P_kW.index, data=[], dtype=float, name="E_system_cum_kWh")
            idx = P_kW.index
            dt_s_series = pd.Series(index=idx, data=_compute_dt_seconds(idx))
            
            SAMPLING_SECONDS = cfg.get('sampling_seconds')
            reg = _check_cadence(dt_s_series, SAMPLING_SECONDS)
            
            if SAMPLING_SECONDS is not None and reg["is_regular"]:
                dt_h = float(SAMPLING_SECONDS) / 3600.0
                E = np.cumsum(P_kW.values) * dt_h
            else:
                dt_h = dt_s_series.values / 3600.0
                P = P_kW.values.astype(float)
                inc = P * dt_h # Left-rectangle method
                E = np.cumsum(inc)
                
            return pd.Series(index=idx, data=E, name="E_system_cum_kWh")
        
        per_device_power = {}
        for dev, df in self.dfs_by_device.items():
            P_kW = _prep_device_power(df, self.config)
            if not P_kW.empty:
                per_device_power[dev] = P_kW

        pow_df = pd.DataFrame(per_device_power).sort_index()
        P_system_kW = pow_df.fillna(0.0).sum(axis=1)
        
        self.dc_system_cumulative_energy = _cumulative_energy_from_power(P_system_kW, self.config)

    def _run_dc_soc_analysis(self):
        SOC_COL = self.config['dc_soc_col']
        IS_SOC_PERCENT = self.config.get('dc_is_soc_percent', True)
        CLIP_MIN, CLIP_MAX = (0.0, 100.0) if IS_SOC_PERCENT else (0.0, 1.0)
        
        soc_by_device = {}
        for dev, df in self.dfs_by_device.items():
            if df is None or df.empty or (SOC_COL not in df.columns):
                continue
            s = pd.to_numeric(df[SOC_COL], errors="coerce").astype(float)
            s = s.clip(CLIP_MIN, CLIP_MAX)
            soc_by_device[dev] = s.sort_index()

        if not soc_by_device:
            warnings.warn("No device had a valid SOC series. Skipping SOC analysis.")
            return

        soc_df_union = pd.DataFrame(soc_by_device).sort_index()
        
        first_valid_per_dev = {c: soc_df_union[c].first_valid_index() for c in soc_df_union.columns}
        last_valid_per_dev  = {c: soc_df_union[c].last_valid_index() for c in soc_df_union.columns}
        drop_cols = [c for c in soc_df_union.columns if first_valid_per_dev[c] is None or last_valid_per_dev[c] is None]
        
        if drop_cols:
            soc_df_union = soc_df_union.drop(columns=drop_cols)
            first_valid_per_dev = {c: soc_df_union[c].first_valid_index() for c in soc_df_union.columns}
            last_valid_per_dev  = {c: soc_df_union[c].last_valid_index() for c in soc_df_union.columns}

        if soc_df_union.empty:
            warnings.warn("All devices dropped due to missing SOC. Skipping SOC analysis.")
            return

        common_start = max(first_valid_per_dev.values())
        common_end = min(last_valid_per_dev.values())
        if (common_start is None) or (common_end is None) or (common_start >= common_end):
            warnings.warn("No overlapping time window where all devices have SOC. Skipping SOC analysis.")
            return

        soc_df = soc_df_union[(soc_df_union.index >= common_start) & (soc_df_union.index <= common_end)]
        soc_df_ff = soc_df.ffill().clip(CLIP_MIN, CLIP_MAX)

        N = soc_df_ff.shape[1]
        self.dc_system_soc = soc_df_ff.sum(axis=1) / float(N)
        self.dc_system_soc.name = "SOC_total"

# =======================================================================
# SECTION 4: PLOTTING HELPER FUNCTIONS (for Streamlit)
# =======================================================================

def get_dc_efficiency_bar_plot(analyzer: DcCapacityTestAnalyzer) -> go.Figure:
    """Creates the DC per-device efficiency bar plot."""
    fig = go.Figure()
    
    summary_df = analyzer.dc_rte_summary
    
    if summary_df is None or summary_df.empty or 'eta_dc' not in summary_df.columns:
        return fig.update_layout(title="No DC efficiency data to plot.")

    # Ensure eta_dc is numeric and multiply by 100 for percentage
    plot_data = summary_df.copy()
    plot_data['eta_dc_pct'] = pd.to_numeric(plot_data['eta_dc'], errors='coerce') * 100
    plot_data = plot_data.dropna(subset=['eta_dc_pct', 'Device'])
    # Sort by efficiency for a cleaner plot
    plot_data = plot_data.sort_values('eta_dc_pct', ascending=False)

    fig.add_trace(go.Bar(
        x=plot_data['Device'],
        y=plot_data['eta_dc_pct'],
        name='DC Efficiency',
        text=plot_data['eta_dc_pct'].apply(lambda x: f"{x:.2f}%"), # Text on bars
        textposition='auto',
        marker_color='#2ca02c' # Green
    ))
    
    # Add average line
    system_totals = analyzer.dc_rte_system_totals
    if system_totals and 'eta_dc_system' in system_totals and pd.notna(system_totals['eta_dc_system']):
        avg_eta_pct = system_totals['eta_dc_system'] * 100
        fig.add_hline(
            y=avg_eta_pct,
            line_dash="dash",
            line_color="red",
            annotation_text=f"System Avg: {avg_eta_pct:.2f}%",
            annotation_position="bottom right"
        )

    fig.update_layout(
        title="Per-Device DC Efficiency (RTE)",
        xaxis_title="Device",
        yaxis_title="Efficiency (RTE) %",
        template="plotly_white",
        xaxis_tickangle=-45 # Angle labels if many devices
    )
    return fig


def get_dc_energy_plot(analyzer: DcCapacityTestAnalyzer) -> go.Figure:
    """Creates the DC cumulative energy Plotly figure."""
    dc_energy_data = analyzer.dc_system_cumulative_energy
    fig = go.Figure()
    
    if dc_energy_data is None or dc_energy_data.empty:
        return fig.update_layout(title="No DC cumulative energy data to plot.")

    fig.add_trace(go.Scatter(
        x=dc_energy_data.index,
        y=dc_energy_data.values,
        mode='lines',
        name='System DC Cumulative Energy (kWh)',
        line=dict(color='blue', width=2)
    ))
    fig.update_layout(
        title="System Cumulative DC Energy (signed, kWh)",
        xaxis_title="Time",
        yaxis_title="Energy (kWh)",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

def get_dc_soc_plot(analyzer: DcCapacityTestAnalyzer) -> go.Figure:
    """Creates the DC system SOC Plotly figure."""
    dc_soc_data = analyzer.dc_system_soc
    fig = go.Figure()

    if dc_soc_data is None or dc_soc_data.empty:
        return fig.update_layout(title="No DC SOC data to plot (check for overlapping time windows).")

    is_percent = analyzer.config.get('dc_is_soc_percent', True)
    soc_unit = "(%)" if is_percent else "(fraction)"
    n_devices = len(analyzer.dfs_by_device)

    fig.add_trace(go.Scatter(
        x=dc_soc_data.index,
        y=dc_soc_data.values,
        mode='lines',
        name=f'System SOC (Avg. N={n_devices})',
        line=dict(color='green', width=2)
    ))
    fig.update_layout(
        title="System State of Charge (Simple Mean Across Devices)",
        xaxis_title="Time",
        yaxis_title=f"SOC {soc_unit}",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

# =======================================================================
# SECTION 5: STREAMLIT APPLICATION
# =======================================================================

st.set_page_config(layout="wide")
st.title("🔋 BESS DC-Side Capacity Test Analyzer")

# --- 1. CONFIGURE SIDEBAR ---
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload DC Data File (MVPS)", type=["csv"])

st.sidebar.subheader("Test Time Windows")
# Use a default date to make inputs easier
default_date = pd.to_datetime("2024-02-28").date()

c1, c2 = st.sidebar.columns(2)
with c1:
    ch_start_date = st.date_input("Charge Start Date", default_date)
    ch_start_time = st.time_input("Charge Start Time", pd.to_datetime("10:30:00").time())
with c2:
    ch_end_date = st.date_input("Charge End Date", default_date)
    ch_end_time = st.time_input("Charge End Time", pd.to_datetime("13:53:00").time())

c3, c4 = st.sidebar.columns(2)
with c3:
    dis_start_date = st.date_input("Discharge Start Date", default_date)
    dis_start_time = st.time_input("Discharge Start Time", pd.to_datetime("14:59:30").time())
with c4:
    dis_end_date = st.date_input("Discharge End Date", default_date)
    dis_end_time = st.time_input("Discharge End Time", pd.to_datetime("15:58:40").time())

# Combine date and time
charge_start_ts = pd.Timestamp(f"{ch_start_date} {ch_start_time}")
charge_end_ts = pd.Timestamp(f"{ch_end_date} {ch_end_time}")
discharge_start_ts = pd.Timestamp(f"{dis_start_date} {dis_start_time}")
discharge_end_ts = pd.Timestamp(f"{dis_end_date} {dis_end_time}")

st.sidebar.subheader("Column Names")
cfg_dc_device_col = st.sidebar.text_input("Device Column", "Device")
cfg_dc_power_col = st.sidebar.text_input("Power Column", "DcTotWatt")
cfg_dc_soc_col = st.sidebar.text_input("SOC Column", "Bat.SOCTot")

st.sidebar.subheader("Calculation Settings")
cfg_sampling_seconds = st.sidebar.number_input("Sampling (seconds)", min_value=1, value=1)
cfg_dc_is_power_in_watts = st.sidebar.checkbox("Power is in Watts (not kW)", False)
cfg_dc_discharge_positive = st.sidebar.checkbox("Discharge is Positive", True)
cfg_dc_is_soc_percent = st.sidebar.checkbox("SOC is 0-100 %", True)


# --- 2. BUILD CONFIG & RUN ANALYSIS ---

# Build the master config from the sidebar inputs
master_config = {
    "charge_start": charge_start_ts,
    "charge_end": charge_end_ts,
    "discharge_start": discharge_start_ts,
    "discharge_end": discharge_end_ts,
    "sampling_seconds": cfg_sampling_seconds,
    "rte_min_charge_kwh": 0.01,
    "dc_device_col": cfg_dc_device_col,
    "dc_power_col": cfg_dc_power_col,
    "dc_soc_col": cfg_dc_soc_col,
    "dc_discharge_positive": cfg_dc_discharge_positive,
    "dc_is_power_in_watts": cfg_dc_is_power_in_watts,
    "dc_p_eps_kw": 0.0,
    "dc_is_soc_percent": cfg_dc_is_soc_percent,
}

# --- 3. MAIN APP AREA ---
if uploaded_file is None:
    st.info("Please upload your DC data file to begin.")
else:
    if st.sidebar.button("Run Analysis", use_container_width=True, type="primary"):
        
        # Load data
        df_dc = load_and_prep_dc_data(uploaded_file)
        
        # Instantiate and run
        # We store the analyzer in session_state to keep its results
        st.session_state.analyzer = DcCapacityTestAnalyzer(master_config, df_dc)
        st.session_state.analyzer.run_analysis()
    
    # --- 4. DISPLAY RESULTS ---
    # Check if analyzer has been run and stored in session state
    if "analyzer" in st.session_state:
        analyzer = st.session_state.analyzer
        
        st.header("DC-Side Test Results")
        
        # --- System Totals ---
        if analyzer.dc_rte_system_totals:
            st.subheader("System RTE Summary")
            totals = analyzer.dc_rte_system_totals
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total DC Energy In (kWh)", f"{totals['Total_E_dc_in_kWh']:,.2f}")
            col2.metric("Total DC Energy Out (kWh)", f"{totals['Total_E_dc_out_kWh']:,.2f}")
            if pd.notna(totals['eta_dc_system']):
                col3.metric("System DC RTE", f"{totals['eta_dc_system'] * 100:,.2f} %")
            else:
                col3.metric("System DC RTE", "N/A")
        
        # --- Per-Device Table ---
        if analyzer.dc_rte_summary is not None:
            st.subheader("Per-Device RTE Summary")
            
            # --- MODIFICATION: Select only desired columns for display ---
            display_cols = ["Device", "E_dc_in_kWh", "E_dc_out_kWh", "eta_dc"]
            cols_to_show = [col for col in display_cols if col in analyzer.dc_rte_summary.columns]
            
            if cols_to_show:
                df_display = analyzer.dc_rte_summary[cols_to_show]
                st.dataframe(df_display.style.format({
                    "E_dc_in_kWh": "{:,.2f}",
                    "E_dc_out_kWh": "{:,.2f}",
                    "eta_dc": "{:.2%}",
                }))
            else:
                st.warning("Could not generate per-device summary.")

        # --- Per-Device Efficiency Bar Plot ---
        fig_eff = get_dc_efficiency_bar_plot(analyzer)
        st.plotly_chart(fig_eff, use_container_width=True)
        # --- NEW: Download Button ---
        st.download_button(
            label="Download Efficiency Plot (HTML)",
            data=fig_eff.to_html(),
            file_name="dc_efficiency_plot.html",
            mime="text/html"
        )
        
        # --- System-Wide Plots ---
        st.header("DC-Side System-Wide Plots")
        
        # Plot 1: Cumulative Energy
        fig_energy = get_dc_energy_plot(analyzer)
        st.plotly_chart(fig_energy, use_container_width=True)
        # --- NEW: Download Button ---
        st.download_button(
            label="Download Energy Plot (HTML)",
            data=fig_energy.to_html(),
            file_name="dc_energy_plot.html",
            mime="text/html"
        )
        
        # Plot 2: SOC
        fig_soc_dc = get_dc_soc_plot(analyzer)
        st.plotly_chart(fig_soc_dc, use_container_width=True)
        # --- NEW: Download Button ---
        st.download_button(
            label="Download SOC Plot (HTML)",
            data=fig_soc_dc.to_html(),
            file_name="dc_soc_plot.html",
            mime="text/html"
        )
        
    else:
        st.info("Click 'Run Analysis' in the sidebar to process the data.")