import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
import re
from typing import Optional, Dict, List, Tuple, Set
from pathlib import Path
import io

# =======================================================================
# SECTION 1: AC DATA LOADING FUNCTION
# (This section is unchanged)
# =======================================================================

@st.cache_data
def load_hycon_hybrid_fast(uploaded_file,
                            sep=';',
                            encoding='latin1',
                            strip_trailing_hyphen=True,
                            parse_timestamp_utc=False):
    """
    Loads the AC-side CSV file from an uploaded file object.
    """
    # Read the file from the upload buffer
    file_buffer = io.StringIO(uploaded_file.getvalue().decode(encoding))
    
    # --- Read only the 6th and 8th lines quickly (no pandas) ---
    row6 = row8 = None
    file_buffer.seek(0) # Rewind buffer to start
    for i, line in enumerate(file_buffer, start=1):  # 1-based line index
        if i == 6:
            row6 = line.rstrip('\n')
        elif i == 8:
            row8 = line.rstrip('\n')
            break  # we have both; stop reading early
    if row6 is None or row8 is None:
        raise ValueError("File is too short or missing required header lines 6 and 8.")

    # Split into header cells and combine cell-wise
    h6_parts = [p.strip() for p in row6.split(sep)]
    h8_parts = [p.strip() for p in row8.split(sep)]

    # Align lengths
    width = max(len(h6_parts), len(h8_parts))
    if len(h6_parts) < width:
        h6_parts += [''] * (width - len(h6_parts))
    if len(h8_parts) < width:
        h8_parts += [''] * (width - len(h8_parts))

    combined = [f"{a} {b}".strip() for a, b in zip(h6_parts, h8_parts)]
    combined = [pd.Series([c]).str.replace(r'\s+', ' ', regex=True).iloc[0].strip() for c in combined]
    if strip_trailing_hyphen:
        combined = [pd.Series([c]).str.replace(r'\s*-\s*$', '', regex=True).iloc[0] for c in combined]

    if combined:
        combined[0] = 'Timestamp'

    # --- Read data once with the prepared header ---
    file_buffer.seek(0) # Rewind buffer again for pd.read_csv
    df = pd.read_csv(
        file_buffer,
        sep=sep,
        skiprows=8,
        header=None,
        names=combined,
        encoding=encoding,
        low_memory=False,
    )
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=parse_timestamp_utc)
    return df

# =======================================================================
# SECTION 2: CONSOLIDATED HELPER FUNCTIONS
# (This section is unchanged)
# =======================================================================

def _sanitize_time_col(d: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Helper to clean and sort the time column."""
    d = d.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col])
    d = d.sort_values(time_col).reset_index(drop=True)
    return d

def _check_cadence(dt_s: pd.Series,
                   expected_seconds: Optional[float],
                   rtol: float = 0.02,
                   atol: float = 0.5) -> dict:
    """Consolidated cadence checker."""
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

# --- Numeric Cleaning Helpers ---
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
    
    if verbose and diagnostics:
        # We removed the st.expander wrapper from here
        for c, info in diagnostics.items():
            st.text(f"- {c}: {info}")
            
    return df_out

# =======================================================================
# SECTION 2.5: 3-STATE DETECTION FUNCTION
# (This section is unchanged from your last correct version)
# =======================================================================

@st.cache_data
def find_all_events(df_ac: pd.DataFrame,
                      time_col: str,
                      power_col: str,
                      idle_threshold_kw: float,
                      min_duration_s: float
                      ) -> List[Dict]:
    """
    Scans the entire DataFrame and finds all "zero-to-zero" events.
    This version correctly uses 3 states (Idle, Charge, Discharge).
    """
    if df_ac is None:
        return []
    if power_col not in df_ac.columns or time_col not in df_ac.columns:
        st.warning(f"Column '{power_col}' or '{time_col}' not found. Event detection failed.")
        return []
    
    d = df_ac[[time_col, power_col]].copy()
    d = d.dropna()
    if d.empty:
        return []

    d = _sanitize_time_col(d, time_col)
    if len(d) < 2:
        return []

    # 1. Define the 3 states
    # State 0 = Idle
    # State 1 = Discharge
    # State -1 = Charge
    def get_state(power):
        if power > idle_threshold_kw:
            return 1  # Discharge
        elif power < -idle_threshold_kw:
            return -1 # Charge
        else:
            return 0  # Idle

    d['state'] = d[power_col].apply(get_state)
    
    # 2. Find continuous blocks of the *same state*
    # This now correctly separates Charge (state -1) from Discharge (state 1)
    d['state_block'] = (d['state'] != d['state'].shift(1)).cumsum()
    
    events = []
    # 3. Group by blocks
    for name, group in d.groupby('state_block'):
        if group.empty:
            continue
            
        block_state = group['state'].iloc[0]
        
        # 4. Ignore all "idle" blocks (state 0)
        if block_state == 0:
            continue
            
        start_time = group[time_col].iloc[0]
        end_time = group[time_col].iloc[-1]
        duration = (end_time - start_time).total_seconds()
        
        # 5. Filter by minimum duration
        if duration >= min_duration_s:
            
            # 6. Classify by state
            event_type = "Discharge" if block_state == 1 else "Charge"
            
            events.append({
                "type": event_type,
                "start": start_time,
                "end": end_time
            })
                
    return sorted(events, key=lambda x: x['start'])


# =======================================================================
# SECTION 3: ***MODIFIED*** AC-SIDE ANALYSIS FUNCTIONS
# =======================================================================

def compute_nominal_from_poi_plotly(
    df: pd.DataFrame,
    discharge_start,
    discharge_end,
    P_nom_kW: float,
    tol_pct: float = 5.0,
    required_minutes: Optional[float] = None,
    time_col: str = "Timestamp",
    power_col: str = "PoiPwrAt kW",
    # --- MODIFIED: More generic title ---
    title: str = "AC Power Analysis (Plotly)",
    drop_duplicate_timestamps: bool = True,
    ramp_trim_seconds_discharge: int = 0,
    charge_start: Optional[str] = None,
    charge_end: Optional[str] = None,
    sampling_seconds: Optional[float] = None,
    discharge_positive: bool = True,
    ramp_trim_seconds_charge: int = 0,
    warn_irregular: bool = True,
    rte_min_charge_kWh: float = 0.01,
    soc_col: Optional[str] = None, # SOC column
    
    P_nom_charge_kW: Optional[float] = None,
    tol_charge_pct: Optional[float] = None,
    P_nom_discharge_kW: Optional[float] = None,
    tol_discharge_pct: Optional[float] = None
) -> Dict:
    """
    Calculates AC-side KPI, RTE, cumulative energy, and SOC plot.
    *** MODIFIED ***
    This plot now shows data from BOTH charge and discharge windows if provided.
    It will also plot tolerance bands for BOTH events.
    """
    
    ts_start_raw = pd.to_datetime(discharge_start, errors="coerce")
    ts_end_raw = pd.to_datetime(discharge_end, errors="coerce")
    if pd.isna(ts_start_raw) or pd.isna(ts_end_raw):
         raise ValueError("Invalid or missing discharge start/end times.")
    if ts_end_raw <= ts_start_raw:
        raise ValueError("discharge_end must be after discharge_start")
    
    # Get raw charge times for SOC plot
    c_start_raw = pd.to_datetime(charge_start, errors="coerce")
    c_end_raw = pd.to_datetime(charge_end, errors="coerce")

    # --- MODIFIED: Determine full plot window ---
    all_times = [ts_start_raw, ts_end_raw]
    has_charge = pd.notna(c_start_raw) and pd.notna(c_end_raw)
    if has_charge:
        all_times.extend([c_start_raw, c_end_raw])
    
    plot_start = min(all_times)
    plot_end = max(all_times)
    # --- End modification ---

    trim_dis = pd.to_timedelta(int(ramp_trim_seconds_discharge), unit="s")
    ts_start = ts_start_raw + trim_dis
    ts_end = ts_end_raw - trim_dis
    if ts_end <= ts_start:
        raise ValueError("Ramp trim too large for discharge window.")

    if time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not in df")
    if power_col not in df.columns:
        raise KeyError(f"Power column '{power_col}' not in df")

    warnings_list: List[str] = [] # Init warnings list

    # Create list of columns to keep, including optional SOC
    cols_to_keep = {time_col, power_col}
    if soc_col and soc_col in df.columns:
        cols_to_keep.add(soc_col)
    elif soc_col:
        warnings_list.append(f"SOC column '{soc_col}' provided but not found in data.")
    
    d = df[list(cols_to_keep)].copy()
    d = _sanitize_time_col(d, time_col)

    if drop_duplicate_timestamps:
        # Aggregate all columns (power and SOC) by mean on duplicate timestamps
        d = d.groupby(time_col, as_index=False).mean()

    if d.empty:
        raise ValueError("No valid rows after parsing timestamps.")
    
    # --- Generate SOC Plot (uses full data 'd') ---
    fig_soc = go.Figure()
    if soc_col and soc_col in d.columns:
        fig_soc.add_trace(go.Scatter(
            x=d[time_col], y=d[soc_col],
            mode="lines", name="SOC",
            line=dict(color="#FF5733", width=2)
        ))
        
        # Add charge window highlight
        if has_charge:
            fig_soc.add_vrect(
                x0=c_start_raw, x1=c_end_raw,
                fillcolor="green", opacity=0.15, layer="below", line_width=0,
                name="Charge Window"
            )
        # Add discharge window highlight
        fig_soc.add_vrect(
            x0=ts_start_raw, x1=ts_end_raw,
            fillcolor="red", opacity=0.15, layer="below", line_width=0,
            name="Discharge Window"
        )
        
        fig_soc.update_layout(
            title=f"System SOC ({soc_col})",
            xaxis_title="Time", yaxis_title="SOC (%)",
            template="plotly_white", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
        )
    else:
        fig_soc.update_layout(title="SOC Plot (Column Not Found or Not Requested)")
    # --- End SOC Plot ---

    # --- MODIFIED: Create new dPlot for the main figure ---
    dPlot = d[(d[time_col] >= plot_start) & (d[time_col] <= plot_end)].copy()
    if dPlot.empty:
        raise ValueError("No data found in the combined plot window.")
    
    # --- KPI calculation (dKPI) is UNCHANGED, still discharge-only ---
    dKPI = d[(d[time_col] >= ts_start) & (d[time_col] <= ts_end)][[time_col, power_col]].copy()
    if dKPI.empty:
        raise ValueError("No samples inside discharge window.")
    if len(dKPI) < 2:
        raise ValueError("Only one sample in KPI window.")

    dKPI["dt_s"] = dKPI[time_col].diff().dt.total_seconds()
    if not dKPI.empty and np.isnan(dKPI.loc[dKPI.index[0], "dt_s"]) and len(dKPI) >= 2:
        dKPI.loc[dKPI.index[0], "dt_s"] = (
            dKPI.loc[dKPI.index[1], time_col] - dKPI.loc[dKPI.index[0], time_col]
        ).total_seconds()
    dKPI = dKPI[dKPI["dt_s"] > 0].copy()

    dKPI["E_kWh_slice"] = dKPI[power_col] * (dKPI["dt_s"] / 3600.0)
    actual_energy_kWh = float(dKPI["E_kWh_slice"].sum())

    # --- Discharge KPI Band Calculation (unchanged) ---
    band_low = float(P_nom_kW) * (1 - tol_pct / 100.0)
    band_high = float(P_nom_kW) * (1 + tol_pct / 100.0)
    dKPI["in_band"] = (dKPI[power_col] >= band_low) & (dKPI[power_col] <= band_high)

    inband_time_s_cum = float(dKPI.loc[dKPI["in_band"], "dt_s"].sum())
    E_nom_cum_kWh = float(P_nom_kW) * (inband_time_s_cum / 3600.0)

    # Shaded segments for discharge KPI
    segs: List[Tuple[pd.Timestamp, pd.Timestamp, float]] = []
    in_seg = False
    acc_s = 0.0
    seg_start = None
    for i, row in dKPI.iterrows():
        if row["in_band"]:
            if not in_seg:
                in_seg = True
                seg_start = row[time_col]
                acc_s = row["dt_s"]
            else:
                acc_s += row["dt_s"]
        else:
            if in_seg:
                seg_end = row[time_col]
                segs.append((seg_start, seg_end, acc_s))
                in_seg = False
    if in_seg:
        seg_end = dKPI.iloc[-1][time_col]
        segs.append((seg_start, seg_end, acc_s))

    longest_s = max([s for *_, s in segs], default=0.0)
    E_nom_cont_kWh = float(P_nom_kW) * (longest_s / 3600.0)

    compliance_cont = compliance_cum = None
    required_str = None
    if required_minutes is not None:
        req_s = float(required_minutes) * 60.0
        compliance_cont = longest_s >= req_s
        compliance_cum = inband_time_s_cum >= req_s
        required_str = f"{required_minutes:.0f} min"
    
    # --- RTE & Energy Calculations (unchanged) ---
    E_charge_kWh = E_discharge_kWh = np.nan
    RTE_pct = np.nan
    rte_method = None
    dt_expected = float(sampling_seconds) if sampling_seconds is not None else np.nan
    df_calc_poi_egymtr = None 
    
    E_charge_kWh_nom = E_discharge_kWh_nom = np.nan
    RTE_pct_nom = np.nan

    if has_charge: # Use the flag we defined earlier
        
        # --- 1. Cumulative Energy Plot Data ---
        d_ce = d[(d[time_col] >= c_start_raw) & (d[time_col] <= ts_end_raw)][[time_col, power_col]].copy()
        
        if d_ce.empty:
            warnings_list.append(f"No data in Calc-PoiEgyMtr window ({c_start_raw} to {ts_end_raw}).")
            df_calc_poi_egymtr = pd.DataFrame(columns=[time_col, "Calc-PoiEgy", "Calc-PoiEgyMtr"])
        else:
            d_ce["dt_s"] = d_ce[time_col].diff().dt.total_seconds()
            if np.isnan(d_ce.loc[d_ce.index[0], "dt_s"]) and len(d_ce) >= 2:
                d_ce.loc[d_ce.index[0], "dt_s"] = (
                    d_ce.loc[d_ce.index[1], time_col] - d_ce.loc[d_ce.index[0], time_col]
                ).total_seconds()
            
            d_ce = d_ce[d_ce["dt_s"] > 0].copy()
            d_ce["Calc-PoiEgy"] = d_ce[power_col] * (d_ce["dt_s"] / 3600.0)
            d_ce["Calc-PoiEgyMtr"] = d_ce["Calc-PoiEgy"].cumsum()
            df_calc_poi_egymtr = d_ce[[time_col, "Calc-PoiEgy", "Calc-PoiEgyMtr"]].copy()

        # --- 2. Full-Window RTE (Method 2) ---
        dd = d[[time_col, power_col]].copy() 
        P = dd[power_col].to_numpy(dtype=float)
        if not discharge_positive:
            P = -P
        dd["P"] = P

        trim_ch = pd.to_timedelta(int(ramp_trim_seconds_charge), unit="s")
        c_start_trimmed = c_start_raw + trim_ch
        c_end_trimmed = c_end_raw - trim_ch

        d_charge = dd[(dd[time_col] >= c_start_trimmed) & (dd[time_col] <= c_end_trimmed)].copy()
        d_dis = dd[(dd[time_col] >= ts_start) & (dd[time_col] <= ts_end)].copy() # Use trimmed discharge window

        for subset, name in [(d_charge, "charge"), (d_dis, "discharge")]:
            if not subset.empty:
                subset["dt_s"] = subset[time_col].diff().dt.total_seconds()
                if np.isnan(subset.loc[subset.index[0], "dt_s"]) and len(subset) >= 2:
                    subset.loc[subset.index[0], "dt_s"] = (
                        subset.loc[subset.index[1], time_col] - subset.loc[subset.index[0], time_col]
                    ).total_seconds()
                subset.dropna(subset=["dt_s"], inplace=True)
                subset = subset[subset["dt_s"] > 0]
                if name == "charge":
                    d_charge = subset
                else:
                    d_dis = subset

        reg_charge = _check_cadence(
            d_charge["dt_s"] if not d_charge.empty else pd.Series([], dtype=float),
            expected_seconds=sampling_seconds)
        reg_dis = _check_cadence(
            d_dis["dt_s"] if not d_dis.empty else pd.Series([], dtype=float),
            expected_seconds=sampling_seconds)
        
        P_charge_star = (-d_charge["P"]).clip(lower=0.0).to_numpy() if not d_charge.empty else np.array([], dtype=float)
        P_dis_star = d_dis["P"].clip(lower=0.0).to_numpy() if not d_dis.empty else np.array([], dtype=float)

        if sampling_seconds is not None and reg_charge["is_regular"] and reg_dis["is_regular"]:
            dt_h = float(sampling_seconds) / 3600.0
            E_charge_kWh = float(P_charge_star.sum() * dt_h)
            E_discharge_kWh = float(P_dis_star.sum() * dt_h)
            rte_method = f"constant_dt({int(sampling_seconds)}s)"
        else:
            rte_method = "trapezoid_dt"
            msg = "Detected irregular cadence for AC-RTE; using trapezoidal integration."
            if warn_irregular:
                warnings.warn(msg)
            warnings_list.append(msg)
            if not d_charge.empty:
                t_c = d_charge[time_col].astype("int64").to_numpy() / 1e9
                E_charge_kWh = float(np.trapz(P_charge_star, x=t_c) / 3600.0)
            else:
                E_charge_kWh = 0.0
            if not d_dis.empty:
                t_d = d_dis[time_col].astype("int64").to_numpy() / 1e9
                E_discharge_kWh = float(np.trapz(P_dis_star, x=t_d) / 3600.0)
            else:
                E_discharge_kWh = 0.0

        if E_charge_kWh > float(rte_min_charge_kWh):
            RTE_pct = 100.0 * E_discharge_kWh / E_charge_kWh
        else:
            RTE_pct = np.nan
            warnings_list.append(f"E_charge ({E_charge_kWh:.6f}) ≤ guard ({rte_min_charge_kWh:.6f}); RTE=NaN.")

        # --- 3. Nominal Power RTE (Method 1) ---
        if (P_nom_charge_kW is not None) and (tol_charge_pct is not None) and not d_charge.empty:
            P_ch_nom = float(P_nom_charge_kW)
            abs_p_ch = abs(P_ch_nom)
            abs_tol_ch = abs_p_ch * (float(tol_charge_pct) / 100.0)
            ch_band_low = P_ch_nom - abs_tol_ch
            ch_band_high = P_ch_nom + abs_tol_ch
            
            d_charge_nom = d_charge[
                (d_charge["P"] >= ch_band_low) & (d_charge["P"] <= ch_band_high)
            ].copy()
            
            if not d_charge_nom.empty:
                P_charge_nom_star = (-d_charge_nom["P"]).clip(lower=0.0).to_numpy()
                if rte_method == "trapezoid_dt":
                    t_c_nom = d_charge_nom[time_col].astype("int64").to_numpy() / 1e9
                    E_charge_kWh_nom = float(np.trapz(P_charge_nom_star, x=t_c_nom) / 3600.0)
                else: # constant_dt
                    dt_h = float(sampling_seconds) / 3600.0
                    E_charge_kWh_nom = float(P_charge_nom_star.sum() * dt_h)
            else:
                E_charge_kWh_nom = 0.0
                warnings_list.append(f"No charge data found within Nominal Charge Band [{ch_band_low:.0f}, {ch_band_high:.0f}] kW.")

        if (P_nom_discharge_kW is not None) and (tol_discharge_pct is not None) and not d_dis.empty:
            P_dis_nom = float(P_nom_discharge_kW)
            abs_p_dis = abs(P_dis_nom)
            abs_tol_dis = abs_p_dis * (float(tol_discharge_pct) / 100.0)
            dis_band_low = P_dis_nom - abs_tol_dis
            dis_band_high = P_dis_nom + abs_tol_dis
            
            d_dis_nom = d_dis[
                (d_dis["P"] >= dis_band_low) & (d_dis["P"] <= dis_band_high)
            ].copy()

            if not d_dis_nom.empty:
                P_dis_nom_star = d_dis_nom["P"].clip(lower=0.0).to_numpy()
                if rte_method == "trapezoid_dt":
                    t_d_nom = d_dis_nom[time_col].astype("int64").to_numpy() / 1e9
                    E_discharge_kWh_nom = float(np.trapz(P_dis_nom_star, x=t_d_nom) / 3600.0)
                else: # constant_dt
                    dt_h = float(sampling_seconds) / 3600.0
                    E_discharge_kWh_nom = float(P_dis_nom_star.sum() * dt_h)
            else:
                E_discharge_kWh_nom = 0.0
                warnings_list.append(f"No discharge data found within Nominal Discharge Band [{dis_band_low:.0f}, {dis_band_high:.0f}] kW.")
        
        # Calculate final Nominal RTE
        if not np.isnan(E_charge_kWh_nom) and E_charge_kWh_nom > float(rte_min_charge_kWh):
             RTE_pct_nom = 100.0 * E_discharge_kWh_nom / E_charge_kWh_nom
        else:
             RTE_pct_nom = np.nan
             if not np.isnan(E_charge_kWh_nom):
                 warnings_list.append(f"Nominal E_charge ({E_charge_kWh_nom:.3f}) is too low; Nominal RTE=NaN.")
    
    # --- End RTE Calculations ---

    # --- MODIFIED: Plot Figure ---
    fig = go.Figure()
    
    # Use dPlot for the main data trace
    fig.add_trace(go.Scatter(x=dPlot[time_col], y=dPlot[power_col],
                             mode="lines", name=f"{power_col}",
                             line=dict(color="#1f77b4", width=2)))
    
    # Get full plot x-range
    x_min = dPlot[time_col].min()
    x_max = dPlot[time_col].max()

    # Plot Discharge KPI Bands
    fig.add_trace(go.Scatter(x=[x_min, x_max],
                             y=[band_low, band_low], mode="lines", name=f"Discharge Band (±{tol_pct:.0f}%)",
                             line=dict(color="#2ca02c", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=[x_min, x_max],
                             y=[band_high, band_high], mode="lines", name=f"Discharge Band (±{tol_pct:.0f}%)",
                             line=dict(color="#2ca02c", width=1.5, dash="dash"), showlegend=False))

    # Plot Charge Bands (IF they exist)
    if has_charge and (P_nom_charge_kW is not None) and (tol_charge_pct is not None):
        # Recalculate for plotting
        P_ch_nom_plot = float(P_nom_charge_kW)
        abs_p_ch_plot = abs(P_ch_nom_plot)
        abs_tol_ch_plot = abs_p_ch_plot * (float(tol_charge_pct) / 100.0)
        ch_band_low_plot = P_ch_nom_plot - abs_tol_ch_plot
        ch_band_high_plot = P_ch_nom_plot + abs_tol_ch_plot
        
        fig.add_trace(go.Scatter(x=[x_min, x_max],
                                 y=[ch_band_low_plot, ch_band_low_plot], mode="lines", name=f"Charge Band (±{tol_charge_pct:.0f}%)",
                                 line=dict(color="#d62728", width=1.5, dash="dash"))) # Red color
        fig.add_trace(go.Scatter(x=[x_min, x_max],
                                 y=[ch_band_high_plot, ch_band_high_plot], mode="lines", name=f"Charge Band (±{tol_charge_pct:.0f}%)",
                                 line=dict(color="#d62728", width=1.5, dash="dash"), showlegend=False))

    # Plot Discharge KPI compliance segments (unchanged)
    shapes = []
    for (s, e, dur_s) in segs:
        shapes.append(dict(type="rect", xref="x", yref="y",
                           x0=s, x1=e, y0=band_low, y1=band_high,
                           fillcolor="rgba(46,204,113,0.18)", line=dict(width=0),
                           layer="below"))
    fig.update_layout(shapes=shapes)

    # --- MODIFIED: Subtitle ---
    subtitle = [
        f"Discharge KPI: P_nom={P_nom_kW:.0f} kW, tol=±{tol_pct:.1f}%",
    ]
    if has_charge and (P_nom_charge_kW is not None):
        subtitle.append(f"Charge: P_nom={P_nom_charge_kW:.0f} kW, tol=±{tol_charge_pct:.1f}%")
        
    subtitle.append(f"KPI Actual Energy={actual_energy_kWh:.1f} kWh")
    
    if required_minutes is not None:
        subtitle.append(f"Req={required_str}")
    if not np.isnan(RTE_pct):
        subtitle.append(f"Full-RTE={RTE_pct:.2f}%")
    if not np.isnan(RTE_pct_nom):
        subtitle.append(f"Nom-RTE={RTE_pct_nom:.2f}%")
    # --- End Subtitle Modification ---

    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{' | '.join(subtitle)}</sup>", x=0.01),
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        margin=dict(l=60, r=30, t=80, b=40),
    )
    # --- End Plot Figure Modification ---


    # --- Summary Table (unchanged) ---
    metrics = [
        "Window start", "Window end", "P_nom (kW)", "Tolerance (%)",
        "Actual energy (kWh)", "In-band time (continuous, min)", "In-band time (cumulative, min)",
        "Compliance required duration",
        "Ramp trim (discharge, s)", "Total timeframe start", "Total timeframe end"
    ]
    values = [
        ts_start, ts_end, float(P_nom_kW), float(tol_pct),
        round(actual_energy_kWh, 3), round(longest_s/60.0, 3), round(inband_time_s_cum/60.0, 3),
        required_str,
        int(ramp_trim_seconds_discharge), d[time_col].min(), d[time_col].max()
    ]

    if has_charge:
        metrics += [
            "Charge window start", "Charge window end", "Ramp trim (charge, s)",
            "Sampling interval expected (s)",
            "--- Full-Window RTE (Method 2) ---",
            "Full-RTE method", "Full-E_charge (kWh)", "Full-E_discharge (kWh)", "Full-RTE (%)",
            "--- Nominal Power RTE (Method 1) ---",
            "Nom-P_charge (kW)", "Nom-Tol_charge (%)", "Nom-E_charge (kWh)",
            "Nom-P_discharge (kW)", "Nom-Tol_discharge (%)", "Nom-E_discharge (kWh)",
            "Nom-RTE (%)"
        ]
        values += [
            c_start_trimmed if 'c_start_trimmed' in locals() else None,
            c_end_trimmed if 'c_end_trimmed' in locals() else None,
            int(ramp_trim_seconds_charge),
            dt_expected if not np.isnan(dt_expected) else None,
            None, # Separator
            rte_method,
            None if np.isnan(E_charge_kWh) else round(float(E_charge_kWh), 3),
            None if np.isnan(E_discharge_kWh) else round(float(E_discharge_kWh), 3),
            None if np.isnan(RTE_pct) else round(float(RTE_pct), 3),
            None, # Separator
            P_nom_charge_kW,
            tol_charge_pct,
            None if np.isnan(E_charge_kWh_nom) else round(float(E_charge_kWh_nom), 3),
            P_nom_discharge_kW,
            tol_discharge_pct,
            None if np.isnan(E_discharge_kWh_nom) else round(float(E_discharge_kWh_nom), 3),
            None if np.isnan(RTE_pct_nom) else round(float(RTE_pct_nom), 3)
        ]
    summary = pd.DataFrame({"metric": metrics, "value": values})

    return {
        "summary_table": summary,
        "figure": fig,
        "figure_soc": fig_soc, # Return SOC figure
        "df_calc_poi_egymtr": df_calc_poi_egymtr,
        "warnings": warnings_list,
    }

def plot_calc_poi_egymtr(df_energy: pd.DataFrame, 
                         time_col: str = "Timestamp",
                         title: str = "AC-Side Cumulative Energy") -> go.Figure:
    """Plots the cumulative and incremental energy from the AC dataframe."""
    if df_energy is None or df_energy.empty:
        return go.Figure().update_layout(title="No AC cumulative energy data to plot.")
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_energy[time_col], y=df_energy["Calc-PoiEgyMtr"],
        mode="lines", name="Cumulative Energy (kWh)",
        line=dict(color="#FF5733", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=df_energy[time_col], y=df_energy["Calc-PoiEgy"],
        mode="lines", name="Interval Energy (kWh)",
        line=dict(color="#337AFF", width=1, dash="dot"), yaxis="y2"
    ))
    
    fig.update_layout(
        title=title, xaxis_title="Time",
        
        yaxis=dict(title=dict(text="<b>Cumulative Energy (kWh)</b>", font=dict(color="#FF5733")),
                   tickfont=dict(color="#FF5733")),
        
        yaxis2=dict(title=dict(text="<b>Interval Energy (kWh)</b>", font=dict(color="#337AFF")),
                    tickfont=dict(color="#337AFF"),
                    overlaying="y", side="right", showgrid=False),
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
    )
    return fig


# =======================================================================
# SECTION 4: ***CORRECTED*** 3-STATE WORKFLOW
# (This section is unchanged from your last correct version,
#  but I've updated the default nominal power values and label)
# =======================================================================

st.set_page_config(layout="wide")
st.title("⚡ BESS AC-Side Capacity Test Analyzer")

# --- Define Workflow States ---
STATE_INIT = "STATE_INIT" # No file
STATE_PROCESSING = "STATE_PROCESSING" # File uploaded, running detection
STATE_CHOOSE_ANALYSIS = "STATE_CHOOSE_ANALYSIS" # File processed, ask user what to do
STATE_CONFIGURE = "STATE_CONFIGURE" # User has chosen, show sidebar
STATE_RUNNING = "STATE_RUNNING" # Run button clicked
STATE_RESULTS = "STATE_RESULTS" # Analysis complete

# --- Initialize Session State ---
def init_session_state():
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = STATE_INIT
    if 'last_file_id' not in st.session_state:
        st.session_state.last_file_id = None
    if 'df_ac' not in st.session_state:
        st.session_state.df_ac = None
    if 'ac_results' not in st.session_state:
        st.session_state.ac_results = None
    if 'analysis_config' not in st.session_state:
        st.session_state.analysis_config = {}

init_session_state()

# --- Helper function to format event labels ---
def format_event(event):
    start_str = event['start'].strftime('%Y-%m-%d %H:%M:%S')
    end_str = event['end'].strftime('%Y-%m-%d %H:%M:%S')
    duration_min = (event['end'] - event['start']).total_seconds() / 60.0
    # --- This label is now more descriptive ---
    return f"[{event['type']}] {start_str}  →  {end_str} ({duration_min:.1f} min)"


# ==================================
# --- 1. SIDEBAR CONFIGURATION ---
# ==================================
# Sidebar is built FIRST to ensure its values are available to the main app
st.sidebar.header("Configuration")

# --- Basic Config (always visible) ---
uploaded_file = st.sidebar.file_uploader("Upload AC Data File (LogDataFast)", type=["csv"])
st.sidebar.subheader("Column Names")
cfg_ac_time_col = st.sidebar.text_input("Timestamp Column", "Timestamp")
cfg_ac_power_col = st.sidebar.text_input("Power Column", "PoiPwrAt kW")
cfg_ac_soc_col = st.sidebar.text_input("SOC Column (optional)", "SocAvg %")

st.sidebar.subheader("Event Auto-Detection")
# --- Renamed for clarity ---
cfg_idle_kw = st.sidebar.number_input("Idle Threshold (kW)", value=10.0, help="Power level (absolute) to be considered 'idle' or 'zero'.")
cfg_min_duration_s = st.sidebar.number_input("Min Event Duration (s)", value=300.0, help="Shortest event to detect (e.g., 300s = 5 min). Filters out noise and short spikes.")


# --- Analysis-Specific Config (conditionally visible) ---
# These widgets are just placeholders; their visibility is controlled by the workflow state
if st.session_state.workflow_state in [STATE_CONFIGURE, STATE_RUNNING, STATE_RESULTS]:
    
    st.sidebar.subheader("Test Time Windows")
    
    # Get config from session state
    config = st.session_state.analysis_config
    analysis_type = config.get("analysis_type", "MANUAL")

    if analysis_type == "MANUAL":
        st.sidebar.info("Using Manual Time Inputs")
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
            
        # Store selections back in config for the "Run" button
        st.session_state.analysis_config['charge_start'] = pd.Timestamp(f"{ch_start_date} {ch_start_time}")
        st.session_state.analysis_config['charge_end'] = pd.Timestamp(f"{ch_end_date} {ch_end_time}")
        st.session_state.analysis_config['discharge_start'] = pd.Timestamp(f"{dis_start_date} {dis_start_time}")
        st.session_state.analysis_config['discharge_end'] = pd.Timestamp(f"{dis_end_date} {dis_end_time}")

    else: # KPI or RTE
        st.sidebar.success("Using Auto-Detected Times")
        if config.get('charge_start'):
            st.sidebar.markdown(f"**Charge:** {config['charge_start'].strftime('%H:%M:%S')} to {config['charge_end'].strftime('%H:%M:%S')}")
        if config.get('discharge_start'):
            st.sidebar.markdown(f"**Discharge:** {config['discharge_start'].strftime('%H:%M:%S')} to {config['discharge_end'].strftime('%H:%M:%S')}")

    # --- KPI & RTE Settings ---
    st.sidebar.subheader("KPI & RTE Settings")
    st.session_state.analysis_config['P_nom_kW'] = st.sidebar.number_input("Discharge KPI: Nominal Power (kW)", value=24500.0, help="Default changed to match graph.") # <-- Default Changed
    st.session_state.analysis_config['tol_pct'] = st.sidebar.number_input("Discharge KPI: Tolerance (%)", value=5.0)
    st.session_state.analysis_config['required_minutes'] = st.sidebar.number_input("Discharge KPI: Required Duration (min)", value=58)
    st.session_state.analysis_config['sampling_seconds'] = st.sidebar.number_input("Sampling (seconds)", min_value=1, value=1)
    st.session_state.analysis_config['discharge_positive'] = st.sidebar.checkbox("Discharge is Positive", True)
    st.session_state.analysis_config['ramp_trim_charge'] = st.sidebar.number_input("Charge Ramp Trim (s)", value=0)
    st.session_state.analysis_config['ramp_trim_discharge'] = st.sidebar.number_input("Discharge Ramp Trim (s)", value=0)

    # --- Nominal Power RTE (Method 1) Settings ---
    st.sidebar.subheader("Nominal Power RTE (Method 1)")
    c5, c6 = st.sidebar.columns(2)
    with c5:
        st.session_state.analysis_config['P_nom_charge_kW'] = st.number_input("Nominal Charge P (kW)", value=-24500.0, help="Default changed to match graph.") # <-- Default Changed
        st.session_state.analysis_config['tol_charge_pct'] = st.number_input("Charge Tolerance (%)", value=10.0)
    with c6:
        st.session_state.analysis_config['P_nom_discharge_kW'] = st.number_input("Nominal Discharge P (kW)", value=24500.0, help="Default changed to match graph.") # <-- Default Changed
        st.session_state.analysis_config['tol_discharge_pct'] = st.number_input("Discharge Tolerance (%)", value=1.0)
        
    # --- Action Buttons ---
    if st.sidebar.button("Run Analysis", use_container_width=True, type="primary"):
        st.session_state.workflow_state = STATE_RUNNING
        st.rerun()

    if st.session_state.workflow_state == STATE_RESULTS:
        if st.sidebar.button("Analyze Another Event", use_container_width=True):
            st.session_state.workflow_state = STATE_CHOOSE_ANALYSIS
            st.session_state.ac_results = None
            st.session_state.analysis_config = {}
            st.rerun()


# ==========================
# --- 2. MAIN APP AREA ---
# ==========================

# --- Workflow Step 0: Check for file ---
if uploaded_file is None:
    st.session_state.workflow_state = STATE_INIT
    st.info("Please upload your AC data file in the sidebar to begin.")
    st.stop()

# --- Workflow Step 1: New file is uploaded ---
# Check if it's a new file
if uploaded_file.file_id != st.session_state.last_file_id:
    st.session_state.last_file_id = uploaded_file.file_id
    st.session_state.workflow_state = STATE_PROCESSING
    st.session_state.ac_results = None # Clear old results

# --- Workflow Step 2: Process the file ---
if st.session_state.workflow_state == STATE_PROCESSING:
    with st.spinner("Processing file..."):
        try:
            df_ac = load_hycon_hybrid_fast(uploaded_file)
            
            ac_string_cols = {
                cfg_ac_time_col, 'OpStt', 'HybridSysState', 'HybridSysStateTrans', 
                'SignalValidity', 'CtrlModeCmdIn'
            }
            # We must use verbose=False here or it will print diagnostics
            # every time we change a setting.
            df_ac_clean = convert_mixed_numeric_columns(
                df_ac, exclude=ac_string_cols, verbose=False
            )
            
            st.session_state.df_ac = df_ac_clean
            st.session_state.workflow_state = STATE_CHOOSE_ANALYSIS
            st.rerun()
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.session_state.workflow_state = STATE_INIT
            st.session_state.last_file_id = None
            st.stop()

# --- Workflow Step 3: Ask User for Analysis Type ---
if st.session_state.workflow_state == STATE_CHOOSE_ANALYSIS:
    st.subheader("What would you like to analyze?")
    
    # Call find_all_events every time this page is drawn.
    # @st.cache_data will only re-execute if the inputs (like cfg_ac_power_col) change.
    all_events = find_all_events(
        st.session_state.df_ac,
        cfg_ac_time_col,
        cfg_ac_power_col,
        cfg_idle_kw,
        cfg_min_duration_s
    )
    
    # Filter events for the selection boxes
    dis_events = [e for e in all_events if e['type'] == 'Discharge']
    ch_events = [e for e in all_events if e['type'] == 'Charge']

    # --- Analysis Type 1: Discharge KPI ---
    with st.expander("📈 Run Discharge KPI Test", expanded=True):
        if not dis_events:
            st.warning("No discharge events detected with current settings. Try lowering the 'Idle Threshold' or 'Min Duration' in the sidebar.")
        else:
            selected_dis_label = st.selectbox(
                "Select a Discharge Event:",
                options=[format_event(e) for e in dis_events],
                key="kpi_dis_select"
            )
            if st.button("Configure KPI Test", use_container_width=True):
                # Find the selected event dict
                selected_event = next(e for e in dis_events if format_event(e) == selected_dis_label)
                
                st.session_state.analysis_config = {
                    "analysis_type": "KPI",
                    "discharge_start": selected_event['start'],
                    "discharge_end": selected_event['end'],
                    "charge_start": None, # Will be ignored
                    "charge_end": None,
                }
                st.session_state.workflow_state = STATE_CONFIGURE
                st.rerun()

    # --- Analysis Type 2: Round-Trip Efficiency (RTE) ---
    with st.expander("🔄 Run Round-Trip Efficiency (RTE) Test"):
        if not dis_events or not ch_events:
            st.warning("Both a Charge and Discharge event must be detected for RTE. Try adjusting settings in the sidebar.")
        else:
            selected_ch_label = st.selectbox(
                "Step 1: Select a Charge Event",
                options=[format_event(e) for e in ch_events],
                key="rte_ch_select"
            )
            selected_dis_label = st.selectbox(
                "Step 2: Select a Discharge Event",
                options=[format_event(e) for e in dis_events],
                key="rte_dis_select"
            )
            if st.button("Configure RTE Test", use_container_width=True):
                # Find the selected event dicts
                selected_ch_event = next(e for e in ch_events if format_event(e) == selected_ch_label)
                selected_dis_event = next(e for e in dis_events if format_event(e) == selected_dis_label)
                
                st.session_state.analysis_config = {
                    "analysis_type": "RTE",
                    "discharge_start": selected_dis_event['start'],
                    "discharge_end": selected_dis_event['end'],
                    "charge_start": selected_ch_event['start'],
                    "charge_end": selected_ch_event['end'],
                }
                st.session_state.workflow_state = STATE_CONFIGURE
                st.rerun()
                
    # --- Analysis Type 3: Manual ---
    with st.expander("⌨️ Use Manual Time Input"):
        st.info("This will allow you to manually enter all timestamps in the sidebar. This is useful for analyzing custom periods or short/spiky events.")
        if st.button("Configure Manual Test", use_container_width=True):
            st.session_state.analysis_config = {
                "analysis_type": "MANUAL",
            }
            st.session_state.workflow_state = STATE_CONFIGURE
            st.rerun()
            
    # Add a note if the column wasn't found, as a hint
    if (cfg_ac_power_col not in st.session_state.df_ac.columns) or (cfg_ac_time_col not in st.session_state.df_ac.columns):
        st.error(f"Warning: Column name '{cfg_ac_power_col}' or '{cfg_ac_time_col}' not found in the loaded data. Please check sidebar.")


# --- Workflow Step 4: Show Configuration Sidebar ---
if st.session_state.workflow_state == STATE_CONFIGURE:
    st.success("Configuration loaded. Please review settings in the sidebar and click 'Run Analysis'.")
    st.info("The analysis will run on the data you selected.")
    # Show diagnostics if this is the first time
    with st.expander("Numeric Conversion Diagnostics"):
        convert_mixed_numeric_columns(st.session_state.df_ac, exclude={cfg_ac_time_col}, verbose=True)


# --- Workflow Step 5: Run the Analysis ---
if st.session_state.workflow_state == STATE_RUNNING:
    with st.spinner("Running AC analysis..."):
        try:
            # Build the final config from session state
            df_ac = st.session_state.df_ac
            config = st.session_state.analysis_config
            
            # Add non-sidebar items
            config["time_col"] = cfg_ac_time_col
            config["power_col"] = cfg_ac_power_col
            config["soc_col"] = cfg_ac_soc_col
            
            # Rename keys for the function
            final_config = {
                "discharge_start": config.get("discharge_start"),
                "discharge_end": config.get("discharge_end"),
                "P_nom_kW": config.get("P_nom_kW"),
                "tol_pct": config.get("tol_pct"),
                "required_minutes": config.get("required_minutes"),
                "time_col": config.get("time_col"),
                "power_col": config.get("power_col"),
                "ramp_trim_seconds_discharge": int(config.get("ramp_trim_discharge", 0)),
                "charge_start": config.get("charge_start"),
                "charge_end": config.get("charge_end"),
                "sampling_seconds": config.get("sampling_seconds"),
                "discharge_positive": config.get("discharge_positive"),
                "ramp_trim_seconds_charge": int(config.get("ramp_trim_charge", 0)),
                "soc_col": config.get("soc_col"),
                "P_nom_charge_kW": config.get("P_nom_charge_kW"),
                "tol_charge_pct": config.get("tol_charge_pct"),
                "P_nom_discharge_kW": config.get("P_nom_discharge_kW"),
                "tol_discharge_pct": config.get("tol_discharge_pct"),
            }
            
            results = compute_nominal_from_poi_plotly(df_ac, **final_config)
            st.session_state.ac_results = results
            st.session_state.workflow_state = STATE_RESULTS
            st.rerun()
            
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.session_state.workflow_state = STATE_CONFIGURE # Go back to config
            
# --- Workflow Step 6: Display Results ---
if st.session_state.workflow_state == STATE_RESULTS:
    results = st.session_state.ac_results
    if results:
        st.header("AC-Side Test Results")
        
        # --- Plots ---
        # --- MODIFIED: More descriptive title ---
        st.subheader("AC Power Analysis Plot")
        fig_kpi = results['figure']
        st.plotly_chart(fig_kpi, use_container_width=True)
        st.download_button(
            label="Download Power Plot (HTML)",
            data=fig_kpi.to_html(),
            file_name="ac_power_plot.html",
            mime="text/html"
        )
        
        st.subheader("SOC Plot")
        fig_soc_ac = results['figure_soc']
        st.plotly_chart(fig_soc_ac, use_container_width=True)
        st.download_button(
            label="Download SOC Plot (HTML)",
            data=fig_soc_ac.to_html(),
            file_name="ac_soc_plot.html",
            mime="text/html"
        )

        st.subheader("Cumulative Energy Plot (Charge to Discharge)")
        fig_cum_energy = plot_calc_poi_egymtr(results['df_calc_poi_egymtr'], time_col=cfg_ac_time_col)
        st.plotly_chart(fig_cum_energy, use_container_width=True)
        st.download_button(
            label="Download Energy Plot (HTML)",
            data=fig_cum_energy.to_html(),
            file_name="ac_energy_plot.html",
            mime="text/html"
        )
        
        # --- Summary Table ---
        st.subheader("Analysis Summary Table")
        st.dataframe(results['summary_table'])
        
        # --- Warnings ---
        if results['warnings']:
            st.subheader("Analysis Warnings")
            for w in results['warnings']:
                st.warning(w)
    else:
        st.error("No results found. An unknown error may have occurred.")
        st.session_state.workflow_state = STATE_CHOOSE_ANALYSIS