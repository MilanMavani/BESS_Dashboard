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
# =======================================================================

# @st.cache_data tells Streamlit to only run this once per file
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
        with st.expander("[AC Numeric Conversion Diagnostics]"):
            for c, info in diagnostics.items():
                st.text(f"- {c}: {info}")
    return df_out


# =======================================================================
# SECTION 3: AC-SIDE ANALYSIS FUNCTIONS
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
    title: str = "Discharge KPI at POI (Plotly)",
    drop_duplicate_timestamps: bool = True,
    ramp_trim_seconds_discharge: int = 0,
    charge_start: Optional[str] = None,
    charge_end: Optional[str] = None,
    sampling_seconds: Optional[float] = None,
    discharge_positive: bool = True,
    ramp_trim_seconds_charge: int = 0,
    warn_irregular: bool = True,
    rte_min_charge_kWh: float = 0.01,
    soc_col: Optional[str] = None # SOC column
) -> Dict:
    """Calculates AC-side KPI, RTE, cumulative energy, and SOC plot."""
    
    ts_start_raw = pd.to_datetime(discharge_start, errors="raise")
    ts_end_raw = pd.to_datetime(discharge_end, errors="raise")
    if ts_end_raw <= ts_start_raw:
        raise ValueError("discharge_end must be after discharge_start")
    
    # Get raw charge times for SOC plot
    c_start_raw = pd.to_datetime(charge_start, errors="coerce")
    c_end_raw = pd.to_datetime(charge_end, errors="coerce")

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
    
    # --- Generate SOC Plot ---
    fig_soc = go.Figure()
    if soc_col and soc_col in d.columns:
        fig_soc.add_trace(go.Scatter(
            x=d[time_col], y=d[soc_col],
            mode="lines", name="SOC",
            line=dict(color="#FF5733", width=2)
        ))
        
        # Add charge window highlight
        if pd.notna(c_start_raw) and pd.notna(c_end_raw):
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

    # KPI calculation is only on power and time
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

    band_low = float(P_nom_kW) * (1 - tol_pct / 100.0)
    band_high = float(P_nom_kW) * (1 + tol_pct / 100.0)
    dKPI["in_band"] = (dKPI[power_col] >= band_low) & (dKPI[power_col] <= band_high)

    inband_time_s_cum = float(dKPI.loc[dKPI["in_band"], "dt_s"].sum())
    E_nom_cum_kWh = float(P_nom_kW) * (inband_time_s_cum / 3600.0)

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
    
    E_charge_kWh = E_discharge_kWh = np.nan
    RTE_pct = np.nan
    rte_method = None
    dt_expected = float(sampling_seconds) if sampling_seconds is not None else np.nan
    df_calc_poi_egymtr = None 

    if (charge_start is not None) and (charge_end is not None) and pd.notna(c_start_raw) and pd.notna(c_end_raw):
        
        # Use full power data (d) for cumulative energy
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

        # Use full power data (d) for RTE
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dKPI[time_col], y=dKPI[power_col],
                                mode="lines", name=f"{power_col}",
                                line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=[dKPI[time_col].min(), dKPI[time_col].max()],
                                y=[band_low, band_low], mode="lines", name=f"−{tol_pct:.0f}% band",
                                line=dict(color="#2ca02c", width=1.5, dash="dash")))
    
    fig.add_trace(go.Scatter(x=[dKPI[time_col].min(), dKPI[time_col].max()],
                                y=[band_high, band_high], mode="lines", name=f"+{tol_pct:.0f}% band",
                                line=dict(color="#2ca02c", width=1.5, dash="dash")))

    shapes = []
    for (s, e, dur_s) in segs:
        shapes.append(dict(type="rect", xref="x", yref="y",
                            x0=s, x1=e, y0=band_low, y1=band_high,
                            fillcolor="rgba(46,204,113,0.18)", line=dict(width=0),
                            layer="below"))
    fig.update_layout(shapes=shapes)

    subtitle = [
        f"P_nom={P_nom_kW:.0f} kW, tol=±{tol_pct:.1f}%",
        f"Actual={actual_energy_kWh:.1f} kWh",
        f"Nom(cum)={E_nom_cum_kWh:.1f} kWh",
        f"Nom(cont)={E_nom_cont_kWh:.1f} kWh"
    ]
    if required_minutes is not None:
        subtitle.append(f"Req={required_str}")
    if not np.isnan(RTE_pct):
        subtitle.append(f"RTE={RTE_pct:.2f}%")

    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{' | '.join(subtitle)}</sup>", x=0.01),
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        margin=dict(l=60, r=30, t=80, b=40),
    )

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

    if (charge_start is not None) and (charge_end is not None):
        metrics += [
            "Charge window start", "Charge window end", "Ramp trim (charge, s)",
            "Sampling interval expected (s)",
            "RTE method", "E_charge (kWh)", "E_discharge (kWh)", "RTE (%)"
        ]
        values += [
            c_start_trimmed if 'c_start_trimmed' in locals() else None,
            c_end_trimmed if 'c_end_trimmed' in locals() else None,
            int(ramp_trim_seconds_charge),
            dt_expected if not np.isnan(dt_expected) else None,
            rte_method,
            None if np.isnan(E_charge_kWh) else round(float(E_charge_kWh), 3),
            None if np.isnan(E_discharge_kWh) else round(float(E_discharge_kWh), 3),
            None if np.isnan(RTE_pct) else round(float(RTE_pct), 3),
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
# SECTION 4: STREAMLIT APPLICATION
# =======================================================================

st.set_page_config(layout="wide")
st.title("⚡ BESS AC-Side Capacity Test Analyzer")

# --- 1. CONFIGURE SIDEBAR ---
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload AC Data File (LogDataFast)", type=["csv"])

st.sidebar.subheader("Test Time Windows")
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
cfg_ac_time_col = st.sidebar.text_input("Timestamp Column", "Timestamp")
cfg_ac_power_col = st.sidebar.text_input("Power Column", "PoiPwrAt kW")
cfg_ac_soc_col = st.sidebar.text_input("SOC Column (optional)", "SocAvg %")


st.sidebar.subheader("KPI & RTE Settings")
cfg_ac_p_nom_kw = st.sidebar.number_input("Nominal Power (kW)", value=24500)
cfg_ac_tol_pct = st.sidebar.number_input("Tolerance (%)", value=5.0)
cfg_ac_required_minutes = st.sidebar.number_input("Required Duration (min)", value=58)
cfg_sampling_seconds = st.sidebar.number_input("Sampling (seconds)", min_value=1, value=1)
cfg_ac_discharge_positive = st.sidebar.checkbox("Discharge is Positive", True)
cfg_ac_ramp_trim_charge = st.sidebar.number_input("Charge Ramp Trim (s)", value=0)
cfg_ac_ramp_trim_discharge = st.sidebar.number_input("Discharge Ramp Trim (s)", value=0)


# --- 2. MAIN APP AREA ---
if uploaded_file is None:
    st.info("Please upload your AC data file to begin.")
else:
    if st.sidebar.button("Run Analysis", use_container_width=True, type="primary"):
        
        try:
            # Load data
            df_ac = load_hycon_hybrid_fast(uploaded_file)
            
            # --- CRITICAL FIX: Clean numeric columns ---
            st.info(f"Loaded AC file. Found columns: {df_ac.columns.to_list()}")
            
            # Define all known non-numeric columns to exclude them from conversion
            ac_string_cols = {
                cfg_ac_time_col, 'OpStt', 'HybridSysState', 'HybridSysStateTrans', 
                'SignalValidity', 'CtrlModeCmdIn'
            }
            with st.spinner("Cleaning numeric data..."):
                df_ac = convert_mixed_numeric_columns(df_ac, exclude=ac_string_cols, verbose=True)
            st.success("Numeric cleaning complete.")
            
            # --- Build Config ---
            ac_config = {
                "discharge_start": discharge_start_ts,
                "discharge_end": discharge_end_ts,
                "P_nom_kW": cfg_ac_p_nom_kw,
                "tol_pct": cfg_ac_tol_pct,
                "required_minutes": cfg_ac_required_minutes,
                "time_col": cfg_ac_time_col,
                "power_col": cfg_ac_power_col,
                "ramp_trim_seconds_discharge": cfg_ac_ramp_trim_discharge,
                "charge_start": charge_start_ts,
                "charge_end": charge_end_ts,
                "sampling_seconds": cfg_sampling_seconds,
                "discharge_positive": cfg_ac_discharge_positive,
                "ramp_trim_seconds_charge": cfg_ac_ramp_trim_charge,
                "soc_col": cfg_ac_soc_col # Pass SOC column
            }
            
            # --- Run Analysis ---
            with st.spinner("Running AC analysis..."):
                results = compute_nominal_from_poi_plotly(df_ac, **ac_config)
                st.session_state.ac_results = results # Store results
            st.success("AC Analysis Complete!")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            # Clear results if analysis fails
            if 'ac_results' in st.session_state:
                del st.session_state.ac_results
    
    # --- 4. DISPLAY RESULTS ---
    if "ac_results" in st.session_state:
        results = st.session_state.ac_results
        
        st.header("AC-Side Test Results")
        
        # --- Plots ---
        st.subheader("Discharge KPI Plot")
        fig_kpi = results['figure'] # Store figure
        st.plotly_chart(fig_kpi, use_container_width=True)
        # --- NEW: Download Button ---
        st.download_button(
            label="Download KPI Plot (HTML)",
            data=fig_kpi.to_html(),
            file_name="ac_kpi_plot.html",
            mime="text/html"
        )
        
        st.subheader("SOC Plot")
        fig_soc_ac = results['figure_soc'] # Store figure
        st.plotly_chart(fig_soc_ac, use_container_width=True)
        # --- NEW: Download Button ---
        st.download_button(
            label="Download SOC Plot (HTML)",
            data=fig_soc_ac.to_html(),
            file_name="ac_soc_plot.html",
            mime="text/html"
        )

        st.subheader("Cumulative Energy Plot (Charge to Discharge)")
        fig_cum_energy = plot_calc_poi_egymtr(results['df_calc_poi_egymtr'], time_col=cfg_ac_time_col)
        st.plotly_chart(fig_cum_energy, use_container_width=True)
        # --- NEW: Download Button ---
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
        st.info("Click 'Run Analysis' in the sidebar to process the data.")