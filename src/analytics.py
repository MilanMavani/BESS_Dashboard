# src/analytics.py
from __future__ import annotations
import numpy as np
import pandas as pd

DEVICE_ID_COL = "device-address:uid"

# ... [Keep estimate_sample_rate_per_device unchanged] ...
def estimate_sample_rate_per_device(
    df: pd.DataFrame,
    *,
    device_col: str = DEVICE_ID_COL,
    min_points: int = 3,
) -> pd.DataFrame:
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=["device", "dt_t", "label"])

    def _label_from_deltas(median_dt: float, q05: float, q95: float) -> str:
        if np.isnan(median_dt):
            return "unknown"
        common = np.array([0.5, 1, 2, 5, 10, 15, 30, 60], dtype=float)
        nearest = common[np.argmin(np.abs(common - median_dt))]
        spread = q95 - q05
        if spread > 0.6 * median_dt:
            return f"irregular (~{median_dt:.1f} s; 5–95%: {q05:.1f}–{q95:.1f} s)"
        return f"≈{int(nearest) if nearest.is_integer() else nearest:g} s"

    rows = []
    if device_col in df.columns:
        for dev, g in df.groupby(device_col):
            if len(g.index) < min_points:
                rows.append({"device": str(dev), "dt_t": np.nan, "label": "too few points"})
                continue
            dts = g.index.to_series().diff().dt.total_seconds().dropna()
            if dts.empty:
                rows.append({"device": str(dev), "dt_t": np.nan, "label": "unknown"})
                continue
            m = float(dts.median())
            q05 = float(dts.quantile(0.05))
            q95 = float(dts.quantile(0.95))
            rows.append({"device": str(dev), "dt_t": m, "label": _label_from_deltas(m, q05, q95)})
    else:
        if len(df.index) < min_points:
            return pd.DataFrame([{"device": "ALL", "dt_t": np.nan, "label": "too few points"}])
        dts = df.index.to_series().diff().dt.total_seconds().dropna()
        if dts.empty:
            return pd.DataFrame([{"device": "ALL", "dt_t": np.nan, "label": "unknown"}])
        m = float(dts.median())
        q05 = float(dts.quantile(0.05))
        q95 = float(dts.quantile(0.95))
        rows.append({"device": "ALL", "dt_t": m, "label": _label_from_deltas(m, q05, q95)})

    return pd.DataFrame(rows, columns=["device", "dt_t", "label"])


# ... [Keep compute_capacity_ac_features unchanged] ...
def compute_capacity_ac_features(
    df: pd.DataFrame,
    *,
    status_col: str,
    operate_value: str,
    power_col: str,
    use_assume_1s: bool = True,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy()

    out = df.copy()

    if status_col not in out.columns or power_col not in out.columns:
        return out

    op_mask = out[status_col].astype(str).eq(operate_value)
    p_kw = pd.to_numeric(out[power_col], errors="coerce")

    if use_assume_1s:
        inc_kwh = p_kw / 3600.0
    else:
        if isinstance(out.index, pd.DatetimeIndex):
            if DEVICE_ID_COL in out.columns:
                dt_sec = (
                    out.groupby(DEVICE_ID_COL, group_keys=False)
                    .apply(lambda g: g.index.to_series().diff().dt.total_seconds().clip(lower=0).fillna(0))
                )
            else:
                dt_sec = out.index.to_series().diff().dt.total_seconds().clip(lower=0).fillna(0)
            inc_kwh = (p_kw * dt_sec) / 3600.0
        else:
            inc_kwh = p_kw / 3600.0

    inc_kwh = inc_kwh.where(op_mask)

    if DEVICE_ID_COL in out.columns:
        cum_kwh = inc_kwh.groupby(out[DEVICE_ID_COL], group_keys=False).cumsum()
    else:
        cum_kwh = inc_kwh.cumsum()

    out["Calc-PoiEgy"] = inc_kwh
    out["Calc-PoiEgyMtr"] = cum_kwh
    return out

# ... [Keep operate_time_bounds unchanged] ...
def operate_time_bounds(
    df: pd.DataFrame,
    *,
    status_col: str,
    operate_value: str,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if df is None or df.empty or status_col not in df.columns:
        return None, None
    if not isinstance(df.index, pd.DatetimeIndex):
        return None, None

    op_mask = df[status_col].astype(str).eq(operate_value)
    if not op_mask.any():
        return None, None

    idx = df.index[op_mask]
    return idx.min(), idx.max()


def compute_cumulative_deltas(
    df: pd.DataFrame,
    counters: list[str],
) -> pd.DataFrame:

    rows = []
    if df is None or df.empty:
        return pd.DataFrame()

    for col in counters:
        if col not in df.columns:
            continue
            
        series = df[col].dropna()
        
        if series.empty:
            rows.append({
                "Metric": col, 
                "Start (MWh)": 0.0, 
                "End (MWh)": 0.0, 
                "Total Energy (MWh)": 0.0,
                "Note": "No valid data"
            })
            continue

        start_val = float(series.iloc[0])
        end_val = float(series.iloc[-1])
        delta = end_val - start_val
        
        note = "Normal"
        if delta < 0:
             note = "Negative Delta (Reset?)"

        energy_mwh = delta

        rows.append({
            "Metric": col,
            "Start (MWh)": start_val,
            "End (MWh)": end_val,
            "Total Energy (MWh)": energy_mwh,
            "Note": note
        })

    return pd.DataFrame(rows)

# --- UPDATED FUNCTION WITH DC RTE ---
def compute_efficiency_metrics(
    e_ac_in: float,
    e_ac_out: float,
    e_dc_in: float,
    e_dc_out: float
) -> pd.DataFrame:
    """
    Calculates RTE, Charge Eff, Discharge Eff, Battery Eff (DC RTE), and Loss.
    """
    # Avoid division by zero
    rte = (e_ac_out / e_ac_in * 100.0) if e_ac_in > 0 else 0.0
    chg_eff = (e_dc_in / e_ac_in * 100.0) if e_ac_in > 0 else 0.0
    dis_eff = (e_ac_out / e_dc_out * 100.0) if e_dc_out > 0 else 0.0
    
    # DC RTE (Battery Efficiency)
    batt_eff = (e_dc_out / e_dc_in * 100.0) if e_dc_in > 0 else 0.0
    
    # System Loss
    loss = e_ac_in - e_ac_out
    
    metrics = [
        {"Metric": "System RTE", "Value": rte, "Unit": "%", "Formula": "AC_Out / AC_In"},
        {"Metric": "Battery Efficiency ", "Value": batt_eff, "Unit": "%", "Formula": "DC_Out / DC_In"},
        {"Metric": "Charging Efficiency", "Value": chg_eff, "Unit": "%", "Formula": "DC_In / AC_In"},
        {"Metric": "Discharging Efficiency ", "Value": dis_eff, "Unit": "%", "Formula": "AC_Out / DC_Out"},
        {"Metric": "System Loss", "Value": loss, "Unit": "MWh", "Formula": "AC_In - AC_Out"},
    ]
    
    return pd.DataFrame(metrics)