# src/analytics.py
from __future__ import annotations
import numpy as np
import pandas as pd

DEVICE_ID_COL = "device-address:uid"


def estimate_sample_rate_per_device(
    df: pd.DataFrame,
    *,
    device_col: str = DEVICE_ID_COL,
    min_points: int = 3,
) -> pd.DataFrame:
    """
    Estimate median Δt per device with a friendly label, or "irregular" if spread is high.
    Returns columns: ["device", "dt_t", "label"] where dt_t is median seconds.
    """
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=["device", "dt_t", "label"])

    def _label_from_deltas(median_dt: float, q05: float, q95: float) -> str:
        if np.isnan(median_dt):
            return "unknown"
        # Quantize to nearest common seconds (display only)
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


def compute_capacity_ac_features(
    df: pd.DataFrame,
    *,
    status_col: str,
    operate_value: str,
    power_col: str,
    use_assume_1s: bool = True,
) -> pd.DataFrame:
    """
    Compute AC-side energy increment (Calc-PoiEgy, kWh per row) and cumulative meter (Calc-PoiEgyMtr).
    Energy increments are only retained where status == operate_value.
    """
    if df is None or df.empty:
        return df.copy()

    out = df.copy()

    # Guard columns
    if status_col not in out.columns or power_col not in out.columns:
        return out

    # Build mask: rows where status equals the operate value
    op_mask = out[status_col].astype(str).eq(operate_value)

    # Power to numeric (assumed kW)
    p_kw = pd.to_numeric(out[power_col], errors="coerce")

    if use_assume_1s:
        # Legacy: assume 1-second samples → E(kWh) = P(kW) / 3600
        inc_kwh = p_kw / 3600.0
    else:
        # Use actual Δt to previous sample (per device if available)
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
            # Fallback: behave like 1s mode when no DatetimeIndex
            inc_kwh = p_kw / 3600.0

    # Keep values only while operating; elsewhere NaN
    inc_kwh = inc_kwh.where(op_mask)

    # Cumulative meter
    if DEVICE_ID_COL in out.columns:
        cum_kwh = inc_kwh.groupby(out[DEVICE_ID_COL], group_keys=False).cumsum()
    else:
        cum_kwh = inc_kwh.cumsum()

    out["Calc-PoiEgy"] = inc_kwh
    out["Calc-PoiEgyMtr"] = cum_kwh
    return out


def operate_time_bounds(
    df: pd.DataFrame,
    *,
    status_col: str,
    operate_value: str,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Return (min_ts, max_ts) where status == operate_value within df.index (DatetimeIndex).
    """
    if df is None or df.empty or status_col not in df.columns:
        return None, None
    if not isinstance(df.index, pd.DatetimeIndex):
        return None, None

    op_mask = df[status_col].astype(str).eq(operate_value)
    if not op_mask.any():
        return None, None

    idx = df.index[op_mask]
    return idx.min(), idx.max()


# -------------------- DC side (coming soon) --------------------

def compute_capacity_dc_features(
    df: pd.DataFrame,
    *,
    power_col: str,
    soc_col: str | None = None,
    use_assume_1s: bool = True,
) -> pd.DataFrame:
    """
    Placeholder for DC-side features (COMING SOON).

    Intended behavior (to be implemented):
      - Accept DC power signal (e.g., 'DcTotWatt' in W or kW) and optional SOC.
      - Compute per-row DC energy increments and cumulative energy (per device if available),
        using either actual Δt or a legacy 1-second assumption (same pattern as AC).
      - Optionally derive SOC-based KPIs for the operate window.

    Current behavior:
      - Returns df.copy() so the UI can be wired without failing.
    """
    return df.copy()