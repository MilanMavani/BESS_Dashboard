# src/plot_logic_chart.py

from __future__ import annotations
import re
from typing import Optional, List, Dict
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st

from .plot_utils import _build_unique_legend_labels, DEVICE_ID_COLUMN

# ==================== Plotly Helpers ====================
def _legend_layout(placement: str):
    # Returns Plotly legend configuration and figure margins
    placement = (placement or "Right").strip().lower()
    if placement == "right":
        return (
            dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            dict(l=60, r=220, t=70, b=50),
        )
    if placement == "top":
        return (
            dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
            dict(l=60, r=60, t=110, b=60),
        )
    if placement == "bottom":
        return (
            dict(orientation="h", yanchor="top", y=-0.28, xanchor="left", x=0),
            dict(l=60, r=60, t=70, b=150),
        )
    # Default/Inside
    return (
        dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.75)"),
        dict(l=60, r=60, t=70, b=50),
    )


def _sanitize_html_name(name: str) -> str:
    # Cleans filename for HTML download
    name = re.sub(r"[^A-Za-z0-9 ._\-]", "_", (name or "plotly")).strip(" ._-")
    if not name.lower().endswith(".html"):
        name += ".html"
    return name


def _export_ui_under_plot(fig, default_prefix: str, unique_key: str, title_text: str = ""):
    # Adds a Streamlit expander for downloading the Plotly chart as HTML.
    with st.expander("⬇️ Export (Plotly HTML)", expanded=False):
        default_name = f"{default_prefix}_{datetime.now():%Y%m%d_%H%M%S}.html"
        file_name = st.text_input(
            "File name",
            value=default_name,
            key=f"export_name_{unique_key}",
        )
        safe_name = _sanitize_html_name(file_name)
        html = pio.to_html(fig, include_plotlyjs="inline", full_html=True)
        st.download_button(
            "💾 Download HTML",
            data=html.encode("utf-8"),
            file_name=safe_name,
            mime="text/html",
            key=f"download_html_{unique_key}",
        )


# ==================== Plotly Functions ====================
def plot_plotly_dual_y(
    df_long_primary: pd.DataFrame,
    df_long_secondary: Optional[pd.DataFrame],
    secondary_feature: Optional[str],
    title: str,
    legend_placement: str,
):
    """Generates an interactive Plotly chart with primary (left) and secondary (right) Y axes."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    order = []
    base_label_map = {}

    # Primary (Left Y)
    if not df_long_primary.empty:
        for key, g in df_long_primary.groupby("SeriesKey", sort=False):
            order.append(key)
            base_label_map[key] = g["LegendLabel"].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=g["Time"], y=g["Value"], mode="lines", name=key,
                    hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br><b>%{y:.2f}</b><extra>" + g["LegendLabel"].iloc[0] + "</extra>",
                ),
                secondary_y=False,
            )
        unique_labels = _build_unique_legend_labels(order, base_label_map)
        for tr in fig.data:
            if tr.name in unique_labels:
                tr.name = unique_labels[tr.name]

    # Secondary (Right Y) - UPDATED to use solid line (mode="lines")
    if df_long_secondary is not None and not df_long_secondary.empty and secondary_feature:
        for key, g in df_long_secondary.groupby("SeriesKey", sort=False):
            fig.add_trace(
                go.Scatter(
                    x=g["Time"], y=g["Value"], mode="lines", name=g["LegendLabel"].iloc[0] + " (sec)", # line=dict(dash="dot") removed
                    hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br><b>%{y:.2f}</b><extra>" + g["LegendLabel"].iloc[0] + " (sec)</extra>",
                ),
                secondary_y=True,
            )

    # Layout
    prim_feats = df_long_primary["Feature"].unique().tolist() if not df_long_primary.empty else []
    y_left_title = prim_feats[0] if len(prim_feats) == 1 else ("Primary (multiple)" if prim_feats else "Primary")
    y_right_title = secondary_feature if secondary_feature else "Secondary"

    legend_cfg, margins = _legend_layout(legend_placement)
    fig.update_layout(
        template="plotly_white", hovermode="x unified", title=title,
        legend={**legend_cfg, "title": {"text": "Series"}},
        margin=margins,
    )
    fig.update_yaxes(title_text=y_left_title, secondary_y=False)
    fig.update_yaxes(title_text=y_right_title, secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", (title or "plotly_overlay_dualY"))
    _export_ui_under_plot(fig, default_prefix="plotly_overlay_dualY", unique_key=f"dual_{slug}", title_text=title or "")


def plot_plotly_single_axis(df_long: pd.DataFrame, plot_mode: str, legend_placement: str):
    """Generates an interactive Plotly chart with a single Y axis (Overlay or Subplots)."""
    if df_long.empty:
        st.warning("No data for plotting.")
        return

    # Mapped internal names
    if plot_mode == "Overlay Plot (All Features on Single Axis)":
        fig = px.line(
            df_long, x="Time", y="Value", color="SeriesKey",
            title="Plotly Overlay Plot: Features Grouped by Device ID", template="plotly_white",
        )
        default_prefix = "plotly_overlay"
        title_text = "Plotly Overlay Plot: Features Grouped by Device ID"

        # Rename traces to unique LegendLabel using helper
        unique_series_order = list(df_long["SeriesKey"].drop_duplicates())
        base_label_map = {sk: df_long.loc[df_long["SeriesKey"] == sk, "LegendLabel"].iloc[0] for sk in unique_series_order}
        unique_labels = _build_unique_legend_labels(unique_series_order, base_label_map)
        for tr in fig.data:
            sk = tr.name
            if sk in unique_labels:
                tr.name = unique_labels[sk]

    # Mapped internal names
    elif plot_mode == "Subplots (Separate Axis for Each Feature)":
        n_feat = df_long["Feature"].nunique()
        fig = px.line(
            df_long, x="Time", y="Value", color=DEVICE_ID_COLUMN, facet_row="Feature",
            height=max(350 * n_feat, 350),
            title="Plotly Subplots: Each Feature Separated by Device ID", template="plotly_white",
        )
        fig.update_yaxes(matches=None)
        default_prefix = "plotly_subplots"
        title_text = "Plotly Subplots: Each Feature Separated by Device ID"
    else:
        st.error("Invalid Plotly mode selected.")
        return

    legend_cfg, margins = _legend_layout(legend_placement)
    fig.update_layout(
        hovermode="x unified",
        legend={**legend_cfg, "title": {"text": "Series"}},
        margin=margins,
    )

    st.plotly_chart(fig, use_container_width=True)
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", fig.layout.title.text or default_prefix)
    _export_ui_under_plot(fig, default_prefix=default_prefix, unique_key=f"single_{slug}", title_text=title_text)


# ==================== Seaborn Functions ====================
def plot_seaborn_dual_y(
    df_long_primary: pd.DataFrame,
    df_long_secondary: Optional[pd.DataFrame],
    secondary_feature: Optional[str],
    title: str,
):
    """Generates a static Seaborn/Matplotlib chart with primary (left) and secondary (right) Y axes."""
    if df_long_primary.empty and (df_long_secondary is None or df_long_secondary.empty):
        st.warning("No data for plotting.")
        return

    sns.set_theme(style="whitegrid")
    fig, ax_left = plt.subplots(figsize=(14, 7))
    handles, labels = [], []

    # Primary (Left Y)
    if not df_long_primary.empty:
        sns.lineplot(data=df_long_primary, x="Time", y="Value", hue="SeriesKey", dashes=False, ax=ax_left, legend=False)
        prim_feats = df_long_primary["Feature"].unique().tolist()
        ax_left.set_ylabel(prim_feats[0] if len(prim_feats) == 1 else "Primary (multiple)")

        # Create unique legend handles/labels
        h, l = ax_left.get_legend_handles_labels()
        order = list(df_long_primary["SeriesKey"].drop_duplicates())
        base_map = {sk: df_long_primary.loc[df_long_primary["SeriesKey"] == sk, "LegendLabel"].iloc[0] for sk in order}
        unique_map = _build_unique_legend_labels(order, base_map)

        # Matplotlib/Seaborn sometimes adds extra labels, so we check for SeriesKey match
        for idx, key in enumerate(l):
            handles.append(h[idx])
            labels.append(unique_map.get(key, key))

    # Secondary (Right Y)
    if df_long_secondary is not None and not df_long_secondary.empty and secondary_feature:
        ax_right = ax_left.twinx()
        for device, df_g in df_long_secondary.groupby(DEVICE_ID_COLUMN, sort=False):
            # No change needed for Seaborn/Matplotlib, it's already using a solid line by default
            (ln,) = ax_right.plot(
                df_g["Time"], df_g["Value"], linestyle="-", label=f"{device} - {secondary_feature} (sec)"
            )
            handles.append(ln)
            labels.append(ln.get_label())
        ax_right.set_ylabel(secondary_feature)

    ax_left.set_title(title)
    ax_left.set_xlabel("Time")
    ax_left.grid(True)
    plt.xticks(rotation=45)
    ax_left.legend(handles, labels, title="Series", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_seaborn_single_axis(df_long: pd.DataFrame, plot_mode: str):
    """Generates a static Seaborn/Matplotlib chart with a single Y axis (Overlay or Subplots)."""
    if df_long.empty:
        st.warning("No data for plotting.")
        return
    sns.set_theme(style="whitegrid")
    unique_features = df_long["Feature"].unique()
    num_features = len(unique_features)

    # Mapped internal names
    if plot_mode == "Overlay Plot (All Features on Single Axis)":
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=df_long, x="Time", y="Value", hue="SeriesKey", dashes=False, legend="full")
        plt.title("Seaborn Overlay Plot: Features Grouped by Device ID", fontsize=16)

        # Manually create unique legend labels
        order = list(df_long["SeriesKey"].drop_duplicates())
        base_map = {sk: df_long.loc[df_long["SeriesKey"] == sk, "LegendLabel"].iloc[0] for sk in order}
        unique_map = _build_unique_legend_labels(order, base_map)

        h, l = plt.gca().get_legend_handles_labels()
        new_labels = [unique_map.get(label, label) for label in l]
        plt.legend(h, new_labels, title="Series", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        st.pyplot(plt)
        plt.close()

    # Mapped internal names
    elif plot_mode == "Subplots (Separate Axis for Each Feature)":
        fig, axes = plt.subplots(num_features, 1, figsize=(14, 4 * num_features), sharex=True)
        if num_features == 1: axes = [axes]

        for i, feature in enumerate(unique_features):
            ax = axes[i]
            df_feature = df_long[df_long["Feature"] == feature]
            sns.lineplot(
                data=df_feature, x="Time", y="Value", hue=DEVICE_ID_COLUMN, dashes=False, ax=ax,
                legend="full" if i == 0 else False,
            )
            ax.set_title(f"Feature: {feature}", fontsize=14)
            ax.set_ylabel(feature, fontsize=12)
            ax.grid(True)

        axes[-1].set_xlabel("Time", fontsize=14)
        if num_features > 0 and axes[0].get_legend():
            axes[0].get_legend().set_title(DEVICE_ID_COLUMN)
            axes[0].get_legend().set_bbox_to_anchor((1.05, 1))
            axes[0].get_legend().set_loc("upper left")

        fig.suptitle("Seaborn Subplots: Correctly Scaled Y-Axes for Each Feature", y=1.02, fontsize=16)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)