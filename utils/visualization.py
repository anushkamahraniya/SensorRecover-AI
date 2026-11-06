"""All charts, drawn with Plotly in the pastel blue style.

Pale blues are hard to tell apart by colour alone, so every line chart also
varies dash style and marker shape. Tables with the same data sit beside them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.evaluation import (
    HOPFIELD_BINARY,
    HOPFIELD_SINGLE,
    HOPFIELD_WINDOW,
    INTERPOLATION,
    KNN_SINGLE,
    KNN_WINDOW,
    MEAN,
)

NAVY = "#344B64"
GREY = "#71869A"
GRID = "#E9F1F8"
TRUE_LINE = "#344B64"
ACCENT = "#5A7FD0"

# Colour, dash and marker belong to the method, so they never change between charts.
METHOD_STYLE = {
    HOPFIELD_WINDOW: ("#344B64", "solid", "circle"),
    KNN_WINDOW: ("#5A7FD0", "dash", "square"),
    INTERPOLATION: ("#6CBFCC", "dot", "diamond"),
    MEAN: ("#A5AEEA", "dashdot", "triangle-up"),
    HOPFIELD_SINGLE: ("#71869A", "solid", "circle-open"),
    KNN_SINGLE: ("#8DB9E0", "dash", "square-open"),
    HOPFIELD_BINARY: ("#B4A9DC", "longdash", "x"),
    "No repair": ("#C3D3E0", "solid", "cross"),
}
SEQUENTIAL = [[0, "#F3F8FC"], [0.35, "#C5E3F6"], [0.7, "#7FAFD8"], [1, "#344B64"]]
# Diverging: blue for positive, muted violet for negative, pale neutral midpoint.
DIVERGING = [[0, "#7D73B8"], [0.25, "#C3BEE6"], [0.5, "#F5F7FA"], [0.75, "#9CC6E8"], [1, "#2F5E8C"]]


def _layout(fig: go.Figure, title: str = "", height: int = 380, y_title="", x_title="") -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": NAVY}, "x": 0, "xanchor": "left"},
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": "Inter, Segoe UI, sans-serif", "color": GREY, "size": 12.5},
        margin={"l": 56, "r": 22, "t": 54 if title else 18, "b": 96},
        hoverlabel={"bgcolor": "#FFFFFF", "bordercolor": "#C5E3F6", "font": {"color": NAVY}},
        legend={"orientation": "h", "yanchor": "top", "y": -0.3, "x": 0, "font": {"size": 12}},
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, zeroline=False, linecolor="#D6EAF8")
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False, linecolor="#D6EAF8")
    return fig


def method_lines(summary: pd.DataFrame, metric: str, methods: list[str], title="") -> go.Figure:
    """A metric against the share of missing data, one line per method."""
    fig = go.Figure()
    for method in methods:
        rows = summary[summary["method"] == method].sort_values("rate")
        if rows.empty:
            continue
        colour, dash, marker = METHOD_STYLE[method]
        spread = rows["RMSE_std"] if metric == "RMSE" and "RMSE_std" in rows else None
        fig.add_scatter(
            x=rows["rate"] * 100, y=rows[metric], name=method, mode="lines+markers",
            line={"color": colour, "dash": dash, "width": 2.4},
            marker={"symbol": marker, "size": 9, "color": colour},
            error_y=None if spread is None else {"array": spread, "color": colour, "thickness": 1},
            hovertemplate=f"{method}<br>%{{x:.0f}}% missing<br>{metric} %{{y:.3f}}<extra></extra>",
        )
    fig.update_xaxes(ticksuffix="%")
    return _layout(fig, title, y_title=metric, x_title="Share of readings hidden")


def method_bars(table: pd.DataFrame, metric: str, title="", ascending=True) -> go.Figure:
    """Horizontal bars, best method on top. Each bar shows its value."""
    ordered = table.sort_values(metric, ascending=not ascending)
    colours = [METHOD_STYLE.get(m, (ACCENT,))[0] for m in ordered["method"]]
    fig = go.Figure(
        go.Bar(
            x=ordered[metric], y=ordered["method"], orientation="h", marker_color=colours,
            marker={"cornerradius": 5}, text=ordered[metric].map("{:.3f}".format),
            textposition="outside", textfont={"color": NAVY}, cliponaxis=False,
            hovertemplate="%{y}<br>" + metric + " %{x:.3f}<extra></extra>",
        )
    )
    fig = _layout(fig, title, height=max(250, 58 * len(ordered) + 80), x_title=metric)
    fig.update_layout(margin={"l": 160, "r": 58}, showlegend=False, bargap=0.42)
    return fig


def sensor_heatmap(table: pd.DataFrame, metric="MAE", title="") -> go.Figure:
    """Error for each method and sensor, normalised per sensor by the best value."""
    pivot = table.pivot(index="method", columns="sensor", values=metric)
    pivot = pivot[[c for c in table["sensor"].unique()]]
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(), x=list(pivot.columns), y=list(pivot.index),
            colorscale=SEQUENTIAL, colorbar={"title": metric, "thickness": 12},
            xgap=2, ygap=2,
            hovertemplate="%{y}<br>%{x}<br>" + metric + " %{z:.3f}<extra></extra>",
        )
    )
    fig = _layout(fig, title, height=120 + 44 * len(pivot))
    fig.update_layout(margin={"l": 150})
    return fig


def correlation_heatmap(table: pd.DataFrame, title="") -> go.Figure:
    corr = table.corr(numeric_only=True)
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(), x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1,
            colorscale=DIVERGING, xgap=2, ygap=2, colorbar={"thickness": 12},
            text=corr.round(2).to_numpy(), texttemplate="%{text}", textfont={"size": 10.5},
            hovertemplate="%{y} and %{x}<br>correlation %{z:.2f}<extra></extra>",
        )
    )
    fig = _layout(fig, title, height=520)
    fig.update_layout(margin={"l": 110, "b": 100})
    fig.update_yaxes(autorange="reversed")
    return fig


def missing_bars(table: pd.DataFrame, title="") -> go.Figure:
    share = (table.isna().mean() * 100).sort_values()
    fig = go.Figure(
        go.Bar(
            x=share.values, y=share.index, orientation="h", marker_color="#9FC4E4",
            marker={"cornerradius": 4}, text=[f"{v:.1f}%" for v in share.values],
            textposition="outside", textfont={"color": NAVY}, cliponaxis=False,
            hovertemplate="%{y}<br>%{x:.1f}% missing<extra></extra>",
        )
    )
    fig = _layout(fig, title, height=70 + 30 * len(share), x_title="Share of hours with no reading")
    fig.update_layout(margin={"l": 110, "r": 58}, showlegend=False)
    fig.update_xaxes(ticksuffix="%")
    return fig


def distribution(table: pd.DataFrame, column: str, title="") -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=table[column].dropna(), nbinsx=50, marker_color="#9FC4E4",
            marker_line={"color": "#FFFFFF", "width": 1},
            hovertemplate="%{x}<br>%{y} hours<extra></extra>",
        )
    )
    fig = _layout(fig, title, height=300, x_title=column, y_title="Hours")
    fig.update_layout(showlegend=False, bargap=0.02)
    return fig


CELL_STATES = ["Observed", "Natural gap", "Hidden on purpose", "Corrupted on purpose"]
CELL_COLOURS = ["#F1F7FC", "#D9DDF7", "#7FAFD8", "#344B64"]


def cell_map(states: np.ndarray, index, columns, title="", max_rows=336) -> go.Figure:
    """A map of every cell: 0 observed, 1 natural gap, 2 hidden, 3 corrupted."""
    states, index = states[:max_rows], index[:max_rows]
    scale = []
    for i, colour in enumerate(CELL_COLOURS):
        scale += [[i / 4, colour], [(i + 1) / 4, colour]]
    labels = np.array(CELL_STATES)[states]
    fig = go.Figure(
        go.Heatmap(
            z=states.T, x=index, y=list(columns), zmin=0, zmax=4, colorscale=scale,
            showscale=False, ygap=1, customdata=labels.T,
            hovertemplate="%{y}<br>%{x}<br>%{customdata}<extra></extra>",
        )
    )
    for name, colour in zip(CELL_STATES, CELL_COLOURS):
        fig.add_scatter(x=[None], y=[None], mode="markers", name=name,
                        marker={"symbol": "square", "size": 12, "color": colour,
                                "line": {"color": "#C5D5E2", "width": 1}})
    fig = _layout(fig, title, height=120 + 26 * len(columns))
    fig.update_layout(margin={"l": 110, "b": 70}, legend={"y": -0.12})
    return fig


def series_comparison(index, truth, damaged, estimate, hidden, sensor, unit="", title="") -> go.Figure:
    """True line, what the system saw, and the values it reconstructed."""
    fig = go.Figure()
    fig.add_scatter(x=index, y=truth, name="True reading", mode="lines",
                    line={"color": "#A9C6DE", "width": 2},
                    hovertemplate="True %{y:.2f}<extra></extra>")
    if damaged is not None:
        fig.add_scatter(x=index, y=damaged, name="Seen by the model", mode="markers",
                        marker={"color": "#C5E3F6", "size": 4},
                        hovertemplate="Seen %{y:.2f}<extra></extra>")
    fig.add_scatter(x=index[hidden], y=estimate[hidden], name="Reconstructed", mode="markers",
                    marker={"color": NAVY, "size": 7, "line": {"color": "#FFFFFF", "width": 1}},
                    hovertemplate="Reconstructed %{y:.2f}<extra></extra>")
    fig = _layout(fig, title, height=380, y_title=f"{sensor} ({unit})" if unit else sensor)
    fig.update_layout(hovermode="x unified")
    return fig


def truth_vs_estimate(truth, estimate, sensor="", title="") -> go.Figure:
    """Every point on the diagonal would be a perfect reconstruction."""
    low = float(min(np.min(truth), np.min(estimate)))
    high = float(max(np.max(truth), np.max(estimate)))
    fig = go.Figure()
    fig.add_scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect reconstruction",
                    line={"color": "#C3D3E0", "dash": "dash", "width": 1.5}, hoverinfo="skip")
    fig.add_scatter(x=truth, y=estimate, mode="markers", name="Hidden readings",
                    marker={"color": ACCENT, "size": 6, "opacity": 0.55},
                    hovertemplate="True %{x:.2f}<br>Reconstructed %{y:.2f}<extra></extra>")
    return _layout(fig, title, height=380, x_title=f"True {sensor}", y_title=f"Reconstructed {sensor}")


def error_histogram(errors: dict[str, np.ndarray], title="") -> go.Figure:
    fig = go.Figure()
    for method, values in errors.items():
        fig.add_histogram(x=values, name=method, opacity=0.7, nbinsx=60,
                          marker_color=METHOD_STYLE[method][0])
    fig = _layout(fig, title, height=340, x_title="Reconstruction error (standard deviations)",
                  y_title="Readings")
    fig.update_layout(barmode="overlay")
    return fig


def trace_chart(values, y_title, title="") -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=list(range(len(values))), y=values, mode="lines+markers",
            line={"color": NAVY, "width": 2.2}, marker={"size": 7, "color": NAVY},
            hovertemplate="Step %{x}<br>" + y_title + " %{y:.4f}<extra></extra>",
        )
    )
    fig = _layout(fig, title, height=300, x_title="Update step", y_title=y_title)
    fig.update_layout(showlegend=False)
    return fig


def ablation_chart(table: pd.DataFrame, title="") -> go.Figure:
    fig = go.Figure()
    styles = {"Random gaps": (NAVY, "solid", "circle"), "Sensor outages": (ACCENT, "dash", "square")}
    for pattern, rows in table.groupby("pattern"):
        colour, dash, marker = styles.get(pattern, (GREY, "solid", "circle"))
        fig.add_scatter(x=rows["steps"], y=rows["RMSE"], name=pattern, mode="lines+markers",
                        line={"color": colour, "dash": dash, "width": 2.2},
                        marker={"symbol": marker, "size": 8, "color": colour},
                        hovertemplate="%{x} steps<br>RMSE %{y:.3f}<extra></extra>")
    return _layout(fig, title, height=320, x_title="Update steps after the first recall", y_title="RMSE")
