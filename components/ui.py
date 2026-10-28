"""Shared visual styling and chart helpers for the Streamlit dashboard."""

from __future__ import annotations

import io
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

NAVY = "#08111f"
PANEL = "#101d30"
GOLD = "#d8b46a"
MUTED = "#9aa9ba"


def apply_theme() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background: radial-gradient(ellipse at 12% 0%, #14243a 0%, {NAVY} 46%); color:#edf2f7; }}
        [data-testid="stSidebar"] {{ background: #0b1626; border-right:1px solid rgba(216,180,106,.18); }}
        h1,h2,h3 {{ font-family: 'Segoe UI', 'Aptos', sans-serif; letter-spacing:-.025em; }}
        p,div,label,span {{ font-family:'Segoe UI', 'Aptos', sans-serif; }}
        .brand-kicker {{ color:{GOLD}; text-transform:uppercase; letter-spacing:.19em; font-size:.72rem; font-weight:700; }}
        .hero-title {{ font-size:clamp(2.2rem,5vw,4rem); font-weight:800; line-height:1.05; margin:.4rem 0; }}
        .hero-subtitle {{ color:#afbdce; font-size:1.12rem; }}
        .metric-card {{ background:linear-gradient(145deg,rgba(20,36,58,.96),rgba(12,24,40,.96)); border:1px solid rgba(216,180,106,.2); border-radius:16px; padding:18px 20px; min-height:110px; box-shadow:0 12px 30px rgba(0,0,0,.12); }}
        .metric-label {{ color:{MUTED}; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }}
        .metric-value {{ color:#f3f5f8; font-family:'Segoe UI', 'Aptos', sans-serif; font-size:1.5rem; font-weight:700; margin-top:9px; }}
        .gold-rule {{ width:58px; height:3px; background:{GOLD}; border-radius:3px; margin:18px 0 26px; }}
        .notice {{ color:#adbac9; font-size:.86rem; }}
        div[data-testid="stForm"] {{ box-sizing:border-box; width:100%; max-width:720px; }}
        div[data-testid="stForm"] input {{ max-width:100%; }}
        div.stButton > button {{ border-radius:10px; border:1px solid rgba(216,180,106,.55); font-weight:700; }}
        div.stButton > button[kind="primary"] {{ background:{GOLD}; color:#111827; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def format_percent(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}%}" if np.isfinite(value) else "—"


def price_chart(prices: pd.DataFrame, title: str, ylabel: str) -> plt.Figure:
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor="#101d30")
    ax.set_facecolor("#101d30")
    palette = sns.color_palette(["#d8b46a", "#63b3ed", "#a78bfa", "#53c6a5", "#f28b82"])
    for index, column in enumerate(prices.columns):
        ax.plot(prices.index, prices[column], label=column, linewidth=1.8, color=palette[index % len(palette)])
    ax.set_title(title, color="#edf2f7", loc="left", pad=14, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, color="#c1cbd7")
    ax.set_xlabel("Date", color="#c1cbd7")
    ax.tick_params(colors="#aebaca")
    ax.grid(color="#26364b", alpha=.7)
    ax.legend(frameon=False, labelcolor="#dce4ee", ncol=min(4, max(1, len(prices.columns))))
    for spine in ax.spines.values():
        spine.set_color("#34465c")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def correlation_heatmap(correlation: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(5, 1.05 * len(correlation.columns)), 4.6), facecolor="#101d30")
    ax.set_facecolor("#101d30")
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, center=0,
                linewidths=.7, linecolor="#17263b", square=True, cbar_kws={"shrink": .78}, ax=ax)
    ax.set_title("Daily return correlation", color="#edf2f7", loc="left", pad=15, fontsize=13, fontweight="bold")
    ax.tick_params(colors="#d6dfeb", labelrotation=25)
    fig.tight_layout()
    return fig


def risk_return_chart(simulations: pd.DataFrame, minimum: dict, maximum: dict | None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor="#101d30")
    ax.set_facecolor("#101d30")
    points = simulations.replace([np.inf, -np.inf], np.nan).dropna(subset=["Annualized Volatility", "Expected Annualized Return"])
    scatter = ax.scatter(points["Annualized Volatility"], points["Expected Annualized Return"],
                         c=points["Sharpe Ratio"].replace([np.inf, -np.inf], np.nan), cmap="viridis",
                         s=17, alpha=.55, linewidths=0, label="Random portfolios")
    fig.colorbar(scatter, ax=ax, label="Sharpe ratio")
    ax.scatter(minimum["Annualized Volatility"], minimum["Expected Annualized Return"], marker="*", s=240,
               color="#f2c66d", edgecolor="#fff2cf", linewidth=.8, label="Minimum volatility", zorder=5)
    if maximum is not None:
        ax.scatter(maximum["Annualized Volatility"], maximum["Expected Annualized Return"], marker="D", s=105,
                   color="#f28b82", edgecolor="#fff", linewidth=.7, label="Maximum Sharpe", zorder=6)
    ax.set_xlabel("Annualized volatility (decimal)", color="#c1cbd7")
    ax.set_ylabel("Expected annualized return (decimal)", color="#c1cbd7")
    ax.set_title("Portfolio risk and return", color="#edf2f7", loc="left", pad=14, fontsize=13, fontweight="bold")
    ax.tick_params(colors="#aebaca")
    ax.grid(color="#26364b", alpha=.7)
    ax.legend(frameon=False, labelcolor="#dce4ee")
    for spine in ax.spines.values():
        spine.set_color("#34465c")
    fig.tight_layout()
    return fig


def allocation_chart(weights: pd.Series, title: str) -> plt.Figure:
    ordered = weights.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, max(2.7, .42 * len(ordered))), facecolor="#101d30")
    ax.set_facecolor("#101d30")
    ax.barh(ordered.index, ordered.values, color="#d8b46a", edgecolor="#f4d99d", height=.58)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Portfolio weight", color="#c1cbd7")
    ax.set_title(title, color="#edf2f7", loc="left", pad=12, fontsize=12, fontweight="bold")
    ax.tick_params(colors="#d6dfeb")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(axis="x", color="#26364b", alpha=.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#34465c")
    fig.tight_layout()
    return fig


def csv_bytes(frame: pd.DataFrame, index: bool = True) -> bytes:
    stream = io.StringIO()
    frame.to_csv(stream, index=index, float_format="%.10g")
    return stream.getvalue().encode("utf-8-sig")
