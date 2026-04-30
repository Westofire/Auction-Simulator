"""
app.py
------
Streamlit dashboard for the AI Auction Simulator.
Run with: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from simulation import run_simulation

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Auction Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME / GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── root palette ── */
:root {
    --bg:         #0a0a0f;
    --surface:    #12121a;
    --surface2:   #1a1a26;
    --border:     #2a2a3d;
    --accent:     #7b61ff;
    --accent2:    #00e5c0;
    --accent3:    #ff6b6b;
    --accent4:    #ffd166;
    --text:       #e8e8f0;
    --muted:      #6b6b8a;
    --font-head:  'Syne', sans-serif;
    --font-mono:  'IBM Plex Mono', monospace;
}

/* ── base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-head) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── metric cards ── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    font-family: var(--font-mono);
    margin-bottom: 8px;
}
.metric-value {
    font-size: 32px;
    font-weight: 800;
    color: var(--text);
    font-family: var(--font-head);
    line-height: 1;
}
.metric-delta {
    font-size: 12px;
    font-family: var(--font-mono);
    margin-top: 6px;
}
.delta-pos { color: var(--accent2); }
.delta-neg { color: var(--accent3); }

/* ── section headers ── */
.section-header {
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    font-family: var(--font-mono);
    margin: 28px 0 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #12121a 0%, #1a1130 60%, #0a1520 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '⚡';
    position: absolute;
    right: 40px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 72px;
    opacity: 0.08;
}
.hero h1 {
    font-size: 28px;
    font-weight: 800;
    margin: 0 0 6px;
    background: linear-gradient(90deg, #e8e8f0, var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 13px;
    margin: 0;
}

/* ── sidebar controls ── */
.sidebar-section {
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted) !important;
    font-family: var(--font-mono) !important;
    margin: 18px 0 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* ── run button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b41df) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.05em !important;
    padding: 14px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    box-shadow: 0 4px 20px rgba(123,97,255,0.3) !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--font-head) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── plotly containers ── */
.chart-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 4px;
    overflow: hidden;
}

/* hide streamlit chrome */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "ai_bidder":     "#7b61ff",
    "truthful":      "#00e5c0",
    "shaded":        "#ffd166",
    "random":        "#ff6b6b",
    "bandit_bidder": "#ff9f43",   # warm orange — distinct from all existing colours
}
BG      = "#12121a"
GRID    = "#2a2a3d"
TEXT    = "#e8e8f0"
MUTED   = "#6b6b8a"

def base_layout(title="", height=340):
    return dict(
        title=dict(text=title, font=dict(family="Syne", size=14, color=TEXT), x=0.02, xanchor="left"),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="IBM Plex Mono", color=MUTED, size=11),
        height=height,
        margin=dict(l=12, r=12, t=44, b=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=GRID,
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(size=10)),
    )

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="padding: 4px 0 16px;">'
                '<span style="font-size:22px; font-weight:800; font-family:Syne,sans-serif;">⚡ Auction Sim</span>'
                '</div>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section">Auction Settings</p>', unsafe_allow_html=True)
    auction_type = st.selectbox(
        "Auction Type",
        options=["second", "first"],
        format_func=lambda x: "Second-Price (Vickrey)" if x == "second" else "First-Price Sealed-Bid",
    )
    n_rounds = st.slider("Simulation Rounds", min_value=50, max_value=20000, value=200, step=50)

    st.markdown('<p class="sidebar-section">Strategy Settings</p>', unsafe_allow_html=True)
    shade_factor = st.slider("Bid Shade Factor", min_value=0.5, max_value=1.0, value=0.8, step=0.05,
                             help="Fraction of valuation used by the shading strategy.")

    st.markdown('<p class="sidebar-section">AI Bidder (Q-Learning)</p>', unsafe_allow_html=True)
    ai_alpha         = st.slider("Learning Rate (α)",   0.01, 0.5,  0.10, 0.01)
    ai_gamma         = st.slider("Discount Factor (γ)", 0.5,  1.0,  0.95, 0.05)
    ai_epsilon_decay = st.slider("Epsilon Decay",       0.98, 1.0,  0.995, 0.001, format="%.3f")
    ai_n_levels      = st.slider("Bid Levels",          5,    21,   11,   2)

    st.markdown('<p class="sidebar-section">Bandit AI (UCB / Thompson)</p>', unsafe_allow_html=True)
    include_bandit   = st.checkbox("Include Bandit AI", value=False)
    bandit_algorithm = st.selectbox(
        "Bandit Algorithm",
        options=["ucb", "thompson"],
        format_func=lambda x: "UCB (Upper Confidence Bound)" if x == "ucb" else "Thompson Sampling",
        disabled=not include_bandit,
    )
    bandit_ucb_c = st.slider(
        "UCB Exploration (c)", 0.5, 5.0, 2.0, 0.5,
        help="Higher = more exploration. Only used when UCB is selected.",
        disabled=(not include_bandit or bandit_algorithm != "ucb"),
    )

    st.markdown('<p class="sidebar-section">Reproducibility</p>', unsafe_allow_html=True)
    use_seed = st.checkbox("Fix Random Seed", value=False)
    seed_val = st.number_input("Seed", value=42, step=1, disabled=not use_seed)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Simulation")

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>AI Auction Simulator</h1>
  <p>Q-Learning &amp; Bandit AI bidders vs fixed strategies · First &amp; Second-price auctions · Real-time analytics</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RUN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None

if run_btn:
    with st.spinner("Running simulation…"):
        df = run_simulation(
            n_rounds=n_rounds,
            auction_type=auction_type,
            shade_factor=shade_factor,
            ai_alpha=ai_alpha,
            ai_gamma=ai_gamma,
            ai_epsilon_decay=ai_epsilon_decay,
            ai_n_levels=ai_n_levels,
            include_bandit=include_bandit,
            bandit_algorithm=bandit_algorithm,
            bandit_ucb_c=bandit_ucb_c,
            seed=int(seed_val) if use_seed else None,
        )
    st.session_state.df = df

df = st.session_state.df

if df is None:
    st.markdown(
        '<div style="text-align:center; padding: 80px 0; color: #6b6b8a; font-family: IBM Plex Mono; font-size:14px;">'
        '← Configure parameters and click <strong>Run Simulation</strong> to begin.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────
total_rounds   = len(df)
avg_revenue    = df["price_paid"].mean()
avg_efficiency = df["efficiency"].mean()
ai_win_rate    = (df["winner_id"] == "ai_bidder").mean()
ai_total_reward= df["ai_reward"].sum()
avg_ai_ratio   = df["ai_bid_ratio"].mean()

win_counts = df["winner_id"].value_counts()

# Bidder order — bandit appended only when it was active this run
bidder_order = ["ai_bidder", "truthful", "shaded", "random"]
if include_bandit and "bandit_reward" in df.columns:
    bidder_order.append("bandit_bidder")

st.markdown('<div class="section-header">Summary Metrics</div>', unsafe_allow_html=True)

if include_bandit and "bandit_reward" in df.columns:
    bandit_win_rate     = (df["winner_id"] == "bandit_bidder").mean()
    bandit_total_reward = df["bandit_reward"].sum()
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    cards = [
        (c1, "Total Rounds",        f"{total_rounds:,}",          None),
        (c2, "Avg Revenue",         f"{avg_revenue:.2f}",         None),
        (c3, "Avg Efficiency",      f"{avg_efficiency:.1%}",      "delta-pos" if avg_efficiency > 0.85 else "delta-neg"),
        (c4, "AI Win Rate",         f"{ai_win_rate:.1%}",         "delta-pos" if ai_win_rate > 0.20 else "delta-neg"),
        (c5, "AI Total Profit",     f"{ai_total_reward:.1f}",     "delta-pos" if ai_total_reward > 0 else "delta-neg"),
        (c6, "AI Avg Bid Ratio",    f"{avg_ai_ratio:.2f}×",       None),
        (c7, "Bandit Win Rate",     f"{bandit_win_rate:.1%}",     "delta-pos" if bandit_win_rate > 0.20 else "delta-neg"),
        (c8, "Bandit Total Profit", f"{bandit_total_reward:.1f}", "delta-pos" if bandit_total_reward > 0 else "delta-neg"),
    ]
else:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cards = [
        (c1, "Total Rounds",      f"{total_rounds:,}",          None),
        (c2, "Avg Revenue",       f"{avg_revenue:.2f}",         None),
        (c3, "Avg Efficiency",    f"{avg_efficiency:.1%}",      "delta-pos" if avg_efficiency > 0.85 else "delta-neg"),
        (c4, "AI Win Rate",       f"{ai_win_rate:.1%}",         "delta-pos" if ai_win_rate > 0.25 else "delta-neg"),
        (c5, "AI Total Profit",   f"{ai_total_reward:.1f}",     "delta-pos" if ai_total_reward > 0 else "delta-neg"),
        (c6, "AI Avg Bid Ratio",  f"{avg_ai_ratio:.2f}×",       None),
    ]
for col, label, val, cls in cards:
    hint = ""
    if cls:
        arrow = "↑" if cls == "delta-pos" else "↓"
        hint = f'<div class="metric-delta {cls}">{arrow} vs 25% baseline</div>'
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val}</div>'
            f'{hint}'
            f'</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 1 — Strategy Profit Comparison (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">① Strategy Profit Comparison</div>', unsafe_allow_html=True)

profit = {}
for bidder in bidder_order:
    won_mask = df["winner_id"] == bidder
    val_col  = f"{bidder}_value"
    profit[bidder] = (df.loc[won_mask, "price_paid"].values,
                      df.loc[won_mask, val_col].values)

profit_totals  = {}
profit_means   = {}
profit_wins    = {}
for bidder in bidder_order:
    prices, vals = profit[bidder]
    rewards = vals - prices
    profit_totals[bidder] = rewards.sum()
    profit_means[bidder]  = rewards.mean() if len(rewards) > 0 else 0
    profit_wins[bidder]   = len(rewards)

labels      = ["AI Bidder", "Truthful", "Shaded", "Random"]
colors_list = ["#7b61ff", "#00e5c0", "#ffd166", "#ff6b6b"]
if include_bandit and "bandit_bidder" in bidder_order:
    labels.append("Bandit AI")
    colors_list.append("#ff9f43")

fig = go.Figure()
fig.add_trace(go.Bar(
    name="Total Profit",
    x=labels,
    y=[profit_totals[b] for b in bidder_order],
    marker_color=colors_list,
    marker_line=dict(color=BG, width=2),
    text=[f"{profit_totals[b]:.1f}" for b in bidder_order],
    textposition="outside",
    textfont=dict(family="IBM Plex Mono", size=11, color=TEXT),
))
fig.add_trace(go.Bar(
    name="Avg Profit per Win",
    x=labels,
    y=[profit_means[b] for b in bidder_order],
    marker_color=[c.replace("#", "rgba(").rstrip(")") +
                  ",0.45)" if False else
                  "rgba({},{},{},0.45)".format(int(c[1:3],16),int(c[3:5],16),int(c[5:7],16))
                  for c in colors_list],
    marker_line=dict(color=BG, width=2),
    text=[f"{profit_means[b]:.1f}" for b in bidder_order],
    textposition="outside",
    textfont=dict(family="IBM Plex Mono", size=11, color=MUTED),
))
fig.update_layout(
    **base_layout("Total Profit vs Avg Profit per Win — All Strategies", height=380),
    barmode="group",
    bargap=0.25,
    bargroupgap=0.08,
)
st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 2 — Learning Curves (Q-learner + Bandit when active)
# ─────────────────────────────────────────────────────────────────────────────
learning_header = "② AI Learning Curve — Q-Learner vs Bandit" if (include_bandit and "bandit_reward" in df.columns) else "② AI Learning Curve"
st.markdown(f'<div class="section-header">{learning_header}</div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    window = max(1, n_rounds // 20)
    df["ai_reward_ma"]         = df["ai_reward"].rolling(window).mean()
    df["ai_cumulative_reward"] = df["ai_reward"].cumsum()

    fig = go.Figure()
    # Q-learner raw + rolling reward
    fig.add_trace(go.Scatter(
        x=df["round"], y=df["ai_reward"],
        mode="lines", name="Q-Learner (raw)",
        line=dict(color="#7b61ff", width=1),
        opacity=0.25,
    ))
    fig.add_trace(go.Scatter(
        x=df["round"], y=df["ai_reward_ma"],
        mode="lines", name=f"Q-Learner rolling ({window}r)",
        line=dict(color="#7b61ff", width=2.5),
    ))
    # Bandit raw + rolling reward — shown only when active
    if include_bandit and "bandit_reward" in df.columns:
        df["bandit_reward_ma"] = df["bandit_reward"].rolling(window).mean()
        fig.add_trace(go.Scatter(
            x=df["round"], y=df["bandit_reward"],
            mode="lines", name="Bandit (raw)",
            line=dict(color="#ff9f43", width=1),
            opacity=0.25,
        ))
        fig.add_trace(go.Scatter(
            x=df["round"], y=df["bandit_reward_ma"],
            mode="lines", name=f"Bandit rolling ({window}r)",
            line=dict(color="#ff9f43", width=2.5),
        ))
    fig.update_layout(**base_layout("Per-Round Reward: Q-Learner vs Bandit", height=360))
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    # Cumulative profit comparison
    df["ai_cumulative_reward"] = df["ai_reward"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["round"], y=df["ai_cumulative_reward"],
        mode="lines", name="Q-Learner cumulative",
        line=dict(color="#7b61ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(123,97,255,0.07)",
    ))
    if include_bandit and "bandit_reward" in df.columns:
        df["bandit_cumulative_reward"] = df["bandit_reward"].cumsum()
        fig.add_trace(go.Scatter(
            x=df["round"], y=df["bandit_cumulative_reward"],
            mode="lines", name="Bandit cumulative",
            line=dict(color="#ff9f43", width=2.5),
            fill="tozeroy", fillcolor="rgba(255,159,67,0.07)",
        ))
    algo_label = bandit_algorithm.upper() if include_bandit else ""
    title = f"Cumulative Profit: Q-Learner vs {algo_label} Bandit" if include_bandit else "Q-Learner Cumulative Profit"
    fig.update_layout(**base_layout(title, height=360))
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 3 — Revenue Distribution
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">③ Revenue Distribution</div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    nbins = min(60, max(20, n_rounds // 30))
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["price_paid"],
        nbinsx=nbins,
        name="Revenue",
        marker_color="#ffd166",
        marker_line=dict(color=BG, width=0.5),
        opacity=0.85,
    ))
    fig.add_vline(
        x=df["price_paid"].mean(),
        line_dash="dot", line_color="#00e5c0", line_width=2,
        annotation_text=f"Mean {df['price_paid'].mean():.1f}",
        annotation_font_color="#00e5c0",
        annotation_font_size=11,
    )
    fig.update_layout(**base_layout("Distribution of Auction Revenue (Price Paid)", height=340))
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    window3 = max(1, n_rounds // 20)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["round"], y=df["price_paid"],
        mode="lines", name="Revenue",
        line=dict(color="#ffd166", width=1),
        opacity=0.3,
    ))
    fig.add_trace(go.Scatter(
        x=df["round"],
        y=df["price_paid"].rolling(window3).mean(),
        mode="lines", name=f"Rolling Avg ({window3}r)",
        line=dict(color="#ffd166", width=2.5),
    ))
    fig.update_layout(**base_layout("Auction Revenue per Round", height=340))
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 4 — Efficiency Histogram
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">④ Allocative Efficiency</div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["efficiency"],
        nbinsx=nbins,
        name="Efficiency",
        marker_color="#00e5c0",
        marker_line=dict(color=BG, width=0.5),
        opacity=0.85,
        histnorm="percent",
    ))
    fig.add_vline(
        x=df["efficiency"].mean(),
        line_dash="dot", line_color="#7b61ff", line_width=2,
        annotation_text=f"Mean {df['efficiency'].mean():.1%}",
        annotation_font_color="#7b61ff",
        annotation_font_size=11,
    )
    fig.add_vline(
        x=1.0,
        line_dash="dash", line_color=MUTED, line_width=1,
        annotation_text="Optimal",
        annotation_font_color=MUTED,
        annotation_font_size=10,
    )
    fig.update_layout(**base_layout("Efficiency Distribution (% of rounds)", height=340))
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    # Efficiency breakdown by winner — dynamically includes bandit when active
    eff_by_winner = df.groupby("winner_id")["efficiency"].mean().reindex(bidder_order).fillna(0)
    fig = go.Figure(go.Bar(
        x=labels,
        y=eff_by_winner.values,
        marker_color=colors_list,
        marker_line=dict(color=BG, width=2),
        text=[f"{v:.1%}" for v in eff_by_winner.values],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color=TEXT),
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color=MUTED, opacity=0.5,
                  annotation_text="Optimal", annotation_font_color=MUTED)
    fig.update_layout(**base_layout("Avg Efficiency When Each Strategy Wins", height=340))
    fig.update_yaxes(tickformat=".0%")
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RAW DATA TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Round-by-Round Data</div>', unsafe_allow_html=True)

with st.expander("Show full simulation data", expanded=False):
    display_cols = [
        "round", "auction_type", "winner_id", "price_paid",
        "winner_value", "highest_value", "efficiency",
        "ai_bid", "ai_valuation", "ai_bid_ratio", "ai_reward", "ai_epsilon",
    ]
    fmt = {
        "price_paid":    "{:.2f}",
        "winner_value":  "{:.2f}",
        "highest_value": "{:.2f}",
        "efficiency":    "{:.1%}",
        "ai_bid":        "{:.2f}",
        "ai_valuation":  "{:.2f}",
        "ai_bid_ratio":  "{:.2f}",
        "ai_reward":     "{:.2f}",
        "ai_epsilon":    "{:.3f}",
    }
    # Append bandit columns if present
    if include_bandit and "bandit_reward" in df.columns:
        display_cols += ["bandit_bid", "bandit_valuation", "bandit_bid_ratio", "bandit_reward"]
        fmt.update({
            "bandit_bid":       "{:.2f}",
            "bandit_valuation": "{:.2f}",
            "bandit_bid_ratio": "{:.2f}",
            "bandit_reward":    "{:.2f}",
        })
    # Only keep cols that actually exist in df (guard against stale session state)
    display_cols = [c for c in display_cols if c in df.columns]

    display_df = df[display_cols]
    total_cells = display_df.shape[0] * display_df.shape[1]

    # Pandas Styler has a cell-count limit — raise it dynamically or skip styling
    # for large runs to avoid StreamlitAPIException
    try:
        pd.set_option("styler.render.max_elements", max(total_cells, 262144))
        styled = display_df.style.format(fmt).background_gradient(
            subset=["ai_reward"], cmap="Purples"
        )
        st.dataframe(styled, use_container_width=True, height=460)
    except Exception:
        # Fallback: plain dataframe with no styling (always works at any size)
        st.dataframe(display_df, use_container_width=True, height=460)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇  Download Full CSV",
        data=csv,
        file_name="auction_simulation_results.csv",
        mime="text/csv",
    )