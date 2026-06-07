"""
Fraud Intelligence Dashboard (Task 7 + Bonus)
- CSV upload & batch prediction
- Fraud probability & high-risk transactions
- Attention visualization
- Real-time fraud detection simulation
"""

import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Aegis · Fraud Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ═══════════════════════════════
   FONTS
═══════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

/* ═══════════════════════════════
   TOKENS
═══════════════════════════════ */
:root {
    --bg-void:       #03050a;
    --bg-base:       #070c14;
    --bg-surface:    #0c1422;
    --bg-raised:     #111c2e;
    --bg-overlay:    #162035;

    --border-dim:    #1a2638;
    --border-mid:    #243248;
    --border-lit:    #2e4060;

    --accent-cyan:   #00d4ff;
    --accent-blue:   #0085ff;
    --accent-green:  #00e5a0;
    --accent-red:    #ff3d5a;
    --accent-amber:  #ffaa00;
    --accent-purple: #a855f7;

    --text-primary:  #e8f0fe;
    --text-secondary:#7a93b8;
    --text-muted:    #3d5270;
    --text-accent:   #00d4ff;

    --glow-cyan:     0 0 20px rgba(0, 212, 255, 0.15);
    --glow-red:      0 0 20px rgba(255, 61, 90, 0.2);

    --radius-sm:     6px;
    --radius-md:     10px;
    --radius-lg:     16px;

    --font-display:  'Syne', sans-serif;
    --font-mono:     'JetBrains Mono', monospace;
    --font-body:     'Inter', sans-serif;
}

/* ═══════════════════════════════
   BASE RESET
═══════════════════════════════ */
html, body, .stApp {
    background-color: var(--bg-void) !important;
    font-family: var(--font-body);
}

.main {
    background: var(--bg-void);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,133,255,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 40% 30% at 90% 80%, rgba(0,212,255,0.03) 0%, transparent 60%);
    min-height: 100vh;
}

.block-container {
    padding: 2rem 2.5rem;
    max-width: 1600px;
}

/* ═══════════════════════════════
   TOPBAR / HEADER
═══════════════════════════════ */
.aegis-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.4rem 2rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-mid);
    border-radius: var(--radius-lg);
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}

.aegis-topbar::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    opacity: 0.5;
}

.aegis-topbar::after {
    content: '';
    position: absolute;
    top: 0; left: -200px;
    width: 400px; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.03), transparent);
    animation: sweep 6s ease-in-out infinite;
}

@keyframes sweep {
    0%   { left: -400px; }
    100% { left: calc(100% + 400px); }
}

.aegis-wordmark {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.aegis-hex {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--bg-void);
    position: relative;
    flex-shrink: 0;
}

.aegis-name {
    font-family: var(--font-display);
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: var(--text-primary);
    text-transform: uppercase;
}

.aegis-name span {
    color: var(--accent-cyan);
}

.aegis-subtitle {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 2px;
}

.aegis-status-bar {
    display: flex;
    align-items: center;
    gap: 1.8rem;
}

.status-chip {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--text-secondary);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 8px var(--accent-green);
    animation: pulse-dot 2s ease-in-out infinite;
}

.status-dot.alert {
    background: var(--accent-red);
    box-shadow: 0 0 8px var(--accent-red);
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

.sys-tag {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    border: 1px solid var(--border-dim);
    padding: 3px 8px;
    border-radius: 4px;
    letter-spacing: 0.08em;
}

/* ═══════════════════════════════
   METRICS PANEL
═══════════════════════════════ */
[data-testid="metric-container"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.2rem 1.4rem !important;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="metric-container"]:hover {
    border-color: var(--border-lit) !important;
    box-shadow: var(--glow-cyan);
}

[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent-cyan), var(--accent-blue));
    border-radius: 0;
}

[data-testid="metric-container"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    line-height: 1.1 !important;
}

[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
}

/* ═══════════════════════════════
   SIDEBAR
═══════════════════════════════ */
section[data-testid="stSidebar"] {
    background-color: var(--bg-base) !important;
    border-right: 1px solid var(--border-mid) !important;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] * {
    color: var(--text-secondary) !important;
}

section[data-testid="stSidebar"] [data-testid="stSelectbox"],
section[data-testid="stSidebar"] [data-testid="stSlider"] {
    margin-bottom: 1rem;
}

/* Sidebar divider */
.sidebar-section {
    border-top: 1px solid var(--border-dim);
    margin: 1rem 0;
    padding-top: 1rem;
}

.sidebar-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}

/* ═══════════════════════════════
   TABS
═══════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border-mid);
    gap: 0;
    padding: 0;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.8rem 1.4rem !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    position: relative;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
    background: rgba(0, 212, 255, 0.03) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--accent-cyan) !important;
    background: transparent !important;
}

.stTabs [aria-selected="true"]::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0; right: 0;
    height: 2px;
    background: var(--accent-cyan);
    box-shadow: 0 0 8px var(--accent-cyan);
}

/* Tab content area */
.stTabs [data-baseweb="tab-panel"] {
    padding: 1.8rem 0 !important;
}

/* ═══════════════════════════════
   DATAFRAMES / TABLES
═══════════════════════════════ */
.stDataFrame {
    border-radius: var(--radius-md) !important;
    overflow: hidden;
    border: 1px solid var(--border-mid) !important;
}

.stDataFrame iframe {
    border-radius: var(--radius-md);
}

/* ═══════════════════════════════
   BUTTONS
═══════════════════════════════ */
.stButton > button {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    color: var(--accent-cyan) !important;
    border: 1px solid var(--border-lit) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0,212,255,0.06), rgba(0,133,255,0.06));
    opacity: 0;
    transition: opacity 0.2s ease;
}

.stButton > button:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 16px rgba(0,212,255,0.2) !important;
    transform: none !important;
}

.stButton > button:hover::before {
    opacity: 1;
}

/* Primary buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    color: var(--bg-void) !important;
    border: none !important;
    font-weight: 700 !important;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 20px rgba(0,212,255,0.35) !important;
}

/* ═══════════════════════════════
   FORM INPUTS
═══════════════════════════════ */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}

.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within,
.stTextInput > div > div:focus-within {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.1) !important;
}

/* ═══════════════════════════════
   SLIDER
═══════════════════════════════ */
.stSlider [data-baseweb="slider"] {
    padding: 0 !important;
}

.stSlider [data-baseweb="slider"] > div > div {
    background: var(--border-mid) !important;
}

.stSlider [data-baseweb="slider"] > div > div > div {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
}

.stSlider [data-baseweb="thumb"] {
    background: var(--accent-cyan) !important;
    border: 2px solid var(--bg-void) !important;
    box-shadow: 0 0 10px rgba(0,212,255,0.4) !important;
    width: 16px !important;
    height: 16px !important;
}

/* ═══════════════════════════════
   FILE UPLOADER
═══════════════════════════════ */
[data-testid="stFileUploader"] {
    background: var(--bg-surface) !important;
    border: 1px dashed var(--border-lit) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.5rem !important;
    text-align: center;
    transition: border-color 0.2s ease, background 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-cyan) !important;
    background: rgba(0,212,255,0.02) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ═══════════════════════════════
   ALERTS
═══════════════════════════════ */
.stAlert {
    border-radius: var(--radius-md) !important;
    border: none !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}

[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
}

/* Success */
.stSuccess {
    background: rgba(0,229,160,0.06) !important;
    border-left: 3px solid var(--accent-green) !important;
    color: var(--accent-green) !important;
}

/* Error */
.stError {
    background: rgba(255,61,90,0.06) !important;
    border-left: 3px solid var(--accent-red) !important;
    color: var(--accent-red) !important;
}

/* Warning */
.stWarning {
    background: rgba(255,170,0,0.06) !important;
    border-left: 3px solid var(--accent-amber) !important;
    color: var(--accent-amber) !important;
}

/* Info */
.stInfo {
    background: rgba(0,212,255,0.04) !important;
    border-left: 3px solid var(--accent-cyan) !important;
    color: var(--text-secondary) !important;
}

/* ═══════════════════════════════
   SECTION HEADERS
═══════════════════════════════ */
.stMarkdown h3 {
    font-family: var(--font-display) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-dim);
    margin-bottom: 1rem !important;
}

/* Subheader override */
[data-testid="stMarkdownContainer"] h2 {
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--text-primary) !important;
}

/* st.subheader */
div[data-testid="stHeading"] h2 {
    font-family: var(--font-display) !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}

/* Caption */
.stCaption, [data-testid="stCaptionContainer"] {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.08em !important;
}

/* ═══════════════════════════════
   RISK BADGES
═══════════════════════════════ */
.risk-high {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255,61,90,0.1);
    border: 1px solid rgba(255,61,90,0.3);
    color: var(--accent-red);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 4px;
}

.risk-medium {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255,170,0,0.1);
    border: 1px solid rgba(255,170,0,0.3);
    color: var(--accent-amber);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 4px;
}

.risk-low {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(0,229,160,0.08);
    border: 1px solid rgba(0,229,160,0.25);
    color: var(--accent-green);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 4px;
}

/* ═══════════════════════════════
   STAT CARDS (Custom)
═══════════════════════════════ */
.stat-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-mid);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 1px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
    opacity: 0.4;
}

.stat-card-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}

.stat-card-value {
    font-family: var(--font-display);
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1;
}

.stat-card-sub {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    margin-top: 0.35rem;
}

/* ═══════════════════════════════
   SECTION DIVIDER
═══════════════════════════════ */
.section-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.8rem 0 1.2rem;
}

.section-divider-line {
    flex: 1;
    height: 1px;
    background: var(--border-dim);
}

.section-divider-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    white-space: nowrap;
}

/* ═══════════════════════════════
   EXPANDER
═══════════════════════════════ */
.stExpander {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-md) !important;
}

.stExpander summary {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.08em !important;
}

/* ═══════════════════════════════
   TOGGLE
═══════════════════════════════ */
.stCheckbox > label,
.stToggle > label {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.06em !important;
}

/* ═══════════════════════════════
   DOWNLOAD BUTTON
═══════════════════════════════ */
.stDownloadButton > button {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    color: var(--accent-green) !important;
    border: 1px solid rgba(0,229,160,0.3) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
}

.stDownloadButton > button:hover {
    background: rgba(0,229,160,0.06) !important;
    border-color: var(--accent-green) !important;
    box-shadow: 0 0 12px rgba(0,229,160,0.15) !important;
}

/* ═══════════════════════════════
   SIDEBAR MODEL CARD
═══════════════════════════════ */
.model-card {
    background: var(--bg-raised);
    border: 1px solid var(--border-mid);
    border-radius: var(--radius-md);
    padding: 0.9rem 1rem;
    margin-top: 0.5rem;
}

.model-card-name {
    font-family: var(--font-display);
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.model-card-desc {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--text-muted);
    margin-top: 3px;
}

/* ═══════════════════════════════
   TABLE STYLING
═══════════════════════════════ */
table {
    border-radius: var(--radius-md) !important;
    overflow: hidden;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    width: 100%;
}

thead th {
    background: var(--bg-raised) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    padding: 0.75rem 1rem !important;
    border-bottom: 1px solid var(--border-mid) !important;
}

tbody td {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-secondary) !important;
    padding: 0.7rem 1rem !important;
    border-bottom: 1px solid var(--border-dim) !important;
    background: var(--bg-surface) !important;
}

tbody tr:hover td {
    background: var(--bg-overlay) !important;
    color: var(--text-primary) !important;
}

/* ═══════════════════════════════
   REALTIME SIMULATION LOG
═══════════════════════════════ */
.sim-entry {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 1.2rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-mid);
    border-radius: var(--radius-sm);
    margin-bottom: 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-secondary);
    transition: border-color 0.2s ease;
}

.sim-entry:hover {
    border-color: var(--border-lit);
}

.sim-entry.fraud {
    border-left: 3px solid var(--accent-red);
    background: rgba(255,61,90,0.03);
}

.sim-entry.clean {
    border-left: 3px solid var(--accent-green);
}

/* ═══════════════════════════════
   PLOT CONTAINERS
═══════════════════════════════ */
.stPyplot {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden;
    padding: 0.5rem;
}

/* ═══════════════════════════════
   SCROLLBAR
═══════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-lit); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

/* ═══════════════════════════════
   RESPONSIVE
═══════════════════════════════ */
@media (max-width: 768px) {
    .aegis-topbar { flex-direction: column; gap: 1rem; text-align: center; }
    .aegis-status-bar { justify-content: center; flex-wrap: wrap; }
    .block-container { padding: 1rem; }
}

</style>
""", unsafe_allow_html=True)


# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0c1422",
    "axes.facecolor":    "#0c1422",
    "axes.edgecolor":    "#1a2638",
    "axes.labelcolor":   "#7a93b8",
    "axes.titlecolor":   "#e8f0fe",
    "xtick.color":       "#3d5270",
    "ytick.color":       "#3d5270",
    "text.color":        "#7a93b8",
    "grid.color":        "#1a2638",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
})


@st.cache_resource
def load_artifacts():
    from src.inference import load_artifacts as _load
    return _load()


def render_header():
    import datetime
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    st.markdown(f"""
    <div class="aegis-topbar">
        <div class="aegis-wordmark">
            <div class="aegis-hex">⬡</div>
            <div>
                <div class="aegis-name">AEGIS <span>·</span> FRAUD INTELLIGENCE</div>
                <div class="aegis-subtitle">Sequential Transaction Analysis · LSTM + Attention Engine</div>
            </div>
        </div>
        <div class="aegis-status-bar">
            <div class="status-chip">
                <div class="status-dot"></div>
                System Nominal
            </div>
            <div class="sys-tag">LIVE</div>
            <div class="sys-tag">v2.1.0</div>
            <div class="sys-tag">{now}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_risk_badge(level: str) -> str:
    css = {"High Risk": "risk-high", "Medium Risk": "risk-medium", "Low Risk": "risk-low"}
    icons = {"High Risk": "▲", "Medium Risk": "◆", "Low Risk": "●"}
    icon = icons.get(level, "")
    return f'<span class="{css.get(level, "")}">{icon} {level}</span>'


def section_divider(label: str):
    st.markdown(f"""
    <div class="section-divider">
        <div class="section-divider-line"></div>
        <div class="section-divider-label">{label}</div>
        <div class="section-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)


try:
    artifacts = load_artifacts()
    models = artifacts["models"]
    extractor = artifacts["extractor"]
    scaler = artifacts["scaler"]
    models_loaded = len(models) > 0
except Exception as e:
    models_loaded = False
    load_error = str(e)


render_header()

if not models_loaded:
    st.warning(
        "**Models not initialised.** Run training pipeline first:\n\n"
        "```bash\npython -m src.train\n```"
    )
    if "load_error" in dir():
        st.error(f"Loader exception: {load_error}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-label">Model Selection</div>', unsafe_allow_html=True)

    model_labels = {
        "dense":            "Model A — Dense Network",
        "lstm":             "Model B — LSTM",
        "lstm_attention":   "Model C — LSTM + Attention",
        "lstm_pe_attention":"Model D — LSTM + PE + Attention",
    }
    selected_model = st.selectbox(
        "Active Model",
        list(models.keys()),
        format_func=lambda x: model_labels.get(x, x),
        label_visibility="collapsed",
    )

    model_descs = {
        "dense":            "Feedforward layers only. Fast, no sequence context.",
        "lstm":             "Recurrent memory across transaction sequence.",
        "lstm_attention":   "Weighted importance across sequence positions.",
        "lstm_pe_attention":"Full stack with positional encoding. Highest accuracy.",
    }
    st.markdown(f"""
    <div class="model-card">
        <div class="model-card-name">{model_labels.get(selected_model, selected_model)}</div>
        <div class="model-card-desc">{model_descs.get(selected_model, '')}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Detection Thresholds</div>', unsafe_allow_html=True)

    threshold = st.slider("Fraud Threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Probability above which a transaction is flagged as fraud.")
    high_risk_threshold = st.slider("High Risk Threshold", 0.5, 0.95, 0.70, 0.05,
                                    help="Probability above which a transaction is escalated to high risk.")

    st.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Intelligence Engine</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="status-chip" style="padding: 0.6rem 0; gap: 8px;">
        <div class="status-dot"></div>
        Models Loaded
    </div>
    """, unsafe_allow_html=True)

model = models[selected_model]

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_upload, tab_attention, tab_realtime, tab_info = st.tabs([
    "  CSV UPLOAD  ",
    "  ATTENTION PROBE  ",
    "  LIVE SIMULATION  ",
    "  SYSTEM CONTEXT  ",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: CSV Upload
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader("Batch Transaction Analysis")
    st.caption(
        "Required columns: Time · V1–V28 · Amount · "
        "Sequences of 4 prior transactions used to predict next-transaction fraud probability."
    )

    uploaded = st.file_uploader("Drop transaction CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        from src.inference import csv_to_sequences, predict_fraud, get_risk_level

        try:
            df = pd.read_csv(uploaded)

            section_divider("RAW DATA PREVIEW")
            st.dataframe(df.head(10), use_container_width=True)

            X, meta = csv_to_sequences(df, scaler)
            probs, preds = predict_fraud(model, X, threshold)
            risk_levels = [get_risk_level(p) for p in probs]

            results = meta.copy()
            results["Fraud Probability"] = probs.round(4)
            results["Prediction"] = np.where(preds == 1, "🔴 Fraud", "🟢 Legitimate")
            results["Risk Level"] = risk_levels

            section_divider("DETECTION SUMMARY")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Sequences Analysed", len(results))
            with c2:
                fraud_pct = f"{preds.sum() / len(preds) * 100:.1f}%"
                st.metric("Predicted Fraud", int(preds.sum()), delta=fraud_pct)
            with c3:
                hr_count = int((probs >= high_risk_threshold).sum())
                st.metric("High Risk Escalations", hr_count)
            with c4:
                st.metric("Avg Fraud Probability", f"{probs.mean():.4f}")

            section_divider("ALL PREDICTIONS")
            st.dataframe(results, use_container_width=True)

            high_risk = results[results["Fraud Probability"] >= high_risk_threshold]
            section_divider("HIGH RISK ESCALATIONS")

            if len(high_risk) > 0:
                st.error(f"**{len(high_risk)} transaction(s)** exceed the high-risk threshold of {high_risk_threshold:.0%}")
                st.dataframe(
                    high_risk.sort_values("Fraud Probability", ascending=False),
                    use_container_width=True,
                )
            else:
                st.success("No transactions exceeded the high-risk threshold.")

            csv_out = results.to_csv(index=False)
            st.download_button(
                "↓  Export Results as CSV",
                csv_out,
                "aegis_fraud_predictions.csv",
                "text/csv",
            )

            st.session_state["X_sequences"] = X
            st.session_state["results"] = results

        except Exception as exc:
            st.error(f"Processing error: {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Attention Investigation
# ═══════════════════════════════════════════════════════════════════════════════
with tab_attention:
    st.subheader("Attention Probe")
    st.caption(
        "Inspect which transaction within a sequence drove the fraud prediction. "
        "Powered by the dedicated attention extractor (LSTM + PE + Attention)."
    )

    if extractor is None:
        st.info("Attention extractor unavailable. Re-run the training pipeline.")
    elif "X_sequences" not in st.session_state:
        st.info("Upload a CSV in the **CSV Upload** tab first.")
    else:
        from src.inference import analyze_attention
        from src.attention_utils import visualize_attention_bar, visualize_attention_heatmap

        if selected_model not in ("lstm_attention", "lstm_pe_attention"):
            st.warning(
                "Active model is **"
                + model_labels.get(selected_model, selected_model)
                + "**. Attention extractor uses the dedicated LSTM + PE + Attention model — "
                "switch to Model C or D for consistent results."
            )

        X = st.session_state["X_sequences"]
        results = st.session_state["results"]

        seq_idx = st.selectbox(
            "Sequence",
            range(len(X)),
            format_func=lambda i: (
                f"SEQ-{i:04d}  ·  P={results.iloc[i]['Fraud Probability']:.4f}  "
                f"·  {results.iloc[i]['Prediction']}"
            ),
        )

        try:
            analysis = analyze_attention(extractor, X, seq_idx)

            section_divider("SEQUENCE METRICS")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Fraud Probability", f"{analysis['fraud_probability']:.4f}")
            with c2:
                st.metric("Top Influencer", f"TXN-{analysis['top_transaction']}")
            with c3:
                st.metric("Attention Score", f"{analysis['top_score']:.4f}")

            section_divider("INFLUENCE RANKING")
            rank_df = pd.DataFrame(analysis["ranked"])
            st.dataframe(rank_df, use_container_width=True)

            section_divider("VISUALISATIONS")
            col1, col2 = st.columns(2)
            with col1:
                fig_bar = visualize_attention_bar(
                    analysis["importance"],
                    analysis["txn_labels"],
                    title=f"Attention Weights · SEQ-{seq_idx:04d}",
                )
                st.pyplot(fig_bar, use_container_width=True)
                plt.close(fig_bar)

            with col2:
                fig_heat = visualize_attention_heatmap(
                    analysis["attention_matrix"],
                    analysis["txn_labels"],
                    title="Attention Matrix",
                )
                st.pyplot(fig_heat, use_container_width=True)
                plt.close(fig_heat)

            top = analysis["ranked"][0]
            st.success(
                f"**TXN-{top['transaction']}** carries the highest attention weight "
                f"({top['importance']:.4f}), making it the primary driver of the fraud "
                f"prediction for SEQ-{seq_idx:04d}."
            )

        except Exception as exc:
            st.error(f"Attention analysis failed: {exc}")
            with st.expander("Traceback"):
                import traceback
                st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Real-Time Simulation
# ═══════════════════════════════════════════════════════════════════════════════
with tab_realtime:
    st.subheader("Live Transaction Stream")
    st.caption(
        "Simulates incoming transactions one by one. "
        "After 4 transactions are collected, fraud is predicted on each subsequent transaction."
    )

    from src.inference import build_single_sequence_from_history, predict_fraud, get_risk_level

    if "sim_history" not in st.session_state:
        st.session_state.sim_history = []
    if "sim_log" not in st.session_state:
        st.session_state.sim_log = []

    section_divider("TRANSACTION INPUT")

    sim_cols = st.columns([2, 2, 3])
    with sim_cols[0]:
        sim_amount = st.number_input("Amount (USD)", min_value=0.0, value=100.0, key="sim_amt")
    with sim_cols[1]:
        sim_time = st.number_input("Timestamp (s)", min_value=0.0, value=0.0, key="sim_time")
    with sim_cols[2]:
        auto_sim = st.toggle("Auto-stream  (1 txn / sec)", value=False)

    btn_cols = st.columns([2, 1, 5])
    with btn_cols[0]:
        add_txn = st.button("➕  Inject Transaction", type="primary")
    with btn_cols[1]:
        reset = st.button("↺  Reset")

    if reset:
        st.session_state.sim_history = []
        st.session_state.sim_log = []
        st.rerun()

    if add_txn or auto_sim:
        txn = {"Time": sim_time, "Amount": sim_amount}
        for i in range(1, 29):
            txn[f"V{i}"] = float(np.random.randn())
        st.session_state.sim_history.append(txn)

        if len(st.session_state.sim_history) >= 4:
            try:
                X_single = build_single_sequence_from_history(
                    st.session_state.sim_history, scaler
                )
                prob, pred = predict_fraud(model, X_single, threshold)
                risk = get_risk_level(prob[0])

                entry = {
                    "txn_num": len(st.session_state.sim_history),
                    "amount": sim_amount,
                    "fraud_prob": round(float(prob[0]), 4),
                    "prediction": "Fraud" if pred[0] == 1 else "Legitimate",
                    "risk": risk,
                }
                st.session_state.sim_log.append(entry)

                if pred[0] == 1:
                    st.error(f"🚨  **FRAUD FLAGGED** — Probability: {prob[0]:.4f} · {risk}")
                else:
                    st.success(f"✅  **Legitimate** — Probability: {prob[0]:.4f} · {risk}")
            except Exception as exc:
                st.warning(str(exc))
        else:
            remaining = 4 - len(st.session_state.sim_history)
            st.info(
                f"Buffering sequence — {len(st.session_state.sim_history)}/4 transactions "
                f"collected. {remaining} more required."
            )

        if auto_sim:
            time.sleep(1)
            st.rerun()

    if st.session_state.sim_log:
        section_divider("DETECTION LOG")
        log_df = pd.DataFrame(st.session_state.sim_log)
        st.dataframe(log_df, use_container_width=True)

        fraud_count = sum(1 for e in st.session_state.sim_log if e["prediction"] == "Fraud")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Transactions Evaluated", len(st.session_state.sim_log))
        with col2:
            st.metric("Fraud Detections", fraud_count)
        with col3:
            rate = fraud_count / len(st.session_state.sim_log) * 100
            st.metric("Detection Rate", f"{rate:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Business Context
# ═══════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.subheader("System Context")

    section_divider("WHY FRAUD DETECTION IS HARD")

    st.markdown("""
    **1. Extreme Class Imbalance**
    Fraudulent transactions represent < 0.2% of total volume. A naïve model predicting
    "no fraud" for every transaction achieves >99.8% accuracy while catching zero fraud.

    **2. Evolving Fraud Patterns**
    Adversaries continuously adapt their tactics. Historical patterns degrade in relevance.
    Models must generalise beyond memorised attack signatures.

    **3. Sequential Context**
    A single transaction may appear entirely legitimate in isolation. The fraud signal often
    emerges from the *pattern* — rapid small authorisations before a large purchase, or
    unusual spend after inactivity.

    **4. Cost Asymmetry**
    False negatives (missed fraud) cause direct financial loss. False positives (blocking
    legitimate customers) destroy trust and revenue. Neither error is free.

    **5. Anonymised Feature Space**
    PCA-transformed features V1–V28 preserve privacy but eliminate direct interpretability,
    making causality analysis and model debugging substantially harder.
    """)

    section_divider("WHY ACCURACY IS MISLEADING")

    st.markdown("""
    | Scenario | Accuracy | Reality |
    |----------|----------|---------|
    | Predict all transactions as legitimate | **99.8%** | Zero fraud caught |
    | Flag 1% of transactions randomly | ~99% | Massive false positive rate |
    | Optimised recall model | High recall | Poor precision, alert fatigue |

    **Metrics that matter in fraud detection:**
    - **Precision** — Of all flagged transactions, what fraction are genuine fraud?
    - **Recall** — Of all real fraud, what fraction did the model catch?
    - **F1 Score** — Harmonic mean balancing precision and recall.
    - **PR-AUC** — Area under the precision-recall curve; threshold-independent.
    - **ROC-AUC** — Discriminative power across all classification thresholds.
    """)

    section_divider("WHY TRANSACTION ORDER MATTERS")

    st.markdown("""
    Fraud is a **sequence pattern**, not a point event. Examples:

    - **Card testing** — Three sub-£1 authorisations followed by a £2,000 purchase.
    - **Account takeover** — Dormant account suddenly transacts across multiple merchants.
    - **Velocity abuse** — Dozens of transactions within seconds across different terminals.

    **Positional Encoding** injects explicit sequence-position information into the model's
    attention mechanism, enabling it to distinguish *where* a transaction sits in the pattern —
    "Transaction 4 immediately after three small charges" is fundamentally different from
    "Transaction 4 in isolation."
    """)

    comparison_path = ROOT / "outputs" / "model_comparison.json"
    if comparison_path.exists():
        import json
        with open(comparison_path) as f:
            comparison = json.load(f)

        section_divider("MODEL BENCHMARK RESULTS")

        rows = []
        for name, m in comparison.items():
            rows.append({
                "Model": model_labels.get(name, name),
                "Accuracy": f"{m['accuracy']:.4f}",
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1": f"{m['f1']:.4f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)