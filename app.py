"""
IIT Guwahati — Road Health & Predictive Pavement Management System
Granica x IIT Guwahati Hackathon
Estates & Works Section Portal
"""

import os
import io
import time
import json
import base64
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image

from pipeline import (
    RoadVisionEngine,
    AccelerometerSignalEngine,
    PredictivePavementAnalytics,
    GranicaParquetExporter,
    IITG_CAMPUS_ROUTES,
    generate_granica_hackathon_dataset
)
from export_pdf import generate_work_order_pdf

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — NO SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="IIT Guwahati — Road Health Portal",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════════
# ASSET LOADER (Base64 for reliable rendering)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_assets():
    bg_b64 = ""
    logo_b64 = ""
    bg_path = os.path.join("assets", "campus_hero_bg.jpg")
    logo_path = os.path.join("assets", "iitg_logo.png")
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode("utf-8")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("utf-8")
    return bg_b64, logo_b64

bg_b64, logo_b64 = load_assets()

# ═══════════════════════════════════════════════════════════════════════════════
# THEMED CSS — EXACT REPLICATION OF CAMPUS ROAD HEALTH & PAVEMENT PORTAL
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
    /* ── Hide Streamlit Sidebar & Default Chrome ── */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {{ display: none !important; }}
    #MainMenu, header, footer {{ visibility: hidden; }}
    .block-container {{
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px !important;
    }}

    /* ── Modern Typography ── */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif;
        color: #e2e8f0;
        background-color: #0f172a;
    }}

    /* Base Theme Overrides for Streamlit Elements */
    .stApp {{
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }}
    
    p, span, div {{ color: #cbd5e1; }}
    h1, h2, h3, h4, h5, h6 {{ color: #f8fafc !important; font-family: 'Outfit', sans-serif; }}

    /* ── Top Navigation Bar (Glassmorphism) ── */
    .iitg-navbar {{
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 14px 32px !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        border-radius: 16px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    }}
    .navbar-brand {{ display: flex; align-items: center; gap: 16px; }}
    .navbar-logo {{
        display: flex; align-items: center; justify-content: center;
        width: 52px; height: 52px; flex-shrink: 0;
        background: rgba(255,255,255,0.05); border-radius: 12px; padding: 4px;
    }}
    .navbar-titles {{ display: flex; flex-direction: column; }}
    .navbar-title-hi {{ font-size: 14px !important; font-weight: 700 !important; color: #94a3b8 !important; line-height: 1.2 !important; -webkit-text-fill-color: #94a3b8 !important; }}
    .navbar-title-en {{ font-size: 16px !important; font-weight: 800 !important; color: #f1f5f9 !important; line-height: 1.2 !important; -webkit-text-fill-color: #f1f5f9 !important; letter-spacing: 0.5px; }}
    .navbar-divider {{ width: 1px; height: 38px; background: rgba(255,255,255,0.15); margin: 0 12px; }}
    .navbar-section {{ display: flex; flex-direction: column; }}
    .navbar-sec-title {{ font-size: 14px !important; font-weight: 700 !important; color: #38bdf8 !important; line-height: 1.2 !important; -webkit-text-fill-color: #38bdf8 !important; }}
    .navbar-sec-sub {{ font-size: 11px !important; font-weight: 500 !important; color: #94a3b8 !important; line-height: 1.2 !important; -webkit-text-fill-color: #94a3b8 !important; text-transform: uppercase; letter-spacing: 1px; }}
    
    .navbar-links {{ display: flex; align-items: center; gap: 24px; }}
    .nav-link-item {{
        display: inline-flex; align-items: center; gap: 8px;
        font-size: 14px; font-weight: 500; color: #94a3b8;
        text-decoration: none; padding: 8px 12px; border-radius: 8px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .nav-link-item:hover {{ color: #f8fafc; background: rgba(255,255,255,0.05); transform: translateY(-2px); }}
    .nav-link-item.active {{ color: #38bdf8; background: rgba(56, 189, 248, 0.1); font-weight: 600; box-shadow: inset 0 0 0 1px rgba(56,189,248,0.2); }}

    /* ── Hero Section ── */
    .hero-banner-wrapper {{
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 27, 75, 0.85) 50%, rgba(15, 23, 42, 0.4) 100%),
                    url("data:image/jpeg;base64,{bg_b64}") center/cover no-repeat;
        border-radius: 24px; padding: 64px 56px; margin-bottom: 32px;
        box-shadow: 0 24px 48px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        position: relative; overflow: hidden;
    }}
    .hero-banner-wrapper::before {{
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    }}
    .hero-flex {{ display: flex; justify-content: space-between; align-items: center; gap: 48px; flex-wrap: wrap; z-index: 1; position: relative; }}
    .hero-left-box {{ flex: 1 1 500px; max-width: 600px; }}
    .hero-gold-bar {{ width: 60px; height: 4px; background: linear-gradient(90deg, #38bdf8, #818cf8); border-radius: 2px; margin-bottom: 28px; box-shadow: 0 0 12px rgba(56, 189, 248, 0.5); }}
    .hero-main-title {{
        font-family: 'Outfit', sans-serif !important; font-size: 46px !important; font-weight: 800 !important;
        color: #ffffff !important; line-height: 1.15 !important; margin: 0 0 24px 0 !important;
        letter-spacing: -1px !important; -webkit-text-fill-color: #ffffff !important;
        text-shadow: 0 4px 24px rgba(0,0,0,0.5);
    }}
    .hero-main-desc {{
        font-size: 18px !important; color: #cbd5e1 !important; line-height: 1.6 !important; margin: 0 !important;
        font-weight: 300 !important; max-width: 540px !important; -webkit-text-fill-color: #cbd5e1 !important;
    }}
    .hero-right-box {{ flex: 0 0 460px; max-width: 480px; perspective: 1000px; }}

    /* ── Welcome Glass Card ── */
    .welcome-card-box {{
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(20px) !important; -webkit-backdrop-filter: blur(20px) !important;
        border-radius: 24px !important; padding: 40px 32px !important;
        box-shadow: 0 32px 64px -12px rgba(0, 0, 0, 0.5), inset 0 0 0 1px rgba(255, 255, 255, 0.1) !important;
        transform: translateZ(0); transition: transform 0.3s ease;
    }}
    .welcome-card-box:hover {{ transform: translateY(-5px); }}
    .welcome-header-title {{ font-size: 36px !important; font-weight: 800 !important; color: #f8fafc !important; margin: 0 0 8px 0 !important; line-height: 1.2 !important; letter-spacing: -1px !important; -webkit-text-fill-color: #f8fafc !important; }}
    .welcome-header-sub {{ font-size: 15px !important; color: #94a3b8 !important; font-weight: 400 !important; margin: 0 0 28px 0 !important; -webkit-text-fill-color: #94a3b8 !important; }}

    /* ── Interactive Role Cards ── */
    .role-action-link {{
        display: flex; align-items: center; gap: 18px; padding: 18px 20px;
        border-radius: 16px; margin-bottom: 16px; text-decoration: none !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); cursor: pointer;
        background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255,255,255,0.05);
        position: relative; overflow: hidden;
    }}
    .role-action-link::before {{
        content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
        transform: translateX(-100%); transition: transform 0.5s ease;
    }}
    .role-action-link:hover::before {{ transform: translateX(100%); }}
    .role-action-link:hover {{ transform: translateY(-4px) scale(1.02); box-shadow: 0 12px 24px rgba(0,0,0,0.3); border-color: rgba(255,255,255,0.15); }}

    .role-style-guest {{ background: linear-gradient(145deg, rgba(245,158,11,0.1), rgba(15,23,42,0.5)); border-color: rgba(245,158,11,0.2); }}
    .role-style-guest:hover {{ background: linear-gradient(145deg, rgba(245,158,11,0.15), rgba(15,23,42,0.8)); border-color: rgba(245,158,11,0.4); box-shadow: 0 12px 24px rgba(245,158,11,0.15); }}
    .role-style-admin {{ background: linear-gradient(145deg, rgba(56,189,248,0.1), rgba(15,23,42,0.5)); border-color: rgba(56,189,248,0.2); margin-bottom: 0; }}
    .role-style-admin:hover {{ background: linear-gradient(145deg, rgba(56,189,248,0.15), rgba(15,23,42,0.8)); border-color: rgba(56,189,248,0.4); box-shadow: 0 12px 24px rgba(56,189,248,0.15); }}

    .role-icon-container {{
        width: 56px; height: 56px; border-radius: 14px;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
        box-shadow: inset 0 2px 4px rgba(255,255,255,0.1);
    }}
    .role-icon-guest-bg {{ background: linear-gradient(135deg, #fbbf24, #d97706); color: #fff; }}
    .role-icon-admin-bg {{ background: linear-gradient(135deg, #38bdf8, #0369a1); color: #fff; }}

    .role-body-box {{ flex: 1; }}
    .role-title-text {{ font-size: 18px !important; font-weight: 700 !important; color: #f8fafc !important; margin-bottom: 4px !important; line-height: 1.2 !important; -webkit-text-fill-color: #f8fafc !important; }}
    .role-sub-text {{ font-size: 13px !important; color: #94a3b8 !important; line-height: 1.5 !important; font-weight: 400 !important; -webkit-text-fill-color: #94a3b8 !important; }}
    
    .role-arrow-icon {{
        font-size: 24px !important; font-weight: 300 !important; color: #64748b !important;
        -webkit-text-fill-color: #64748b !important; transition: all 0.3s ease; opacity: 0.5;
    }}
    .role-action-link:hover .role-arrow-icon {{
        transform: translateX(8px); color: #f8fafc !important; -webkit-text-fill-color: #f8fafc !important; opacity: 1;
    }}

    /* ── Role Pill Badges ── */
    .pill-guest {{ display: inline-block; background: linear-gradient(90deg, rgba(245,158,11,0.2), rgba(217,119,6,0.2)); border: 1px solid rgba(245,158,11,0.5); color: #fbbf24; font-weight: 700; font-size: 11px; padding: 4px 14px; border-radius: 20px; letter-spacing: 1px; box-shadow: 0 0 10px rgba(245,158,11,0.1); }}
    .pill-admin {{ display: inline-block; background: linear-gradient(90deg, rgba(56,189,248,0.2), rgba(3,105,161,0.2)); border: 1px solid rgba(56,189,248,0.5); color: #38bdf8; font-weight: 700; font-size: 11px; padding: 4px 14px; border-radius: 20px; letter-spacing: 1px; box-shadow: 0 0 10px rgba(56,189,248,0.1); }}

    /* ── Stat Cards (KPIs) ── */
    .stat-row {{ display: flex; gap: 20px; margin-bottom: 32px; flex-wrap: wrap; }}
    .stat-card {{
        flex: 1; min-width: 200px;
        background: rgba(30, 41, 59, 0.4); backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.05); border-radius: 16px; padding: 24px;
        transition: all 0.3s ease; position: relative; overflow: hidden;
    }}
    .stat-card::after {{
        content: ''; position: absolute; bottom: 0; left: 0; width: 100%; height: 3px;
        background: linear-gradient(90deg, #38bdf8, #818cf8); transform: scaleX(0); transform-origin: left; transition: transform 0.3s ease;
    }}
    .stat-card:hover {{ transform: translateY(-5px); background: rgba(30, 41, 59, 0.6); border-color: rgba(255,255,255,0.1); box-shadow: 0 16px 32px rgba(0,0,0,0.3); }}
    .stat-card:hover::after {{ transform: scaleX(1); }}
    .stat-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; font-weight: 600; margin-bottom: 8px; }}
    .stat-val {{ font-size: 32px; font-weight: 800; color: #f8fafc; line-height: 1.2; text-shadow: 0 2px 10px rgba(0,0,0,0.2); }}
    .stat-sub {{ font-size: 13px; color: #64748b; margin-top: 4px; }}

    /* ── Admin Queue Interactive Cards ── */
    .queue-card {{
        background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255,255,255,0.05); border-radius: 16px;
        padding: 20px; margin-bottom: 16px; cursor: pointer; transition: all 0.2s ease;
        display: flex; flex-direction: column; gap: 12px;
    }}
    .queue-card:hover {{
        background: rgba(30, 41, 59, 0.8); border-color: rgba(56,189,248,0.3);
        transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }}
    .queue-card-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 12px; }}
    .queue-card-title {{ font-size: 18px; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 8px; }}
    .queue-card-pdi {{ font-size: 20px; font-weight: 800; color: #ef4444; background: rgba(239, 68, 68, 0.1); padding: 4px 12px; border-radius: 8px; }}
    .queue-card-stats {{ display: flex; gap: 24px; color: #94a3b8; font-size: 14px; }}
    .queue-stat-item {{ display: flex; flex-direction: column; }}
    .queue-stat-val {{ font-size: 16px; font-weight: 600; color: #e2e8f0; }}

    /* ── Action Verdict Boxes ── */
    .verdict-full {{
        background: linear-gradient(135deg, rgba(153, 27, 27, 0.1), rgba(127, 29, 29, 0.2));
        border: 1px solid rgba(239, 68, 68, 0.3); border-left: 6px solid #ef4444;
        border-radius: 16px; padding: 24px; margin-bottom: 20px;
    }}
    .verdict-full-title {{ font-size: 18px; font-weight: 800; color: #fca5a5; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }}
    
    .verdict-spot {{
        background: linear-gradient(135deg, rgba(146, 64, 14, 0.1), rgba(120, 53, 15, 0.2));
        border: 1px solid rgba(245, 158, 11, 0.3); border-left: 6px solid #f59e0b;
        border-radius: 16px; padding: 24px; margin-bottom: 20px;
    }}
    .verdict-spot-title {{ font-size: 18px; font-weight: 800; color: #fcd34d; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }}

    /* ── Savings Highlight ── */
    .savings-box {{
        background: linear-gradient(135deg, rgba(6, 95, 70, 0.2), rgba(4, 120, 87, 0.1));
        border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 16px; padding: 24px; margin-top: 16px;
    }}
    .savings-big {{ font-size: 28px; font-weight: 800; color: #34d399; margin-bottom: 8px; text-shadow: 0 2px 10px rgba(52,211,153,0.2); }}
    .savings-detail {{ font-size: 14px; color: #a7f3d0; line-height: 1.6; }}

    /* ── Site Footer ── */
    .site-footer {{ margin-top: 48px; padding: 24px 0; border-top: 1px solid rgba(255,255,255,0.05); text-align: center; font-size: 13px; color: #64748b; font-weight: 300; letter-spacing: 0.5px; }}
    
    /* ── Fix Streamlit Input Styling ── */
    div[data-baseweb="input"], div[data-baseweb="select"] {{ background-color: rgba(15, 23, 42, 0.6) !important; border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.1) !important; }}
    input, textarea, select {{ color: #f8fafc !important; }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE INITIALIZATION & DATASET LOAD
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_vision_engine():
    return RoadVisionEngine()

@st.cache_resource
def get_signal_engine():
    return AccelerometerSignalEngine(sampling_rate_hz=100.0)

vision_engine = get_vision_engine()
signal_engine = get_signal_engine()

DATASET_PATH = "road_pothole_dataset.parquet"
if "pavement_df" not in st.session_state:
    if os.path.exists(DATASET_PATH):
        try:
            st.session_state.pavement_df = pd.read_parquet(DATASET_PATH)
        except Exception:
            df, _ = generate_granica_hackathon_dataset(120, DATASET_PATH)
            st.session_state.pavement_df = df
    else:
        df, _ = generate_granica_hackathon_dataset(120, DATASET_PATH)
        st.session_state.pavement_df = df

# ── Session State Defaults ──
for key, default in [
    ("portal_view", "gateway"),
    ("user_role", "guest"),
    ("is_admin_logged_in", False),
    ("admin_username", "estates_admin"),
    ("show_admin_login", False),
    ("active_top_tab", "home"),
    ("guest_submissions", []),
    ("uploaded_hazard_images", {}),
    ("cluster_statuses", {}),
    ("guest_tab", "report"),
    ("admin_tab", "queue"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ═══════════════════════════════════════════════════════════════════════════════
# QUERY PARAMETERS HANDLING FOR INSTANT URL NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
qp = st.query_params
if "nav" in qp:
    nav_val = qp["nav"]
    st.session_state.active_top_tab = nav_val
    if nav_val == "home":
        st.session_state.portal_view = "gateway"
    st.query_params.clear()
    st.rerun()

if "action" in qp:
    act = qp["action"]
    st.query_params.clear()
    if act == "guest":
        st.session_state.user_role = "guest"
        st.session_state.portal_view = "guest"
        st.session_state.active_top_tab = "home"
        st.rerun()
    elif act == "admin":
        if st.session_state.is_admin_logged_in:
            st.session_state.portal_view = "admin"
            st.session_state.active_top_tab = "home"
        else:
            st.session_state.show_admin_login = True
        st.rerun()
    elif act == "admin_cancel":
        st.session_state.show_admin_login = False
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TOP NAVBAR RENDERER (Clean, Official IIT Guwahati Brand Bar)
# ═══════════════════════════════════════════════════════════════════════════════
def render_navbar(active_tab="home"):
    h_act = "active" if active_tab == "home" else ""
    i_act = "active" if active_tab == "insights" else ""
    a_act = "active" if active_tab == "about" else ""
    hl_act = "active" if active_tab == "help" else ""

    st.markdown(f"""
    <div class="iitg-navbar">
        <div class="navbar-brand">
            <div class="navbar-logo">
                <img src="data:image/png;base64,{logo_b64}" width="46" height="46" style="object-fit: contain; display: block;" alt="IIT Guwahati Logo" />
            </div>
            <div class="navbar-titles">
                <div class="navbar-title-hi">भारतीय प्रौद्योगिकी संस्थान गुवाहाटी</div>
                <div class="navbar-title-en">Indian Institute of Technology Guwahati</div>
            </div>
            <div class="navbar-divider"></div>
            <div class="navbar-section">
                <div class="navbar-sec-title">Estates &amp; Works Section</div>
                <div class="navbar-sec-sub">Infrastructure · Maintenance · Sustainability</div>
            </div>
        </div>
        <div class="navbar-links">
            <a href="?nav=home" target="_self" class="nav-link-item {h_act}">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg>
                <span>Home</span>
            </a>
            <a href="?nav=insights" target="_self" class="nav-link-item {i_act}">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M5 9.2h3V19H5zM10.6 5h2.8v14h-2.8zm5.6 8H19v6h-2.8z"/></svg>
                <span>Insights</span>
            </a>
            <a href="?nav=about" target="_self" class="nav-link-item {a_act}">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>
                <span>About</span>
            </a>
            <a href="?nav=help" target="_self" class="nav-link-item {hl_act}">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 16h-2v-2h2v2zm1.07-7.75l-.9.92C12.45 11.9 12 12.5 12 14h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H7c0-2.76 2.24-5 5-5s5 2.24 5 5c0 1.04-.42 1.99-1.07 2.75z"/></svg>
                <span>Help</span>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# GATEWAY VIEW — THE STARTING LANDING PAGE (Exact Screenshot Replication)
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.portal_view == "gateway":

    # 1. Top Navbar
    render_navbar(active_tab=st.session_state.active_top_tab)

    # ── TAB: HOME (The Hero Screenshot Page) ──
    if st.session_state.active_top_tab == "home":

        if not st.session_state.show_admin_login:
            # Full layout with 3 interactive role choices
            st.markdown(f"""
            <div class="hero-banner-wrapper">
                <div class="hero-flex">
                    <div class="hero-left-box">
                        <div class="hero-gold-bar"></div>
                        <h1 class="hero-main-title">AI Road Health &amp;<br>Pavement Management</h1>
                        <p class="hero-main-desc">Proactive pothole monitoring, AI defect verification, and predictive full-segment maintenance planning for a safer, more resilient campus infrastructure.</p>
                    </div>
                    <div class="hero-right-box">
                        <div class="welcome-card-box">
                            <h2 class="welcome-header-title">Welcome</h2>
                            <p class="welcome-header-sub">Choose how you would like to continue</p>
                            <!-- Role Option 1: Guest -->
                            <a href="?action=guest" target="_self" class="role-action-link role-style-guest">
                                <div class="role-icon-container role-icon-guest-bg">
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                                        <circle cx="9" cy="7" r="4"></circle>
                                        <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                                        <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
                                    </svg>
                                </div>
                                <div class="role-body-box">
                                    <div class="role-title-text">Continue as Guest</div>
                                    <div class="role-sub-text">Report a road hazard or view nearby reported issues without logging in.</div>
                                </div>
                                <div class="role-arrow-icon">→</div>
                            </a>
                            <!-- Role Option 2: Admin -->
                            <a href="?action=admin" target="_self" class="role-action-link role-style-admin">
                                <div class="role-icon-container role-icon-admin-bg">
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                                    </svg>
                                </div>
                                <div class="role-body-box">
                                    <div class="role-title-text">Login as Admin</div>
                                    <div class="role-sub-text">Access analytics, inspect hazards, manage work orders, and more.</div>
                                </div>
                                <div class="role-arrow-icon">→</div>
                            </a>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Hero with Admin Login Form inside Welcome Card
            hero_col1, hero_col2 = st.columns([1.1, 0.9], gap="large")
            with hero_col1:
                st.markdown(f"""
                <div class="hero-banner-wrapper" style="padding: 44px 38px;">
                    <div class="hero-gold-bar"></div>
                    <h1 class="hero-main-title" style="font-size:34px;">Campus Road Health &amp;<br>Pavement Management Portal</h1>
                    <p class="hero-main-desc">Estates &amp; Works Section Authority Portal. Sign in with administrative credentials to access DBSCAN corridor hazard rankings, engineering work order dispatch, and live contractor lifecycle analytics.</p>
                </div>
                """, unsafe_allow_html=True)
            with hero_col2:
                st.markdown("""
                <div class="welcome-card-box" style="margin-top: 10px;">
                    <h2 class="welcome-header-title" style="font-size:26px;">🔐 Admin Login</h2>
                    <p class="welcome-header-sub">Estates &amp; Works Section Authority</p>
                </div>
                """, unsafe_allow_html=True)
                au = st.text_input("Username", value="estates_admin", key="gw_admin_u")
                ap = st.text_input("Password", value="admin123", type="password", key="gw_admin_p")
                
                ac1, ac2 = st.columns(2)
                with ac1:
                    if st.button("Sign In →", type="primary", use_container_width=True, key="btn_signin"):
                        if au.strip() and ap == "admin123":
                            st.session_state.is_admin_logged_in = True
                            st.session_state.admin_username = au.strip()
                            st.session_state.show_admin_login = False
                            st.session_state.portal_view = "admin"
                            st.rerun()
                        else:
                            st.error("Invalid credentials (try password: admin123)")
                with ac2:
                    if st.button("⚡ 1-Click Demo Login", use_container_width=True, key="btn_demo"):
                        st.session_state.is_admin_logged_in = True
                        st.session_state.admin_username = "estates_admin"
                        st.session_state.show_admin_login = False
                        st.session_state.portal_view = "admin"
                        st.rerun()
                
                if st.button("← Back to Options", use_container_width=True, key="btn_back_opts"):
                    st.session_state.show_admin_login = False
                    st.rerun()

        # Quick stats footer
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-label">Hazards Monitored</div>
                <div class="stat-val">{len(st.session_state.pavement_df)}</div>
                <div class="stat-sub">Across 8 campus routes</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Community Reports</div>
                <div class="stat-val">{len(st.session_state.guest_submissions)}</div>
                <div class="stat-sub">This active session</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Defect Vision</div>
                <div class="stat-val" style="font-size:20px;">YOLOv8 Edge</div>
                <div class="stat-sub">Road surface verification</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Granica Lake</div>
                <div class="stat-val" style="font-size:20px;">Parquet PyArrow</div>
                <div class="stat-sub">12-field strict schema</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB: INSIGHTS ──
    elif st.session_state.active_top_tab == "insights":
        st.markdown('<h2 style="font-size:28px; font-weight:800; color:#f8fafc; margin-bottom:4px;">📊 Campus Road Network Insights</h2>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:15px; color:#94a3b8; margin-bottom:32px;">Aggregated metrics from sensor runs and community defect submissions.</p>', unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Campus Road Network", "42.5 km", "8 Primary Corridors")
        k2.metric("Average Damage Index", f"{st.session_state.pavement_df['road_damage_index'].mean():.2f}", "-0.14 vs last month")
        k3.metric("High Risk Hazards", f"{(st.session_state.pavement_df['road_damage_index'] > 5.0).sum()}", "Needs Estates Attention")
        k4.metric("Est. Preventive Savings", "₹ 24.6 Lakhs", "Lifecycle vs Delay")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🛣️ Corridor Health Status Overview")

        corridor_data = []
        for r in IITG_CAMPUS_ROUTES:
            sub = st.session_state.pavement_df[st.session_state.pavement_df["route_zone"] == r["zone"]]
            n_obs = len(sub)
            avg_rdi = sub["road_damage_index"].mean() if n_obs > 0 else 2.5
            status = "🔴 Critical Resurfacing" if avg_rdi > 6.0 else ("🟡 Routine Patching" if avg_rdi > 3.5 else "🟢 Optimal Condition")
            corridor_data.append({
                "Corridor Name": r["name"],
                "Zone": r["zone"],
                "Traffic Volume Index": f"{r['traffic_weight']:.1f}",
                "Reported Hazards": n_obs,
                "Average RDI": f"{avg_rdi:.2f}",
                "Pavement Status": status
            })
        st.dataframe(pd.DataFrame(corridor_data), use_container_width=True, hide_index=True)

        if st.button("← Return to Home", use_container_width=False, key="ins_back_btn"):
            st.session_state.active_top_tab = "home"
            st.rerun()

    # ── TAB: ABOUT ──
    elif st.session_state.active_top_tab == "about":
        st.markdown('<h2 style="font-size:28px; font-weight:800; color:#f8fafc; margin-bottom:4px;">🏛️ About the System</h2>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:15px; color:#94a3b8; margin-bottom:32px;">Predictive Pavement Management Platform</p>', unsafe_allow_html=True)

        col_a1, col_a2 = st.columns([1, 1], gap="large")
        with col_a1:
            st.markdown("""
            ##### System Architecture
            This platform empowers campus authorities and students to collaborate on proactive road maintenance:
            - **Multimodal AI Defect Verification:** Computer vision via YOLOv8 identifies potholes, cracks, and surface failures while strictly rejecting spam or non-road photos.
            - **Inertial Vibration Analysis:** 100 Hz 3-axis accelerometer peak processing isolates high vertical shock events ($Z > 2.0g$) to quantify physical impact.
            - **DBSCAN Spatial Clustering:** Automatically groups clustered complaints along identical 50-meter road corridor segments to prevent redundant work orders.
            """)
        with col_a2:
            st.markdown("""
            ##### Lifecycle Economics & Granica Data Lake
            - **Resurface vs Patch Decision Engine:** Compares localized cold-mix patching versus full-segment profile milling based on defect density and traffic weighting.
            - **Preventive Cost Savings:** Early intervention saves up to **65%** compared to reactive road structural failure after monsoon rain degradation.
            - **Granica Parquet Integration:** Strict 12-field PyArrow schema ensures seamless querying and enterprise data lake synchronization.
            """)

        if st.button("← Return to Home", use_container_width=False, key="about_back_btn"):
            st.session_state.active_top_tab = "home"
            st.rerun()

    # ── TAB: HELP ──
    elif st.session_state.active_top_tab == "help":
        st.markdown('<h2 style="font-size:28px; font-weight:800; color:#f8fafc; margin-bottom:4px;">❓ Resident Help & Reporting Guide</h2>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:15px; color:#94a3b8; margin-bottom:32px;">Guidelines for students, faculty, and residents on reporting campus road hazards.</p>', unsafe_allow_html=True)

        h1, h2 = st.columns(2, gap="large")
        with h1:
            st.markdown("""
            ##### 📸 How to Take an Effective Photo
            1. **Clear Surface View:** Capture the pothole or depression clearly from 2-3 meters away.
            2. **Include Context:** Ensure the asphalt road surface is visible so the AI verification model can confirm the road context.
            3. **Avoid Non-Road Objects:** Submissions of indoor flooring, vegetation, or pets will be automatically flagged and rejected.
            4. **Select Accurate Landmark:** Choose the closest hostel, academic core, or canteen from the corridor dropdown.
            """)
        with h2:
            st.markdown("""
            ##### 📞 Estates Emergency Helplines
            - **Estates & Works Control Room:** Ext. 2000 / +91-361-258-2000
            - **Campus Maintenance Helpdesk:** `maintenance@iitg.ac.in`
            - **Security Emergency Dispatch:** Ext. 2111 / +91-361-258-2111
            - **Office Location:** Administrative Building, 2nd Floor, IIT Guwahati, Assam 781039
            """)

        if st.button("← Return to Home", use_container_width=False, key="help_back_btn"):
            st.session_state.active_top_tab = "home"
            st.rerun()

    st.markdown('<div class="site-footer">Estates and Works Section · Indian Institute of Technology Guwahati · Assam 781039</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# GUEST & STUDENT PORTAL (Report Hazard & Track Community Issues)
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.portal_view == "guest":

    # Top brand bar
    render_navbar(active_tab="home")

    # Portal navigation
    nav_cols = st.columns([1, 1.2, 1.2, 1.2])
    with nav_cols[0]:
        if st.button("← Back to Home", use_container_width=True, key="g_back"):
            st.session_state.portal_view = "gateway"
            st.rerun()
    with nav_cols[1]:
        if st.button("📸 Report Hazard", use_container_width=True, key="g_nav_report",
                      type="primary" if st.session_state.guest_tab == "report" else "secondary"):
            st.session_state.guest_tab = "report"
            st.rerun()
    with nav_cols[2]:
        if st.button("📋 Recent Submissions", use_container_width=True, key="g_nav_view",
                      type="primary" if st.session_state.guest_tab == "view" else "secondary"):
            st.session_state.guest_tab = "view"
            st.rerun()
    with nav_cols[3]:
        st.markdown("<div style='text-align:right; padding-top:6px;'><span class='pill-guest'>👤 GUEST ACCESS</span></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── REPORT TAB ──
    if st.session_state.guest_tab == "report":
        st.markdown('<div class="section-head" style="font-size:28px; font-weight:800; color:#f8fafc;">📸 Report a Road Hazard</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:15px; color:#94a3b8; margin-bottom:32px;">Upload a photograph and provide basic details to help us identify and fix the issue.</p>', unsafe_allow_html=True)

        col_form, col_ai = st.columns([1, 1], gap="large")

        with col_form:
            st.markdown("##### Photo Evidence")
            st.caption("Try a test sample or upload a photo from your phone/camera.")
            sc1, sc2, sc3 = st.columns(3)
            sample_damaged = os.path.join("sample_data", "sample_road_pothole.jpg")
            sample_clean = os.path.join("sample_data", "sample_clean_road.jpg")
            sample_spam = os.path.join("sample_data", "sample_spam_indoor.jpg")

            sample_pick = None
            if sc1.button("📸 Pothole", key="gs1"):
                sample_pick = sample_damaged
            if sc2.button("✨ Clean", key="gs2"):
                sample_pick = sample_clean
            if sc3.button("🚫 Spam", key="gs3"):
                sample_pick = sample_spam

            uploaded_photo = st.file_uploader("Upload road photo", type=["jpg", "jpeg", "png"], key="g_up")

            photo = None
            if uploaded_photo is not None:
                photo = Image.open(uploaded_photo)
            elif sample_pick and os.path.exists(sample_pick):
                photo = Image.open(sample_pick)
                st.info(f"Loaded: `{os.path.basename(sample_pick)}`")

            st.markdown("##### Location & Landmark")
            route_names = [r["name"] for r in IITG_CAMPUS_ROUTES]
            sel_route_idx = st.selectbox("Campus Road / Sector", range(len(route_names)),
                                         format_func=lambda i: route_names[i])
            route = IITG_CAMPUS_ROUTES[sel_route_idx]

            landmark = st.text_input("Specific spot near landmark",
                                     value="Near hostel turn opposite canteen",
                                     placeholder="e.g., Left lane opposite Core 4 bus stop")

            severity = st.selectbox("How bad is it?", [
                "Deep pothole — vehicles swerving",
                "Moderate bump / depression",
                "Surface cracking & loose gravel"
            ])

        with col_ai:
            st.markdown("##### AI Verification & YOLO Detection")
            if photo is not None:
                img_bgr = cv2.cvtColor(np.array(photo), cv2.COLOR_RGB2BGR)
                with st.spinner("Analyzing road surface…"):
                    vr = vision_engine.detect_defects(img_bgr)

                if not vr["verified_road"]:
                    st.error(f"❌ **Spam Rejection:** {vr['rejection_reason']}")
                    st.caption("Only genuine road surfaces are accepted.")
                    st.image(photo, caption="Rejected by AI classifier", use_container_width=True)
                else:
                    st.success("✅ Verified — genuine road surface detected")
                    ann = cv2.cvtColor(vr["annotated_frame"], cv2.COLOR_BGR2RGB)
                    st.image(ann, caption="YOLOv8 Defect Detection", use_container_width=True)

                    n_pot = vr.get("defect_count", 0)
                    area = vr.get("total_damage_area_sqm", 0.0)
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Potholes", f"{n_pot}")
                    mc2.metric("Defect Area", f"{area:.2f} m²")

                    if st.button("🚀 Submit Hazard Report", type="primary", use_container_width=True, key="g_submit"):
                        obs_id = f"COMPLAINT_{len(st.session_state.pavement_df)+1:04d}"
                        now_utc = datetime.now(timezone.utc)
                        dp = int(np.random.randint(*route["paving_age_days_range"]))
                        rdi = PredictivePavementAnalytics.calculate_rdi(area, 1.0, route["traffic_weight"])
                        pdi = PredictivePavementAnalytics.calculate_pdi(rdi, dp)

                        rec = {
                            "observation_id": obs_id, "timestamp": now_utc,
                            "data_source": "STUDENT_APP" if st.session_state.user_role == "student" else "GUEST_APP",
                            "route_zone": route["zone"],
                            "gps_lat_long": f"{route['center_lat']:.5f}, {route['center_lon']:.5f}",
                            "pothole_area_sqm": float(area), "z_accel_peak_g": 1.0,
                            "traffic_volume_index": float(route["traffic_weight"]),
                            "road_damage_index": float(rdi), "days_since_paved": int(dp),
                            "predictive_degradation_score": float(pdi), "synthetic_flag": False
                        }

                        st.session_state.uploaded_hazard_images[obs_id] = {
                            "annotated": ann, "landmark": landmark,
                            "route_name": route["name"], "defect_count": n_pot,
                            "area_sqm": area, "rdi": rdi, "pdi": pdi
                        }
                        st.session_state.guest_submissions.append({
                            "Report ID": obs_id,
                            "Date": now_utc.strftime("%d %b %Y, %H:%M"),
                            "Location": route["name"],
                            "Landmark": landmark,
                            "Potholes": n_pot,
                            "Area (m²)": f"{area:.2f}",
                            "Status": "Pending Inspection"
                        })
                        st.session_state.pavement_df = pd.concat(
                            [st.session_state.pavement_df, pd.DataFrame([rec])], ignore_index=True)
                        GranicaParquetExporter.export_to_parquet(st.session_state.pavement_df, DATASET_PATH)
                        st.balloons()
                        st.success(f"🎉 Report **{obs_id}** submitted successfully to Estates & Works Section!")
            else:
                st.info("👆 Upload a road photograph or choose a test sample on the left to begin.")

    # ── VIEW SUBMISSIONS TAB ──
    elif st.session_state.guest_tab == "view":
        st.markdown('<div class="section-head" style="font-size:28px; font-weight:800; color:#f8fafc;">📋 Recent Community Submissions</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:15px; color:#94a3b8; margin-bottom:32px;">Track road hazard reports submitted by the community.</p>', unsafe_allow_html=True)

        if len(st.session_state.guest_submissions) > 0:
            st.dataframe(pd.DataFrame(st.session_state.guest_submissions), use_container_width=True)

            for sub in reversed(st.session_state.guest_submissions[-5:]):
                cid = sub["Report ID"]
                with st.expander(f"📌 {cid} — {sub['Location']} → {sub['Landmark']}", expanded=False):
                    ic1, ic2 = st.columns([1, 2])
                    with ic1:
                        if cid in st.session_state.uploaded_hazard_images:
                            st.image(st.session_state.uploaded_hazard_images[cid]["annotated"],
                                     caption=f"YOLO Defect Analysis: {cid}", use_container_width=True)
                    with ic2:
                        st.markdown(f"**Location:** {sub['Location']}")
                        st.markdown(f"**Landmark:** {sub['Landmark']}")
                        st.markdown(f"**Potholes Detected:** `{sub['Potholes']}` · **Area:** `{sub['Area (m²)']} m²`")
                        st.markdown(f"**Inspection Status:** {sub['Status']}")
        else:
            st.info("No reports submitted in this session yet. Use the **Report Hazard** tab to create the first one!")

    st.markdown('<div class="site-footer">Estates and Works Section · Indian Institute of Technology Guwahati · Assam 781039</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN PORTAL (Estates & Works Section Authority)
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.portal_view == "admin":
    if not st.session_state.is_admin_logged_in:
        st.session_state.portal_view = "gateway"
        st.rerun()

    # Top brand bar
    render_navbar(active_tab="home")

    # Admin sub-nav
    nav_cols = st.columns([1, 1.2, 1.2, 1.2, 1.2, 1.4])
    with nav_cols[0]:
        if st.button("← Logout", use_container_width=True, key="a_back"):
            st.session_state.is_admin_logged_in = False
            st.session_state.portal_view = "gateway"
            st.rerun()
    tabs_map = {"queue": "📋 Ranked Queue", "map": "🗺️ Campus Map",
                "sensors": "📹 Sensors", "export": "📦 Export"}
    for i, (k, label) in enumerate(tabs_map.items()):
        with nav_cols[i + 1]:
            if st.button(label, use_container_width=True, key=f"an_{k}",
                          type="primary" if st.session_state.admin_tab == k else "secondary"):
                st.session_state.admin_tab = k
                st.rerun()
    with nav_cols[5]:
        st.markdown(f"<div style='text-align:right; padding-top:6px;'><span class='pill-admin'>ADMIN: {st.session_state.admin_username}</span></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Run clustering for all admin tabs
    df_current = st.session_state.pavement_df
    df_clustered, work_orders = PredictivePavementAnalytics.spatial_dbscan_clustering(
        df_current, eps_km=0.15, min_samples=2
    )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: RANKED QUEUE
    # ═══════════════════════════════════════════════════════════════════════
    if st.session_state.admin_tab == "queue":
        st.markdown('<div class="section-head" style="font-size:28px; font-weight:800; color:#f8fafc;">📋 Priority Work-Order Queue</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:15px; color:#94a3b8; margin-bottom:32px;">Click any corridor card below to inspect AI defect evidence, engineering recommendations, and lifecycle cost savings.</p>', unsafe_allow_html=True)

        if len(work_orders) == 0:
            st.info("No active high-density hazard clusters detected.")
        else:
            # Interactive Queue Cards
            st.markdown("##### 🔍 Select Corridor to Inspect")
            
            # Using session state to track selected card
            if "selected_cluster_idx" not in st.session_state:
                st.session_state.selected_cluster_idx = 0
            
            # Create a grid of cards
            card_cols = st.columns(3)
            for i, wo in enumerate(work_orders):
                col_idx = i % 3
                with card_cols[col_idx]:
                    cid = wo["cluster_id"]
                    status = st.session_state.cluster_statuses.get(cid, "Ready for Dispatch")
                    is_active = (i == st.session_state.selected_cluster_idx)
                    border_color = "rgba(56,189,248,0.8)" if is_active else "rgba(255,255,255,0.05)"
                    bg_color = "rgba(30,41,59,0.8)" if is_active else "rgba(30,41,59,0.5)"
                    
                    # Create clickable card using markdown and a transparent button overlaid
                    st.markdown(f"""
                    <div style="background:{bg_color}; border:2px solid {border_color}; border-radius:16px; padding:20px; margin-bottom:16px; transition:all 0.3s ease; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:12px; margin-bottom:12px;">
                            <div style="font-size:16px; font-weight:700; color:#f8fafc;">{wo['zone']}</div>
                            <div style="background:rgba(239,68,68,0.15); color:#fca5a5; padding:4px 10px; border-radius:20px; font-size:12px; font-weight:700;">PDI {wo['priority_pdi']:.1f}</div>
                        </div>
                        <div style="display:flex; gap:16px; color:#94a3b8; font-size:13px; margin-bottom:12px;">
                            <div><b>{wo['hazard_count']}</b> Hazards</div>
                            <div><b>{wo['total_pothole_area_sqm']:.1f}m²</b> Area</div>
                        </div>
                        <div style="font-size:12px; color:#38bdf8; font-weight:600;">Status: {status}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("Inspect", key=f"btn_card_{i}", use_container_width=True):
                        st.session_state.selected_cluster_idx = i
                        st.rerun()

            st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 32px 0;'>", unsafe_allow_html=True)

            selected_wo = work_orders[st.session_state.selected_cluster_idx]
            sel_cid = selected_wo["cluster_id"]
            current_status = st.session_state.cluster_statuses.get(sel_cid, "Ready for Dispatch")

            insp_col1, insp_col2 = st.columns([1, 1], gap="large")

            with insp_col1:
                st.markdown("##### 📸 YOLO Defect Evidence")
                found_custom_img = None
                for obs_id, info in st.session_state.uploaded_hazard_images.items():
                    if obs_id in df_clustered[df_clustered["cluster_id"] == sel_cid]["observation_id"].values:
                        found_custom_img = info
                        break

                if found_custom_img is not None:
                    st.image(found_custom_img["annotated"],
                             caption=f"AI Detection: {found_custom_img['route_name']} — {found_custom_img['defect_count']} defects, {found_custom_img['area_sqm']:.2f} m²",
                             use_container_width=True)
                    st.caption(f"📍 Landmark: {found_custom_img['landmark']}")
                else:
                    sample_path = os.path.join("sample_data", "sample_road_pothole.jpg")
                    if os.path.exists(sample_path):
                        st.image(sample_path,
                                 caption=f"Corridor Evidence: {selected_wo['zone']} (Representative Defect)",
                                 use_container_width=True)

                st.markdown("##### Corridor Telemetry")
                tel1, tel2, tel3 = st.columns(3)
                tel1.metric("Defects", f"{selected_wo['hazard_count']}")
                tel2.metric("Total Area", f"{selected_wo['total_pothole_area_sqm']:.1f} m²")
                tel3.metric("Peak Shock", f"{selected_wo['max_z_accel_peak_g']:.2f}g")

            with insp_col2:
                st.markdown("##### 🛠️ Fix Action & Engineering Rationale")
                is_full = "FULL" in selected_wo["recommended_action"].upper()

                if is_full:
                    st.markdown(f"""
                    <div class="verdict-full">
                        <div class="verdict-full-title">🚨 Verdict: Full Segment Milling &amp; Resurfacing</div>
                        <p style="font-size:13px; color:#7f1d1d; line-height:1.55; margin:0 0 8px 0;">
                            <b>Rationale:</b> High defect cluster density ({selected_wo['total_pothole_area_sqm']:.1f} m² total area) across {selected_wo['zone']}.
                            Spot patching is non-viable — the sub-base layer has suffered shear fatigue. Complete 50-meter segment resurfacing is required.
                        </p>
                        <div style="font-size:12px; color:#991b1b;">
                            <b>Scope:</b> 50 mm Asphalt Concrete Overlay · Bituminous Tack Coat · Sub-base Compaction
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-spot">
                        <div class="verdict-spot-title">🔧 Verdict: Targeted High-Pressure Spot Patching</div>
                        <p style="font-size:13px; color:#78350f; line-height:1.55; margin:0 0 8px 0;">
                            <b>Rationale:</b> Defect density is isolated ({selected_wo['total_pothole_area_sqm']:.1f} m² across {selected_wo['hazard_count']} points).
                            Sub-base integrity remains intact. Rapid cold/warm-mix pothole injection will restore serviceability.
                        </p>
                        <div style="font-size:12px; color:#92400e;">
                            <b>Scope:</b> Jet-patcher bitumen injection · Edge sealing · Traffic reopened in 2 hours
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("##### 💰 Lifecycle Cost & Savings Analysis")
                proactive_cost = selected_wo["estimated_cost_inr"]
                reactive_cost = proactive_cost * 2.85
                savings = reactive_cost - proactive_cost

                st.markdown(f"""
                <div class="savings-box">
                    <div class="savings-big">₹ {savings:,.0f} Net Savings</div>
                    <div class="savings-detail">
                        <b>Proactive Cost (Now):</b> ₹ {proactive_cost:,.0f}<br>
                        <b>Reactive Cost (Delayed 6 mos):</b> ₹ {reactive_cost:,.0f}<br>
                        Acting now avoids monsoon water ingress, sub-base gravel washout, and vehicle suspension liability claims.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                cost_df = pd.DataFrame({
                    "Scenario": ["Proactive (Today)", "Delayed (Reactive)"],
                    "Cost (₹)": [proactive_cost, reactive_cost]
                })
                st.bar_chart(cost_df.set_index("Scenario"), height=160)

                st.markdown("##### ⚡ Dispatch & Status Controls")
                status_cols = st.columns(3)
                with status_cols[0]:
                    if st.button("🚀 Dispatch Order", type="primary", use_container_width=True, key=f"d_{sel_cid}"):
                        st.session_state.cluster_statuses[sel_cid] = "Dispatched to Contractor"
                        st.rerun()
                with status_cols[1]:
                    if st.button("🔧 In Progress", use_container_width=True, key=f"ip_{sel_cid}"):
                        st.session_state.cluster_statuses[sel_cid] = "Repairs In Progress"
                        st.rerun()
                with status_cols[2]:
                    if st.button("✅ Mark Fixed", use_container_width=True, key=f"mf_{sel_cid}"):
                        st.session_state.cluster_statuses[sel_cid] = "Resolved & Verified"
                        st.rerun()

                st.caption(f"Current Status: **{current_status}**")

                pdf_buf = generate_work_order_pdf(
                    cluster_id=sel_cid,
                    zone=selected_wo["zone"],
                    hazard_count=selected_wo["hazard_count"],
                    total_area=selected_wo["total_pothole_area_sqm"],
                    peak_shock=selected_wo["max_z_accel_peak_g"],
                    priority_pdi=selected_wo["priority_pdi"],
                    action=selected_wo["recommended_action"],
                    cost_inr=selected_wo["estimated_cost_inr"]
                )
                st.download_button(
                    label=f"📄 Download PDF Work Order ({sel_cid})",
                    data=pdf_buf,
                    file_name=f"IITG_Work_Order_{sel_cid}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"dl_pdf_{sel_cid}"
                )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: CAMPUS MAP
    # ═══════════════════════════════════════════════════════════════════════
    elif st.session_state.admin_tab == "map":
        st.markdown('<div class="section-head" style="font-size:20px; font-weight:800; color:#0f172a;">🗺️ IIT Guwahati Campus Pavement Map</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:13.5px; color:#64748b; margin-bottom:20px;">Geospatial distribution of hazard observations with DBSCAN spatial cluster overlay across campus roads.</p>', unsafe_allow_html=True)

        try:
            import folium
            from streamlit_folium import st_folium

            iitg_center = [26.1878, 91.6916]
            m = folium.Map(location=iitg_center, zoom_start=15, tiles="OpenStreetMap")

            for _, row in df_clustered.iterrows():
                try:
                    lat_str, lon_str = row["gps_lat_long"].split(",")
                    lat, lon = float(lat_str.strip()), float(lon_str.strip())
                except Exception:
                    continue

                cid = row.get("cluster_id", "NOISE")
                color = "red" if cid != "NOISE" else "blue"
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5 if cid == "NOISE" else 8,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"Obs: {row['observation_id']}<br>Cluster: {cid}<br>RDI: {row['road_damage_index']:.1f}<br>PDI: {row['predictive_degradation_score']:.1f}"
                ).add_to(m)

            st_folium(m, width=None, height=520)
        except ImportError:
            st.warning("folium or streamlit_folium not installed. Showing coordinate table:")
            st.dataframe(df_clustered[["observation_id", "gps_lat_long", "route_zone", "road_damage_index"]].head(20))

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: SENSORS
    # ═══════════════════════════════════════════════════════════════════════
    elif st.session_state.admin_tab == "sensors":
        st.markdown('<div class="section-head" style="font-size:20px; font-weight:800; color:#0f172a;">📹 Multimodal Edge Sensor Feeds</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:13.5px; color:#64748b; margin-bottom:20px;">Telemetry feeds from campus bus / e-rickshaw mounted cameras and 100 Hz inertial measurement units.</p>', unsafe_allow_html=True)

        sc1, sc2 = st.columns([1, 1], gap="large")

        with sc1:
            st.markdown("##### YOLOv8 Vision Feed")
            sample_path = os.path.join("sample_data", "sample_road_pothole.jpg")
            if os.path.exists(sample_path):
                img = cv2.imread(sample_path)
                res = vision_engine.detect_defects(img)
                ann = cv2.cvtColor(res["annotated_frame"], cv2.COLOR_BGR2RGB)
                st.image(ann, caption=f"Defects: {res['defect_count']} · Total Area: {res['total_damage_area_sqm']:.2f} m²", use_container_width=True)

        with sc2:
            st.markdown("##### 100 Hz Accelerometer Telemetry")
            np.random.seed(42)
            t = np.linspace(0, 10, 1000)
            base_vib = np.random.normal(0, 0.15, 1000)
            spike_idx = [250, 480, 720]
            for si in spike_idx:
                base_vib[si:si+15] += np.random.uniform(1.8, 3.2, 15)

            accel_df = pd.DataFrame({"Time (s)": t, "Z-Axis Shock (g)": base_vib})
            st.line_chart(accel_df.set_index("Time (s)"), height=260)
            st.caption("Threshold: Vertical shock > 1.8g triggers automatic road pothole correlation event.")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: EXPORT
    # ═══════════════════════════════════════════════════════════════════════
    elif st.session_state.admin_tab == "export":
        st.markdown('<div class="section-head" style="font-size:20px; font-weight:800; color:#0f172a;">📦 Granica Hackathon Dataset Export</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:13.5px; color:#64748b; margin-bottom:20px;">Download validated telemetry in strict 12-field PyArrow Parquet format or export consolidated PDF reports.</p>', unsafe_allow_html=True)

        ec1, ec2 = st.columns(2, gap="large")

        with ec1:
            st.markdown("##### Parquet Lake Export")
            st.write(f"Total Records: **{len(df_current)}**")
            st.write("Format: **Apache Parquet (PyArrow Engine)**")

            parquet_buf = io.BytesIO()
            df_current.to_parquet(parquet_buf, engine="pyarrow", index=False)
            parquet_bytes = parquet_buf.getvalue()

            st.download_button(
                label="📥 Download Parquet Dataset",
                data=parquet_bytes,
                file_name="iitg_road_pothole_dataset.parquet",
                mime="application/octet-stream",
                type="primary",
                use_container_width=True
            )

        with ec2:
            st.markdown("##### Consolidated Work Order PDF")
            st.write(f"Active Priority Clusters: **{len(work_orders)}**")
            st.write("Summary: Work orders grouped for contractor procurement.")

            top_wo = work_orders[0] if len(work_orders) > 0 else None
            if top_wo:
                summary_pdf = generate_work_order_pdf(
                    cluster_id="CONSOLIDATED_ALL",
                    zone="Entire IITG Campus Network",
                    hazard_count=sum(w["hazard_count"] for w in work_orders),
                    total_area=sum(w["total_pothole_area_sqm"] for w in work_orders),
                    peak_shock=max(w["max_z_accel_peak_g"] for w in work_orders),
                    priority_pdi=max(w["priority_pdi"] for w in work_orders),
                    action="Comprehensive Campus Resurfacing & Jet-Patching Campaign",
                    cost_inr=sum(w["estimated_cost_inr"] for w in work_orders)
                )
                st.download_button(
                    label="📄 Download Consolidated PDF",
                    data=summary_pdf,
                    file_name="IITG_Estates_Consolidated_Work_Orders.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    st.markdown('<div class="site-footer">Estates and Works Section · Indian Institute of Technology Guwahati · Assam 781039</div>', unsafe_allow_html=True)
