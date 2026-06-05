import streamlit as st
import os, sys, json, re
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE       = os.path.dirname(os.path.abspath(__file__))
VP_FILE    = os.path.join(BASE, "verified_papers.json")
CODE_DIR   = os.path.join(BASE, "generated_code")
GOLDEN_DIR = os.path.join(BASE, "tests", "golden")

QUICK_PAPERS = [
    ("1512.03385","ResNet"),("1609.02907","GCN"),
    ("1706.03762","Transformer"),("1509.02971","DDPG"),
    ("1708.05031","NCF"),
]

st.set_page_config(page_title="Cogniterra", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

# ── SPLASH SCREEN ─────────────────────────────────────────────────────────────
if "splash_done" not in st.session_state:
    st.session_state.splash_done = False

if not st.session_state.splash_done:
    splash = st.empty()
    splash.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,.stApp{background:#0f1117!important;margin:0;padding:0;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0!important;}
.splash{position:fixed;inset:0;background:#0f1117;display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:9999;animation:splashFade 0.4s ease both;}
@keyframes splashFade{from{opacity:0;}to{opacity:1;}}
.splash::before{content:'';position:absolute;inset:0;background-image:linear-gradient(rgba(249,115,22,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(249,115,22,0.04) 1px,transparent 1px);background-size:60px 60px;animation:gridMove 8s linear infinite;}
@keyframes gridMove{from{background-position:0 0;}to{background-position:60px 60px;}}
.splash-orb{position:absolute;width:320px;height:320px;border-radius:50%;background:radial-gradient(circle,rgba(249,115,22,0.12) 0%,transparent 70%);animation:orbPulse 3s ease-in-out infinite;}
@keyframes orbPulse{0%,100%{transform:scale(1);opacity:0.6;}50%{transform:scale(1.15);opacity:1;}}
.splash-content{position:relative;z-index:1;display:flex;flex-direction:column;align-items:center;}
.splash-logo{width:80px;height:80px;background:linear-gradient(135deg,#f97316,#ea580c);border-radius:22px;display:flex;align-items:center;justify-content:center;font-family:'Sora',sans-serif;font-size:1.8rem;font-weight:800;color:white;box-shadow:0 0 40px rgba(249,115,22,0.5),0 0 80px rgba(249,115,22,0.2);animation:logoIn 0.6s cubic-bezier(0.34,1.56,0.64,1) 0.2s both;margin-bottom:28px;}
@keyframes logoIn{from{opacity:0;transform:scale(0.4) rotate(-10deg);}to{opacity:1;transform:scale(1) rotate(0deg);}}
.splash-name{font-family:'Sora',sans-serif;font-size:3.2rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.05em;line-height:1;animation:nameIn 0.5s ease 0.5s both;margin-bottom:10px;}
.splash-name span{color:#f97316;}
@keyframes nameIn{from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:translateY(0);}}
.splash-tag{font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#475569;letter-spacing:0.08em;animation:nameIn 0.5s ease 0.7s both;margin-bottom:52px;}
.splash-bar-wrap{width:220px;height:2px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;animation:nameIn 0.4s ease 0.9s both;}
.splash-bar{height:100%;width:0%;background:linear-gradient(90deg,#f97316,#fb923c);border-radius:2px;animation:barFill 2.2s cubic-bezier(0.4,0,0.2,1) 1s forwards;box-shadow:0 0 8px rgba(249,115,22,0.6);}
@keyframes barFill{0%{width:0%;}60%{width:75%;}85%{width:88%;}100%{width:100%;}}
.splash-loading{font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#334155;letter-spacing:0.1em;margin-top:16px;animation:nameIn 0.4s ease 1s both;}
.splash-loading span{display:inline-block;animation:dotBlink 1.4s ease-in-out infinite;}
.splash-loading span:nth-child(2){animation-delay:0.2s;}
.splash-loading span:nth-child(3){animation-delay:0.4s;}
@keyframes dotBlink{0%,80%,100%{opacity:0;}40%{opacity:1;}}
</style>
<div class="splash">
  <div class="splash-orb"></div>
  <div class="splash-content">
    <div class="splash-logo">CT</div>
    <div class="splash-name">Cogni<span>terra</span></div>
    <div class="splash-tag">ML PAPER REPLICATION ENGINE</div>
    <div class="splash-bar-wrap"><div class="splash-bar"></div></div>
    <div class="splash-loading">INITIALISING<span>.</span><span>.</span><span>.</span></div>
  </div>
</div>
""", unsafe_allow_html=True)
    import time
    time.sleep(3)
    splash.empty()
    st.session_state.splash_done = True
    st.rerun()

@st.cache_resource
def load_agents():
    from agents.latex_ingestion import parse_paper, get_last_source_type   # Upgrade 1
    from agents.rag_agent import get_relevant_context
    from agents.coder_agent import generate_code
    from agents.domain_detector import detect_domain, get_code_domain
    from agents.tester_agent import generate_html_report
    from agents.scoring_engine import run_test                              # Upgrade 2
    return parse_paper, get_last_source_type, get_relevant_context, generate_code, detect_domain, get_code_domain, run_test, generate_html_report

parse_paper, get_last_source_type, get_relevant_context, generate_code, detect_domain, get_code_domain, run_test, generate_html_report = load_agents()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{--bg:#0f1117;--surf:#1a1d27;--surf2:#22263a;--border:#2e3248;--border2:#3a3f5c;--accent:#f97316;--accent2:#fb923c;--text:#f1f5f9;--text2:#94a3b8;--text3:#475569;--success:#22c55e;--error:#ef4444;--warn:#f59e0b;}
*,*::before,*::after{box-sizing:border-box;}
html,body,.stApp{background:var(--bg)!important;font-family:'Inter',sans-serif!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stSidebar"]{display:none;}
.topbar{background:var(--surf);border-bottom:1px solid var(--border);padding:0 1.5rem;height:60px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:200;box-shadow:0 1px 20px rgba(0,0,0,0.4);}
.tb-left{display:flex;align-items:center;gap:12px;}
.tb-logo{width:34px;height:34px;background:linear-gradient(135deg,#f97316,#ea580c);border-radius:9px;display:flex;align-items:center;justify-content:center;font-family:'Sora',sans-serif;font-size:0.76rem;font-weight:800;color:white;box-shadow:0 0 14px rgba(249,115,22,0.45);}
.tb-name{font-family:'Sora',sans-serif;font-size:1.04rem;font-weight:700;color:var(--text);letter-spacing:-0.02em;}
.tb-tag{font-size:0.59rem;color:var(--text3);background:var(--surf2);padding:2px 7px;border-radius:4px;border:1px solid var(--border);}
.sidebar{background:var(--surf);border-right:1px solid var(--border);padding:1.5rem 0;display:flex;flex-direction:column;position:sticky;top:60px;height:calc(100vh - 60px);overflow-y:auto;}
.sb-label{font-size:0.59rem;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:0.13em;padding:0 1.2rem;margin-bottom:8px;}
.sb-step{display:flex;align-items:flex-start;gap:10px;padding:10px 12px;border-radius:9px;margin:0 0.5rem 2px;border-left:3px solid transparent;transition:all 0.2s;}
.sb-step.active{background:rgba(249,115,22,0.1);border-left-color:var(--accent);}
.sb-step.done{background:rgba(34,197,94,0.05);border-left-color:var(--success);}
.sb-step.waiting{opacity:0.4;}
.sb-num{width:22px;height:22px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:0.62rem;font-weight:800;margin-top:1px;transition:all 0.3s;}
.sb-num.active{background:var(--accent);color:white;animation:pulse-o 2s infinite;}
.sb-num.done{background:var(--success);color:white;}
.sb-num.waiting{background:var(--surf2);color:var(--text3);border:1px solid var(--border);}
@keyframes pulse-o{0%,100%{box-shadow:0 0 8px rgba(249,115,22,0.5);}50%{box-shadow:0 0 18px rgba(249,115,22,0.85);}}
.sb-info{flex:1;}
.sb-step-title{font-size:0.82rem;font-weight:600;color:var(--text);margin-bottom:2px;}
.sb-step-desc{font-size:0.69rem;color:var(--text2);line-height:1.35;}
.sb-step.active .sb-step-title{color:var(--accent);}
.sb-step.done .sb-step-title{color:var(--success);}
.sb-div{height:1px;background:var(--border);margin:1rem 1.2rem;}
.sb-hist{font-size:0.59rem;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:0.13em;padding:0 1.2rem;margin-bottom:8px;}
.vp-item{display:flex;align-items:center;gap:8px;padding:7px 1.2rem;transition:background 0.15s;}
.vp-item:hover{background:var(--surf2);}
.vp-d{width:6px;height:6px;border-radius:50%;flex-shrink:0;}
.vp-pass{background:var(--success);}
.vp-fail{background:var(--error);}
.vp-warn{background:var(--warn);}
.vp-pid{flex:1;font-family:'JetBrains Mono',monospace;font-size:0.71rem;color:var(--text2);}
.vp-sc{font-size:0.69rem;font-weight:700;color:var(--accent);}
.main{padding:2rem 2.5rem;animation:fadeUp 0.32s ease;}
@keyframes fadeUp{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}
.pipe-bar{display:flex;align-items:center;background:var(--surf);border:1px solid var(--border);border-radius:12px;padding:0.85rem 1.2rem;margin-bottom:2rem;}
.pipe-node{flex:1;display:flex;align-items:center;justify-content:center;gap:8px;font-size:0.76rem;font-weight:600;color:var(--text3);padding:5px 0;border-radius:7px;transition:all 0.3s;}
.pipe-node.done{color:var(--success);}
.pipe-node.active{color:var(--accent);background:rgba(249,115,22,0.07);}
.pipe-ball{width:25px;height:25px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:0.63rem;font-weight:800;transition:all 0.3s;}
.pipe-ball.done{background:var(--success);color:white;}
.pipe-ball.active{background:var(--accent);color:white;animation:pulse-o 2s infinite;}
.pipe-ball.waiting{background:var(--surf2);color:var(--text3);border:1px solid var(--border);}
.pipe-line{flex:0 0 24px;height:1px;background:var(--border);margin:0 4px;}
.pipe-line.done{background:var(--success);}
.eyebrow{font-size:0.65rem;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:0.15em;margin-bottom:5px;}
.page-title{font-family:'Sora',sans-serif;font-size:2rem;font-weight:800;color:var(--text);letter-spacing:-0.03em;line-height:1.15;margin-bottom:5px;}
.page-desc{font-size:0.87rem;color:var(--text2);margin-bottom:1.8rem;}
.card{background:var(--surf);border:1px solid var(--border);border-radius:14px;padding:1.4rem 1.5rem;margin-bottom:1rem;transition:border-color 0.2s,box-shadow 0.2s;}
.card:hover{border-color:var(--border2);box-shadow:0 4px 22px rgba(0,0,0,0.35);}
.card-label{font-size:0.62rem;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:0.12em;margin-bottom:1rem;}
.paper-banner{display:flex;align-items:center;gap:12px;background:var(--surf);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:10px;padding:0.8rem 1.1rem;margin-bottom:1.5rem;}
.paper-banner-title{font-size:0.9rem;font-weight:600;color:var(--text);flex:1;}
.paper-banner-id{font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--text3);}
.row{display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border);}
.row:last-child{border-bottom:none;}
.row-label{font-size:0.78rem;color:var(--text2);}
.row-value{font-size:0.8rem;font-weight:600;color:var(--text);}
.hw{display:flex;align-items:flex-start;gap:10px;padding:9px 0;border-bottom:1px solid var(--border);}
.hw:last-child{border-bottom:none;}
.hw-n{width:22px;height:22px;border-radius:50%;background:linear-gradient(135deg,var(--accent),#ea580c);color:white;font-size:0.62rem;font-weight:800;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px;}
.hw-t{font-size:0.8rem;color:var(--text2);line-height:1.45;}
.domain-tag{display:inline-flex;align-items:center;gap:8px;background:rgba(249,115,22,0.12);border:1px solid rgba(249,115,22,0.3);color:var(--accent2);padding:7px 16px;border-radius:30px;font-size:0.8rem;font-weight:700;margin-bottom:10px;}
.conf-track{background:var(--surf2);border-radius:6px;height:6px;overflow:hidden;margin-top:8px;}
.conf-fill{height:6px;background:linear-gradient(90deg,var(--accent),#ea580c);border-radius:6px;transition:width 0.9s ease;}
.sec-row{display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border);}
.sec-row:last-child{border-bottom:none;}
.sec-row-name{font-size:0.8rem;color:var(--text2);}
.sec-row-tick{color:var(--success);font-weight:800;font-size:0.8rem;}
.code-hdr{background:#0a0d14;border-radius:10px 10px 0 0;padding:10px 16px;display:flex;align-items:center;gap:7px;border:1px solid var(--border);border-bottom:none;}
.dot-r{width:12px;height:12px;border-radius:50%;background:#ff5f57;}
.dot-y{width:12px;height:12px;border-radius:50%;background:#ffbd2e;}
.dot-g{width:12px;height:12px;border-radius:50%;background:#28c840;}
.code-fn{font-size:0.69rem;color:var(--text3);margin-left:8px;font-family:'JetBrains Mono',monospace;}
.mg{display:flex;gap:10px;margin-bottom:10px;flex-wrap:wrap;}
.mc{background:var(--surf);border:1px solid var(--border);border-radius:12px;padding:1.1rem 1.2rem;transition:all 0.2s;flex:1;min-width:120px;}
.mc:hover{border-color:var(--border2);transform:translateY(-1px);}
.mc.pass{border-top:2px solid var(--success);}
.mc.fail{border-top:2px solid var(--error);}
.mc.neu{border-top:2px solid var(--accent);}
.mc.warn{border-top:2px solid var(--warn);}
.mc-lbl{font-size:0.6rem;color:var(--text3);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;font-weight:700;}
.mc-val{font-size:1.65rem;font-weight:800;color:var(--text);letter-spacing:-0.04em;line-height:1;font-family:'Sora',sans-serif;}
.mc-unt{font-size:0.74rem;font-weight:500;color:var(--text2);margin-left:2px;}
.score-wrap{text-align:center;padding:1.8rem 1rem;}
.score-num{font-family:'Sora',sans-serif;font-size:5.5rem;font-weight:800;letter-spacing:-0.07em;line-height:1;background:linear-gradient(135deg,var(--accent),#ea580c);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.score-num.fail{background:linear-gradient(135deg,var(--error),#dc2626);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.score-num.warn{background:linear-gradient(135deg,var(--warn),#d97706);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.score-lbl{font-size:0.62rem;color:var(--text3);text-transform:uppercase;letter-spacing:0.14em;margin-top:6px;font-weight:700;}
.score-badge{display:inline-flex;align-items:center;gap:6px;padding:6px 16px;border-radius:8px;margin-top:14px;font-size:0.73rem;font-weight:700;}
.bdg-pass{background:rgba(34,197,94,0.12);color:var(--success);border:1px solid rgba(34,197,94,0.3);}
.bdg-fail{background:rgba(239,68,68,0.12);color:var(--error);border:1px solid rgba(239,68,68,0.3);}
.bdg-warn{background:rgba(245,158,11,0.12);color:var(--warn);border:1px solid rgba(245,158,11,0.3);}
.et{width:100%;border-collapse:collapse;font-size:0.78rem;}
.et th{text-align:left;padding:7px 10px;color:var(--text3);font-weight:700;border-bottom:1px solid var(--border);font-size:0.63rem;text-transform:uppercase;letter-spacing:0.09em;}
.et td{padding:8px 10px;border-bottom:1px solid var(--border);color:var(--text2);}
.et tr:last-child td{border-bottom:none;}
.et tr:hover td{background:var(--surf2);}
.et-final td{color:var(--text)!important;font-weight:700!important;}
.final-tag{background:rgba(34,197,94,0.12);color:var(--success);border:1px solid rgba(34,197,94,0.3);font-size:0.62rem;font-weight:700;padding:2px 8px;border-radius:5px;}
/* 4-dimension scoring styles */
.dim-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1.2rem;}
.dim-card{background:var(--surf);border:1px solid var(--border);border-radius:12px;padding:1rem 1.1rem;position:relative;overflow:hidden;}
.dim-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.dim-exec::before{background:linear-gradient(90deg,#22c55e,#16a34a);}
.dim-meth::before{background:linear-gradient(90deg,#f97316,#ea580c);}
.dim-res::before{background:linear-gradient(90deg,#3b82f6,#2563eb);}
.dim-comp::before{background:linear-gradient(90deg,#a855f7,#9333ea);}
.dim-label{font-size:0.58rem;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:0.12em;margin-bottom:4px;}
.dim-score{font-family:'Sora',sans-serif;font-size:2rem;font-weight:800;color:var(--text);letter-spacing:-0.04em;line-height:1;}
.dim-weight{font-size:0.62rem;color:var(--text3);margin-top:4px;}
.dim-bar-wrap{background:var(--surf2);border-radius:4px;height:4px;margin-top:8px;overflow:hidden;}
.dim-bar{height:4px;border-radius:4px;transition:width 1s ease;}
.dim-exec .dim-bar{background:linear-gradient(90deg,#22c55e,#16a34a);}
.dim-meth .dim-bar{background:linear-gradient(90deg,#f97316,#ea580c);}
.dim-res .dim-bar{background:linear-gradient(90deg,#3b82f6,#2563eb);}
.dim-comp .dim-bar{background:linear-gradient(90deg,#a855f7,#9333ea);}
.sub-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:1rem;}
.sub-card{background:var(--surf2);border:1px solid var(--border);border-radius:8px;padding:0.7rem 0.8rem;text-align:center;}
.sub-label{font-size:0.55rem;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:5px;line-height:1.3;}
.sub-val{font-size:1.1rem;font-weight:800;font-family:'Sora',sans-serif;}
.sub-full{color:var(--success);}
.sub-half{color:var(--warn);}
.sub-zero{color:var(--error);}
.weight-formula{background:var(--surf2);border:1px solid var(--border);border-radius:8px;padding:0.8rem 1rem;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--text2);margin-top:0.5rem;}
.weight-formula span{color:var(--accent);font-weight:700;}
.cta-box{background:linear-gradient(135deg,rgba(249,115,22,0.08),rgba(124,58,237,0.08));border:1px solid rgba(249,115,22,0.2);border-radius:16px;padding:2.5rem;text-align:center;margin-top:2rem;}
.cta-title{font-family:'Sora',sans-serif;font-size:1.3rem;font-weight:800;color:var(--text);letter-spacing:-0.02em;margin-bottom:6px;}
.cta-sub{font-size:0.82rem;color:var(--text2);margin-bottom:1.6rem;}
.stButton>button{font-family:'Inter',sans-serif!important;font-weight:600!important;border-radius:9px!important;height:42px!important;font-size:0.83rem!important;transition:all 0.15s ease!important;}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#f97316,#ea580c)!important;border:none!important;color:white!important;box-shadow:0 2px 12px rgba(249,115,22,0.3)!important;}
.stButton>button[kind="primary"]:hover{transform:translateY(-1px)!important;box-shadow:0 4px 20px rgba(249,115,22,0.5)!important;filter:brightness(1.08)!important;}
.stButton>button[kind="secondary"]{background:var(--surf2)!important;border:1px solid var(--border)!important;color:var(--text2)!important;}
.stButton>button[kind="secondary"]:hover{border-color:var(--accent)!important;color:var(--accent)!important;}
div[data-testid="stFileUploader"]{background:var(--surf2)!important;border:1.5px dashed var(--border2)!important;border-radius:10px!important;}
div[data-testid="stFileUploader"]:hover{border-color:var(--accent)!important;}
div[data-testid="stFileUploader"] section{padding:0.6rem 0.9rem!important;min-height:0!important;}
div[data-testid="stFileUploader"] section small{display:none!important;}
.stTextInput>div>div>input{background:var(--surf2)!important;border:1px solid var(--border)!important;border-radius:9px!important;color:var(--text)!important;font-family:'JetBrains Mono',monospace!important;font-size:0.86rem!important;height:42px!important;}
.stTextInput>div>div>input:focus{border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(249,115,22,0.15)!important;}
.stTextInput label{color:var(--text2)!important;font-size:0.73rem!important;font-weight:600!important;}
.stSelectbox>div>div{background:var(--surf2)!important;border:1px solid var(--border)!important;border-radius:9px!important;color:var(--text)!important;}
.stSelectbox label{color:var(--text2)!important;font-size:0.73rem!important;font-weight:600!important;}
.stProgress>div>div>div>div{background:linear-gradient(90deg,#f97316,#ea580c)!important;}
div[data-testid="stExpander"]>div:first-child{background:var(--surf2)!important;border:1px solid var(--border)!important;border-radius:9px!important;color:var(--text2)!important;font-weight:600!important;}
hr{border-color:var(--border)!important;margin:1.2rem 0!important;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_vp():
    if not os.path.exists(VP_FILE): return []
    try:
        with open(VP_FILE) as f: return json.load(f)
    except: return []

def save_vp(pid, score, status):
    papers = [p for p in load_vp() if p.get("paper_id") != pid]
    papers.insert(0, {"paper_id": pid, "score": score, "status": status, "ts": datetime.now().strftime("%d %b, %H:%M")})
    try:
        with open(VP_FILE, "w") as f: json.dump(papers[:20], f, indent=2)
    except: pass

def get_epoch_accs(stdout):
    return [(int(e), float(a)) for e, a in re.findall(r'Epoch\s+\[?(\d+)[/\d\]]*\s+(?:Loss:[^\n]+\n)?.*?(?:Accuracy|accuracy):\s*([\d.]+)%', stdout, re.I | re.S)]

def get_title(text):
    for line in [l.strip() for l in text.split('\n') if l.strip()][:8]:
        if 10 < len(line) < 130 and not line.startswith('#') and not line.startswith('http'):
            return line
    return "Research Paper"

def get_deps(code):
    r = []
    for lib, name in [("torch","PyTorch"),("torchvision","Torchvision"),("numpy","NumPy"),("sklearn","Scikit-Learn"),("pandas","Pandas"),("gymnasium","Gymnasium"),("gym","OpenAI Gym")]:
        if lib in code: r.append(name)
    return r or ["PyTorch", "NumPy"]

def step_cls(n):
    order = ["upload", "analysis", "codegen", "results"]
    ci = order.index(st.session_state.get("stage", "upload"))
    return "active" if n-1 == ci else ("done" if n-1 < ci else "waiting")

def reset_all():
    for k in ["stage","paper_id","text","detection","rag","code","result","ep_accs","title","stdout","stderr","pipe","domain"]:
        st.session_state.pop(k, None)

def render_pipe(active):
    steps = [("📡","Fetching",1),("🔍","Parsing",2),("⚡","Code Gen",3),("✅","Verify",4)]
    def cls(i):
        if active == 0: return "waiting"
        if i < active: return "done"
        if i == active: return "active"
        return "waiting"
    def inn(i): c = cls(i); return "✓" if c=="done" else ("●" if c=="active" else str(i))
    h = '<div class="pipe-bar">'
    for ico, lbl, i in steps:
        c = cls(i)
        h += f'<div class="pipe-node {c}"><div class="pipe-ball {c}">{inn(i)}</div>{ico} {lbl}</div>'
        if i < 4:
            lc = "done" if active > i else ""
            h += f'<div class="pipe-line {lc}"></div>'
    h += '</div>'
    st.markdown(h, unsafe_allow_html=True)

def render_sidebar():
    steps = [(1,"Upload Paper","Drop PDF or enter\nArXiv ID"),(2,"Analysis","Domain & parameter\nextraction"),(3,"Code Generation","PyTorch via\nGroq · Llama 3.3 70B"),(4,"Verification","Run & score\nresults")]
    vps = load_vp()
    h = '<div class="sidebar"><div class="sb-label">Pipeline</div>'
    for num, title, desc in steps:
        cls = step_cls(num)
        tick = "✓" if cls=="done" else str(num)
        h += f'<div class="sb-step {cls}"><div class="sb-num {cls}">{tick}</div><div class="sb-info"><div class="sb-step-title">{title}</div><div class="sb-step-desc">{desc}</div></div></div>'
    h += '<div class="sb-div"></div><div class="sb-hist">Verified Papers</div>'
    if not vps:
        h += '<div style="padding:0 1.2rem;font-size:0.75rem;color:var(--text3)">No papers verified yet</div>'
    else:
        for vp in vps[:6]:
            pid = vp.get("paper_id",""); sc = int(vp.get("score",0)); stat = vp.get("status","")
            dc = "vp-pass" if "PASS" in stat else ("vp-fail" if "FAIL" in stat else "vp-warn")
            h += f'<div class="vp-item"><div class="vp-d {dc}"></div><span class="vp-pid">{pid}</span><span class="vp-sc">{sc}/100</span></div>'
    h += '</div>'
    st.markdown(h, unsafe_allow_html=True)

def render_dimension_scores(res):
    """Render the 4-dimension scoring breakdown cards."""
    dims = res.get("dimension_scores", {})
    e_sc = dims.get("execution",    0)
    m_sc = dims.get("methodology",  0)
    r_sc = dims.get("results",      0)
    c_sc = dims.get("completeness", 0)

    def score_color(s):
        if s >= 75: return "var(--success)"
        if s >= 50: return "var(--warn)"
        return "var(--error)"

    st.markdown(f"""
    <div class="dim-grid">
      <div class="dim-card dim-exec">
        <div class="dim-label">Execution</div>
        <div class="dim-score" style="color:{score_color(e_sc)}">{e_sc:.0f}</div>
        <div class="dim-weight">weight 0.20</div>
        <div class="dim-bar-wrap"><div class="dim-bar" style="width:{e_sc}%"></div></div>
      </div>
      <div class="dim-card dim-meth">
        <div class="dim-label">Methodology</div>
        <div class="dim-score" style="color:{score_color(m_sc)}">{m_sc:.0f}</div>
        <div class="dim-weight">weight 0.35</div>
        <div class="dim-bar-wrap"><div class="dim-bar" style="width:{m_sc}%"></div></div>
      </div>
      <div class="dim-card dim-res">
        <div class="dim-label">Results</div>
        <div class="dim-score" style="color:{score_color(r_sc)}">{r_sc:.0f}</div>
        <div class="dim-weight">weight 0.30</div>
        <div class="dim-bar-wrap"><div class="dim-bar" style="width:{r_sc}%"></div></div>
      </div>
      <div class="dim-card dim-comp">
        <div class="dim-label">Completeness</div>
        <div class="dim-score" style="color:{score_color(c_sc)}">{c_sc:.0f}</div>
        <div class="dim-weight">weight 0.15</div>
        <div class="dim-bar-wrap"><div class="dim-bar" style="width:{c_sc}%"></div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Methodology sub-scores
    method_detail = res.get("methodology_detail", {})
    if method_detail:
        def sub_cls(v):
            if v >= 0.9: return "sub-full"
            if v >= 0.4: return "sub-half"
            return "sub-zero"
        def sub_sym(v):
            if v >= 0.9: return "1.0"
            if v >= 0.4: return "0.5"
            return "0.0"

        arch  = method_detail.get("architecture_match", 0)
        opt   = method_detail.get("optimizer_match", 0)
        loss  = method_detail.get("loss_match", 0)
        hp    = method_detail.get("hyperparams_present", 0)
        contrib = method_detail.get("contribution_implemented", 0)
        reasoning = method_detail.get("reasoning", "")

        st.markdown(f"""
        <p class="card-label" style="margin-top:0.8rem">Methodology Sub-Scores</p>
        <div class="sub-grid">
          <div class="sub-card">
            <div class="sub-label">Architecture</div>
            <div class="sub-val {sub_cls(arch)}">{sub_sym(arch)}</div>
          </div>
          <div class="sub-card">
            <div class="sub-label">Optimizer</div>
            <div class="sub-val {sub_cls(opt)}">{sub_sym(opt)}</div>
          </div>
          <div class="sub-card">
            <div class="sub-label">Loss Fn</div>
            <div class="sub-val {sub_cls(loss)}">{sub_sym(loss)}</div>
          </div>
          <div class="sub-card">
            <div class="sub-label">Hyperparams</div>
            <div class="sub-val {sub_cls(hp)}">{sub_sym(hp)}</div>
          </div>
          <div class="sub-card">
            <div class="sub-label">Contribution</div>
            <div class="sub-val {sub_cls(contrib)}">{sub_sym(contrib)}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if reasoning:
            st.markdown(f'<p style="font-size:0.73rem;color:var(--text3);font-style:italic;margin-bottom:0.5rem">"{reasoning}"</p>', unsafe_allow_html=True)

    # Completeness reason
    comp_reason = res.get("completeness_reason", "")
    if comp_reason:
        st.markdown(f'<p style="font-size:0.73rem;color:var(--text3);margin-top:0.3rem"><b style="color:var(--text2)">Completeness:</b> {comp_reason}</p>', unsafe_allow_html=True)

    # Weighted formula display
    final = res.get("final_score", res.get("reproducibility_score", 0))
    st.markdown(f"""
    <div class="weight-formula">
      Final = <span>0.20</span>×{e_sc:.0f} + <span>0.35</span>×{m_sc:.0f} + <span>0.30</span>×{r_sc:.0f} + <span>0.15</span>×{c_sc:.0f} = <span>{final}</span> / 100
    </div>
    """, unsafe_allow_html=True)

# ── Session defaults ──────────────────────────────────────────────────────────
for k, v in [("stage","upload"),("paper_id",""),("text",""),("detection",None),("rag",""),("code",""),("result",None),("ep_accs",[]),("title",""),("stdout",""),("stderr",""),("pipe",0),("domain",""),("fast_mode",True)]:
    if k not in st.session_state: st.session_state[k] = v

# ── TOPBAR ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="tb-left">
    <div class="tb-logo">CT</div>
    <span class="tb-name">Cogniterra</span>
    <span class="tb-tag">v2.0</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── GRID ──────────────────────────────────────────────────────────────────────
sb_col, main_col = st.columns([1, 3.2], gap="small")

with sb_col:
    render_sidebar()

with main_col:
    st.markdown('<div class="main">', unsafe_allow_html=True)

    # ── PAGE 1 — UPLOAD ───────────────────────────────────────────────────────
    if st.session_state.stage == "upload":
        render_pipe(0)
        st.markdown('<div class="eyebrow">Step 1 of 4</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-title">Upload a Research Paper</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-desc">Enter an ArXiv ID or upload a PDF -- Cogniterra handles the rest</div>', unsafe_allow_html=True)

        col_l, col_r = st.columns([3,2], gap="large")
        with col_l:
            with st.container(border=True):
                uploaded = st.file_uploader("📎  Drop PDF here or click to browse", type=["pdf"], label_visibility="visible")
                st.markdown('<div style="display:flex;align-items:center;gap:12px;margin:10px 0"><div style="flex:1;height:1px;background:var(--border)"></div><span style="font-size:0.68rem;color:var(--text3);font-weight:600;letter-spacing:0.06em">OR</span><div style="flex:1;height:1px;background:var(--border)"></div></div>', unsafe_allow_html=True)
                arxiv = st.text_input("ArXiv Paper ID", placeholder="e.g.  1512.03385  (ResNet)", value=st.session_state.paper_id, label_visibility="visible")
                st.markdown('<p style="font-size:0.65rem;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:0.1em;margin:12px 0 8px">Quick Papers</p>', unsafe_allow_html=True)
                qc = st.columns(5)
                for i, (pid, lbl) in enumerate(QUICK_PAPERS):
                    with qc[i]:
                        if st.button(pid, key=f"q{i}", use_container_width=True, type="secondary"):
                            st.session_state.paper_id = pid; st.rerun()
            c1, c2 = st.columns([4,1], gap="small")
            with c1: start_btn = st.button("Start Analysis ->", use_container_width=True, type="primary")
            with c2:
                if st.button("Clear", use_container_width=True, type="secondary"): reset_all(); st.rerun()

        with col_r:
            with st.container(border=True):
                st.markdown('<p class="card-label">How Cogniterra Works</p>', unsafe_allow_html=True)
                for n, t in [("1","Enter an ArXiv ID or upload a PDF"),("2","Engine fetches & parses the paper"),("3","PyTorch code generated via Groq LLM"),("4","Experiment runs & results are verified")]:
                    st.markdown(f'<div class="hw"><div class="hw-n">{n}</div><div class="hw-t">{t}</div></div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('<p class="card-label">Supported Domains</p>', unsafe_allow_html=True)
                for tag in ["Image Classification","NLP & Transformers","Reinforcement Learning","Graph Neural Networks","Generative Models","Recommendation Systems"]:
                    st.markdown(f'<div class="row"><span class="row-label">{tag}</span><span style="color:var(--success);font-size:0.78rem;font-weight:700">✓</span></div>', unsafe_allow_html=True)

        if start_btn:
            pid = arxiv.strip() if arxiv.strip() else st.session_state.paper_id
            if not pid and not uploaded: st.error("Please enter an ArXiv ID or upload a PDF.")
            elif uploaded: st.warning("PDF upload coming soon -- please use an ArXiv ID for now.")
            else:
                st.session_state.paper_id = pid; st.session_state.pipe = 1; st.session_state.stage = "analysis"; st.rerun()

    # ── PAGE 2 — ANALYSIS ─────────────────────────────────────────────────────
    elif st.session_state.stage == "analysis":
        render_pipe(st.session_state.pipe)
        if not st.session_state.text:
            st.info("📡  Fetching paper -- please wait...")
            prog = st.progress(10)
            try:
                # Upgrade 1: uses latex_ingestion.parse_paper
                text = parse_paper(st.session_state.paper_id)
                if not text.strip(): raise ValueError("No text extracted. Check the ArXiv ID.")
                source_type = get_last_source_type()
                prog.progress(40)
                st.session_state.text = text
                st.session_state.title = get_title(text)
                st.session_state.pipe = 2

                # FIX: pass paper_id to detect_domain for structure cache override
                detection = detect_domain(text, paper_id=st.session_state.paper_id)
                st.session_state.detection = detection

                df_early = get_code_domain(detection)
                st.session_state.domain = df_early
                prog.progress(75)
                st.session_state.rag = get_relevant_context(text, domain=df_early)
                prog.progress(100)
                st.session_state.source_type = source_type
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                if st.button("<- Go back"): st.session_state.stage = "upload"; st.rerun()
                st.stop()

        det  = st.session_state.detection or {}
        dom  = det.get("domain", "unknown"); conf = det.get("confidence", 0)
        kws  = det.get("matched_keywords", [])[:4]; ds = det.get("dataset", "MNIST")
        emoji = {"image_classification":"🖼","nlp":"📝","recommendation":"🎯","reinforcement_learning":"🤖","algorithm":"⚙️","graph":"🕸","generative":"🎨"}.get(dom, "📄")
        src_type = st.session_state.get("source_type", "unknown")
        src_emoji = {"latex": "📄 LaTeX", "pdf": "📋 PDF", "html": "🌐 HTML"}.get(src_type, "❓ Unknown")

        st.markdown('<div class="eyebrow">Step 2 of 4</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-title">Paper Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-desc">Extracted methodology, architecture and key parameters</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="paper-banner"><span style="font-size:1.1rem">📄</span><span class="paper-banner-title">{st.session_state.title}</span><span class="paper-banner-id">{st.session_state.paper_id}</span></div>', unsafe_allow_html=True)

        col_l, col_r = st.columns([3,2], gap="large")
        with col_l:
            with st.container(border=True):
                st.markdown('<p class="card-label">Detected Domain</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="domain-tag">{emoji}&nbsp; {dom.replace("_"," ").title()}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="conf-track"><div class="conf-fill" style="width:{conf}%"></div></div><p style="font-size:0.71rem;color:var(--text3);margin-top:5px">{conf}% confidence</p>', unsafe_allow_html=True)
                st.divider()
                st.markdown('<p class="card-label">Extracted Parameters</p>', unsafe_allow_html=True)
                for lbl, val in [("Source",src_emoji),("Dataset",ds.split("(")[0].strip()),("Key Terms",", ".join(kws) if kws else "Detected"),("Confidence",f"{conf}%"),("Paper ID",st.session_state.paper_id)]:
                    st.markdown(f'<div class="row"><span class="row-label">{lbl}</span><span class="row-value">{val}</span></div>', unsafe_allow_html=True)
        with col_r:
            with st.container(border=True):
                st.markdown('<p class="card-label">Sections Extracted</p>', unsafe_allow_html=True)
                for sec in ["Methodology","Architecture","Training Setup","Experiments","Results"]:
                    st.markdown(f'<div class="sec-row"><span class="sec-row-name">{sec}</span><span class="sec-row-tick">✓</span></div>', unsafe_allow_html=True)
                chars = len(st.session_state.text)
                st.markdown(f'<div style="margin-top:12px;background:var(--surf2);border-radius:8px;padding:10px 12px;border:1px solid var(--border)"><p style="font-size:0.62rem;color:var(--text3);font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:3px">Total Extracted</p><p style="font-size:0.95rem;font-weight:800;color:var(--accent);font-family:JetBrains Mono,monospace;margin:0">{chars:,} chars</p></div>', unsafe_allow_html=True)
            with st.expander("📖  Preview extracted text"):
                st.code(st.session_state.text[:1500]+"...", language=None)

        st.markdown("")
        c1, c2, c3, c4 = st.columns([2,2,1,1], gap="small")
        with c1: gen_btn = st.button("Generate Code ->", use_container_width=True, type="primary")
        with c2: override = st.selectbox("Override domain", "Auto-detect,ml,nlp,recommendation,rl,graph,algorithm".split(","), label_visibility="visible")
        with c3: st.session_state["fast_mode"] = st.checkbox("⚡ Fast Mode", value=True, help="5x faster: fewer epochs + dataset subset. ~2-5% accuracy drop.")
        with c4:
            if st.button("<- Back", use_container_width=True, type="secondary"): st.session_state.stage = "upload"; st.rerun()

        if gen_btn:
            with st.spinner("⚡  Generating PyTorch code via Groq..."):
                try:
                    st.session_state.pipe = 3
                    df = get_code_domain(det) if override == "Auto-detect" else override
                    st.session_state.domain = df
                    code = generate_code(st.session_state.rag, domain=df, paper_id=st.session_state.paper_id, fast_mode=st.session_state.get("fast_mode", True))
                    st.session_state.code = code; st.session_state.stage = "codegen"; st.rerun()
                except Exception as e: st.error(f"Code generation failed: {e}")

    # ── PAGE 3 — CODE GEN ─────────────────────────────────────────────────────
    elif st.session_state.stage == "codegen":
        render_pipe(3)
        code = st.session_state.code; dp = get_deps(code)
        st.markdown('<div class="eyebrow">Step 3 of 4</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-title">Generated Code</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-desc">PyTorch implementation derived from the paper methodology</div>', unsafe_allow_html=True)

        col_l, col_r = st.columns([3,2], gap="large")
        with col_l:
            st.markdown('<div class="code-hdr"><div class="dot-r"></div><div class="dot-y"></div><div class="dot-g"></div><span class="code-fn">solution.py -- generated by Cogniterra</span></div>', unsafe_allow_html=True)
            st.code(code, language="python")
        with col_r:
            with st.container(border=True):
                st.markdown('<p class="card-label">Dependencies</p>', unsafe_allow_html=True)
                for d in dp:
                    st.markdown(f'<div class="row"><span class="row-label">{d}</span><span style="color:var(--success);font-weight:700">✓</span></div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('<p class="card-label">Code Stats</p>', unsafe_allow_html=True)
                for lbl, val in [("Lines",str(code.count('\n'))),("Characters",f"{len(code):,}"),("Domain",st.session_state.get("domain","auto")),("Paper ID",st.session_state.paper_id)]:
                    st.markdown(f'<div class="row"><span class="row-label">{lbl}</span><span class="row-value" style="font-family:JetBrains Mono,monospace;font-size:0.76rem">{val}</span></div>', unsafe_allow_html=True)

        st.markdown("")
        c1, c2, c3 = st.columns([2,2,1], gap="small")
        with c1: st.download_button("Download .py", data=code, file_name=f"{st.session_state.paper_id}_solution.py", mime="text/plain", use_container_width=True)
        with c2: verify_btn = st.button("Run & Verify ->", use_container_width=True, type="primary")
        with c3:
            if st.button("<- Back", use_container_width=True, type="secondary"): st.session_state.stage = "analysis"; st.rerun()

        if verify_btn:
            pid = st.session_state.paper_id
            os.makedirs(CODE_DIR, exist_ok=True)
            code_path = os.path.join(CODE_DIR, f"{pid}_solution.py")
            with open(code_path, "w", encoding="utf-8") as f2: f2.write(code)
            st.session_state.pipe = 4
            prog = st.progress(10)
            st.info("🐳  Running in Docker -- first run takes 5-10 min to build image...")
            try:
                from utils.docker_helper import run_code_in_docker
                prog.progress(20)
                success, stdout, stderr = run_code_in_docker(code, pid)
                prog.progress(70)

                gp = os.path.join(GOLDEN_DIR, f"{pid}_expected.json")
                if not os.path.exists(gp):
                    os.makedirs(GOLDEN_DIR, exist_ok=True)
                    gp = os.path.join(GOLDEN_DIR, f"{pid}_temp.json")
                    with open(gp, "w") as gf:
                        json.dump({"expected_accuracy": None, "tolerance": 5.0}, gf)

                # Upgrade 2: pass filtered_text and code for methodology + completeness scoring
                res = run_test(
                    pid, stdout, stderr, gp,
                    filtered_text=st.session_state.text,
                    code=st.session_state.code,
                )
                prog.progress(95)
                st.session_state.ep_accs = get_epoch_accs(stdout)
                st.session_state.result  = res
                st.session_state.stdout  = stdout
                st.session_state.stderr  = stderr
                save_vp(pid, res["reproducibility_score"], res["status"])
                prog.progress(100)
                st.session_state.stage = "results"; st.rerun()
            except Exception as e: st.error(f"Execution error: {e}")

    # ── PAGE 4 — RESULTS ──────────────────────────────────────────────────────
    elif st.session_state.stage == "results":
        render_pipe(5)
        res    = st.session_state.result; ep = st.session_state.ep_accs
        score  = res["reproducibility_score"]; actual = res.get("actual_accuracy")
        expect = res.get("expected_accuracy"); diff = res.get("difference")
        status = res.get("status",""); tol = res.get("tolerance", 2.0)
        if "PASS" in status:   scls, bcls, bi = "", "bdg-pass", "✓  Reproduced"
        elif "FAIL" in status: scls, bcls, bi = "fail", "bdg-fail", "✗  Failed"
        else:                  scls, bcls, bi = "warn", "bdg-warn", "⚠  Partial"
        mcls = "pass" if "PASS" in status else ("fail" if "FAIL" in status else "neu")

        st.markdown('<div class="eyebrow">Step 4 of 4</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-title">Verification Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-desc">Experiment summary and reproducibility metrics</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="paper-banner"><span style="font-size:1.1rem">🔬</span><span class="paper-banner-title">{st.session_state.title}</span><span class="paper-banner-id">{st.session_state.paper_id}</span></div>', unsafe_allow_html=True)

        # ── Score + metric cards ──────────────────────────────────────────────
        col_sc, col_mx = st.columns([1,3], gap="large")
        with col_sc:
            with st.container(border=True):
                st.markdown(f'<div class="score-wrap"><div class="score-num {scls}" id="ct-score">0</div><div class="score-lbl">Reproducibility Score / 100</div><div><span class="score-badge {bcls}">{bi}</span></div></div>', unsafe_allow_html=True)
                st.components.v1.html(f"""<script>
                var t={int(score)},c=0,el=window.parent.document.getElementById('ct-score');
                var iv=setInterval(function(){{c+=Math.ceil((t-c)/8);if(c>=t){{c=t;clearInterval(iv);}}if(el)el.textContent=c;}},28);
                </script>""", height=0)
        with col_mx:
            es   = f"{expect:.1f}" if expect is not None else "N/A"
            acs  = f"{actual:.1f}" if actual is not None else "N/A"
            ds_  = f"{diff:.2f}"   if diff   is not None else "N/A"
            ecol = "var(--error)" if res.get("has_errors") else "var(--success)"
            et   = "Yes" if res.get("has_errors") else "None"
            st.markdown(f'<div class="mg"><div class="mc {mcls}"><div class="mc-lbl">Expected</div><div class="mc-val">{es}<span class="mc-unt">%</span></div></div><div class="mc {mcls}"><div class="mc-lbl">Achieved</div><div class="mc-val">{acs}<span class="mc-unt">%</span></div></div><div class="mc neu"><div class="mc-lbl">Difference</div><div class="mc-val">{ds_}<span class="mc-unt">%</span></div></div><div class="mc neu"><div class="mc-lbl">Tolerance</div><div class="mc-val">±{tol}<span class="mc-unt">%</span></div></div><div class="mc neu"><div class="mc-lbl">Epochs</div><div class="mc-val">{len(ep) if ep else 5}</div></div><div class="mc neu"><div class="mc-lbl">Errors</div><div class="mc-val" style="font-size:1rem;color:{ecol}">{et}</div></div></div>', unsafe_allow_html=True)

        st.markdown("")

        # ── 4-Dimension scoring breakdown ─────────────────────────────────────
        with st.container(border=True):
            st.markdown('<p class="card-label">4-Dimension Scoring Breakdown</p>', unsafe_allow_html=True)
            render_dimension_scores(res)

        st.markdown("")

        # ── Training curve + epoch table ──────────────────────────────────────
        col_l, col_r = st.columns([1,1], gap="large")
        with col_l:
            with st.container(border=True):
                st.markdown('<p class="card-label">Per-Epoch Accuracy</p>', unsafe_allow_html=True)
                if ep:
                    rows = ""
                    for e, a in ep:
                        last = (e == ep[-1][0]); cls = ' class="et-final"' if last else ""
                        badge = '<span class="final-tag">final</span>' if last else ""
                        rows += f'<tr{cls}><td>{e}</td><td>{a:.2f}%</td><td>{badge}</td></tr>'
                    st.markdown(f'<table class="et"><thead><tr><th>Epoch</th><th>Accuracy</th><th></th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="font-size:0.78rem;color:var(--text3)">No per-epoch data captured.</p>', unsafe_allow_html=True)
        with col_r:
            with st.container(border=True):
                st.markdown('<p class="card-label">Accuracy Curve</p>', unsafe_allow_html=True)
                if ep and len(ep) >= 2:
                    df = pd.DataFrame({"Epoch":[e for e,_ in ep],"Accuracy (%)":[a for _,a in ep]}).set_index("Epoch")
                    st.line_chart(df, use_container_width=True, height=195)
                elif actual is not None:
                    df = pd.DataFrame({"Epoch":[1,2,3,4,5],"Accuracy (%)":[round(actual*.72,2),round(actual*.83,2),round(actual*.90,2),round(actual*.96,2),round(actual,2)]}).set_index("Epoch")
                    st.line_chart(df, use_container_width=True, height=195)
                    st.markdown('<p style="font-size:0.63rem;color:var(--text3)">Estimated curve from final accuracy</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="font-size:0.78rem;color:var(--text3)">No data to plot.</p>', unsafe_allow_html=True)

        if st.session_state.stdout:
            with st.expander("📋  Training logs"): st.code(st.session_state.stdout, language=None)
        if st.session_state.stderr:
            with st.expander("⚠️  Errors / warnings"): st.code(st.session_state.stderr, language=None)

        st.markdown("")
        c1, c2 = st.columns([1,1], gap="large")
        with c1:
            cp = os.path.join(CODE_DIR, f"{st.session_state.paper_id}_solution.py")
            try:
                rp = generate_html_report(res, cp)
                with open(rp, "r", encoding="utf-8") as f: hc = f.read()
                st.download_button("Download Report", data=hc, file_name=f"{st.session_state.paper_id}_report.html", mime="text/html", use_container_width=True)
            except: st.button("Report unavailable", disabled=True, use_container_width=True)
        with c2:
            if st.button("<- Back to Code", use_container_width=True, type="secondary"): st.session_state.stage = "codegen"; st.rerun()

        st.markdown('<div class="cta-box"><div class="cta-title">Ready to replicate another paper?</div><div class="cta-sub">Start fresh with a new ArXiv ID or PDF upload</div></div>', unsafe_allow_html=True)
        _, cc, _ = st.columns([1,2,1])
        with cc:
            if st.button("New Paper ->", use_container_width=True, type="primary", key="new_upload"): reset_all(); st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)