"""
상명대학교 문서 AI 비서 - Streamlit 웹 앱
실행: streamlit run app.py
"""

import streamlit as st
import os
import re
import sys
import base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from agents.scanner import FileScanner
from agents.parser import parse_file
from agents.embedder import EmbeddingAgent
from agents.searcher import SemanticSearchAgent
from agents.master import MasterAgent, ClaudeMasterAgent, OpenAIMasterAgent, GeminiMasterAgent
from utils.vector_db import VectorDB

# 경로 상수
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "index",    "faiss.index")
META_PATH = os.path.join(DATA_DIR, "index",    "metadata.json")
STATE_PATH = os.path.join(DATA_DIR, "metadata", "scan_state.json")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
LOGO_PATH = os.path.join(BASE_DIR, "assets",   "smu_logo.svg")
os.makedirs(UPLOAD_DIR, exist_ok=True)

EMBEDDING_DIM = 768
ENV_PATH = os.path.join(BASE_DIR, ".env")


def _save_env_key(key: str, value: str):
    """키=값을 .env 파일에 저장 (없으면 추가, 있으면 덮어씀)"""
    lines = []
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
    updated = False
    for i, line in enumerate(lines):
        if re.match(rf"^{key}\s*=", line):
            lines[i] = f"{key}={value}\n"
            updated = True
            break
    if not updated:
        lines.append(f"{key}={value}\n")
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

# .env 자동 로드
from dotenv import load_dotenv
load_dotenv(ENV_PATH, override=False)

NAVY = "#1B3A6B"
BLUE = "#2563EB"
LBLUE = "#3B82F6"
XBLUE = "#EFF6FF"

def _logo_b64() -> str:
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

LOGO_B64 = _logo_b64()

st.set_page_config(
    page_title="문서 AI 비서 | 상명대학교",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap');

* {{ font-family: 'Noto Sans KR', sans-serif; }}

.stApp {{ background-color: #EEF2FA; }}

/* ── 사이드바 ───────────────────────── */
[data-testid="stSidebar"] > div:first-child {{
    background: linear-gradient(180deg, {NAVY} 0%, #0F2447 100%);
    padding-top: 0 !important;
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown {{
    color: rgba(255,255,255,0.90) !important;
}}
[data-testid="stSidebar"] .stCaption {{
    color: rgba(255,255,255,0.55) !important;
}}
[data-testid="stSidebar"] input {{
    background: rgba(255,255,255,0.10) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    border-radius: 6px !important;
}}
[data-testid="stSidebar"] input::placeholder {{
    color: rgba(255,255,255,0.35) !important;
}}
[data-testid="stSidebar"] [data-baseweb="input"] {{
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    border-radius: 6px !important;
}}
[data-testid="stSidebar"] [data-baseweb="input"] input {{
    background: transparent !important;
    color: white !important;
}}
[data-testid="stSidebar"] [data-baseweb="base-input"] {{
    background: rgba(255,255,255,0.10) !important;
}}
[data-testid="stSidebar"] [data-baseweb="base-input"] input {{
    color: white !important;
    background: transparent !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    color: white !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] svg {{ fill: white !important; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{ color: rgba(255,255,255,0.85) !important; }}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {{ color: rgba(255,255,255,0.85) !important; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.15) !important; }}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {{
    color: white !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {{
    color: rgba(255,255,255,0.65) !important;
    font-size: 0.78rem !important;
}}
[data-testid="stSidebar"] [data-testid="stMetric"] {{
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 8px 12px;
}}

/* 사이드바 버튼 */
[data-testid="stSidebar"] .stButton button {{
    background: rgba(255,255,255,0.10) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
    font-size: 0.85em !important;
    transition: all 0.18s ease;
    letter-spacing: 0.01em;
}}
[data-testid="stSidebar"] .stButton button:hover {{
    background: rgba(255,255,255,0.20) !important;
    border-color: rgba(255,255,255,0.40) !important;
}}
[data-testid="stSidebar"] .stButton button[kind="primary"] {{
    background: {BLUE} !important;
    border: none !important;
    font-weight: 600 !important;
}}
[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {{
    background: #1D4ED8 !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.4) !important;
}}

/* ── 메인 버튼 ──────────────────────── */
.stButton button[kind="primary"] {{
    background: linear-gradient(135deg, {NAVY}, {BLUE}) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.18s ease;
    box-shadow: 0 2px 8px rgba(27,58,107,0.22);
}}
.stButton button[kind="primary"]:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(27,58,107,0.32) !important;
}}

/* ── 채팅 입력 ──────────────────────── */
[data-testid="stChatInput"] textarea {{
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 10px !important;
    background: white !important;
    font-size: 0.95em !important;
}}
[data-testid="stChatInput"] textarea:focus {{
    border-color: {LBLUE} !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}}
[data-testid="stChatInputSubmitButton"] button {{
    background: {NAVY} !important;
    border-radius: 7px !important;
}}

/* ── 채팅 메시지 ────────────────────── */
[data-testid="stChatMessage"] {{
    background: white;
    border-radius: 12px;
    box-shadow: 0 1px 6px rgba(27,58,107,0.07);
    margin: 4px 0;
}}

/* ── Expander ───────────────────────── */
[data-testid="stExpander"] {{
    background: white;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px;
    overflow: hidden;
}}
[data-testid="stExpander"] summary {{
    color: {NAVY} !important;
    font-weight: 600;
    font-size: 0.9em;
}}

/* ── 출처/근거 카드 ─────────────────── */
.source-card {{
    background: white;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 8px 0;
    border-left: 4px solid {NAVY};
    box-shadow: 0 1px 6px rgba(27,58,107,0.07);
    transition: box-shadow 0.18s ease, transform 0.18s ease;
}}
.source-card:hover {{
    box-shadow: 0 4px 16px rgba(27,58,107,0.14);
    transform: translateY(-1px);
}}
.passage-card {{
    background: {XBLUE};
    border-radius: 6px;
    padding: 10px 14px;
    margin: 4px 0 6px;
    font-size: 0.875em;
    color: #1E3A5F;
    border-left: 3px solid {LBLUE};
    line-height: 1.75;
}}
.passage-label {{
    font-size: 0.70em;
    font-weight: 700;
    letter-spacing: 0.07em;
    color: {LBLUE};
    text-transform: uppercase;
    margin-bottom: 5px;
}}
.badge {{
    background: {XBLUE};
    color: {BLUE};
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.76em;
    font-weight: 700;
    letter-spacing: 0.03em;
    white-space: nowrap;
}}
.meta-text {{
    font-size: 0.79em;
    color: #94A3B8;
    margin-top: 4px;
    word-break: break-all;
    overflow-wrap: break-word;
}}

/* ── 사이드바 Expander - 어두운 배경으로 ── */
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
    color: rgba(255,255,255,0.82) !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] p,
[data-testid="stSidebar"] [data-testid="stExpander"] span,
[data-testid="stSidebar"] [data-testid="stExpander"] div {{
    color: rgba(255,255,255,0.80) !important;
}}

/* ── 사이드바 파일 업로더 ─────────────── */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {{
    background: rgba(255,255,255,0.07) !important;
    border: 1.5px dashed rgba(255,255,255,0.25) !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section button {{
    color: white !important;
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.28) !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {{
    color: rgba(255,255,255,0.60) !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {{
    background: rgba(255,255,255,0.10) !important;
    border-color: rgba(255,255,255,0.20) !important;
}}

/* ── 사이드바 Alert ───────────────────── */
[data-testid="stSidebar"] [data-testid="stAlert"] p,
[data-testid="stSidebar"] [data-testid="stAlert"] div,
[data-testid="stSidebar"] [data-testid="stAlert"] span {{
    color: #1E293B !important;
}}

/* ── Alert ──────────────────────────── */
[data-testid="stAlert"] {{ border-radius: 9px !important; }}

/* ── Progress ───────────────────────── */
[data-testid="stProgressBar"] > div {{
    background: linear-gradient(90deg, {NAVY}, {BLUE}) !important;
    border-radius: 4px;
}}

/* ── Streamlit 기본 실행 애니메이션 제거 ── */
[data-testid="stStatusWidget"] {{ display: none !important; }}
#MainMenu {{ visibility: hidden; }}
header {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

/* ── Scrollbar ──────────────────────── */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: #E8EEF9; }}
::-webkit-scrollbar-thumb {{ background: #93B4D8; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {NAVY}; }}

/* ── 섹션 레이블 ────────────────────── */
.section-label {{
    font-size: 0.72em;
    font-weight: 700;
    letter-spacing: 0.10em;
    color: rgba(255,255,255,0.50);
    text-transform: uppercase;
    margin: 16px 0 8px;
}}
</style>
""", unsafe_allow_html=True)


# 스플래시 UI  (체크리스트 방식)
_SPLASH_STEPS = [
    "데이터베이스 연결",
    "임베딩 에이전트 준비",
    "AI 모델 로딩",
    "검색 엔진 초기화",
    "파일 스캐너 초기화",
]

def _splash_html(current: int) -> str:
    logo_tag = (
        f'<img style="height:62px;filter:brightness(0) invert(1);margin-bottom:22px;" src="data:image/svg+xml;base64,{LOGO_B64}" />'
        if LOGO_B64 else
        '<div style="font-size:2.2em;font-weight:900;color:white;margin-bottom:22px;">SMU</div>'
    )
    bar_pct = int(current / len(_SPLASH_STEPS) * 100)

    rows = ""
    for i, label in enumerate(_SPLASH_STEPS):
        if i < current:
            icon   = '<span style="color:rgba(255,255,255,0.30);">✓</span>'
            color  = "rgba(255,255,255,0.28)"
            weight = "400"
        elif i == current:
            icon   = '<span style="display:inline-block;animation:spin 0.9s linear infinite;">⟳</span>'
            color  = "white"
            weight = "700"
        else:
            icon   = '<span style="color:rgba(255,255,255,0.75);">○</span>'
            color  = "rgba(255,255,255,0.80)"
            weight = "500"
        rows += f'<div style="display:flex;align-items:center;gap:14px;padding:9px 0;border-bottom:1px solid rgba(255,255,255,0.06);"><div style="width:22px;text-align:center;">{icon}</div><div style="font-size:0.88em;font-weight:{weight};color:{color};">{label}</div></div>'

    done_msg = '<div style="color:#4ADE80;font-weight:700;font-size:0.88em;margin-top:14px;">준비 완료!</div>' if current >= len(_SPLASH_STEPS) else ""

    return (
        f'<style>'
        f'@keyframes fadeIn{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:translateY(0)}}}}'
        f'@keyframes spin{{from{{transform:rotate(0deg)}}to{{transform:rotate(360deg)}}}}'
        f'#spl{{position:fixed;inset:0;background:linear-gradient(145deg,{NAVY} 0%,#0F2447 55%,#0D3280 100%);z-index:99999;display:flex;flex-direction:column;align-items:center;justify-content:center;font-family:"Noto Sans KR",sans-serif;}}'
        f'</style>'
        f'<div id="spl">'
        f'{logo_tag}'
        f'<div style="font-size:1.85em;font-weight:900;color:white;letter-spacing:-0.01em;animation:fadeIn 0.4s ease both;">문서 AI 비서</div>'
        f'<div style="font-size:0.78em;color:rgba(255,255,255,0.45);margin-top:6px;letter-spacing:0.08em;animation:fadeIn 0.4s ease 0.1s both;">DOCUMENT AI ASSISTANT</div>'
        f'<div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.10);border-radius:16px;padding:22px 32px;min-width:300px;margin-top:32px;animation:fadeIn 0.5s ease 0.2s both;">'
        f'{rows}'
        f'{done_msg}'
        f'<div style="width:100%;height:4px;background:rgba(255,255,255,0.12);border-radius:2px;margin-top:18px;overflow:hidden;">'
        f'<div style="height:100%;border-radius:2px;background:linear-gradient(90deg,{LBLUE},{BLUE});width:{bar_pct}%;transition:width 0.4s ease;"></div>'
        f'</div>'
        f'</div>'
        f'<div style="position:fixed;bottom:30px;text-align:center;animation:fadeIn 0.5s ease 0.3s both;">'
        f'<div style="font-size:0.86em;font-weight:700;color:rgba(255,255,255,0.70);">이선역</div>'
        f'<div style="font-size:0.68em;color:rgba(255,255,255,0.30);margin-top:3px;letter-spacing:0.06em;">상명대학교 휴먼AI공학전공</div>'
        f'</div>'
        f'</div>'
    )


# 에이전트 초기화 (단계별 진행)
_splash = st.empty()

import time

if "agents_ready" not in st.session_state:
    _splash.markdown(_splash_html(0), unsafe_allow_html=True)
    db = VectorDB(INDEX_PATH, META_PATH, dim=EMBEDDING_DIM)
    time.sleep(0.5)

    _splash.markdown(_splash_html(1), unsafe_allow_html=True)
    embedder = EmbeddingAgent(db)
    time.sleep(0.5)

    _splash.markdown(_splash_html(2), unsafe_allow_html=True)
    embedder.warmup()                  # 실제로 오래 걸림 → sleep 불필요

    _splash.markdown(_splash_html(3), unsafe_allow_html=True)
    searcher = SemanticSearchAgent(embedder, db)
    time.sleep(0.5)

    _splash.markdown(_splash_html(4), unsafe_allow_html=True)
    scanner = FileScanner(STATE_PATH)
    if db.total == 0 and scanner.state:
        scanner.state = {}
        scanner._save_state()
    time.sleep(0.5)

    _splash.markdown(_splash_html(5), unsafe_allow_html=True)
    time.sleep(0.8)

    st.session_state.agents_ready = True
    st.session_state._db      = db
    st.session_state._embedder = embedder
    st.session_state._searcher = searcher
    st.session_state._scanner  = scanner
else:
    db       = st.session_state._db
    embedder = st.session_state._embedder
    searcher = st.session_state._searcher
    scanner  = st.session_state._scanner

_splash.empty()

if "messages"   not in st.session_state: st.session_state.messages   = []
if "master"     not in st.session_state: st.session_state.master     = None
if "master_key" not in st.session_state: st.session_state.master_key = None


# 색인 함수
def run_indexing(folder_path: str, force: bool = False) -> str:
    folder = os.path.normpath(folder_path)
    if not os.path.isdir(folder):
        return f"폴더를 찾을 수 없습니다: {folder}"
    files, deleted = (scanner.full_scan(folder), []) if force else scanner.scan(folder)
    for fp in deleted:
        db.remove_by_file(fp)
    if not files:
        db.save()
        return "변경된 파일이 없습니다. (이미 최신 상태)"
    prog = st.sidebar.progress(0, text="색인 준비 중...")
    total_chunks = 0
    for i, fi in enumerate(files):
        prog.progress((i + 1) / len(files), text=f"({i+1}/{len(files)}) {fi['file_name'][:28]}")
        db.remove_by_file(fi["file_path"])
        total_chunks += embedder.index_file(fi, parse_file(fi))
    db.save()
    prog.empty()
    return f"{len(files)}개 파일 · {total_chunks}개 청크 색인 완료 (총 {db.total:,}개 벡터)"


def simple_indexer_fn(folder_path: str, force: bool = False) -> str:
    folder = os.path.normpath(folder_path)
    if not os.path.isdir(folder):
        return f"폴더를 찾을 수 없습니다: {folder}"
    files, deleted = (scanner.full_scan(folder), []) if force else scanner.scan(folder)
    for fp in deleted:
        db.remove_by_file(fp)
    if not files:
        db.save()
        return "변경된 파일이 없습니다."
    total_chunks = 0
    for fi in files:
        db.remove_by_file(fi["file_path"])
        total_chunks += embedder.index_file(fi, parse_file(fi))
    db.save()
    return f"{len(files)}개 파일, {total_chunks}개 청크 색인 완료 (총 {db.total:,}개 벡터)"


def get_master(backend: str, model: str):
    key = f"{backend}::{model}"
    if st.session_state.master_key != key:
        kwargs = dict(searcher=searcher, indexer_fn=simple_indexer_fn, model=model)
        if   backend == "claude": agent = ClaudeMasterAgent(**kwargs)
        elif backend == "openai": agent = OpenAIMasterAgent(**kwargs)
        elif backend == "gemini": agent = GeminiMasterAgent(**kwargs)
        else:                     agent = MasterAgent(**kwargs)
        st.session_state.master     = agent
        st.session_state.master_key = key
    return st.session_state.master


# 출처 표시
_EXT_ICONS = {
    "pdf": "📄", "docx": "📝", "doc": "📝",
    "pptx": "📊", "ppt": "📊",
    "xlsx": "📗", "xls": "📗",
    "txt": "📃", "md": "📃",
}

def _file_icon(src: dict) -> str:
    if src.get("file_path", "").startswith("https://www.notion.so"):
        return "🔗"
    ext = src.get("extension", "").lower().lstrip(".")
    return _EXT_ICONS.get(ext, "📁")


def show_sources(sources: list):
    if not sources:
        return
    with st.expander(f"📎 참조 문서 {len(sources)}건", expanded=False):
        for i, src in enumerate(sources, 1):
            score_pct = min(int(src.get("relevance_score", 0) * 100), 100)
            is_notion = src.get("file_path", "").startswith("https://www.notion.so")
            icon      = _file_icon(src)
            link_html = (
                f'<a href="{src["file_path"]}" target="_blank" '
                f'style="color:{BLUE};font-size:0.78em;text-decoration:none;white-space:nowrap;">열기 ↗</a>'
            ) if is_notion else ""
            st.markdown(f"""
<div class="source-card">
  <div style="display:flex;gap:12px;align-items:flex-start;">
    <div style="font-size:1.3em;line-height:1.2;flex-shrink:0;margin-top:1px;">{icon}</div>
    <div style="flex:1;min-width:0;">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px;">
        <div style="font-weight:700;color:{NAVY};font-size:0.92em;min-width:0;word-break:break-word;">
          {src['file_name']}&nbsp;{link_html}
        </div>
        <span class="badge">관련도 {score_pct}%</span>
      </div>
      <div style="height:3px;background:#E2E8F0;border-radius:2px;margin:0 0 6px;overflow:hidden;">
        <div style="height:100%;width:{score_pct}%;background:linear-gradient(90deg,{LBLUE},{BLUE});border-radius:2px;"></div>
      </div>
      <div class="meta-text">{src['file_path']}&nbsp;·&nbsp;{src.get('modified_at', '-')}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
            for j, passage in enumerate(src.get("key_passages", [])[:3], 1):
                st.markdown(f"""
<div class="passage-card">
  <div class="passage-label">근거 {j}</div>
  {passage[:500]}{'…' if len(passage)>500 else ''}
</div>""", unsafe_allow_html=True)


# 사이드바
with st.sidebar:

    # ── 로고 헤더 ─────────────────────────────────────────────
    logo_img = (
        f'<img src="data:image/svg+xml;base64,{LOGO_B64}" '
        f'style="height:40px;filter:brightness(0) invert(1);" />'
        if LOGO_B64 else
        f'<span style="font-size:1.6em;font-weight:900;color:white;letter-spacing:0.06em;">SMU</span>'
    )
    st.markdown(f"""
<div style="padding:22px 20px 16px;border-bottom:1px solid rgba(255,255,255,0.12);">
  {logo_img}
  <div style="margin-top:10px;">
    <div style="font-size:0.98em;font-weight:700;color:white;line-height:1.3;">
      문서 AI 비서
    </div>
    <div style="font-size:0.70em;color:rgba(255,255,255,0.45);margin-top:3px;letter-spacing:0.05em;">
      상명대학교 휴먼AI공학전공
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── LLM 설정 ──────────────────────────────────────────────
    st.markdown('<div class="section-label">LLM 설정</div>', unsafe_allow_html=True)
    backend_label = st.radio("백엔드", [
        "Ollama (로컬)", "Claude API", "OpenAI GPT", "Google Gemini",
    ], label_visibility="collapsed")

    if "Ollama" in backend_label:
        backend_key = "ollama"
        try:
            import ollama as _ollama
            available_models = [m.model for m in _ollama.Client().list().models]
        except Exception:
            available_models = []
        if available_models:
            default_idx = next((i for i, m in enumerate(available_models) if "qwen2.5:14b" in m), 0)
            model_choice = st.selectbox("모델", available_models, index=default_idx)
        else:
            model_choice = st.text_input("모델명", "qwen2.5:14b")
            st.warning("Ollama 연결 실패 — ollama serve 확인")

    elif "Claude" in backend_label:
        backend_key  = "claude"
        model_choice = st.selectbox("모델", ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"])
        v = st.text_input("Anthropic API Key", type="password", value=os.environ.get("ANTHROPIC_API_KEY", ""))
        if v: os.environ["ANTHROPIC_API_KEY"] = v
        elif not os.environ.get("ANTHROPIC_API_KEY"): st.warning("console.anthropic.com")

    elif "OpenAI" in backend_label:
        backend_key  = "openai"
        model_choice = st.selectbox("모델", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-mini"])
        v = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
        if v: os.environ["OPENAI_API_KEY"] = v
        elif not os.environ.get("OPENAI_API_KEY"): st.warning("platform.openai.com")

    else:
        backend_key  = "gemini"
        model_choice = st.selectbox("모델", ["gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-preview-05-06", "gemini-2.0-flash", "gemini-2.0-flash-lite"])
        v = st.text_input("Google API Key", type="password", value=os.environ.get("GOOGLE_API_KEY", ""))
        if v: os.environ["GOOGLE_API_KEY"] = v
        elif not os.environ.get("GOOGLE_API_KEY"): st.warning("aistudio.google.com (무료)")

    # ── 색인 현황 ─────────────────────────────────────────────
    st.markdown('<div class="section-label">색인 현황</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("벡터 수", f"{db.total:,}")
    c2.metric("파일 수", len(scanner.state))
    if scanner.state:
        with st.expander("파일 목록"):
            for fp in list(scanner.state.keys()):
                st.caption(os.path.basename(fp))

    # ── 폴더 색인 ─────────────────────────────────────────────
    st.markdown('<div class="section-label">폴더 색인</div>', unsafe_allow_html=True)
    folder_input = st.text_input("경로", placeholder=r"C:\Users\나\Documents", label_visibility="collapsed")
    force_check  = st.checkbox("강제 전체 재색인")
    if st.button("색인 시작", use_container_width=True, type="primary"):
        if not folder_input:
            st.warning("폴더 경로를 입력하세요")
        elif not os.path.isdir(folder_input):
            st.error("경로를 찾을 수 없습니다")
        else:
            with st.spinner("색인 중..."):
                st.success(run_indexing(folder_input, force=force_check))
            st.rerun()

    # ── 파일 업로드 ───────────────────────────────────────────
    st.markdown('<div class="section-label">파일 업로드</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "파일 선택", type=["pdf", "docx", "pptx", "txt"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    if uploaded_files and st.button("업로드 및 색인", use_container_width=True):
        for uf in uploaded_files:
            with open(os.path.join(UPLOAD_DIR, uf.name), "wb") as f:
                f.write(uf.getbuffer())
        with st.spinner("색인 중..."):
            st.success(run_indexing(UPLOAD_DIR, force=False))
        st.rerun()

    # ── Notion 연결 ───────────────────────────────────────────
    st.markdown('<div class="section-label">Notion 연결</div>', unsafe_allow_html=True)

    _notion_pages = sum(1 for m in db.metadata if m.get("file_path","").startswith("https://www.notion.so"))
    _notion_token_set = bool(os.environ.get("NOTION_TOKEN", ""))
    if _notion_token_set and _notion_pages > 0:
        st.markdown(f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;"><span style="width:8px;height:8px;border-radius:50%;background:#4ADE80;display:inline-block;"></span><span style="font-size:0.80em;color:rgba(255,255,255,0.75);">연결됨 · Notion {_notion_pages}개 청크 색인</span></div>', unsafe_allow_html=True)
    elif _notion_token_set:
        st.markdown('<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;"><span style="width:8px;height:8px;border-radius:50%;background:#FACC15;display:inline-block;"></span><span style="font-size:0.80em;color:rgba(255,255,255,0.75);">토큰 있음 · 미색인</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;"><span style="width:8px;height:8px;border-radius:50%;background:rgba(255,255,255,0.25);display:inline-block;"></span><span style="font-size:0.80em;color:rgba(255,255,255,0.40);">미연결</span></div>', unsafe_allow_html=True)

    notion_token = st.text_input(
        "Integration Token", type="password",
        value=os.environ.get("NOTION_TOKEN", ""),
        placeholder="secret_xxxxxxxxxxxx",
    )
    if notion_token:
        os.environ["NOTION_TOKEN"] = notion_token
        if st.button("토큰 저장", use_container_width=True, key="notion_save"):
            _save_env_key("NOTION_TOKEN", notion_token)
            st.success("저장 완료! 다음부터 자동으로 불러와요.")
        notion_force = st.checkbox("강제 재색인", key="notion_force")
        ca, cb = st.columns(2)
        notion_msg = st.empty()  # 결과 메시지 전용 플레이스홀더
        if ca.button("연결 확인", use_container_width=True, key="notion_verify"):
            try:
                from agents.notion_connector import NotionConnector
                ok, msg = NotionConnector(notion_token).verify_token()
                (notion_msg.success if ok else notion_msg.error)(msg)
            except Exception as e:
                notion_msg.error(str(e))
        if cb.button("Notion 색인", use_container_width=True, key="notion_index"):
            try:
                from agents.notion_connector import NotionConnector
                nc = NotionConnector(notion_token)
                with st.spinner("페이지 가져오는 중..."):
                    items, deleted = (nc.full_scan(), []) if notion_force else nc.scan()
                for url in deleted: db.remove_by_file(url)
                if not items:
                    notion_msg.info("변경된 페이지가 없습니다.")
                else:
                    prog = st.sidebar.progress(0)
                    total_chunks = 0
                    for i, (fi, text) in enumerate(items):
                        prog.progress((i+1)/len(items), text=fi["file_name"][:25])
                        db.remove_by_file(fi["file_path"])
                        total_chunks += embedder.index_file(fi, text)
                    db.save(); prog.empty()
                    notion_msg.success(f"{len(items)}개 페이지 색인 완료 (총 {db.total:,}개 벡터)")
                    st.rerun()
            except Exception as e:
                notion_msg.error(f"Notion 오류: {e}")
    else:
        st.caption("토큰 입력 시 활성화")

    # ── 대화 초기화 + 크레딧 ──────────────────────────────────
    st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.master: st.session_state.master.reset_history()
        st.rerun()

    st.markdown("""
<div style="margin-top:20px;padding:12px 16px;
            border-top:1px solid rgba(255,255,255,0.08);
            text-align:center;">
  <div style="font-size:0.82em;font-weight:600;color:rgba(255,255,255,0.70);">이선역</div>
  <div style="font-size:0.68em;color:rgba(255,255,255,0.30);margin-top:2px;">상명대학교 휴먼AI공학전공</div>
</div>
""", unsafe_allow_html=True)


# 메인 화면

# 헤더
logo_html = (
    f'<img src="data:image/svg+xml;base64,{LOGO_B64}" '
    f'style="height:48px;filter:brightness(0) invert(1);margin-right:18px;flex-shrink:0;" />'
    if LOGO_B64 else
    f'<span style="font-size:1.6em;font-weight:900;color:white;margin-right:14px;">SMU</span>'
)

st.markdown(f"""
<div style="
  background: linear-gradient(135deg, {NAVY} 0%, {BLUE} 100%);
  border-radius: 12px;
  padding: 20px 26px;
  margin-bottom: 18px;
  display: flex;
  align-items: center;
  box-shadow: 0 4px 18px rgba(27,58,107,0.22);
">
  {logo_html}
  <div style="flex:1;">
    <div style="font-size:1.35em;font-weight:800;color:white;letter-spacing:-0.01em;line-height:1.2;">
      문서 AI 비서
    </div>
    <div style="font-size:0.78em;color:rgba(255,255,255,0.68);margin-top:5px;line-height:1.5;">
      상명대학교 휴먼AI공학전공 &nbsp;·&nbsp;
      <span style="color:rgba(255,255,255,0.85);font-weight:600;">{model_choice}</span>
      &nbsp;·&nbsp; 벡터 <span style="color:rgba(255,255,255,0.85);font-weight:600;">{db.total:,}</span>개
    </div>
  </div>
  <div style="text-align:right;flex-shrink:0;margin-left:16px;">
    <div style="font-size:0.88em;font-weight:700;color:rgba(255,255,255,0.80);">이선역</div>
    <div style="font-size:0.66em;color:rgba(255,255,255,0.35);margin-top:2px;letter-spacing:0.04em;">휴먼AI공학전공</div>
  </div>
</div>
""", unsafe_allow_html=True)

# DB 비어 있을 때 안내
if db.total == 0:
    st.markdown(f"""
<div style="text-align:center;padding:40px 20px 24px;">
  <div style="font-size:2.6em;margin-bottom:16px;">📄</div>
  <div style="font-size:1.25em;font-weight:800;color:{NAVY};margin-bottom:8px;">문서를 먼저 추가해 주세요</div>
  <div style="font-size:0.88em;color:#64748B;max-width:420px;margin:0 auto;line-height:1.8;">
    왼쪽 사이드바에서 문서를 색인하면 AI가 내용을 분석하고 질문에 답변합니다.
  </div>
</div>
<div style="display:flex;gap:14px;max-width:560px;margin:0 auto 28px;flex-wrap:wrap;justify-content:center;">
  <div style="background:white;border-radius:12px;padding:18px 22px;flex:1;min-width:150px;
              box-shadow:0 1px 8px rgba(27,58,107,0.08);border-top:3px solid {BLUE};">
    <div style="font-size:1.4em;margin-bottom:8px;">🗂</div>
    <div style="font-weight:700;color:{NAVY};font-size:0.90em;margin-bottom:4px;">폴더 색인</div>
    <div style="color:#64748B;font-size:0.80em;line-height:1.6;">로컬 폴더를 경로로<br>등록하면 자동 스캔</div>
  </div>
  <div style="background:white;border-radius:12px;padding:18px 22px;flex:1;min-width:150px;
              box-shadow:0 1px 8px rgba(27,58,107,0.08);border-top:3px solid {BLUE};">
    <div style="font-size:1.4em;margin-bottom:8px;">📤</div>
    <div style="font-weight:700;color:{NAVY};font-size:0.90em;margin-bottom:4px;">파일 업로드</div>
    <div style="color:#64748B;font-size:0.80em;line-height:1.6;">PDF · DOCX · PPTX · TXT<br>파일 직접 업로드</div>
  </div>
  <div style="background:white;border-radius:12px;padding:18px 22px;flex:1;min-width:150px;
              box-shadow:0 1px 8px rgba(27,58,107,0.08);border-top:3px solid {BLUE};">
    <div style="font-size:1.4em;margin-bottom:8px;">🔗</div>
    <div style="font-weight:700;color:{NAVY};font-size:0.90em;margin-bottom:4px;">Notion 연결</div>
    <div style="color:#64748B;font-size:0.80em;line-height:1.6;">Integration 토큰으로<br>페이지 자동 색인</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.stop()

st.divider()

# 채팅 기록
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

# 채팅 입력
if prompt := st.chat_input("문서 내용에 대해 무엇이든 질문하세요…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    master = get_master(backend_key, model_choice)
    with st.chat_message("assistant"):
        status_ph = st.empty()
        answer_ph = st.empty()
        full_answer = ""
        first_text  = True
        try:
            for kind, data in master.stream_chat(prompt):
                if kind == "status":
                    status_ph.markdown(
                        f'<span style="color:#94A3B8;font-size:0.88em;">{data}</span>',
                        unsafe_allow_html=True,
                    )
                elif kind == "text":
                    if first_text:
                        status_ph.empty()
                        first_text = False
                    full_answer += data
                    answer_ph.markdown(full_answer + "▌")
            answer_ph.markdown(full_answer)
        except Exception as e:
            status_ph.empty()
            full_answer = f"오류가 발생했습니다: {e}"
            answer_ph.markdown(full_answer)

        sources = getattr(master, "last_search_results", [])
        if sources:
            show_sources(sources)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_answer,
        "sources": getattr(master, "last_search_results", []),
    })
