"""Streamlit Web 对话调试入口。"""
from __future__ import annotations

import html
import base64
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

import streamlit as st

from agent import ReActAgent, warm_agent
from tools import warm_mcp_tools
from app.agent.console_stream import make_console_stream_callback
from app.agent.model_client import safe_print
from app.agent.kb_body_images import (
    normalize_kb_image_path,
    split_kb_image_markers,
    strip_kb_image_markers,
)


from app.application.conversation.services import ChatStore, SessionManager
from app.infrastructure.cache import RedisCache
from app.infrastructure.logging_setup import configure_project_logging
from app.infrastructure.observability import AppMetrics, bind_llm_metrics, log_timing
from db.client import load_mysql_config
from app.skills.system import build_skill_system


logger = configure_project_logging(ROOT, logger_name=__name__, log_basename="streamlit")
metrics = AppMetrics()
bind_llm_metrics(metrics)

WEB_STREAM_FLUSH_INTERVAL_SECONDS = 0.025
WEB_USER_COOKIE_NAME = "wechat_agent_web_user_id"
WEB_USER_QUERY_PARAM = "web_user_id"
WEB_USER_COOKIE_DAYS = int(os.getenv("WEB_USER_COOKIE_DAYS") or "365")
WEB_BOT_ID = (os.getenv("WEB_BOT_ID") or "web").strip() or "web"
WEB_MCP_WARMUP_ENABLED = (os.getenv("WEB_MCP_WARMUP_ENABLED") or "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
WEB_AGENT_WARMUP_ENABLED = (os.getenv("WEB_AGENT_WARMUP_ENABLED") or "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
WEB_ASSET_DIR = ROOT / "app" / "assets" / "web"
WEB_APP_ICON_PATH = WEB_ASSET_DIR / "web_app_icon.png"
WEB_UI_MOCKUP_PATH = WEB_ASSET_DIR / "web_ui_mockup.png"

_web_session_locks: dict[tuple[str, str], threading.Lock] = {}
_web_session_locks_mu = threading.Lock()

# Web 页面中知识库图片最大宽度，避免撑爆页面
WEB_KB_IMAGE_MAX_WIDTH = int(os.getenv("WEB_KB_IMAGE_MAX_WIDTH") or "520")
WEB_CODE_WRAP_COLUMN = int(os.getenv("WEB_CODE_WRAP_COLUMN") or "120")


def _asset_data_uri(path: Path) -> str:
    """功能：将本地图片资产转为可嵌入 HTML 的 data URI。
    参数：
    - path：本地图片文件路径。
    返回值：
    - str：data URI 字符串；文件不存在时返回空字符串。
    """
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _page_icon():
    """功能：读取页面 favicon 图标；失败时返回 None 让 Streamlit 使用默认图标。
    参数：
    - 无。
    返回值：
    - PIL.Image 或 None：可用图标对象或 None。
    """
    if not WEB_APP_ICON_PATH.exists():
        return None
    try:
        from PIL import Image

        return Image.open(WEB_APP_ICON_PATH)
    except Exception:
        return None


def _inject_app_css() -> None:
    """功能：注入 Web 页面自定义 CSS 样式。
    参数：
    - 无。
    返回值：
    - 无。
    """
    st.markdown(
        """
        <style>
        :root {
            --wa-bg: #f6f8f7;
            --wa-panel: #ffffff;
            --wa-panel-soft: #f1f5f4;
            --wa-border: #dfe7e4;
            --wa-text: #17202a;
            --wa-muted: #63717c;
            --wa-green: #19a974;
            --wa-blue: #2563eb;
            --wa-amber: #f59e0b;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(25, 169, 116, 0.08), transparent 34rem),
                linear-gradient(180deg, #fbfcfd 0%, var(--wa-bg) 100%);
            color: var(--wa-text);
        }

        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu,
        footer {
            visibility: hidden;
            height: 0;
        }

        [data-testid="stSidebarHeader"],
        [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }

        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 920px;
            padding: 1.35rem 2rem 0;
        }

        [data-testid="stSidebar"] {
            background: #f7f7f8;
            border-right: 1px solid #e5e7eb;
            width: 370px !important;
            min-width: 370px !important;
            max-width: 370px !important;
            flex: 0 0 370px !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            width: 370px !important;
            min-width: 370px !important;
            max-width: 370px !important;
            padding: 0.9rem 0.85rem 1rem;
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.45rem;
        }

        .wa-sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.72rem;
            margin: 0 0 1.6rem;
        }

        .wa-logo {
            width: 2.25rem;
            height: 2.25rem;
            border-radius: 8px;
            display: grid;
            place-items: center;
            color: #ffffff;
            background: linear-gradient(135deg, var(--wa-green), var(--wa-blue));
            font-weight: 800;
            letter-spacing: 0;
            box-shadow: 0 10px 22px rgba(31, 115, 88, 0.18);
        }

        .wa-sidebar-title {
            font-size: 1rem;
            font-weight: 760;
            letter-spacing: 0;
        }

        .wa-sidebar-subtitle {
            color: var(--wa-muted);
            font-size: 0.78rem;
            line-height: 1.35;
            margin-top: 0.1rem;
        }

        .wa-sidebar-label {
            margin: 1.35rem 0 0.55rem;
            color: #394651;
            font-size: 0.77rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        .wa-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.35rem;
        }

        .wa-chip {
            border: 1px solid var(--wa-border);
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            background: rgba(255,255,255,0.72);
            color: #40505b;
            font-size: 0.76rem;
            line-height: 1.2;
        }

        .wa-chip strong {
            color: #16232e;
            font-weight: 720;
        }

        [data-testid="stSidebar"] .stButton > button {
            border-radius: 8px;
            border: 1px solid #d9dde3;
            background: #ffffff;
            color: #202123;
            min-height: 2.25rem;
            box-shadow: none;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: #c9cdd4;
            background: #f4f4f5;
            color: #111827;
        }

        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
            gap: 0.45rem;
            align-items: center;
        }

        [data-testid="stSidebar"] [class*="st-key-web_new_session_action"] button {
            height: 2.25rem !important;
            min-height: 2.25rem !important;
            border-radius: 8px !important;
            border: 1px solid #d9dde3 !important;
            background: #ffffff !important;
            color: #202123 !important;
            font-size: 0.9rem !important;
            font-weight: 560 !important;
            justify-content: flex-start !important;
            padding: 0 0.85rem !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_refresh_sessions_action"] button,
        [data-testid="stSidebar"] [class*="st-key-web_delete_all_sessions_action"] button {
            width: 2.25rem !important;
            height: 2.25rem !important;
            min-height: 2.25rem !important;
            padding: 0 !important;
            border-radius: 8px !important;
            border: 1px solid #d9dde3 !important;
            background: #ffffff !important;
            color: #202123 !important;
            font-size: 1rem !important;
            line-height: 1.15 !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_delete_all_sessions_action"] button:hover {
            border-color: #f1b9b9 !important;
            background: #fff1f1 !important;
            color: #b42318 !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_new_session_action"] button:hover,
        [data-testid="stSidebar"] [class*="st-key-web_refresh_sessions_action"] button:hover {
            border-color: #c9cdd4 !important;
            background: #f4f4f5 !important;
            color: #111827 !important;
        }

        [data-testid="stSidebar"] [data-testid="stTextInput"] {
            margin-top: 0.2rem;
        }

        [data-testid="stSidebar"] [data-testid="stTextInput"] label {
            color: #4b5563;
            font-size: 0.82rem;
            font-weight: 520;
            padding-bottom: 0.25rem;
        }

        [data-testid="stSidebar"] [data-testid="stTextInput"] input {
            height: 2.35rem;
            border-radius: 8px;
            border-color: #e5e7eb;
            background: #ffffff;
            font-size: 0.9rem;
        }

        [data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
            border-color: #c7cbd1;
            box-shadow: 0 0 0 1px #c7cbd1;
        }

        .wa-sidebar-hero {
            margin: 0 0 0.75rem;
            padding: 0.05rem 0.1rem 0.85rem;
            border-bottom: 1px solid #e5e7eb;
        }

        .wa-sidebar-hero-title {
            color: #202123 !important;
            font-size: 1.42rem;
            font-weight: 820;
            letter-spacing: 0;
            line-height: 1.1;
            margin: 0;
        }

        .wa-sidebar-hero-subtitle {
            color: #4b5563 !important;
            font-size: 0.82rem;
            line-height: 1.42;
            margin-top: 0.4rem;
            overflow-wrap: anywhere;
        }

        .wa-sidebar-hero-meta {
            display: flex;
            flex-wrap: nowrap;
            gap: 0.38rem;
            margin-top: 0.62rem;
        }

        .wa-sidebar-hero-meta span {
            border: 1px solid var(--wa-border);
            border-radius: 8px;
            padding: 0.24rem 0.48rem;
            background: #ffffff;
            color: #4b5c67;
            font-size: 0.73rem;
            line-height: 1.2;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid var(--wa-border);
            border-radius: 8px;
            padding: 0.72rem 0.88rem;
            margin: 0.74rem 0;
            box-shadow: 0 8px 28px rgba(38, 50, 56, 0.045);
            max-width: 100%;
            overflow: hidden;
        }

        [data-testid="stChatMessage"] p {
            line-height: 1.65;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            max-width: 100%;
            min-width: 0;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre {
            max-width: 100%;
            max-height: 28rem;
            box-sizing: border-box;
            overflow: auto;
            white-space: pre-wrap !important;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre code {
            white-space: pre-wrap !important;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        /* 流式占位外层 container：无边框、不撑满高度，避免「正在处理」大块空白 */
        [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
            min-height: 0 !important;
        }

        [data-testid="stImage"] {
            max-width: 520px !important;
        }

        [data-testid="stImage"] img {
            max-width: min(100%, 520px) !important;
            height: auto !important;
            border-radius: 8px;
            border: 1px solid var(--wa-border);
            box-shadow: 0 8px 22px rgba(38, 50, 56, 0.08);
        }

        .wa-empty-state {
            margin: 1rem 0 1.2rem;
            border: 1px solid var(--wa-border);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.8);
            padding: 1rem 1.05rem;
        }

        .wa-empty-title {
            font-size: 0.95rem;
            font-weight: 760;
            color: #20303a;
            margin-bottom: 0.3rem;
        }

        .wa-empty-subtitle {
            color: var(--wa-muted);
            font-size: 0.87rem;
            line-height: 1.55;
            margin-bottom: 0.78rem;
        }

        .wa-empty-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }

        .wa-empty-chip {
            border-radius: 999px;
            border: 1px solid var(--wa-border);
            background: #f9fbfa;
            padding: 0.28rem 0.62rem;
            font-size: 0.78rem;
            color: #465763;
        }

        .thought-panel {
            border: 1px solid #dbe4e1;
            border-radius: 8px;
            background: #f8faf9;
            margin-bottom: 0.75rem;
            overflow: hidden;
        }

        .thought-panel summary {
            cursor: pointer;
            padding: 0.62rem 0.78rem;
            color: #32414c;
            font-weight: 700;
            font-size: 0.88rem;
            border-bottom: 1px solid #e4ebe8;
        }

        .thought-panel-body {
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            word-break: break-word;
            line-height: 1.58;
            font-size: 0.88rem;
            color: #4c5b65;
            padding: 0.72rem 0.78rem 0.78rem;
            font-family: ui-monospace, Consolas, "Segoe UI", sans-serif;
        }

        div[data-testid="stChatInput"] {
            max-width: 860px;
            margin: 0 auto;
            padding: 0 2rem 1.45rem;
        }

        div[data-testid="stChatInput"] > div {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }

        div[data-testid="stChatInput"] form {
            position: relative;
            border: 1.5px solid #ff8a4c !important;
            border-radius: 30px !important;
            background: #ffffff !important;
            min-height: 4.65rem;
            padding: 0 !important;
            box-shadow: 0 18px 38px rgba(35, 48, 59, 0.10);
            overflow: hidden;
        }

        div[data-testid="stChatInput"] textarea {
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            background: transparent !important;
            border-radius: 30px !important;
            min-height: 4.65rem !important;
            padding: 1.42rem 4.8rem 1.1rem 1.28rem !important;
            line-height: 1.45;
            font-size: 1rem;
            resize: none;
        }

        div[data-testid="stChatInput"] textarea:focus {
            border: none !important;
            box-shadow: none !important;
        }

        div[data-testid="stChatInput"] textarea::placeholder {
            color: #b0b7c3;
        }

        div[data-testid="stChatInput"] button {
            position: absolute;
            right: 0.62rem;
            top: 50%;
            transform: translateY(-50%);
            border-radius: 14px !important;
            width: 2.5rem !important;
            height: 2.5rem !important;
            min-height: 2.5rem !important;
            border: none !important;
            background: linear-gradient(135deg, #ff9a5f, #ff7a2f) !important;
            box-shadow: 0 8px 18px rgba(255, 122, 47, 0.28);
            z-index: 2;
        }

        div[data-testid="stChatInput"] button:hover {
            background: linear-gradient(135deg, #ff8d4f, #ff6d1f) !important;
        }

        @media (max-width: 760px) {
            [data-testid="stAppViewContainer"] > .main .block-container {
                padding: 2.25rem 1rem 7rem;
            }
            div[data-testid="stChatInput"] {
                padding-inline: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        :root {
            --wa-design-bg: #f4f7fb;
            --wa-design-sidebar: #f7faff;
            --wa-design-text: #172033;
            --wa-design-muted: #7a8aa5;
            --wa-design-line: #e4eaf3;
            --wa-design-blue: #5674f1;
            --wa-design-blue-2: #5f77f4;
            --wa-design-cyan: #21aeea;
            --wa-design-orange: #ff7a2b;
            --wa-design-card: #ffffff;
            --wa-chat-rail: calc(100vw - 370px - 3rem);
            --wa-input-rail: min(1120px, calc(100vw - 520px));
            --wa-chat-pad: 1.5rem;
            --wa-welcome-rail: min(900px, calc(100vw - 420px));
        }

        .stApp {
            background: var(--wa-design-bg) !important;
            color: var(--wa-design-text) !important;
            height: 100vh !important;
            overflow: hidden !important;
        }

        [data-testid="stAppViewContainer"] > .main {
            background: var(--wa-design-bg) !important;
            height: 100vh !important;
            overflow: hidden !important;
        }

        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1120px !important;
            padding: 2rem 2.4rem 0 !important;
        }

        body:has(.wa-mode-chat) [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: none !important;
            padding: 0 var(--wa-chat-pad) 0 !important;
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1120px !important;
            padding: 2rem 2.4rem 0 !important;
        }

        body:has(.wa-mode-welcome) [data-testid="stMainBlockContainer"] {
            width: var(--wa-welcome-rail) !important;
            max-width: var(--wa-welcome-rail) !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-top: 2rem !important;
            padding-bottom: 0 !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            padding-inline: 0 !important;
            min-height: 0 !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
            box-sizing: border-box !important;
        }

        body:has(.wa-mode-welcome) [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {
            width: 100% !important;
            max-width: 100% !important;
        }

        body:has(.wa-mode-welcome) [data-testid="stAppViewContainer"] > .main .block-container {
            padding-left: 0 !important;
            padding-right: 0 !important;
            padding-inline: 0 !important;
        }

        body:has(.wa-mode-welcome) .wa-welcome {
            width: 100% !important;
            max-width: 100% !important;
        }

        body:has(.wa-mode-chat) [data-testid="stMainBlockContainer"] {
            max-width: none !important;
            padding: 0.75rem var(--wa-chat-pad) 0 !important;
            min-height: 0 !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
            display: block !important;
            box-sizing: border-box !important;
        }

        [data-testid="stSidebar"] {
            width: 370px !important;
            min-width: 370px !important;
            max-width: 370px !important;
            flex: 0 0 370px !important;
            background: var(--wa-design-sidebar) !important;
            border-right: 1px solid #eef2f8 !important;
            box-shadow: 14px 0 36px rgba(120, 137, 160, 0.06);
            height: 100vh !important;
            overflow: hidden !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            width: 370px !important;
            min-width: 370px !important;
            max-width: 370px !important;
            height: 100vh !important;
            max-height: 100vh !important;
            padding: 1.5rem 1.2rem 0.75rem !important;
            overflow: hidden !important;
            box-sizing: border-box !important;
        }

        [data-testid="stSidebar"] > div:first-child > [data-testid="stVerticalBlock"] {
            display: block !important;
            overflow: visible !important;
            gap: 0.7rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stElementContainer"]:has([class*="st-key-web_session_list_scroll"]) {
            display: block !important;
            overflow: visible !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_session_list_scroll"] {
            display: block !important;
            flex: none !important;
            width: 100% !important;
            max-height: calc(100vh - 22rem) !important;
            min-height: 5rem !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            margin: 0.15rem -0.35rem 0 0 !important;
            padding: 0.35rem 0.35rem 0.5rem 0 !important;
            box-sizing: border-box !important;
            -webkit-overflow-scrolling: touch;
        }

        [data-testid="stSidebar"] [class*="st-key-web_session_list_scroll"] [data-testid="stVerticalBlock"] {
            display: block !important;
            min-height: auto !important;
            height: auto !important;
            overflow: visible !important;
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.7rem !important;
        }

        .wa-sidebar-hero,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"],
        [data-testid="stSidebar"] [data-testid="stTextInput"] {
            flex-shrink: 0 !important;
        }

        .wa-sidebar-hero {
            border-bottom: 1px solid #eef2f8 !important;
            margin: 0 -1.2rem 0.95rem !important;
            padding: 0 1.2rem 1.15rem !important;
        }

        .wa-sidebar-brandline {
            display: flex;
            align-items: center;
            gap: 0.625rem;
            margin-bottom: 0.875rem;
        }

        .wa-sidebar-brandtext {
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 0.125rem;
            min-width: 0;
            flex: 1 1 auto;
            max-height: 44px;
            overflow: hidden;
        }

        .wa-sidebar-brandline > div:not(.wa-brand-icon) {
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 0.125rem;
            min-width: 0;
            flex: 1 1 auto;
            max-height: 44px;
            overflow: hidden;
        }

        .wa-brand-icon {
            display: inline-grid;
            place-items: center;
            flex: 0 0 auto;
            border-radius: 15px;
            background: linear-gradient(135deg, #6682ff 0%, #4c63e9 100%);
            box-shadow: 0 14px 30px rgba(86, 116, 241, 0.24);
            color: #ffffff;
            font-weight: 900;
            line-height: 1;
            position: relative;
            font-size: 0 !important;
        }

        .wa-brand-icon::before {
            content: "";
            position: absolute;
            left: 50%;
            top: 50%;
            width: 45%;
            height: 34%;
            transform: translate(-50%, -40%);
            border: 3px solid #ffffff;
            border-radius: 7px;
            box-sizing: border-box;
        }

        .wa-brand-icon::after {
            content: "";
            position: absolute;
            left: 50%;
            top: 50%;
            width: 4px;
            height: 4px;
            transform: translate(-7px, -3px);
            border-radius: 50%;
            background: #ffffff;
            box-shadow:
                10px 0 0 #ffffff,
                5px -14px 0 -1px #ffffff,
                5px -10px 0 -2px #ffffff;
        }

        .wa-brand-icon-sm {
            width: 44px;
            height: 44px;
            border-radius: 12px;
            font-size: 18px;
        }

        .wa-brand-icon-lg {
            width: 66px;
            height: 66px;
            border-radius: 20px;
            font-size: 31px;
            margin: 0 auto 1.05rem;
        }

        .wa-sidebar-hero-title {
            color: #1a2332 !important;
            font-size: 1rem !important;
            line-height: 1.15 !important;
            font-weight: 700 !important;
            margin: 0 !important;
            padding: 0 !important;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .wa-sidebar-hero-subtitle {
            color: #8a96a8 !important;
            font-size: 0.75rem !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.15 !important;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        [data-testid="stSidebar"] .wa-sidebar-hero h1,
        [data-testid="stSidebar"] .wa-sidebar-hero-title {
            padding: 0 !important;
            margin: 0 !important;
        }

        .wa-sidebar-hero-meta {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 0.56rem !important;
            margin-top: 0 !important;
        }

        .wa-sidebar-hero-meta span {
            border: 0 !important;
            border-radius: 8px !important;
            background: #edf3ff !important;
            color: #2f69f5 !important;
            padding: 0.47rem 0.68rem !important;
            font-size: 0.78rem !important;
            font-weight: 650 !important;
            box-shadow: none !important;
        }

        .wa-sidebar-hero-meta span:nth-child(2) {
            background: #eaf8ff !important;
            color: #0c99db !important;
        }

        .wa-sidebar-hero-meta span:nth-child(3) {
            background: #f1eeff !important;
            color: #7764e7 !important;
        }

        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
            gap: 0.74rem !important;
            align-items: center !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_new_session_action"] button {
            height: 2.55rem !important;
            min-height: 2.55rem !important;
            border-radius: 10px !important;
            border: 1px solid #d6dce8 !important;
            background: #ffffff !important;
            color: #2f69f5 !important;
            font-size: 0.9rem !important;
            font-weight: 720 !important;
            justify-content: center !important;
            box-shadow: 0 4px 12px rgba(38, 58, 86, 0.04) !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_refresh_sessions_action"] button,
        [data-testid="stSidebar"] [class*="st-key-web_delete_all_sessions_action"] button {
            width: 2.55rem !important;
            height: 2.55rem !important;
            min-height: 2.55rem !important;
            border-radius: 9px !important;
            border: 1px solid #dfe5ef !important;
            background: #ffffff !important;
            color: #71819c !important;
            font-size: 1rem !important;
            box-shadow: 0 4px 12px rgba(38, 58, 86, 0.04) !important;
        }

        [data-testid="stSidebar"] [data-testid="stTextInput"] {
            margin: 0.35rem 0 0.95rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stTextInput"] label {
            display: none !important;
        }

        [data-testid="stSidebar"] [data-testid="stTextInput"] input {
            height: 2.35rem !important;
            border-radius: 12px !important;
            border: 1px solid #e2e8f2 !important;
            background: #ffffff !important;
            color: #36435a !important;
            font-size: 0.86rem !important;
            padding-left: 0.9rem !important;
            box-shadow: 0 2px 10px rgba(50, 70, 100, 0.03) !important;
        }

        [data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder {
            color: #a0adc1 !important;
        }

        .wa-welcome {
            width: 100%;
            max-width: 100%;
            margin: 0 auto;
            text-align: center;
            box-sizing: border-box;
        }

        body:has(.wa-mode-welcome) [data-testid="stAppScrollToBottomContainer"] {
            width: var(--wa-welcome-rail) !important;
            max-width: var(--wa-welcome-rail) !important;
            height: 100vh !important;
            max-height: 100vh !important;
            margin-left: calc((100vw - 370px - var(--wa-welcome-rail)) / 2) !important;
            margin-right: auto !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            box-sizing: border-box !important;
        }

        [data-testid="stAppScrollToBottomContainer"] {
            box-sizing: border-box !important;
        }

        body:has(.wa-mode-chat) [data-testid="stAppViewContainer"] > div:not([data-testid="stSidebar"]) {
            height: calc(100vh - 0.5rem) !important;
            max-height: calc(100vh - 0.5rem) !important;
            min-height: 0 !important;
            overflow: hidden !important;
        }

        body:has(.wa-mode-chat) [data-testid="stAppScrollToBottomContainer"] {
            width: var(--wa-chat-rail) !important;
            max-width: var(--wa-chat-rail) !important;
            margin-left: var(--wa-chat-pad) !important;
            margin-right: var(--wa-chat-pad) !important;
            height: calc(100vh - 10.2rem) !important;
            max-height: calc(100vh - 10.2rem) !important;
            min-height: 0 !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding-top: 0.35rem !important;
            padding-bottom: 0.5rem !important;
            box-sizing: border-box !important;
            -webkit-overflow-scrolling: touch;
        }

        .wa-welcome-title {
            margin: 0;
            color: #172033;
            font-size: 1.55rem;
            line-height: 1.2;
            font-weight: 850;
            letter-spacing: 0;
        }

        .wa-welcome-subtitle {
            margin: 0.72rem 0 0.55rem;
            color: #71809c;
            font-size: 1rem;
            line-height: 1.6;
        }

        .wa-capability-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            width: 100%;
            margin: 0 auto 1.85rem;
            text-align: left;
            box-sizing: border-box;
        }

        .wa-capability-card {
            min-height: 9.25rem;
            border-radius: 16px;
            border: 1px solid #d8e2ff;
            background: #f2f5ff;
            padding: 1.28rem 1.18rem;
            box-shadow: 0 12px 34px rgba(88, 116, 166, 0.06);
        }

        .wa-capability-card:nth-child(2) {
            border-color: #afe2fb;
            background: #e9f8ff;
        }

        .wa-capability-card:nth-child(3) {
            border-color: #ffd8a8;
            background: #fff4e4;
        }

        .wa-cap-icon {
            display: grid;
            place-items: center;
            width: 2.55rem;
            height: 2.55rem;
            border-radius: 12px;
            color: #ffffff;
            font-size: 1.24rem;
            font-weight: 850;
            margin-bottom: 0.78rem;
            background: var(--wa-design-blue);
        }

        .wa-capability-card:nth-child(2) .wa-cap-icon {
            background: var(--wa-design-cyan);
        }

        .wa-capability-card:nth-child(3) .wa-cap-icon {
            background: var(--wa-design-orange);
        }

        .wa-capability-title {
            color: #1d2940;
            font-size: 1rem;
            font-weight: 820;
            margin-bottom: 0.55rem;
        }

        .wa-capability-copy {
            color: #6d7c96;
            font-size: 0.84rem;
            line-height: 1.55;
        }

        .wa-prompt-title {
            width: 100%;
            max-width: var(--wa-welcome-rail);
            margin: 0 auto 0.68rem;
            color: #7a8aa5;
            text-align: left;
            font-size: 0.88rem;
            font-weight: 760;
            box-sizing: border-box;
        }

        [class*="st-key-web_suggestion_"] {
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 auto 0.54rem !important;
        }

        [class*="st-key-web_suggestion_"] button {
            height: 2.45rem !important;
            min-height: 2.45rem !important;
            width: 100% !important;
            border-radius: 12px !important;
            border: 1px solid #e0e6f0 !important;
            background: #ffffff !important;
            color: #39475f !important;
            box-shadow: 0 5px 14px rgba(40, 54, 80, 0.04) !important;
            justify-content: flex-start !important;
            padding: 0 1rem !important;
            font-size: 0.92rem !important;
        }

        [class*="st-key-web_suggestion_"] button::before {
            content: "◌";
            color: var(--wa-design-orange);
            font-size: 1.2rem;
            margin-right: 0.66rem;
        }

        [class*="st-key-web_suggestion_"] button::after {
            content: "↗";
            margin-left: auto;
            color: #b7c4d7;
            font-weight: 800;
        }

        .wa-quick-actions {
            max-width: 900px;
            margin: 1.35rem auto 0.78rem;
        }

        [class*="st-key-web_quick_"] {
            margin-top: 0 !important;
            position: relative !important;
            top: 0 !important;
            z-index: 961 !important;
        }

        [class*="st-key-web_quick_"] button {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            min-height: 2rem !important;
            height: 2rem !important;
            border-radius: 999px !important;
            border: 1px solid #dbe2ec !important;
            background: #ffffff !important;
            color: #71809a !important;
            font-size: 0.88rem !important;
            line-height: 1.15 !important;
            padding: 0 0.92rem !important;
            box-shadow: 0 4px 14px rgba(40, 54, 80, 0.08) !important;
            white-space: nowrap !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease !important;
        }

        [class*="st-key-web_quick_"] button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 24px rgba(40, 54, 80, 0.14) !important;
            border-color: #c5d0e0 !important;
        }

        [class*="st-key-web_quick_"] button p {
            display: block !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            color: inherit !important;
        }

        [data-testid="stHorizontalBlock"]:has([class*="st-key-web_quick_"]) {
            width: var(--wa-input-rail) !important;
            max-width: var(--wa-input-rail) !important;
            margin: 0 auto 1.05rem !important;
            gap: 0.72rem !important;
        }

        [data-testid="stBottom"],
        [data-testid="stBottom"] > div {
            background: transparent !important;
        }

        [data-testid="stBottomBlockContainer"] {
            background: transparent !important;
        }

        [data-testid="stBottom"] {
            position: fixed !important;
            left: 370px !important;
            right: 0 !important;
            bottom: 0 !important;
            width: auto !important;
            height: 8.3rem !important;
            z-index: 900 !important;
            display: flex !important;
            align-items: flex-start !important;
            justify-content: center !important;
            padding-top: 0.95rem !important;
        }

        [data-testid="stBottomBlockContainer"] {
            width: var(--wa-input-rail) !important;
            max-width: var(--wa-input-rail) !important;
            height: auto !important;
        }

        [data-testid="stChatMessage"]:last-of-type {
            margin-bottom: 2.5rem !important;
        }

        [data-testid="stChatMessage"] {
            border: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 1.9rem 0 !important;
            overflow: visible !important;
            display: flex !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            justify-content: flex-end !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            justify-content: flex-end !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            justify-content: flex-start !important;
            gap: 0.78rem !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            justify-content: flex-start !important;
            gap: 0.78rem !important;
        }

        [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
            display: none !important;
        }

        [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] {
            display: none !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
            max-width: min(560px, 72vw) !important;
            border-radius: 17px !important;
            background: linear-gradient(135deg, #5e7bf4, #536def) !important;
            color: #ffffff !important;
            padding: 0.78rem 1.35rem !important;
            box-shadow: 0 10px 22px rgba(82, 109, 239, 0.22) !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            max-width: min(560px, 72vw) !important;
            flex: 0 1 auto !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
            border-radius: 17px !important;
            background: linear-gradient(135deg, #5e7bf4, #536def) !important;
            color: #ffffff !important;
            padding: 0.78rem 1.35rem !important;
            box-shadow: 0 10px 22px rgba(82, 109, 239, 0.22) !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
            margin: 0 !important;
            line-height: 1.45 !important;
            font-weight: 650 !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
            margin: 0 !important;
            line-height: 1.45 !important;
            font-weight: 650 !important;
        }

        [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
            width: 2rem !important;
            height: 2rem !important;
            min-width: 2rem !important;
            border-radius: 9px !important;
            background: linear-gradient(135deg, #6682ff, #4c63e9) !important;
            color: #ffffff !important;
            box-shadow: 0 9px 18px rgba(86, 116, 241, 0.18) !important;
        }

        [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
            width: 2rem !important;
            height: 2rem !important;
            min-width: 2rem !important;
            border-radius: 9px !important;
            background: linear-gradient(135deg, #6682ff, #4c63e9) !important;
            color: #ffffff !important;
            box-shadow: 0 9px 18px rgba(86, 116, 241, 0.18) !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:last-child {
            max-width: calc(100% - 2.75rem) !important;
            width: calc(100% - 2.75rem) !important;
            border-radius: 14px !important;
            border: 1px solid #e0e6ef !important;
            background: #ffffff !important;
            padding: 1.18rem 1.28rem !important;
            box-shadow: 0 8px 24px rgba(38, 54, 78, 0.06) !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
            max-width: calc(100% - 2.75rem) !important;
            width: calc(100% - 2.75rem) !important;
            border-radius: 14px !important;
            border: 1px solid #e0e6ef !important;
            background: #ffffff !important;
            padding: 1.18rem 1.28rem !important;
            box-shadow: 0 8px 24px rgba(38, 54, 78, 0.06) !important;
        }

        [data-testid="stChatMessage"] table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            overflow: hidden;
            border: 1px solid #e2e7ef;
            border-radius: 12px;
        }

        [data-testid="stChatMessage"] th {
            background: #f6f8fb;
            color: #66748d;
            font-weight: 760;
        }

        [data-testid="stChatMessage"] td,
        [data-testid="stChatMessage"] th {
            border-bottom: 1px solid #edf1f6;
            padding: 0.72rem 0.86rem;
        }

        .wa-chat-rail {
            width: min(900px, calc(100vw - 520px));
            max-width: 900px;
            margin: 0 auto;
        }

        .wa-chat-row {
            width: 100% !important;
            max-width: 100% !important;
            margin: 0.85rem 0 !important;
            box-sizing: border-box;
        }

        .wa-chat-row-user {
            display: flex;
            justify-content: flex-end;
            padding-right: 0 !important;
        }

        .wa-user-bubble {
            display: inline-block;
            max-width: min(680px, 82vw);
            border-radius: 17px;
            background: linear-gradient(135deg, #5e7bf4, #536def);
            color: #ffffff;
            padding: 0.78rem 1.35rem;
            box-shadow: 0 10px 22px rgba(82, 109, 239, 0.22);
            line-height: 1.45;
            font-weight: 650;
            overflow-wrap: anywhere;
        }

        .wa-assistant-shell {
            display: flex;
            align-items: flex-start;
            gap: 0.78rem;
        }

        .wa-assistant-avatar {
            width: 2rem;
            height: 2rem;
            flex: 0 0 2rem;
            border-radius: 9px;
            background: linear-gradient(135deg, #6682ff, #4c63e9);
            box-shadow: 0 9px 18px rgba(86, 116, 241, 0.18);
            position: relative;
        }

        .wa-assistant-avatar::before {
            content: "";
            position: absolute;
            left: 50%;
            top: 50%;
            width: 48%;
            height: 34%;
            transform: translate(-50%, -38%);
            border: 2px solid #ffffff;
            border-radius: 5px;
            box-sizing: border-box;
        }

        .wa-assistant-avatar::after {
            content: "";
            position: absolute;
            left: 50%;
            top: 50%;
            width: 3px;
            height: 3px;
            transform: translate(-6px, -2px);
            border-radius: 50%;
            background: #ffffff;
            box-shadow: 8px 0 0 #ffffff, 4px -10px 0 -1px #ffffff;
        }

        [class*="st-key-wa_msg_assistant_"] {
            width: 100% !important;
            max-width: 100% !important;
            margin: 0.85rem 0 !important;
        }

        [class*="st-key-wa_msg_assistant_"]:first-of-type {
            margin-top: 0.25rem !important;
        }

        [class*="st-key-wa_msg_assistant_"] [data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }

        [class*="st-key-wa_card_"] {
            width: calc(100% - 2.75rem) !important;
            max-width: calc(100% - 2.75rem) !important;
            border-radius: 14px !important;
            border: 1px solid #e0e6ef !important;
            background: #ffffff !important;
            padding: 1.18rem 1.28rem !important;
            box-shadow: 0 8px 24px rgba(38, 54, 78, 0.06) !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }

        [class*="st-key-wa_card_"]:has(.wa-loading-state) {
            display: flex !important;
            align-items: center !important;
            min-height: 3.1rem !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }

        [class*="st-key-wa_card_"] [data-testid="stMarkdownContainer"] {
            color: #172033 !important;
            line-height: 1.7 !important;
            overflow-wrap: anywhere !important;
        }

        [class*="st-key-wa_card_"] table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            overflow: hidden;
            border: 1px solid #e2e7ef;
            border-radius: 12px;
        }

        [class*="st-key-wa_card_"] th {
            background: #f6f8fb;
            color: #66748d;
            font-weight: 760;
        }

        [class*="st-key-wa_card_"] td,
        [class*="st-key-wa_card_"] th {
            border-bottom: 1px solid #edf1f6;
            padding: 0.72rem 0.86rem;
        }

        .wa-loading-state {
            display: flex;
            align-items: center;
            min-height: 3.1rem;
        }

        .wa-loading-dots {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.34rem;
            height: 1rem;
            padding: 0;
            line-height: 0;
        }

        .wa-loading-dots span {
            width: 0.42rem;
            height: 0.42rem;
            flex: 0 0 0.42rem;
            display: block;
            border-radius: 999px;
            background: #5976f3;
            opacity: 0.32;
            animation: wa-dot-pulse 1.05s infinite ease-in-out;
        }

        .wa-loading-dots span:nth-child(2) {
            animation-delay: 0.16s;
        }

        .wa-loading-dots span:nth-child(3) {
            animation-delay: 0.32s;
        }

        @keyframes wa-dot-pulse {
            0%, 80%, 100% {
                transform: translateY(0);
                opacity: 0.32;
            }
            40% {
                transform: translateY(-3px);
                opacity: 1;
            }
        }

        div[data-testid="stChatInput"] {
            width: 100% !important;
            max-width: var(--wa-input-rail) !important;
            padding: 0 !important;
            margin: 0 auto 1.55rem !important;
            background: #ffffff !important;
            border: 1px solid #dfe6ef !important;
            border-radius: 21px !important;
            box-shadow: 0 12px 30px rgba(40, 54, 80, 0.08) !important;
        }

        div[data-testid="stChatInput"] div,
        div[data-testid="stChatInput"] textarea {
            background: transparent !important;
        }

        div[data-testid="stChatInput"] form {
            min-height: 4.05rem !important;
            border: 1px solid #dfe6ef !important;
            border-radius: 21px !important;
            box-shadow: 0 12px 30px rgba(40, 54, 80, 0.08) !important;
            background: #ffffff !important;
        }

        div[data-testid="stChatInput"] textarea {
            min-height: 4.05rem !important;
            border-radius: 21px !important;
            padding: 1.34rem 5rem 1rem 1.25rem !important;
            font-size: 0.98rem !important;
            color: #334158 !important;
        }

        div[data-testid="stChatInput"] textarea::placeholder {
            color: #9aa9bd !important;
        }

        div[data-testid="stChatInput"] button {
            width: 2.55rem !important;
            height: 2.55rem !important;
            min-height: 2.55rem !important;
            right: 0.86rem !important;
            border-radius: 50% !important;
            background: linear-gradient(135deg, #ff8d37, #ff7722) !important;
            box-shadow: 0 9px 18px rgba(255, 122, 43, 0.28) !important;
        }

        .wa-footer-note {
            position: fixed;
            left: 370px;
            right: 0;
            bottom: 0.38rem;
            text-align: center;
            color: #c1ccdc;
            font-size: 0.78rem;
            pointer-events: none;
            z-index: 900;
        }

        body:has(.wa-mode-chat) [data-testid="stMarkdownContainer"]:has(.wa-footer-note),
        body:has(.wa-mode-chat) [data-testid="stElementContainer"]:has(.wa-footer-note) {
            height: 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: visible !important;
        }

        /* Final alignment pass: keep the sidebar meta and chat rail matching the mockups. */
        .wa-sidebar-hero-meta {
            flex-wrap: wrap !important;
            gap: 0.46rem !important;
            align-items: center !important;
        }

        .wa-sidebar-hero-meta span {
            padding: 0.4rem 0.5rem !important;
            font-size: 0.74rem !important;
            line-height: 1 !important;
            white-space: nowrap !important;
        }

        [data-testid="stChatMessage"] {
            width: var(--wa-chat-rail) !important;
            max-width: var(--wa-chat-rail) !important;
            margin: 1.75rem 0 !important;
            box-sizing: border-box !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            margin-left: auto !important;
            margin-right: 0 !important;
            width: fit-content !important;
            min-width: 0 !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
            display: inline-block !important;
            min-width: 0 !important;
            max-width: min(560px, 100%) !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            justify-content: flex-start !important;
            align-items: flex-start !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
            margin-left: 0 !important;
            margin-right: auto !important;
            width: calc(100% - 2.75rem) !important;
            max-width: calc(100% - 2.75rem) !important;
            flex: 0 1 auto !important;
        }

        [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
            justify-content: flex-end !important;
            align-items: flex-start !important;
        }

        [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [aria-label="Chat message from user"] {
            margin-left: auto !important;
            margin-right: 0 !important;
            width: fit-content !important;
            max-width: min(560px, 64vw) !important;
            flex: 0 1 auto !important;
        }

        [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stMarkdownContainer"] {
            display: inline-block !important;
            width: auto !important;
            min-width: 0 !important;
            max-width: 100% !important;
            border-radius: 17px !important;
            background: linear-gradient(135deg, #5e7bf4, #536def) !important;
            color: #ffffff !important;
            padding: 0.78rem 1.35rem !important;
            box-shadow: 0 10px 22px rgba(82, 109, 239, 0.22) !important;
        }

        [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
            margin: 0 !important;
            line-height: 1.45 !important;
            font-weight: 650 !important;
        }

        [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) {
            justify-content: flex-start !important;
            align-items: flex-start !important;
            gap: 0.78rem !important;
        }

        [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) [aria-label="Chat message from assistant"] {
            margin-left: 0 !important;
            margin-right: auto !important;
            width: calc(100% - 2.75rem) !important;
            max-width: calc(100% - 2.75rem) !important;
            flex: 0 1 auto !important;
            border-radius: 14px !important;
            border: 1px solid #e0e6ef !important;
            background: #ffffff !important;
            padding: 1.18rem 1.28rem !important;
            box-shadow: 0 8px 24px rgba(38, 54, 78, 0.06) !important;
        }

        body:has(.wa-mode-chat) [class*="st-key-web_suggestion_"],
        body:has(.wa-mode-chat) .wa-prompt-title {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }

        [data-testid="stBottomBlockContainer"],
        div[data-testid="stChatInput"],
        div[data-testid="stChatInput"] > div,
        div[data-testid="stChatInput"] form {
            width: var(--wa-input-rail) !important;
            max-width: var(--wa-input-rail) !important;
            box-sizing: border-box !important;
        }

        .wa-prompt-title,
        [class*="st-key-web_suggestion_"],
        .wa-quick-actions,
        [data-testid="stHorizontalBlock"]:has([class*="st-key-web_quick_"]) {
            width: var(--wa-input-rail) !important;
            max-width: var(--wa-input-rail) !important;
            box-sizing: border-box !important;
        }

        [data-testid="stHorizontalBlock"]:has([class*="st-key-web_quick_"]) {
            display: flex !important;
            flex-wrap: wrap !important;
            align-items: center !important;
            justify-content: flex-start !important;
            gap: 0.55rem !important;
            overflow: visible !important;
        }

        [data-testid="stHorizontalBlock"]:has([class*="st-key-web_quick_"]) > div {
            flex: 0 1 auto !important;
            min-width: 0 !important;
            width: auto !important;
        }

        [class*="st-key-web_quick_"] {
            top: 0 !important;
            margin-top: 0 !important;
        }

        [class*="st-key-web_quick_"] button {
            width: auto !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }

        .wa-quick-actions {
            height: 0 !important;
            margin: 0 auto !important;
        }

        [data-testid="stHorizontalBlock"]:has([class*="st-key-web_quick_"]) {
            position: fixed !important;
            left: calc(370px + (100vw - 370px - var(--wa-input-rail)) / 2) !important;
            right: auto !important;
            width: var(--wa-input-rail) !important;
            bottom: 10.15rem !important;
            z-index: 960 !important;
            margin: 0 !important;
            padding: 0 !important;
            pointer-events: auto !important;
            filter: drop-shadow(0 6px 18px rgba(40, 54, 80, 0.10)) !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_session_row_"] {
            position: relative !important;
            display: block !important;
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
            margin: 0.34rem 0 !important;
            overflow: visible !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_session_row_"] [data-testid="stElementContainer"],
        [data-testid="stSidebar"] [class*="st-key-web_session_row_"] [data-testid="stButton"] {
            position: static !important;
            width: 100% !important;
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_session_row_"] button {
            position: relative !important;
            width: 100% !important;
            min-height: 4.05rem !important;
            height: auto !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            text-align: left !important;
            border-radius: 14px !important;
            border: 2px solid transparent !important;
            background: transparent !important;
            color: #8a99ae !important;
            padding: 0.72rem 0.95rem 0.7rem 2.85rem !important;
            box-shadow: none !important;
            overflow: hidden !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_session_row_selected_"] button {
            background: #eef3ff !important;
            border-color: #4d74f4 !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_session_row_"]:not([class*="selected"]) button:hover {
            background: #f3f6fc !important;
            border-color: transparent !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_session_row_selected_"] button:hover {
            background: #eef3ff !important;
            border-color: #4d74f4 !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_session_row_"] button::before {
            content: "";
            position: absolute;
            left: 0.95rem;
            top: 50%;
            width: 0.95rem;
            height: 0.72rem;
            transform: translateY(-54%);
            border: 2px solid #b8c5da;
            border-radius: 5px;
            background: transparent;
            box-sizing: border-box;
        }

        [data-testid="stSidebar"] [class*="st-key-web_session_row_"] button::after {
            content: "";
            position: absolute;
            left: 0.98rem;
            top: 50%;
            width: 0.34rem;
            height: 0.34rem;
            transform: translateY(0.1rem) rotate(-12deg);
            border-left: 2px solid #b8c5da;
            border-bottom: 2px solid #b8c5da;
            background: transparent;
        }

        [data-testid="stSidebar"] [class*="st-key-web_session_row_selected_"] button::before,
        [data-testid="stSidebar"] [class*="st-key-web_session_row_selected_"] button::after {
            border-color: #4d74f4 !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_session_row_"] button > div {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            height: auto !important;
            display: block !important;
            text-align: left !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_session_row_"] button p {
            position: static !important;
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            color: #8a99ae !important;
            font-size: 0.78rem !important;
            font-weight: 500 !important;
            line-height: 1.45 !important;
            white-space: pre-line !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            text-align: left !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_session_row_"] button p::first-line {
            color: #162033 !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            line-height: 1.35 !important;
        }

        [data-testid="stSidebar"] [class*="st-key-web_session_row_"] .wa-session-time {
            display: none !important;
        }

        [data-testid="stBottomBlockContainer"] > div {
            width: 100% !important;
            max-width: 100% !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            height: 100% !important;
            max-height: 100% !important;
            overflow: visible !important;
        }

        div[data-testid="stChatInput"] form {
            min-height: 4rem !important;
        }

        [data-testid="stBottom"] {
            right: 14px !important;
            width: auto !important;
            height: 9.6rem !important;
            max-height: 9.6rem !important;
            overflow: visible !important;
            background: transparent !important;
            z-index: 940 !important;
        }

        [data-testid="stBottomBlockContainer"] {
            padding-left: 0 !important;
            padding-right: 0 !important;
            height: 8.8rem !important;
            max-height: 8.8rem !important;
            overflow: visible !important;
            background: linear-gradient(
                180deg,
                rgba(244, 247, 251, 0) 0%,
                rgba(244, 247, 251, 0.88) 36%,
                var(--wa-design-bg) 72%,
                var(--wa-design-bg) 100%
            ) !important;
        }

        body:has(.wa-mode-chat) [data-testid="stBottom"] {
            left: calc(370px + (100vw - 370px - var(--wa-input-rail)) / 2) !important;
            right: calc((100vw - 370px - var(--wa-input-rail)) / 2) !important;
            width: auto !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            box-sizing: border-box !important;
        }

        body:has(.wa-mode-welcome) [data-testid="stBottom"] {
            left: calc(370px + (100vw - 370px - var(--wa-input-rail)) / 2) !important;
            right: calc((100vw - 370px - var(--wa-input-rail)) / 2) !important;
            width: auto !important;
            max-width: none !important;
        }

        body:has(.wa-mode-welcome) [data-testid="stBottomBlockContainer"],
        body:has(.wa-mode-welcome) div[data-testid="stChatInput"],
        body:has(.wa-mode-welcome) div[data-testid="stChatInput"] > div,
        body:has(.wa-mode-welcome) div[data-testid="stChatInput"] form {
            width: var(--wa-input-rail) !important;
            max-width: var(--wa-input-rail) !important;
            margin-left: auto !important;
            margin-right: auto !important;
            box-sizing: border-box !important;
        }

        body:has(.wa-mode-welcome) .wa-prompt-title,
        body:has(.wa-mode-welcome) [class*="st-key-web_suggestion_"],
        body:has(.wa-mode-welcome) [class*="st-key-wa_welcome_actions"] {
            width: var(--wa-welcome-rail) !important;
            max-width: var(--wa-welcome-rail) !important;
            margin-left: auto !important;
            margin-right: auto !important;
            box-sizing: border-box !important;
        }

        body:has(.wa-mode-welcome) [class*="st-key-wa_welcome_actions"] [data-testid="stVerticalBlock"] {
            width: 100% !important;
            gap: 0.54rem !important;
        }

        body:has(.wa-mode-welcome) [data-testid="stHorizontalBlock"]:has([class*="st-key-web_quick_"]) {
            left: calc(370px + (100vw - 370px - var(--wa-input-rail)) / 2) !important;
            width: var(--wa-input-rail) !important;
            max-width: var(--wa-input-rail) !important;
            display: flex !important;
            visibility: visible !important;
            pointer-events: auto !important;
            height: auto !important;
            overflow: visible !important;
        }

        body:has(.wa-mode-welcome) .wa-quick-actions {
            height: 0 !important;
            margin: 0 auto !important;
            overflow: visible !important;
        }

        body:has(.wa-mode-welcome) [data-testid="stMainBlockContainer"].block-container,
        body:has(.wa-mode-welcome) [data-testid="stMainBlockContainer"] {
            padding-left: 0 !important;
            padding-right: 0 !important;
            padding-inline: 0 !important;
        }

        body:has(.wa-mode-welcome) [data-testid="stMainBlockContainer"] > div,
        body:has(.wa-mode-welcome) [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"],
        body:has(.wa-mode-welcome) [data-testid="stMainBlockContainer"] [data-testid="stElementContainer"] {
            width: 100% !important;
            max-width: 100% !important;
        }

        body:not(:has(.wa-mode-chat)):not(:has(.wa-mode-welcome)) [data-testid="stBottom"] {
            left: calc(370px + (100vw - 370px - var(--wa-input-rail)) / 2) !important;
            right: calc((100vw - 370px - var(--wa-input-rail)) / 2) !important;
            width: auto !important;
        }

        @media (prefers-reduced-motion: reduce) {
            .wa-loading-dots span {
                animation: none !important;
            }
            [class*="st-key-web_quick_"] button:hover {
                transform: none !important;
            }
        }

        @media (max-width: 900px) {
            :root {
                --wa-welcome-rail: calc(100vw - 2rem);
            }

            [data-testid="stAppViewContainer"] > .main .block-container {
                padding: 2.5rem 1rem 8rem !important;
            }
            .wa-capability-grid {
                grid-template-columns: 1fr !important;
            }
            .wa-footer-note {
                left: 0;
            }
            body:has(.wa-mode-welcome) [data-testid="stAppScrollToBottomContainer"] {
                margin-left: 1rem !important;
            }
            body:has(.wa-mode-welcome) [data-testid="stBottom"] {
                left: 1rem !important;
            }
            [data-testid="stBottomBlockContainer"],
            div[data-testid="stChatInput"],
            div[data-testid="stChatInput"] > div,
            div[data-testid="stChatInput"] form,
            .wa-prompt-title,
            [class*="st-key-web_suggestion_"],
            [class*="st-key-wa_welcome_actions"],
            .wa-quick-actions,
            [data-testid="stHorizontalBlock"]:has([class*="st-key-web_quick_"]) {
                width: calc(100vw - 2rem) !important;
                max-width: calc(100vw - 2rem) !important;
            }
            [data-testid="stHorizontalBlock"]:has([class*="st-key-web_quick_"]) {
                left: 1rem !important;
                bottom: 10.15rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_app_header(model_name: str, tool_count: int, mcp_server_count: int) -> None:
    """功能：渲染页面顶部模型与工具概览信息。
    参数：
    - model_name：当前使用的模型名。
    - tool_count：已注册工具数量。
    - mcp_server_count：MCP 服务数量。
    返回值：
    - 无。
    """
    st.markdown(
        f"""
        <section class="wa-sidebar-hero">
            <div class="wa-sidebar-brandline">
                <div class="wa-brand-icon wa-brand-icon-sm">▣</div>
                <div class="wa-sidebar-brandtext">
                    <h1 class="wa-sidebar-hero-title">WeChat Agent</h1>
                    <div class="wa-sidebar-hero-subtitle">ReAct 智能助手 · 知识库 · MCP · 企微</div>
                </div>
            </div>
            <div class="wa-sidebar-hero-meta">
                <span>模型：{html.escape(model_name)}</span>
                <span>工具：{tool_count}</span>
                <span>MCP：{mcp_server_count}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _mcp_server_count() -> int:
    """功能：统计当前配置中启用的 MCP 服务数量。
    参数：
    - 无。
    返回值：
    - int：启用服务数；无法解析配置时返回 0。
    """
    raw = (os.getenv("MCP_SERVERS_JSON") or "").strip()
    payload = None
    if raw:
        try:
            payload = json.loads(raw)
        except Exception:
            payload = None
    if payload is None:
        config_path = Path(os.getenv("MCP_CONFIG_PATH") or (ROOT / "config" / "mcp_servers.json"))
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        if config_path.exists():
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
    if isinstance(payload, dict):
        server_map = payload.get("mcpServers")
        if isinstance(server_map, dict):
            return sum(1 for item in server_map.values() if not isinstance(item, dict) or item.get("enabled", True))
    if isinstance(payload, list):
        return sum(1 for item in payload if isinstance(item, dict) and item.get("enabled", True))
    return 0


def _tool_count(agent: ReActAgent | None) -> int:
    """功能：统计 Agent 已注册工具数量。
    参数：
    - agent：ReActAgent 实例；None 时返回 0。
    返回值：
    - int：工具数量。
    """
    if agent is None:
        return 0
    tools = getattr(getattr(agent, "tool_registry", None), "_tools", {}) or {}
    return len(tools)


def _get_web_session_lock(user_id: str, session_id: str) -> threading.Lock:
    """功能：获取或创建 Web 会话级进程内互斥锁。
    参数：
    - user_id：Web 用户标识。
    - session_id：会话标识。
    返回值：
    - threading.Lock：该会话对应的锁对象。
    """
    key = (user_id, session_id)
    with _web_session_locks_mu:
        lock = _web_session_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _web_session_locks[key] = lock
        return lock


def _selector_elapsed_ms(agent: Any) -> float:
    """功能：读取 Agent 上下文选择器最近一次耗时（毫秒）。
    参数：
    - agent：ReActAgent 或兼容对象。
    返回值：
    - float：耗时毫秒数；无法读取时返回 0.0。
    """
    try:
        runtime = getattr(agent, "runtime", None)
        selector = getattr(runtime, "context_selector", None)
        return float(getattr(selector, "last_elapsed_ms", 0) or 0)
    except Exception:
        return 0.0


def _append_turn_background(chat_store: ChatStore, **kwargs: Any) -> None:
    """功能：在后台线程异步持久化一轮问答，避免阻塞 UI。
    参数：
    - chat_store：ChatStore 实例。
    - **kwargs：传递给 chat_store.append_turn 的参数。
    返回值：
    - 无。
    """
    def _run() -> None:
        """功能：后台线程执行消息写入。
        参数：
        - 无。
        返回值：
        - 无。失败时通过 safe_print 记录错误。
        """
        try:
            chat_store.append_turn(**kwargs)
        except Exception as exc:  # noqa: BLE001
            safe_print(f"[Streamlit] Web 会话消息写入失败: {exc}")

    threading.Thread(target=_run, daemon=True).start()


@st.cache_resource(show_spinner="正在加载工具信息（首次约 10-20 秒）...")
def _get_web_tool_count(model_name: str) -> int:
    """功能：创建 Agent 并统计工具数量，结果由 Streamlit 缓存。
    参数：
    - model_name：OpenRouter 模型名。
    返回值：
    - int：已注册工具数量。
    """
    return _tool_count(_create_agent(model_name))


@st.cache_resource(show_spinner="正在加载技能与工具（首次约 10-20 秒）...")
def _get_skill_system():
    """功能：构建并缓存项目技能系统，供 Agent 复用。
    参数：
    - 无。
    返回值：
    - SkillSystem：初始化完成的技能系统实例。
    """
    return build_skill_system(project_directory=str(ROOT))


def _create_agent(model_name: str) -> ReActAgent:
    """功能：创建并缓存 Streamlit 会话使用的 ReActAgent。
    参数：
    - model_name：OpenRouter 模型名。
    返回值：
    - ReActAgent：初始化完成的智能体实例。
    """
    return ReActAgent(
        model=model_name,
        project_directory=str(ROOT),
        skill_system=_get_skill_system(),
        key_channel="web",
    )


@st.cache_resource(show_spinner="正在连接 Web 会话存储...")
def _get_web_services(model_name: str, bot_id: str):
    """功能：初始化 Web 聊天存储、会话管理器，并在首次加载时可选预热 MCP/Agent。
    参数：
    - model_name：OpenRouter 模型名。
    - bot_id：Web 端 bot 标识。
    返回值：
    - tuple[ChatStore, SessionManager]：聊天存储与会话管理器实例。
    """
    state_cache = RedisCache()
    chat_store = ChatStore(bot_id=bot_id)
    session_manager = SessionManager(
        agent_factory=lambda: _create_agent(model_name),
        chat_store=chat_store,
        ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "1800")),
        cache=state_cache,
        bot_id=bot_id,
    )
    if WEB_MCP_WARMUP_ENABLED:
        try:
            warm_mcp_tools(str(ROOT), force_refresh=True)
        except Exception as exc:  # noqa: BLE001
            safe_print(f"[Streamlit] MCP 预热失败: {exc}")
            metrics.inc("mcp_warmup_failure_total")
    if WEB_AGENT_WARMUP_ENABLED:
        try:
            warm_agent(
                model=model_name,
                project_directory=str(ROOT),
                skill_system=_get_skill_system(),
                key_channel="web",
            )
        except Exception as exc:  # noqa: BLE001
            safe_print(f"[Streamlit] Agent 预热失败: {exc}")
    return chat_store, session_manager


def _new_web_session_id() -> str:
    """功能：生成新的 Web 会话 ID。
    参数：
    - 无。
    返回值：
    - str：形如 `web-{uuid}` 的会话标识。
    """
    return f"web-{uuid.uuid4().hex}"


def _new_web_user_id() -> str:
    """功能：生成新的 Web 匿名用户 ID。
    参数：
    - 无。
    返回值：
    - str：形如 `web-user-{hex32}` 的用户标识。
    """
    return f"web-user-{uuid.uuid4().hex}"


def _query_param_first(name: str) -> str:
    """功能：读取 Streamlit URL 查询参数的首个值。
    参数：
    - name：查询参数名。
    返回值：
    - str：参数值字符串；不存在或异常时返回空字符串。
    """
    try:
        raw = st.query_params.get(name, "")
    except Exception:
        return ""
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    return str(raw or "").strip()


def _context_cookie(name: str) -> str:
    """功能：从 Streamlit 请求上下文读取 cookie 值。
    参数：
    - name：cookie 名称。
    返回值：
    - str：cookie 值；不存在或异常时返回空字符串。
    """
    try:
        return str(st.context.cookies.get(name) or "").strip()
    except Exception:
        return ""


def _is_web_user_id(value: str) -> bool:
    """功能：校验字符串是否为合法的 Web 用户 ID 格式。
    参数：
    - value：待校验字符串。
    返回值：
    - bool：符合 `web-user-{32位hex}` 格式时返回 True。
    """
    return bool(re.fullmatch(r"web-user-[0-9a-f]{32}", str(value or "").strip()))


def _embed_hidden_html(html_content: str) -> None:
    """功能：在页面中注入不可见 iframe，用于 cookie / localStorage 脚本。
    参数：
    - html_content：iframe 内嵌 HTML/JS 内容。
    返回值：
    - 无。
    """
    hidden_html = (
        "<style>html,body{margin:0;padding:0;overflow:hidden;height:1px;width:1px;opacity:0;}</style>"
        f"{html_content}"
    )
    st.iframe(hidden_html, height=1, width=1, tab_index=-1)


def _bootstrap_web_user_identity() -> None:
    """功能：通过隐藏 iframe 脚本初始化或恢复 Web 用户 ID（cookie + localStorage）。
    参数：
    - 无。
    返回值：
    - 无。必要时将 user_id 写入 URL 查询参数。
    """
    escaped_name = json.dumps(WEB_USER_COOKIE_NAME)
    escaped_query = json.dumps(WEB_USER_QUERY_PARAM)
    max_age = max(1, WEB_USER_COOKIE_DAYS) * 24 * 60 * 60
    _embed_hidden_html(
        f"""
        <script>
        (() => {{
          const cookieName = {escaped_name};
          const queryName = {escaped_query};
          const maxAge = {max_age};
          const pattern = /^web-user-[0-9a-f]{{32}}$/;
          const readCookie = () => {{
            const parts = document.cookie.split(";").map((item) => item.trim());
            const row = parts.find((item) => item.startsWith(`${{cookieName}}=`));
            return row ? decodeURIComponent(row.slice(cookieName.length + 1)) : "";
          }};
          const randomHex = () => {{
            const bytes = new Uint8Array(16);
            crypto.getRandomValues(bytes);
            return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
          }};
          let stored = "";
          try {{ stored = localStorage.getItem(cookieName) || ""; }} catch (error) {{}}
          let userId = readCookie();
          if (!pattern.test(userId) && pattern.test(stored)) userId = stored;
          if (!pattern.test(userId)) userId = `web-user-${{randomHex()}}`;
          document.cookie = `${{cookieName}}=${{encodeURIComponent(userId)}}; path=/; max-age=${{maxAge}}; SameSite=Lax`;
          try {{ localStorage.setItem(cookieName, userId); }} catch (error) {{}}
          let targetWindow = window;
          try {{
            if (window.parent && window.parent.location) targetWindow = window.parent;
          }} catch (error) {{}}
          const url = new URL(targetWindow.location.href);
          if (url.searchParams.get(queryName) !== userId) {{
            url.searchParams.set(queryName, userId);
            targetWindow.location.replace(url.toString());
          }}
        }})();
        </script>
        """,
    )


def _persist_web_user_identity(user_id: str) -> None:
    """功能：将 Web 用户 ID 持久化到 cookie、localStorage 并同步 URL 参数。
    参数：
    - user_id：Web 用户标识。
    返回值：
    - 无。
    """
    escaped_name = json.dumps(WEB_USER_COOKIE_NAME)
    escaped_query = json.dumps(WEB_USER_QUERY_PARAM)
    escaped_user = json.dumps(user_id)
    max_age = max(1, WEB_USER_COOKIE_DAYS) * 24 * 60 * 60
    _embed_hidden_html(
        f"""
        <script>
        (() => {{
          const cookieName = {escaped_name};
          const queryName = {escaped_query};
          const userId = {escaped_user};
          const maxAge = {max_age};
          let topWindow = window;
          let doc = window.document;
          let storage = null;
          try {{
            if (window.parent && window.parent.location && window.parent.location.origin === window.location.origin) {{
              topWindow = window.parent;
              doc = topWindow.document;
            }}
          }} catch (error) {{}}
          try {{ storage = topWindow.localStorage || window.localStorage; }} catch (error) {{}}
          const cookieValue = encodeURIComponent(userId);
          doc.cookie = `${{cookieName}}=${{cookieValue}}; path=/; max-age=${{maxAge}}; SameSite=Lax`;
          try {{ window.document.cookie = `${{cookieName}}=${{cookieValue}}; path=/; max-age=${{maxAge}}; SameSite=Lax`; }} catch (error) {{}}
          try {{ if (storage) storage.setItem(cookieName, userId); }} catch (error) {{}}
          try {{
            const url = new URL(topWindow.location.href);
            const hasCookie = doc.cookie.includes(`${{cookieName}}=`) || window.document.cookie.includes(`${{cookieName}}=`);
            if (url.searchParams.get(queryName) !== userId && !hasCookie) {{
              url.searchParams.set(queryName, userId);
              topWindow.location.replace(url.toString());
            }} else if (url.searchParams.has(queryName) && hasCookie) {{
              url.searchParams.delete(queryName);
              topWindow.history.replaceState(null, "", url.toString());
            }}
          }} catch (error) {{
            try {{ window.parent.postMessage({{ type: "wechat-agent-web-user", queryName, userId }}, "*"); }} catch (ignored) {{}}
          }}
        }})();
        </script>
        """,
    )


def _resolve_web_user_id() -> str:
    """功能：从 cookie、URL 参数或 session_state 解析 Web 用户 ID，必要时新建。
    参数：
    - 无。
    返回值：
    - str：合法的 Web 用户标识。
    """
    user_id = _context_cookie(WEB_USER_COOKIE_NAME)
    if not _is_web_user_id(user_id):
        user_id = _query_param_first(WEB_USER_QUERY_PARAM)
    if not _is_web_user_id(user_id):
        user_id = str(st.session_state.get("web_user_id") or "").strip()
    if not _is_web_user_id(user_id):
        user_id = _new_web_user_id()
    st.session_state.web_user_id = user_id
    _persist_web_user_identity(user_id)
    return user_id


def _load_web_messages(chat_store: ChatStore, *, user_id: str, session_id: str) -> list[dict]:
    """功能：从数据库加载会话消息并转换为 UI 所需格式。
    参数：
    - chat_store：ChatStore 实例。
    - user_id：Web 用户标识。
    - session_id：会话标识。
    返回值：
    - list[dict]：含 role 与 content 的消息字典列表。
    """
    return [
        {"role": message["role"], "content": message["content"]}
        for message in chat_store.load_messages(user_id=user_id, session_id=session_id)
    ]


def _invalidate_session_summaries_cache() -> None:
    """功能：清除侧栏会话列表的 session_state 缓存。
    参数：
    - 无。
    返回值：
    - 无。
    """
    st.session_state.pop("web_session_summaries_cache", None)


def _list_web_session_summaries(chat_store: ChatStore, *, user_id: str, query: str, force_refresh: bool = False):
    """功能：列出用户会话摘要，优先命中 session_state 缓存。
    参数：
    - chat_store：ChatStore 实例。
    - user_id：Web 用户标识。
    - query：可选搜索关键词。
    - force_refresh：为 True 时跳过缓存重新查询。
    返回值：
    - list：ChatSessionSummary 或兼容对象的列表。
    """
    cache = st.session_state.get("web_session_summaries_cache") or {}
    cache_key = f"{user_id}:{(query or '').strip()}"
    if not force_refresh and cache.get("key") == cache_key and cache.get("items") is not None:
        return cache["items"]
    items = chat_store.list_sessions(user_id=user_id, query=query, limit=80)
    st.session_state.web_session_summaries_cache = {"key": cache_key, "items": items}
    return items


def _select_web_session(chat_store: ChatStore, *, user_id: str, session_id: str) -> None:
    """功能：切换当前 Web 会话并加载历史消息到 session_state。
    参数：
    - chat_store：ChatStore 实例。
    - user_id：Web 用户标识。
    - session_id：目标会话标识。
    返回值：
    - 无。
    """
    st.session_state.current_web_session_id = session_id
    st.session_state.loaded_web_session_id = session_id
    st.session_state.messages = _load_web_messages(chat_store, user_id=user_id, session_id=session_id)
    st.session_state.pending_assistant = False
    st.session_state.draft_web_session_id = ""
    st.session_state.pop("agent", None)
    st.session_state.pop("agent_session_id", None)


def _reset_to_welcome() -> None:
    """功能：重置 UI 到欢迎页状态，清空当前会话与 Agent 缓存。
    参数：
    - 无。
    返回值：
    - 无。
    """
    st.session_state.current_web_session_id = ""
    st.session_state.loaded_web_session_id = ""
    st.session_state.messages = []
    st.session_state.pending_assistant = False
    st.session_state.draft_web_session_id = ""
    st.session_state.pop("agent", None)
    st.session_state.pop("agent_session_id", None)
    _invalidate_session_summaries_cache()


def _ensure_current_web_session(chat_store: ChatStore, *, user_id: str) -> str:
    """功能：确保 session_state 中的当前会话已加载，必要时触发加载。
    参数：
    - chat_store：ChatStore 实例。
    - user_id：Web 用户标识。
    返回值：
    - str：当前会话 ID；无选中会话时返回空字符串。
    """
    session_id = str(st.session_state.get("current_web_session_id") or "").strip()
    if not session_id:
        st.session_state.current_web_session_id = ""
        st.session_state.loaded_web_session_id = ""
        return ""
    if st.session_state.get("loaded_web_session_id") != session_id:
        _select_web_session(chat_store, user_id=user_id, session_id=session_id)
    return session_id


def _relative_session_time(value) -> str:
    """功能：将会话时间戳转换为中文相对时间描述。
    参数：
    - value：datetime 或可解析的 ISO 时间字符串。
    返回值：
    - str：如「刚刚」「3分钟前」；无法解析时返回空字符串。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return ""
    if not isinstance(value, datetime):
        return ""
    delta = datetime.now() - value.replace(tzinfo=None)
    if delta < timedelta(minutes=1):
        return "刚刚"
    if delta < timedelta(hours=1):
        return f"{max(1, int(delta.total_seconds() // 60))}分钟前"
    if delta < timedelta(days=1):
        return f"{max(1, int(delta.total_seconds() // 3600))}小时前"
    if delta < timedelta(days=7):
        return f"{max(1, delta.days)}天前"
    if delta < timedelta(days=30):
        return f"{max(1, delta.days // 7)}周前"
    return value.strftime("%m-%d")


def _format_session_title(title: str, *, max_len: int = 14) -> str:
    """功能：格式化侧栏会话标题，超长时截断并加省略号。
    参数：
    - title：原始会话标题。
    - max_len：最大显示字符数。
    返回值：
    - str：格式化后的标题。
    """
    clean_title = " ".join(str(title or "新会话").split()) or "新会话"
    if len(clean_title) > max_len:
        return f"{clean_title[:max_len]}..."
    return clean_title


def _format_session_button_label(title: str, session_time: str, model_name: str = "") -> str:
    """功能：生成侧栏会话项按钮文案（标题 + 模型 · 相对时间）。
    参数：
    - title：会话标题。
    - session_time：相对时间描述。
    - model_name：可选模型名。
    返回值：
    - str：两行按钮标签文本。
    """
    model_label = str(model_name or "").strip()
    meta = f"{model_label} · {session_time}" if model_label else session_time
    return f"{_format_session_title(title)}\n{meta}"


def _format_session_time(updated_at=None) -> str:
    """功能：格式化会话更新时间为相对时间，缺省显示「刚刚」。
    参数：
    - updated_at：可选时间戳或 datetime。
    返回值：
    - str：相对时间描述。
    """
    relative = _relative_session_time(updated_at)
    return relative or "刚刚"


def _queue_web_prompt(text: str) -> None:
    """功能：将快捷提示词写入 session_state 待发送队列。
    参数：
    - text：用户点击的提示词文本。
    返回值：
    - 无。空文本时直接返回。
    """
    value = str(text or "").strip()
    if not value:
        return
    st.session_state.queued_web_prompt = value
    st.session_state.pending_assistant = False


def _render_quick_actions() -> None:
    """功能：渲染欢迎页下方的快捷操作按钮行。
    参数：
    - 无。
    返回值：
    - 无。点击按钮会将提示词入队并触发 rerun。
    """
    st.markdown('<div class="wa-quick-actions"></div>', unsafe_allow_html=True)
    quick_items = ["生成测试用例", "查询接口说明", "分析报错原因", "整理复现步骤"]
    quick_cols = st.columns([1, 1, 1, 1], gap="small")
    for idx, text in enumerate(quick_items):
        with quick_cols[idx]:
            if st.button(text, key=f"web_quick_{idx}", width="stretch"):
                _queue_web_prompt(text)
                st.rerun()


def _render_empty_state() -> None:
    """功能：在无历史消息时展示空状态引导区。
    参数：
    - 无。
    返回值：
    - 无。
    """
    st.markdown(
        """
        <section class="wa-welcome">
            <div class="wa-brand-icon wa-brand-icon-lg">▣</div>
            <h2 class="wa-welcome-title">你好！我是 WeChat Agent</h2>
            <p class="wa-welcome-subtitle">可以帮你查询知识库、联网搜索、调用 MCP 工具并完成日常问答</p>
            <div class="wa-capability-grid">
                <article class="wa-capability-card">
                    <div class="wa-cap-icon">□</div>
                    <div class="wa-capability-title">知识库问答</div>
                    <div class="wa-capability-copy">查询项目文档、接口说明、配置规范、版本说明等内容。</div>
                </article>
                <article class="wa-capability-card">
                    <div class="wa-cap-icon">⌕</div>
                    <div class="wa-capability-title">测试辅助</div>
                    <div class="wa-capability-copy">辅助分析测试问题，整理复现步骤和排查建议。</div>
                </article>
                <article class="wa-capability-card">
                    <div class="wa-cap-icon">▤</div>
                    <div class="wa-capability-title">日常问答</div>
                    <div class="wa-capability-copy">解答常见问题，提供操作建议与最佳实践指导。</div>
                </article>
            </div>
            <div class="wa-prompt-title">试试这样问我</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    suggestions = [
        "帮我生成一组登录接口的功能测试用例",
        "查询支付回调接口的配置说明文档",
        "接口返回超时错误，帮我分析可能的原因",
    ]
    with st.container(key="wa_welcome_actions"):
        for idx, text in enumerate(suggestions):
            if st.button(text, key=f"web_suggestion_{idx}", width="stretch"):
                _queue_web_prompt(text)
                st.rerun()


def _reset_agent_state(agent: ReActAgent) -> None:
    """功能：清空 Agent 技能会话与对话上下文。
    参数：
    - agent：待重置的智能体实例。
    返回值：
    - 无。
    """
    agent.skill_session.active_skills.clear()
    agent.skill_session.completed_skills.clear()
    agent.skill_session.dismissed_skills.clear()
    agent.skill_session.current_request_id = 0
    agent.messages = [{"role": "system", "content": agent.render_system_prompt()}]
    agent.conversation_turns = []


def _sanitize_trace_display(text: str) -> str:
    """功能：清理执行过程文本中的 ReAct/XML 协议标签。
    参数：
    - text：原始 trace 文本。
    返回值：
    - str：适合页面展示的执行过程正文。
    """
    if not text:
        return ""
    s = str(text)
    s = re.sub(r"<action>.*?</action>", "", s, flags=re.DOTALL)
    s = re.sub(r"<observation>.*?</observation>", "", s, flags=re.DOTALL)
    s = re.sub(r"<final_answer>.*?</final_answer>", "", s, flags=re.DOTALL)
    s = re.sub(r"</?thought>", "", s)
    s = re.sub(r"</?[^>]+>", "", s)
    return s


def _format_trace_lines(raw: str) -> str:
    """功能：将 trace 文本格式化为带 `- ` 前缀的多行列表。
    参数：
    - raw：原始 trace 文本。
    返回值：
    - str：格式化后的多行文本。
    """
    body = _sanitize_trace_display(raw)
    lines = []
    for line in body.splitlines():
        item = line.strip()
        if not item:
            continue
        if item.startswith("- "):
            lines.append(item)
        else:
            lines.append(f"- {item}")
    return "\n".join(lines)


def _thinking_markup(raw: str) -> str:
    """功能：将执行过程正文包装为可折叠 HTML 区块。
    参数：
    - raw：原始 trace 文本。
    返回值：
    - str：HTML 字符串；无内容时返回空串。
    """
    body = _format_trace_lines(raw).strip()
    if not body:
        return ""
    escaped = html.escape(body)
    return (
        '<details class="thought-panel">'
        "<summary>执行过程</summary>"
        f'<div class="thought-panel-body">{escaped}</div>'
        "</details>"
    )


def _format_answer_block(raw: str) -> str:
    """功能：格式化最终答案文本并软换行长代码块。
    参数：
    - raw：含可选 KB 图片标记的原始答案。
    返回值：
    - str：适合 Markdown 渲染的正文。
    """
    body = strip_kb_image_markers(raw or "").strip()
    if not body:
        return ""
    return _wrap_markdown_code_blocks(body, width=WEB_CODE_WRAP_COLUMN)


def _wrap_long_line(line: str, *, width: int) -> str:
    """功能：按固定宽度软换行单行文本。
    参数：
    - line：原始单行文本。
    - width：最大行宽。
    返回值：
    - str：换行后的文本。
    """
    if width <= 0 or len(line) <= width:
        return line
    chunks = [line[i : i + width] for i in range(0, len(line), width)]
    return "\n".join(chunks)


def _wrap_markdown_code_blocks(text: str, *, width: int = 120) -> str:
    """功能：对 Markdown 代码块内的超长行进行软换行。
    参数：
    - text：Markdown 文本。
    - width：代码行最大宽度。
    返回值：
    - str：换行处理后的 Markdown 文本。
    """
    if not text or width <= 0:
        return text

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_fence = False
    for raw_line in lines:
        newline = "\n" if raw_line.endswith("\n") else ""
        line = raw_line[:-1] if newline else raw_line
        stripped = line.lstrip()
        if stripped.startswith("```"):
            out.append(raw_line)
            in_fence = not in_fence
            continue

        is_indented_code = line.startswith("    ") or line.startswith("\t")
        if in_fence or is_indented_code:
            out.append(_wrap_long_line(line, width=width) + newline)
        else:
            out.append(raw_line)
    return "".join(out)


def _build_answer_segments(raw: str) -> list[tuple[str, str]]:
    """功能：将含 [KB_IMAGE:...] 的回复拆分为 text/image 片段列表。
    参数：
    - raw：完整助手回复文本。
    返回值：
    - list[tuple[str, str]]：`(kind, payload)` 片段列表。
    """
    segments: list[tuple[str, str]] = []
    for kind, payload in split_kb_image_markers(raw or ""):
        if kind == "text" and payload:
            if segments and segments[-1][0] == "text":
                segments[-1] = ("text", segments[-1][1] + payload)
            else:
                segments.append(("text", payload))
        elif kind == "image" and payload:
            segments.append(("image", payload))
    return segments


def _segments_to_marked_text(parts: list[tuple[str, str]]) -> str:
    """功能：将渲染片段还原为含 [KB_IMAGE:...] 标记的历史文本。
    参数：
    - parts：text/image 片段列表。
    返回值：
    - str：可再次渲染的完整文本。
    """
    chunks: list[str] = []
    for kind, payload in parts:
        if kind == "text" and payload:
            chunks.append(payload)
        elif kind == "image" and payload:
            chunks.append(f"[KB_IMAGE:{payload}]")
    return "".join(chunks)


class _WebKbImageStreamSplitter:
    """功能：流式拆分 final_answer，缓冲未闭合的 [KB_IMAGE:...] 标记。
    参数：
    - 无。
    返回值：
    - 无。
    """

    _prefix = "[KB_IMAGE:"

    def __init__(self) -> None:
        """功能：初始化流式 KB 图片标记拆分器的内部缓冲。
        参数：
        - 无。
        返回值：
        - 无。
        """
        self._buffer = ""

    def feed(self, chunk: str):
        """功能：接收流式文本块并产出已解析的 text/image 片段。
        参数：
        - chunk：新增的流式文本块。
        返回值：
        - Generator：`(kind, payload)` 元组的生成器。
        """
        if not chunk:
            return
        self._buffer += chunk
        yield from self._drain(final=False)

    def flush(self):
        """功能：刷新缓冲区，产出剩余片段（含未闭合标记的兜底处理）。
        参数：
        - 无。
        返回值：
        - Generator：`(kind, payload)` 元组的生成器。
        """
        yield from self._drain(final=True)

    def _drain(self, *, final: bool):
        """功能：从内部缓冲区解析并产出 text/image 片段。
        参数：
        - final：为 True 时强制输出剩余缓冲内容。
        返回值：
        - Generator：`(kind, payload)` 元组的生成器。
        """
        while self._buffer:
            start = self._buffer.find(self._prefix)
            if start == -1:
                keep = "" if final else self._partial_prefix_suffix(self._buffer)
                emit_len = len(self._buffer) - len(keep)
                if emit_len > 0:
                    text = self._buffer[:emit_len]
                    self._buffer = self._buffer[emit_len:]
                    yield ("text", text)
                if final and self._buffer:
                    text = self._buffer
                    self._buffer = ""
                    yield ("text", text)
                return

            if start > 0:
                text = self._buffer[:start]
                self._buffer = self._buffer[start:]
                yield ("text", text)
                continue

            end = self._buffer.find("]", len(self._prefix))
            if end == -1:
                if final:
                    text = self._buffer
                    self._buffer = ""
                    yield ("text", text)
                return

            path = self._buffer[len(self._prefix) : end].strip()
            self._buffer = self._buffer[end + 1 :]
            if path:
                yield ("image", path)

    @classmethod
    def _partial_prefix_suffix(cls, text: str) -> str:
        """功能：检测文本末尾是否为 `[KB_IMAGE:` 前缀的部分匹配，需保留缓冲。
        参数：
        - text：待检测文本。
        返回值：
        - str：需保留在缓冲区末尾的后缀；无部分匹配时返回空字符串。
        """
        max_len = min(len(text), len(cls._prefix) - 1)
        for size in range(max_len, 0, -1):
            suffix = text[-size:]
            if cls._prefix.startswith(suffix):
                return suffix
        return ""


def _render_answer_parts(placeholder, parts: list[tuple[str, str]], *, render_images: bool = True) -> None:
    """功能：按片段顺序渲染 Markdown 文本与内联图片。
    参数：
    - placeholder：Streamlit empty 占位符。
    - parts：text/image 片段列表。
    - render_images：流式过程中可设为 False，仅在最终刷新时渲染图片，避免重复调用 st.image。
    返回值：
    - 无。
    """
    if placeholder is None:
        return
    with placeholder.container():
        for kind, payload in parts:
            if kind == "text" and payload:
                st.markdown(payload)
            elif kind == "image" and payload and render_images:
                resolved = normalize_kb_image_path(payload, ROOT)
                if resolved is not None:
                    st.image(str(resolved), caption=None, width="stretch")
                else:
                    st.caption(f"[图示文件不可用] {payload}")


def _render_answer_text(placeholder, raw: str) -> None:
    """功能：统一渲染 Agent 回复，自动拆分并展示知识库图片。
    参数：
    - placeholder：Streamlit 占位符。
    - raw：原始回复文本。
    返回值：
    - 无。
    """
    if placeholder is None:
        return
    content = raw or ""
    if "[KB_IMAGE:" in content:
        _render_answer_parts(placeholder, _build_answer_segments(content))
    else:
        placeholder.markdown(_format_answer_block(content), unsafe_allow_html=True)


def _split_assistant_history(raw: str) -> tuple[str, str]:
    """功能：拆分历史助手消息中的思考区 HTML 与正文。
    参数：
    - raw：存储在 session 中的助手消息全文。
    返回值：
    - tuple[str, str]：`(thinking_html, answer_text)`。
    """
    text = raw or ""
    marker = '<details class="thought-panel"'
    if not text.strip().startswith(marker):
        return "", text
    close = "</details>"
    end = text.find(close)
    if end == -1:
        return "", text
    thinking = text[: end + len(close)]
    answer = text[end + len(close) :].strip()
    if answer.startswith("\n\n"):
        answer = answer[2:].strip()
    return thinking, answer


def _render_assistant_message(content: str, *, key_prefix: str) -> None:
    """功能：渲染单条助手消息（含可选思考区与正文）。
    参数：
    - content：助手消息 Markdown/HTML 内容。
    - key_prefix：Streamlit 组件 key 前缀。
    返回值：
    - 无。
    """
    _thinking, answer = _split_assistant_history(content)
    if answer:
        _render_answer_text(st.container(key=f"{key_prefix}_answer"), answer)


def _loading_dots_html() -> str:
    """功能：生成助手回复加载中的动画 HTML。
    参数：
    - 无。
    返回值：
    - str：含三点动画的 HTML 字符串。
    """
    return (
        '<div class="wa-loading-state" aria-label="AI 正在生成回复">'
        '<div class="wa-loading-dots">'
        '<span></span><span></span><span></span></div>'
        '</div>'
    )


def _render_user_bubble(content: str) -> None:
    """功能：渲染用户消息气泡 HTML。
    参数：
    - content：用户消息文本。
    返回值：
    - 无。
    """
    st.markdown(
        f"""
        <div class="wa-chat-row wa-chat-row-user">
            <div class="wa-user-bubble">{html.escape(content or "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_assistant_shell_open() -> None:
    """功能：渲染助手消息容器开头 HTML（含头像占位）。
    参数：
    - 无。
    返回值：
    - 无。
    """
    st.markdown(
        """
        <div class="wa-assistant-shell">
            <div class="wa-assistant-avatar" aria-hidden="true"></div>
        """,
        unsafe_allow_html=True,
    )


def _render_assistant_shell_close() -> None:
    """功能：渲染助手消息容器结尾 HTML。
    参数：
    - 无。
    返回值：
    - 无。
    """
    st.markdown("</div>", unsafe_allow_html=True)


class _BufferedPlaceholder:
    """功能：节流刷新 Streamlit 占位符，降低高频 stream 重绘开销。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(self, placeholder, render, *, unsafe_allow_html: bool = False):
        """功能：初始化节流占位符刷新器。
        参数：
        - placeholder：Streamlit empty 占位符。
        - render：将原始文本转为展示内容的渲染函数。
        - unsafe_allow_html：是否以 unsafe_allow_html 模式写入 Markdown。
        返回值：
        - 无。
        """
        self.placeholder = placeholder
        self.render = render
        self.unsafe_allow_html = unsafe_allow_html
        self.last_flush = 0.0
        self.dirty = False

    def mark_dirty(self) -> None:
        """功能：标记占位符内容已变更，待下次 flush 刷新。
        参数：
        - 无。
        返回值：
        - 无。
        """
        self.dirty = True

    def flush(self, raw: str, *, force: bool = False) -> None:
        """功能：按节流间隔刷新占位符内容。
        参数：
        - raw：原始文本，经 render 函数转换后展示。
        - force：为 True 时跳过节流立即刷新。
        返回值：
        - 无。
        """
        if self.placeholder is None or not self.dirty:
            return
        now = time.monotonic()
        if not force and now - self.last_flush < WEB_STREAM_FLUSH_INTERVAL_SECONDS:
            return
        self.placeholder.markdown(self.render(raw), unsafe_allow_html=self.unsafe_allow_html)
        self.last_flush = now
        self.dirty = False


def _run_agent_with_web_stream_adapter(
    agent: ReActAgent,
    user_text: str,
    *,
    stream_callback,
) -> str:
    """功能：包装 agent.arun，供 Web 流式 UI 后台线程调用。
    在 worker 线程里用 asyncio.run() 驱动 arun()，不阻塞 Streamlit 主线程。
    参数：
    - agent：ReActAgent 实例。
    - user_text：用户输入。
    - stream_callback：流式事件回调。
    返回值：
    - str：最终答案文本。
    """
    import asyncio
    return asyncio.run(agent.arun(user_text, stream_callback=stream_callback))


def _run_agent_stream_ui(
    agent: ReActAgent,
    user_text: str,
    *,
    answer_out: dict,
    errors_out: list,
    show_thinking: bool,
    thought_placeholder,
    answer_placeholder,
) -> tuple[str, str, list[tuple[str, str]]]:
    """功能：后台运行 Agent 并在主线程刷新思考区与答案区。
    参数：
    - agent：ReActAgent 实例。
    - user_text：用户输入。
    - answer_out：用于接收最终答案的字典。
    - errors_out：用于收集异常的列表。
    - show_thinking：是否展示执行过程。
    - thought_placeholder：思考区占位符。
    - answer_placeholder：答案区占位符。
    返回值：
    - tuple[str, str, list]：思考 trace、答案文本与渲染片段列表。
    """
    q: "queue.Queue[object]" = queue.Queue()
    done = object()
    console_callback, _ = make_console_stream_callback(web_mode=True)

    def stream_callback(section: str, chunk: str) -> None:
        """功能：Agent 流式事件回调，分发到控制台与 UI 队列。
        参数：
        - section：流式事件类型（如 final_answer、tool_start）。
        - chunk：事件文本块。
        返回值：
        - 无。
        """
        console_callback(section, chunk)
        if section in ("final_answer", "final_answer_flush"):
            q.put((section, chunk or ""))
            return
        if not chunk:
            return
        if show_thinking and section in ("agent_start", "model_decision", "tool_start", "tool_end", "agent_finish", "error"):
            q.put((section, chunk))

    def worker() -> None:
        """功能：后台线程运行 Agent 并将结果写入 answer_out。
        参数：
        - 无。
        返回值：
        - 无。异常写入 errors_out，结束时向队列发送 done 信号。
        """
        try:
            answer_out["text"] = _run_agent_with_web_stream_adapter(
                agent,
                user_text,
                stream_callback=stream_callback,
            )
        except Exception as exc:  # noqa: BLE001
            errors_out.append(exc)
        finally:
            q.put((done, ""))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    thought_parts: list[str] = []
    answer_segments: list[tuple[str, str]] = []
    answer_text_buf = ""
    kb_splitter = _WebKbImageStreamSplitter()
    thought_buffer = _BufferedPlaceholder(
        thought_placeholder,
        _thinking_markup,
        unsafe_allow_html=True,
    )

    def _append_answer_text(text: str) -> None:
        """功能：追加答案文本到缓冲区与渲染片段列表。
        参数：
        - text：待追加的文本块。
        返回值：
        - 无。
        """
        nonlocal answer_text_buf
        if not text:
            return
        answer_text_buf += text
        if answer_segments and answer_segments[-1][0] == "text":
            answer_segments[-1] = ("text", answer_segments[-1][1] + text)
        else:
            answer_segments.append(("text", text))

    def _consume_final_answer_chunk(chunk: str) -> None:
        """功能：处理 final_answer 流式块，拆分 KB 图片并增量渲染。
        参数：
        - chunk：final_answer 文本块。
        返回值：
        - 无。
        """
        for kind, payload in kb_splitter.feed(chunk):
            if kind == "text":
                _append_answer_text(payload)
            elif kind == "image" and payload:
                answer_segments.append(("image", payload))
                answer_segments.append(("text", ""))
        _render_answer_parts(answer_placeholder, answer_segments, render_images=False)

    while True:
        item = q.get()
        if item[0] is done:
            break
        kind, chunk = item
        if kind in ("agent_start", "model_decision", "tool_start", "tool_end", "agent_finish", "error"):
            thought_parts.append(chunk)
            if not chunk.endswith("\n"):
                thought_parts.append("\n\n")
            thought_buffer.mark_dirty()
        elif kind == "final_answer":
            _consume_final_answer_chunk(chunk)
        elif kind == "final_answer_flush":
            for seg_kind, payload in kb_splitter.flush():
                if seg_kind == "text":
                    _append_answer_text(payload)
                elif seg_kind == "image" and payload:
                    answer_segments.append(("image", payload))
                    answer_segments.append(("text", ""))
            _render_answer_parts(answer_placeholder, answer_segments)
        thought_buffer.flush("".join(thought_parts))

    thread.join(timeout=300)
    thought_buffer.flush("".join(thought_parts), force=True)
    runtime_raw = (answer_out.get("text") or "").strip()
    if runtime_raw and "[KB_IMAGE:" in runtime_raw:
        answer_segments = _build_answer_segments(runtime_raw)
        answer_text_buf = strip_kb_image_markers(runtime_raw)
    if answer_segments:
        _render_answer_parts(answer_placeholder, answer_segments)
    return "".join(thought_parts), answer_text_buf, answer_segments


def main() -> None:
    """功能：Streamlit 脚本入口，校验运行上下文并启动 UI。
    参数：
    - 无。
    返回值：
    - 无。
    """
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
    except ImportError:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

    if get_script_run_ctx() is None:
        script = Path(__file__).resolve()
        print(
            "\n[Streamlit] 请使用下列命令启动（勿用 `python` 直接运行本文件）：\n\n"
            f'    "{sys.executable}" -m streamlit run "{script}"\n',
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        _main_ui()
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)
        safe_print(f"[Streamlit] 界面致命错误: {exc}")


def _main_ui() -> None:
    """功能：构建 Streamlit 主界面、侧边栏与聊天交互逻辑。
    参数：
    - 无。
    返回值：
    - 无。
    """
    st.set_page_config(page_title="WeChat Agent", layout="wide", initial_sidebar_state="expanded")
    _inject_app_css()

    model_name = (os.getenv("OPENROUTER_MODEL") or "").strip()
    if not model_name:
        st.error("缺少环境变量 OPENROUTER_MODEL，请在项目根目录 `.env` 中配置。")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_assistant" not in st.session_state:
        st.session_state.pending_assistant = False

    try:
        mysql_cfg = load_mysql_config()
        if not mysql_cfg.get("enabled"):
            st.error("Web 多会话历史需要 MYSQL_ENABLED=true，请先配置 MySQL 后再上线 Web 入口。")
            st.stop()
        web_user_id = _resolve_web_user_id()
        chat_store, session_manager = _get_web_services(model_name, WEB_BOT_ID)
        current_session_id = _ensure_current_web_session(chat_store, user_id=web_user_id)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Web 会话存储初始化失败：{exc}")
        st.stop()

    agent: ReActAgent | None = None
    agent_session_id = str(st.session_state.get("agent_session_id") or "").strip()
    if current_session_id and agent_session_id == current_session_id:
        agent = st.session_state.get("agent")
    tool_count = _tool_count(agent) if agent is not None else _get_web_tool_count(model_name)
    mcp_server_count = _mcp_server_count()

    with st.sidebar:
        _render_app_header(model_name, tool_count, mcp_server_count)
        show_thinking = False
        st.session_state.ui_show_thinking = False
        action_cols = st.columns([1, 0.31, 0.31])
        with action_cols[0]:
            new_clicked = st.button("+ 新会话", key="web_new_session_action", width="stretch")
        with action_cols[1]:
            refresh_clicked = st.button("↻", key="web_refresh_sessions_action", help="刷新对话列表", width="stretch")
        with action_cols[2]:
            clear_all_clicked = st.button("⌫", key="web_delete_all_sessions_action", help="删除所有会话", width="stretch")
        if new_clicked:
            _reset_to_welcome()
            st.rerun()
        if refresh_clicked:
            _invalidate_session_summaries_cache()
            st.rerun()
        if clear_all_clicked:
            chat_store.delete_user_sessions(user_id=web_user_id)
            session_manager.discard_user(user_id=web_user_id)
            _reset_to_welcome()
            st.rerun()
        session_query = st.text_input("搜索对话", key="web_session_search", placeholder="搜索历史对话")
        try:
            summaries = _list_web_session_summaries(
                chat_store,
                user_id=web_user_id,
                query=session_query,
                force_refresh=refresh_clicked,
            )
        except Exception as exc:  # noqa: BLE001
            summaries = []
            st.warning(f"读取历史会话失败：{exc}")
        with st.container(key="web_session_list_scroll"):
            for summary in summaries:
                if not summary.session_id:
                    continue
                is_selected = summary.session_id == current_session_id
                row_key_prefix = "web_session_row_selected" if is_selected else "web_session_row"
                button_key_prefix = "web_session_selected" if is_selected else "web_session"
                session_time = _format_session_time(summary.last_active or summary.updated_at or summary.created_at)
                with st.container(key=f"{row_key_prefix}_{summary.session_id}"):
                    if st.button(
                        _format_session_button_label(summary.title, session_time, model_name),
                        key=f"{button_key_prefix}_{summary.session_id}",
                        width="stretch",
                    ):
                        _select_web_session(chat_store, user_id=web_user_id, session_id=summary.session_id)
                        st.rerun()

    queued_prompt = str(st.session_state.pop("queued_web_prompt", "") or "").strip()
    if queued_prompt:
        if not current_session_id:
            current_session_id = _new_web_session_id()
            st.session_state.current_web_session_id = current_session_id
            st.session_state.loaded_web_session_id = current_session_id
            st.session_state.draft_web_session_id = current_session_id
        st.session_state.messages.append({"role": "user", "content": queued_prompt})
        st.session_state.pending_assistant = True
        st.rerun()

    if not st.session_state.messages:
        st.markdown('<div class="wa-mode-welcome" aria-hidden="true"></div>', unsafe_allow_html=True)
        _render_empty_state()
    else:
        st.markdown('<div class="wa-mode-chat" aria-hidden="true"></div>', unsafe_allow_html=True)

    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "assistant":
            with st.container(key=f"wa_msg_assistant_hist_{idx}"):
                avatar_col, card_col = st.columns([0.06, 0.94], gap="small")
                with avatar_col:
                    st.markdown('<div class="wa-assistant-avatar" aria-hidden="true"></div>', unsafe_allow_html=True)
                with card_col:
                    with st.container(key=f"wa_card_hist_{idx}"):
                        _render_assistant_message(message["content"], key_prefix=f"hist_{idx}")
        else:
            _render_user_bubble(message["content"])

    _render_quick_actions()
    st.markdown('<div class="wa-footer-note">WeChat Agent · ReAct + RAG + MCP + 企业微信</div>', unsafe_allow_html=True)

    chat_prompt = st.chat_input("输入问题，按 Enter 发送")
    prompt = chat_prompt or ""
    if prompt:
        if not current_session_id:
            current_session_id = _new_web_session_id()
            st.session_state.current_web_session_id = current_session_id
            st.session_state.loaded_web_session_id = current_session_id
            st.session_state.draft_web_session_id = current_session_id
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_assistant = True
        st.rerun()

    if not st.session_state.pending_assistant:
        return

    st.session_state.pending_assistant = False
    if not st.session_state.messages or st.session_state.messages[-1]["role"] != "user":
        return
    if not current_session_id:
        current_session_id = str(st.session_state.get("current_web_session_id") or "").strip()
    if not current_session_id:
        current_session_id = _new_web_session_id()
        st.session_state.current_web_session_id = current_session_id
        st.session_state.loaded_web_session_id = current_session_id
        st.session_state.draft_web_session_id = current_session_id
    if agent is None:
        try:
            state = session_manager.get(user_id=web_user_id, session_id=current_session_id, touch=False)
            agent = state.agent
            st.session_state.agent = agent
            st.session_state.agent_session_id = current_session_id
        except Exception as exc:  # noqa: BLE001
            st.error(f"Agent 初始化失败：{exc}")
            st.stop()

    user_text = st.session_state.messages[-1]["content"]
    turn_idx = len(st.session_state.messages)
    request_started = time.time()
    lease_owner = f"web-{uuid.uuid4().hex}"
    session_lock = _get_web_session_lock(web_user_id, current_session_id)

    answer_holder: dict = {"text": ""}
    errors: list = []
    answer_for_persist = ""
    lease_acquire_ms = 0.0
    arun_ms = 0.0
    selector_ms = 0.0
    run_success = True

    lease_started = time.time()
    acquired = session_manager.acquire_lease(web_user_id, current_session_id, lease_owner)
    lease_acquire_ms = (time.time() - lease_started) * 1000.0
    if not acquired:
        metrics.inc("session_lease_conflicts_total")
        conflict_msg = "当前会话正在处理中，请稍后重试。"
        st.warning(conflict_msg)
        st.session_state.messages.append({"role": "assistant", "content": conflict_msg})
        log_timing(
            logger,
            metrics,
            "web_request_timing",
            phases={"lease_acquire": lease_acquire_ms},
            total_ms=(time.time() - request_started) * 1000.0,
            user_id=web_user_id,
            session_id=current_session_id,
            success=False,
            lease_conflict=True,
        )
        st.rerun()
        return

    with st.container(key=f"wa_msg_assistant_stream_{turn_idx}"):
        avatar_col, card_col = st.columns([0.06, 0.94], gap="small")
        with avatar_col:
            st.markdown('<div class="wa-assistant-avatar" aria-hidden="true"></div>', unsafe_allow_html=True)
        with card_col:
            card = st.container(key=f"wa_card_stream_{turn_idx}")
    with card:
        # 外层 container(key) 固定多轮位置；内层 empty 流式整块替换（不用 st.spinner，避免大块空白条）
        _stream_box = dict(border=False, height="content")
        thought_ph = None
        if show_thinking:
            thought_ph = st.container(key=f"stream_{turn_idx}_thought", **_stream_box).empty()
        answer_ph = st.container(key=f"stream_{turn_idx}_answer", **_stream_box).empty()
        answer_ph.markdown(_loading_dots_html(), unsafe_allow_html=True)

        try:
            with session_lock:
                arun_started = time.time()
                thinking_raw, answer_raw, answer_stream_parts = _run_agent_stream_ui(
                    agent,
                    user_text,
                    answer_out=answer_holder,
                    errors_out=errors,
                    show_thinking=show_thinking,
                    thought_placeholder=thought_ph,
                    answer_placeholder=answer_ph,
                )
                arun_ms = (time.time() - arun_started) * 1000.0
                selector_ms = _selector_elapsed_ms(agent)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            thinking_raw, answer_raw, answer_stream_parts = "", "", []
            run_success = False
            if arun_ms <= 0:
                arun_ms = (time.time() - arun_started) * 1000.0 if "arun_started" in locals() else 0.0
        finally:
            session_manager.release_lease(web_user_id, current_session_id, lease_owner)

        if errors:
            err = errors[0]
            safe_print(f"[Streamlit] Agent 运行错误: {err}")
            st.error(str(err))
            fallback = (answer_holder.get("text") or "").strip() or f"执行失败：{err}"
            _render_answer_text(answer_ph, fallback)
            reply_md = fallback
            answer_for_persist = fallback
            run_success = False
        else:
            final_from_runtime = (answer_holder.get("text") or "").strip()
            answer_from_stream = (answer_raw or "").strip()
            render_parts: list[tuple[str, str]] = list(answer_stream_parts)
            if answer_from_stream:
                answer_display = answer_from_stream
            else:
                answer_display = strip_kb_image_markers(final_from_runtime) or final_from_runtime
            if not answer_display and not render_parts:
                answer_display = "（未收到模型输出，请检查网络与 API Key。）"
                _render_answer_text(answer_ph, answer_display)
                st.warning(answer_display)
            else:
                if not render_parts and final_from_runtime and "[KB_IMAGE:" in final_from_runtime:
                    render_parts = _build_answer_segments(final_from_runtime)
                if render_parts:
                    _render_answer_parts(answer_ph, render_parts)
                else:
                    _render_answer_text(answer_ph, answer_display)

            if render_parts:
                answer_for_history = _segments_to_marked_text(render_parts)
            elif final_from_runtime and "[KB_IMAGE:" in final_from_runtime:
                answer_for_history = final_from_runtime
            else:
                answer_for_history = answer_display
            answer_for_persist = answer_for_history

            if show_thinking:
                think_block = _sanitize_trace_display(thinking_raw).strip()
                parts: list[str] = []
                if think_block:
                    parts.append(_thinking_markup(think_block))
                parts.append(answer_for_history)
                reply_md = "\n\n".join(parts)
            else:
                reply_md = answer_for_history

    log_timing(
        logger,
        metrics,
        "web_request_timing",
        phases={
            "lease_acquire": lease_acquire_ms,
            "selector": selector_ms,
            "arun": arun_ms,
        },
        total_ms=(time.time() - request_started) * 1000.0,
        user_id=web_user_id,
        session_id=current_session_id,
        success=run_success and not errors,
    )
    if run_success and not errors:
        metrics.inc("agent_runs_success_total")
        metrics.observe_ms("agent_run_duration", arun_ms)
    elif errors:
        metrics.inc("agent_runs_failure_total")

    _append_turn_background(
        chat_store,
        user_id=web_user_id,
        session_id=current_session_id,
        user_text=user_text,
        answer=answer_for_persist or reply_md,
    )
    if str(st.session_state.get("draft_web_session_id") or "") == current_session_id:
        st.session_state.draft_web_session_id = ""
    st.session_state.loaded_web_session_id = current_session_id
    _invalidate_session_summaries_cache()
    st.session_state.messages.append({"role": "assistant", "content": reply_md})
    st.rerun()


if __name__ == "__main__":
    main()
