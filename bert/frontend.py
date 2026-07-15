"""
电商评论情感分析 - 前端页面
启动: streamlit run frontend.py
依赖: pip install streamlit requests
"""
import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000/predict"

# ============================================================
# 颜色 & 图标映射（与 index.html 一致）
# ============================================================
SENT_COLORS = {"好评": "#10B981", "中评": "#F59E0B", "差评": "#EF4444"}
SENT_EMOJI  = {"好评": "😊", "中评": "😐", "差评": "😡"}

ASPECT_COLORS = {"正面": "#10B981", "负面": "#EF4444", "未提及": "#6B7280"}
ASPECT_EMOJI  = {
    "商品质量": "📦", "外观设计": "🎨", "使用体验": "⚡",
    "物流配送": "🚚", "价格": "💰", "客服服务": "🎧",
}
ASPECTS = ["商品质量", "外观设计", "使用体验", "物流配送", "价格", "客服服务"]

EXAMPLES = [
    ("👍 好评示例", "手机屏幕很大很清晰，电池也耐用，推荐购买！"),
    ("🤔 中评示例", "物流太慢包装压坏，但东西质量还不错"),
    ("👎 差评示例", "客服态度极其恶劣，退货拖一星期，再也不买了"),
]

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(page_title="电商评论情感分析", page_icon="🛒", layout="centered")

# ============================================================
# 自定义 CSS（还原 index.html 的视觉风格）
# ============================================================
st.markdown("""
<style>
    /* 全局 */
    .stApp { background: #F0F2F5; }

    /* 标题 */
    .header-icon { font-size: 3rem; display: block; text-align: center; }
    .main-title { font-size: 2rem; font-weight: 800; text-align: center; color: #1F2937; }
    .sub-title { text-align: center; color: #6B7280; font-size: 1rem; margin-bottom: 1rem; }

    /* 卡片容器 */
    .card {
        background: #FFFFFF; border-radius: 20px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.08); padding: 1.5rem 2rem;
        margin-bottom: 1.2rem;
    }

    /* 情感徽章 */
    .sentiment-badge {
        display: inline-block; padding: 0.6rem 2.4rem; border-radius: 50px;
        font-size: 1.5rem; font-weight: 800; color: #fff; letter-spacing: 0.02em;
    }

    /* 置信度条 */
    .conf-row { display: flex; align-items: center; gap: 0.8rem; margin: 0.5rem 0; }
    .conf-label { width: 50px; font-weight: 600; font-size: 0.9rem; }
    .conf-track { flex: 1; height: 10px; background: #E5E7EB; border-radius: 5px; overflow: hidden; }
    .conf-fill { height: 100%; border-radius: 5px; transition: width 0.6s ease; }

    /* 方面级 */
    .aspect-row {
        display: flex; align-items: center; padding: 0.75rem 0;
        border-bottom: 1px solid #F3F4F6;
    }
    .aspect-row:last-child { border-bottom: none; }
    .aspect-icon { font-size: 1.3rem; width: 2rem; text-align: center; }
    .aspect-name { width: 110px; font-weight: 600; font-size: 0.95rem; flex-shrink: 0; }
    .aspect-bar-wrap { flex: 1; margin: 0 1rem; }
    .aspect-bar-track { height: 8px; background: #E5E7EB; border-radius: 4px; overflow: hidden; }
    .aspect-bar-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
    .aspect-label {
        display: inline-block; padding: 0.3rem 1rem; border-radius: 18px;
        font-size: 0.82rem; font-weight: 700; color: #fff; min-width: 64px;
        text-align: center; flex-shrink: 0;
    }

    /* 按钮 */
    div.stButton > button {
        border-radius: 14px !important; font-weight: 700 !important; border: none !important;
        transition: all 0.2s !important;
    }
    div.stButton > button:active { transform: scale(0.97) !important; }

    /* 分析按钮 */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3B82F6, #8B5CF6) !important;
        color: #fff !important;
    }
    div.stButton > button[kind="primary"]:hover { opacity: 0.88 !important; }

    /* 清空按钮 */
    div.stButton > button[kind="secondary"] {
        background: #F3F4F6 !important; color: #374151 !important; font-weight: 500 !important;
    }
    div.stButton > button[kind="secondary"]:hover { background: #E5E7EB !important; }

    /* 示例标签按钮 */
    button.example-tag {
        font-size: 0.8rem !important; background: #F3F4F6 !important; color: #4B5563 !important;
        border: 1px solid #E5E7EB !important; border-radius: 20px !important;
        padding: 0.3rem 0.8rem !important; height: auto !important; font-weight: 500 !important;
        white-space: nowrap !important; cursor: pointer !important;
    }
    button.example-tag:hover { background: #E5E7EB !important; border-color: #3B82F6 !important; color: #3B82F6 !important; }

    /* 隐藏 Streamlit 默认元素 */
    #MainMenu, footer { visibility: hidden; }

    /* 响应式 */
    @media (max-width: 500px) {
        .aspect-name { width: 80px; font-size: 0.82rem; }
        .aspect-icon { font-size: 1rem; width: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State 初始化
# ============================================================
if "text_value" not in st.session_state:
    st.session_state.text_value = ""

# ============================================================
# 头部
# ============================================================
st.markdown('<span class="header-icon">🛒</span>', unsafe_allow_html=True)
st.markdown('<div class="main-title">电商评论情感分析</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">三分类整体情感 + 六维度方面级分析</div>', unsafe_allow_html=True)

# ============================================================
# 输入卡片
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("**📝 输入评论文本**")

# 快捷示例 — 放在 text_area 前面，避免 session_state 修改冲突
st.markdown("**快捷示例：**")
ex_cols = st.columns(len(EXAMPLES))
for i, (label, text) in enumerate(EXAMPLES):
    with ex_cols[i]:
        if st.button(label, key=f"ex_{i}", use_container_width=True, type="secondary"):
            st.session_state.text_value = text
            st.rerun()

text_input = st.text_area(
    "评论内容",
    key="text_value",
    placeholder="请输入京东商品评论，例如：屏幕很大很清晰，但电池不耐用...",
    label_visibility="collapsed",
    height=110,
)

# 分析 & 清空按钮
btn_col1, btn_col2 = st.columns([5, 1.5])
with btn_col1:
    analyze_clicked = st.button("🔍 开始分析", type="primary", use_container_width=True)
with btn_col2:
    if st.button("🗑️ 清空", type="secondary", use_container_width=True):
        st.session_state.text_value = ""
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 分析逻辑
# ============================================================
def render_result(result):
    """渲染分析结果，与 index.html 的 renderResult() 保持一致"""
    sentiment = result.get("sentiment", "未知")
    color = SENT_COLORS.get(sentiment, "#6B7280")
    emoji = SENT_EMOJI.get(sentiment, "")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    # ---- 整体情感 ----
    st.markdown("#### 📊 整体情感")
    st.markdown(
        f'<div style="text-align:center;">'
        f'<span class="sentiment-badge" style="background:{color};">{emoji} {sentiment}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 置信度条（index.html 的核心功能）
    probs = result.get("probabilities")
    if probs:
        for label, prob in probs.items():
            pct = round(prob * 100)
            bar_color = SENT_COLORS.get(label, "#6B7280")
            st.markdown(
                f'<div class="conf-row">'
                f'<span class="conf-label">{label}</span>'
                f'<div class="conf-track">'
                f'<div class="conf-fill" style="width:{pct}%; background:{bar_color};"></div>'
                f'</div>'
                f'<span style="font-size:0.85rem; font-weight:600; width:40px; text-align:right;">{pct}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ---- 方面级分析 ----
    aspects = result.get("aspects")
    if aspects:
        st.markdown("#### 🏷️ 方面级分析")
        st.caption("每条评论在六个维度的细粒度情感判断")

        for aspect_name in ASPECTS:
            label = aspects.get(aspect_name, "未提及")
            label_color = ASPECT_COLORS.get(label, "#6B7280")
            aspect_emoji = ASPECT_EMOJI.get(aspect_name, "")

            bar_val = 100 if label == "正面" else (5 if label == "负面" else 50)
            bar_color = "#D1D5DB" if label == "未提及" else label_color

            st.markdown(
                f'<div class="aspect-row">'
                f'<span class="aspect-icon">{aspect_emoji}</span>'
                f'<span class="aspect-name">{aspect_name}</span>'
                f'<div class="aspect-bar-wrap">'
                f'<div class="aspect-bar-track">'
                f'<div class="aspect-bar-fill" style="width:{bar_val}%; background:{bar_color};"></div>'
                f'</div></div>'
                f'<span class="aspect-label" style="background:{label_color};">{label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


if analyze_clicked:
    text = st.session_state.text_value.strip()

    if not text:
        st.warning("⚠️ 请先输入评论内容")
    else:
        with st.spinner("AI 正在分析中..."):
            try:
                resp = requests.post(
                    API_URL,
                    json={"text": text},
                    timeout=12,
                )

                if not resp.ok:
                    st.error(f"❌ 服务器返回错误 ({resp.status_code})")
                else:
                    result = resp.json()
                    render_result(result)

            except requests.exceptions.ConnectionError:
                st.error("❌ 无法连接到后端服务，请先启动 `python app.py`")
            except requests.exceptions.Timeout:
                st.error("⏱️ 请求超时，请检查后端服务是否正常运行")
            except Exception as e:
                st.error(f"❌ 请求失败: {e}")
