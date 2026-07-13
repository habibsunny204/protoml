import streamlit as st


def setup_page():
    """Initializes page config and injects the global CSS payload."""
    st.set_page_config(
        page_title="ProtoML | Optimizer",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ── Global Typography & Background ── */
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background-color: #0A0A0B; color: #E8E8E8; }

/* ── Sidebar Base ── */
[data-testid="stSidebar"] {
    background-color: #121214 !important;
    border-right: 1px solid #2A2A2D !important;
}
[data-testid="stSidebar"] * {
    color: #E8E8E8;
}

/* ── Hero & Status ── */
.hero-title { font-size: 2.5rem; font-weight: 800; letter-spacing: -0.02em; }
.hero-accent { color: #5B9AFF; }
.hero-sub { color: #8C8C91; font-size: 1.05rem; margin-top: 4px; font-weight: 500; }
.status-pill { border: 1px solid #238636; background: #23863622; color: #3fb950; display: inline-flex; align-items: center; padding: 4px 12px; border-radius: 30px; font-size: 0.8rem; font-weight: 600; float: right; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: #3fb950; margin-right: 6px; box-shadow: 0 0 8px #3fb950; }

/* ── Metric Cards ── */
.metric-card { background: #121214; border: 1px solid #2A2A2D; border-radius: 12px; padding: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.4); display: flex; flex-direction: column; position: relative; overflow: hidden; }
.metric-card::before { content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 2px; }
.metric-card.cyan::before { background: #5B9AFF; }
.metric-card.teal::before { background: #3FB950; }
.metric-card.indigo::before { background: #8A63FF; }
.metric-card.amber::before { background: #D29630; }

.metric-label { font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: #8C8C91; letter-spacing: 0.05em; margin-bottom: 8px; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 2.2rem; font-weight: 700; line-height: 1.1; }
.metric-value.cyan { color: #5B9AFF; }
.metric-value.teal { color: #3FB950; }
.metric-value.indigo { color: #8A63FF; }
.metric-value.amber { color: #D29630; }
.metric-delta { font-size: 0.8rem; font-weight: 500; color: #A1A1A5; margin-top: 6px; }

/* ── Tabs & Buttons ── */
[data-testid="stTabs"] [data-baseweb="tab"] { color: #8C8C91; font-weight: 600; font-size: 0.95rem; }
[data-testid="stTabs"] [aria-selected="true"] { color: #5B9AFF !important; border-bottom-color: #5B9AFF !important; }
div.stButton > button { background: linear-gradient(135deg, #238636 0%, #2ea043 100%); color: #fff !important; font-weight: 600; border: none; padding: 12px 0; border-radius: 8px; box-shadow: 0 4px 14px rgba(35,134,54,0.3); }
div.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
hr { border-color: #2A2A2D; }

/* ── Expanders (Model Categories) ── */
[data-testid="stSidebar"] [data-testid="stExpander"] { background-color: transparent !important; border: 1px solid #2A2A2D !important; border-radius: 8px !important; margin-bottom: 12px !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary { background-color: #1A1A1D !important; padding: 12px 15px !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary p { color: #FFFFFF !important; font-weight: 600 !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover p, [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover svg { color: #5B9AFF !important; fill: #5B9AFF !important; }
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] { background-color: #121214 !important; border-top: 1px solid #2A2A2D !important; }
.stMultiSelect [data-baseweb="select"] { background-color: #0A0A0B !important; border-color: #2A2A2D !important; }

/* ── File Uploader ── */
[data-testid="stFileUploadDropzone"] { background-color: #161619 !important; border: 2px dashed #2A2A2D !important; border-radius: 8px !important; }
[data-testid="stFileUploadDropzone"] button { background-color: #1E1E22 !important; border: 1px solid #3A3A3D !important; color: #FFFFFF !important; font-weight: 600 !important; }
[data-testid="stFileUploadDropzone"] button:hover { border-color: #5B9AFF !important; color: #5B9AFF !important; }
[data-testid="stFileUploadDropzone"] p, [data-testid="stFileUploadDropzone"] div, [data-testid="stFileUploadDropzone"] span, [data-testid="stFileUploadDropzone"] small, [data-testid="stFileUploadDropzone"] button { color: #000000 !important; }
[data-testid="stFileUploadDropzone"] button { border-color: #000000 !important; background-color: transparent !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] *, [data-testid="stSidebar"] div[data-baseweb="select"] span { color: #000000 !important; }
[data-testid="stSidebar"] span[data-baseweb="tag"] { background-color: #E04343 !important; border: none !important; border-radius: 4px !important; }
[data-testid="stSidebar"] span[data-baseweb="tag"] span { color: #FFFFFF !important; font-weight: 600 !important; }
[data-testid="stSidebar"] span[data-baseweb="tag"] svg { fill: #FFFFFF !important; }

/* ─────────────────────────────────────────────
   Chat Premium UI Animations
───────────────────────────────────────────── */
@keyframes slideUpFade {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}


/* ─────────────────────────────────────────────
   File Uploader White Text Fix
───────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] button {
    color: #000000 !important;
}

/* ─────────────────────────────────────────────
   Multiselect Selected Tags
───────────────────────────────────────────── */
[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background-color: #E04343 !important;
    border: none !important;
    border-radius: 4px !important;
}

[data-testid="stSidebar"] span[data-baseweb="tag"] span {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] span[data-baseweb="tag"] svg {
    fill: #FFFFFF !important;
}


/* ─────────────────────────────────────────────
   Text Input Box (URL / Path)
───────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    color: #000000 !important;
    background-color: #FFFFFF !important;
    border-radius: 6px !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #666666 !important;
}

/* ─────────────────────────────────────────────
   Error Box
───────────────────────────────────────────── */
.error-box {
    background: linear-gradient(135deg, #2A1111, #1A0F0F);
    border: 1px solid #5C1F1F;
    border-left: 4px solid #FF4D4D;
    padding: 16px;
    border-radius: 12px;
    margin-top: 10px;
    color: #F5D0D0;
    animation: slideUpFade 0.3s ease-out;
}

.error-title {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    color: #FF8A8A;
}

.error-text {
    margin-top: 8px;
    font-size: 0.92rem;
    line-height: 1.5;
    color: #E5BABA;
}
</style>
    """,
        unsafe_allow_html=True,
    )


def initialize_session_state():
    """No-op: per-engine state is now initialised directly in app.py."""
    pass


def show_ai_dependency_error():
    st.markdown(
        '<div style="'
        "background: linear-gradient(135deg, #1A1A24, #121214);"
        "border: 1px solid #2A2A2D;"
        "border-left: 3px solid #8A63FF;"
        "padding: 15px 20px;"
        "border-radius: 12px 12px 12px 0px;"
        "margin-bottom: 12px;"
        "margin-right: 15%;"
        "box-shadow: 0 4px 15px rgba(0,0,0,0.2);"
        '">'
        '<p style="'
        "color: #8A63FF;"
        "font-size: 0.75rem;"
        "font-weight: 700;"
        "text-transform: uppercase;"
        "margin-bottom: 5px;"
        '">✨ ProtoML Assistant</p>'
        '<p style="'
        "color: #D1D1D6;"
        "font-size: 0.95rem;"
        "line-height: 1.6;"
        "margin: 0 0 10px 0;"
        '">The <code style="'
        "background: rgba(138,99,255,0.15);"
        "color: #8A63FF;"
        "padding: 2px 6px;"
        "border-radius: 4px;"
        "font-size: 0.85rem;"
        '">google-genai</code> package is not installed. Please install it to enable AI chat.</p>'
        '<div style="'
        "background: rgba(138,99,255,0.08);"
        "border: 1px solid rgba(138,99,255,0.25);"
        "padding: 6px 12px;"
        "border-radius: 8px;"
        "display: inline-block;"
        "font-family: monospace;"
        "font-size: 0.85rem;"
        "color: #8A63FF;"
        '">pip install google-genai</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    
    return
