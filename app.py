"""
app.py — Streamlit demo for IT-ticket classification (thesis PoC)

Launch:
    streamlit run app.py --server.port 8501

The VM must be running and have GPU access.
Models are pulled from HuggingFace Hub on first load, then cached in VRAM.
"""

import math
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px

# import plotly.graph_objects as go
import streamlit as st

# ── make project utils importable ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="IT Ticket Classifier",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────
LOGO_PATH = ROOT / "ThesisLogo.png"
LABELS_CFG_PATH = ROOT / "config" / "prompt_template_student.yaml"

MODELS = {
    "LLaMA 3.1-8B  (fine-tuned)": {
        "config": ROOT / "config" / "model_config_llama.yaml",
        "hub_id": "staerkjo/checkpoints-llama",
        "eos_token": "<|eot_id|>",
        "load_in_4bit": True,
        "max_seq_length": 2048,
        "key": "llama",
    },
    "Ministral-8B  (fine-tuned)": {
        "config": ROOT / "config" / "model_config_mistral.yaml",
        "hub_id": "staerkjo/checkpoints-mistral",
        "eos_token": "</s>",
        "load_in_4bit": True,
        "max_seq_length": 2048,
        "key": "mistral",
    },
}

FALLBACK_LABEL = "1 Misc incidents"
CONFIDENCE_WARN_THRESHOLD = 0.30

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── global ── */
    [data-testid="stAppViewContainer"] {background: #F4F6F9;}
    [data-testid="stSidebar"]          {background: #0D2137;}
    [data-testid="stSidebar"] * {color: #E8EEF4 !important;}

    /* ── sidebar: no scroll, fixed layout ── */
    [data-testid="stSidebar"] > div:first-child {
        height: 100vh;
        overflow: hidden !important;
        display: flex;
        flex-direction: column;
    }

    /* sidebar logo */
    [data-testid="stSidebar"] [data-testid="stImage"] img {
        border-radius: 10px;
        padding: 10px;
        background: rgba(255,255,255,1);
    }

    /* model selectbox: fix text + background inside the dropdown */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #1C3D6B !important;
        border: 1px solid #4A90D9 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #E8EEF4 !important;
    }
    /* dropdown option list (rendered outside sidebar, so target globally) */
    [data-baseweb="menu"] {
        background-color: #1C3D6B !important;
        border: 1px solid #4A90D9 !important;
    }
    [data-baseweb="menu"] li {
        color: #E8EEF4 !important;
    }
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] [aria-selected="true"] {
        background-color: #4A90D9 !important;
        color: #ffffff !important;
    }

    /* ── sidebar icon colours ── */
    /* eye (show/hide password) — sits on dark blue input bg → use light colour */
    [data-testid="stSidebar"] [data-testid="stTextInput"] button svg {
        color: #A8C6E8 !important;
        fill: #A8C6E8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"] button svg path {
        stroke: #A8C6E8 !important;
        fill: #A8C6E8 !important;
    }
    /* ? tooltip icon — sits directly on dark sidebar bg → same light colour */
    [data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg,
    [data-testid="stSidebar"] button[kind="icon"] svg {
        color: #E8EEF4 !important;
        fill: #FFFFFF !important;
    }
    [data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg path,
    [data-testid="stSidebar"] button[kind="icon"] svg path {
        stroke: #7AAFD4 !important;
        fill: none !important;
    }

    /* metric cards */
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #E0E6ED;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,.06);
    }

    /* tab headers */
    button[data-baseweb="tab"] {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #1E3A5F !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #4A90D9 !important;
    }

    /* section headings */
    h2 {color: #1E3A5F !important; font-weight: 700;}
    h3 {color: #2C5282 !important;}

    /* sidebar model label */
    .model-badge {
        background: #4A90D9;
        color: white !important;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 12px;
        font-weight: 600;
    }

    /* info box */
    .info-box {
        background: #EBF4FF;
        border-left: 4px solid #4A90D9;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 12px;
        color: #1A365D;
        font-size: 14px;
    }

    /* warn box */
    .warn-box {
        background: #FFFBEB;
        border-left: 4px solid #F6AD55;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 12px;
        color: #744210;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _parse_label(tag: str):
    """Split a predicted tag into (l1, l2, l3) or ('Unknown', …)."""
    m = re.match(r"\(([^,]+),\s*([^)]+)\),\s*(.+)", str(tag).strip())
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return "Unknown", "Unknown", str(tag).strip()


def _extract_tag(text: str) -> str:
    m = re.search(r"Tag:\s*(.+?)(?:\n|$)", str(text), re.IGNORECASE)
    if m:
        return m.group(1).strip()
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            return line
    return text


def load_excel(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    # Build the 'text' field exactly as during training
    parts = []
    for col in ["Title", "Description", "Assignment group"]:
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))
    if not parts:
        raise ValueError(
            "Excel must contain at least one of: Title, Description, Assignment group"
        )
    df["text"] = parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep=" ")
    df["text"] = df["text"].str.strip()

    # Keep the incident number if present
    if "Number" in df.columns:
        df = df.rename(columns={"Number": "number"})

    return df


# ── model loading (cached in VRAM for the session) ────────────────────────────


@st.cache_resource(show_spinner=False)
def load_model(hub_id: str, eos_token: str, max_seq_length: int, load_in_4bit: bool):
    """Load a fine-tuned model from HuggingFace Hub — kept alive in GPU memory."""
    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hub_id,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer.eos_token = eos_token
    tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    return model, tokenizer


# ── inference ─────────────────────────────────────────────────────────────────


def run_inference(
    df: pd.DataFrame, model, tokenizer, batch_size: int = 8
) -> pd.DataFrame:
    """
    Run batched inference and return df with prediction columns added.
    Uses the same prompt builder and scoring logic as scripts/eval.py.
    """
    try:
        # Import directly from the module file to avoid the __init__.py
        # eagerly pulling in unsloth (which is only available on GPU VMs).
        import importlib.util as _ilu

        def _import_from_file(module_name: str, file_path: str):
            spec = _ilu.spec_from_file_location(module_name, file_path)
            mod = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod

        _wandb_eval = _import_from_file(
            "wandb_eval", str(ROOT / "utils" / "evaluation" / "wandb_eval.py")
        )
        _llm_handler = _import_from_file(
            "llm_handler", str(ROOT / "utils" / "training" / "llm_handler.py")
        )
        # Patch the internal cross-import so wandb_eval can call build_system_prompt
        import sys as _sys

        _sys.modules["utils.training.llm_handler"] = _llm_handler  # type: ignore[assignment]

        run_inference_batch = _wandb_eval.run_inference_batch
        load_labels_config = _llm_handler.load_labels_config

    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Missing GPU dependency: {e}. "
            "Make sure unsloth, torch, and all project dependencies are installed on the VM."
        ) from e

    labels_cfg = load_labels_config(str(LABELS_CFG_PATH))

    # run_inference_batch expects a 'text' column; pass a minimal df
    input_df = df[["text"]].copy()
    if "number" in df.columns:
        input_df["number"] = df["number"]

    result_df = run_inference_batch(
        model=model,
        tokenizer=tokenizer,
        df=input_df,
        labels_cfg=labels_cfg,
        max_new_tokens=250,
        batch_size=batch_size,
    )

    # Parse label into hierarchy levels
    result_df["predicted_tag"] = result_df["predicted_label"].apply(_extract_tag)
    parsed = result_df["predicted_tag"].apply(_parse_label)
    result_df["L1"] = [p[0] for p in parsed]
    result_df["L2"] = [p[1] for p in parsed]
    result_df["L3"] = [p[2] for p in parsed]

    return result_df


# ── analytics helpers ─────────────────────────────────────────────────────────

BLUE_SEQ = px.colors.sequential.Blues[3:]
PALETTE = px.colors.qualitative.Set2


def fig_l1_bar(df: pd.DataFrame):
    counts = df["L1"].value_counts().reset_index()
    counts.columns = ["Category (L1)", "Count"]
    fig = px.bar(
        counts,
        x="Count",
        y="Category (L1)",
        orientation="h",
        color="Count",
        color_continuous_scale="Blues",
        text="Count",
        title="Predicted L1 Category Distribution",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=420,
    )
    return fig


def fig_l3_bar(df: pd.DataFrame, top_n: int = 15):
    counts = df["L3"].value_counts().head(top_n).reset_index()
    counts.columns = ["Scenario (L3)", "Count"]
    fig = px.bar(
        counts,
        x="Scenario (L3)",
        y="Count",
        color="Count",
        color_continuous_scale="Blues",
        text="Count",
        title=f"Top {top_n} Predicted Scenarios (L3)",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_tickangle=-35,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=30, t=40, b=120),
        height=420,
    )
    return fig


def fig_confidence_hist(df: pd.DataFrame):
    conf = df["student_confidence"].dropna()
    fig = px.histogram(
        conf,
        nbins=30,
        title="Student Confidence Distribution",
        labels={"value": "Confidence", "count": "Tickets"},
        color_discrete_sequence=["#4A90D9"],
    )
    fig.add_vline(
        x=CONFIDENCE_WARN_THRESHOLD,
        line_dash="dash",
        line_color="#E53E3E",
        annotation_text=f"Low-confidence threshold ({CONFIDENCE_WARN_THRESHOLD})",
        annotation_position="top right",
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=360,
        showlegend=False,
    )
    return fig


def fig_latency_hist(df: pd.DataFrame):
    lat = df["latency_ms"].dropna()
    fig = px.histogram(
        lat,
        nbins=30,
        title="Inference Latency per Ticket (ms)",
        labels={"value": "Latency (ms)", "count": "Tickets"},
        color_discrete_sequence=["#68D391"],
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=360,
        showlegend=False,
    )
    return fig


def fig_confidence_scatter(df: pd.DataFrame):
    df2 = df.copy()
    df2["text_len"] = df2["text"].str.len()
    fig = px.scatter(
        df2,
        x="text_len",
        y="student_confidence",
        color="is_fallback",
        color_discrete_map={True: "#FC8181", False: "#4A90D9"},
        labels={
            "text_len": "Ticket text length (chars)",
            "student_confidence": "Confidence",
        },
        title="Confidence vs Ticket Length",
        opacity=0.55,
    )
    fig.add_hline(
        y=CONFIDENCE_WARN_THRESHOLD, line_dash="dash", line_color="#E53E3E", opacity=0.7
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=360,
        legend_title_text="Fallback",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.markdown("## 🎓 IT Ticket Classifier")

    st.markdown("---")
    st.markdown("### Model")

    selected_model_name = st.selectbox(
        "Select fine-tuned model",
        list(MODELS.keys()),
        label_visibility="collapsed",
    )
    model_cfg = MODELS[selected_model_name]

    st.markdown("---")
    st.markdown("### Inference settings")
    batch_size = st.slider(
        "Batch size",
        min_value=1,
        max_value=16,
        value=4,
        step=1,
        help="Larger batches are faster but use more VRAM",
    )
    conf_threshold = st.slider(
        "Flag tickets below confidence",
        0.0,
        1.0,
        CONFIDENCE_WARN_THRESHOLD,
        0.05,
        help="Tickets with student confidence below this value appear in the review table",
    )

    st.markdown("---")
    st.markdown("### HuggingFace token")
    hf_token = st.text_input(
        "HF_TOKEN (if private model)",
        type="password",
        value=os.environ.get("HF_TOKEN", ""),
        help="Leave blank if already set as env variable",
    )
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    st.markdown("---")
    st.caption("Thesis: IT Ticket Classification · 2026")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — three tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_upload, tab_run, tab_results = st.tabs(
    ["📂  Upload & Preview", "⚙️  Run Inference", "📊  Results & Analytics"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Upload & Preview
# ─────────────────────────────────────────────────────────────────────────────

with tab_upload:
    st.markdown("## Upload daily ticket export")
    st.markdown(
        '<div class="info-box">Upload one or more Excel files from the daily exports '
        "(e.g. <code>02_Dec/03-Dec.xlsx</code>). Files are joined automatically.</div>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Drop ServiceNow Excel export(s) here",
        type=["xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        with st.spinner("Reading file(s)…"):
            try:
                frames = [load_excel(f) for f in uploaded_files]
                df_raw = pd.concat(frames, ignore_index=True)
                df_raw = df_raw.drop_duplicates(
                    subset=["number"] if "number" in df_raw.columns else None
                )
                st.session_state["df_raw"] = df_raw
                st.success(
                    f"Loaded **{len(df_raw):,} tickets** from {len(uploaded_files)} file(s)."
                )
            except Exception as e:
                st.error(f"Could not load file(s): {e}")
                st.stop()

        # Quick stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Total tickets", f"{len(df_raw):,}")
        c2.metric("Files loaded", len(uploaded_files))
        missing_text = int(df_raw["text"].str.strip().eq("").sum())
        c3.metric(
            "Empty text rows",
            missing_text,
            delta=None if missing_text == 0 else f"{missing_text} will be skipped",
            delta_color="off",
        )

        st.markdown("### Preview (first 10 rows)")
        preview_cols = ["number", "text"] if "number" in df_raw.columns else ["text"]
        st.dataframe(
            df_raw[preview_cols].head(10).style.set_properties(**{"font-size": "13px"}),
            use_container_width=True,
            height=300,
        )

        # Show text length distribution
        df_raw["_text_len"] = df_raw["text"].str.len()
        fig_len = px.histogram(
            df_raw["_text_len"],
            nbins=40,
            title="Ticket text length distribution (chars)",
            labels={"value": "Characters", "count": "Tickets"},
            color_discrete_sequence=["#4A90D9"],
        )
        fig_len.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=40, b=10),
            height=280,
        )
        st.plotly_chart(fig_len, use_container_width=True)

    else:
        st.markdown(
            '<div class="warn-box">No file uploaded yet. '
            "Grab a daily .xlsx from <code>ServiceNow</code> and drop it above.</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Run Inference
# ─────────────────────────────────────────────────────────────────────────────

with tab_run:
    st.markdown("## Run inference")

    df_raw = st.session_state.get("df_raw")
    if df_raw is None:
        st.markdown(
            '<div class="warn-box">Upload tickets first on the <b>Upload & Preview</b> tab.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # Filter empty text
    df_clean = df_raw[df_raw["text"].str.strip().ne("")].copy()
    st.markdown(
        f'<div class="info-box">Ready to classify <b>{len(df_clean):,} tickets</b> '
        f"with <b>{selected_model_name}</b>  ·  batch size {batch_size}</div>",
        unsafe_allow_html=True,
    )

    run_btn = st.button(
        "▶  Run classification", type="primary", use_container_width=True
    )

    if run_btn:
        prog_bar = st.progress(0, text="Loading model…")
        status = st.empty()

        # Load model
        try:
            with st.spinner(f"Loading {selected_model_name} into GPU…"):
                model, tokenizer = load_model(
                    hub_id=model_cfg["hub_id"],
                    eos_token=model_cfg["eos_token"],
                    max_seq_length=model_cfg["max_seq_length"],
                    load_in_4bit=model_cfg["load_in_4bit"],
                )
            prog_bar.progress(10, text="Model loaded — running inference…")
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            st.info(
                "Make sure the VM has a GPU, unsloth is installed, and HF_TOKEN is set."
            )
            st.stop()

        # Inference — run in batches with live progress
        try:
            t_start = time.perf_counter()
            result_df = run_inference(df_clean, model, tokenizer, batch_size=batch_size)
            elapsed = time.perf_counter() - t_start

            prog_bar.progress(100, text="Done!")
            status.success(
                f"Classified {len(result_df):,} tickets in {elapsed:.1f}s "
                f"({len(result_df)/elapsed:.1f} tickets/sec)"
            )
            st.session_state["result_df"] = result_df
            st.session_state["model_name"] = selected_model_name
            st.session_state["elapsed"] = elapsed

        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.exception(e)
            st.stop()

        st.info("Head to the **Results & Analytics** tab to explore the predictions.")

    elif "result_df" in st.session_state:
        st.markdown(
            '<div class="info-box">Previous results are loaded — see the '
            "<b>Results & Analytics</b> tab. Click Run again to re-classify.</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Results & Analytics
# ─────────────────────────────────────────────────────────────────────────────

with tab_results:
    result_df = st.session_state.get("result_df")

    if result_df is None:
        st.markdown(
            '<div class="warn-box">No results yet — run classification first.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    model_name = st.session_state.get("model_name", "")
    elapsed = st.session_state.get("elapsed", 0)
    n = len(result_df)

    st.markdown(f"## Results — {model_name}")

    # ── KPI row ──────────────────────────────────────────────────────────────
    format_rate = (
        result_df["format_compliant"].mean()
        if "format_compliant" in result_df.columns
        else float("nan")
    )
    fallback_rate = (
        result_df["is_fallback"].mean()
        if "is_fallback" in result_df.columns
        else float("nan")
    )
    mean_conf = (
        result_df["student_confidence"].mean()
        if "student_confidence" in result_df.columns
        else float("nan")
    )
    mean_lat = (
        result_df["latency_ms"].mean()
        if "latency_ms" in result_df.columns
        else float("nan")
    )
    low_conf_n = int((result_df["student_confidence"].fillna(1) < conf_threshold).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tickets classified", f"{n:,}")
    c2.metric(
        "Format compliance",
        f"{format_rate*100:.1f}%" if not math.isnan(format_rate) else "—",
    )
    c3.metric(
        "Fallback rate",
        f"{fallback_rate*100:.1f}%" if not math.isnan(fallback_rate) else "—",
        help="% classified as '1 Misc incidents' — high value may indicate model uncertainty",
    )
    c4.metric(
        "Mean confidence", f"{mean_conf:.3f}" if not math.isnan(mean_conf) else "—"
    )
    c5.metric("Mean latency", f"{mean_lat:.0f} ms" if not math.isnan(mean_lat) else "—")

    st.markdown("---")

    # ── Charts row 1 ─────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(fig_l1_bar(result_df), use_container_width=True)
    with col_r:
        st.plotly_chart(fig_l3_bar(result_df), use_container_width=True)

    # ── Charts row 2 ─────────────────────────────────────────────────────────
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        if "student_confidence" in result_df.columns:
            st.plotly_chart(fig_confidence_hist(result_df), use_container_width=True)
    with col_r2:
        if "latency_ms" in result_df.columns:
            st.plotly_chart(fig_latency_hist(result_df), use_container_width=True)

    # ── Confidence vs ticket length scatter ───────────────────────────────────
    if "student_confidence" in result_df.columns:
        st.plotly_chart(fig_confidence_scatter(result_df), use_container_width=True)

    # ── Low-confidence review table ───────────────────────────────────────────
    st.markdown(f"### ⚠️ Low-confidence tickets  (confidence < {conf_threshold})")
    low_conf_df = result_df[
        result_df["student_confidence"].fillna(1) < conf_threshold
    ].copy()
    if len(low_conf_df) == 0:
        st.success(f"No tickets below confidence threshold {conf_threshold}.")
    else:
        st.warning(
            f"{len(low_conf_df)} ticket(s) ({len(low_conf_df)/n*100:.1f}%) "
            f"have confidence below {conf_threshold} — consider manual review."
        )
        show_cols = [
            c
            for c in [
                "number",
                "text",
                "predicted_tag",
                "L1",
                "L3",
                "student_confidence",
                "is_fallback",
            ]
            if c in low_conf_df.columns
        ]
        st.dataframe(
            low_conf_df[show_cols].sort_values("student_confidence"),
            use_container_width=True,
            height=280,
        )

    # ── Full predictions table ────────────────────────────────────────────────
    with st.expander("📋 Full predictions table"):
        show_cols_full = [
            c
            for c in [
                "number",
                "text",
                "predicted_tag",
                "L1",
                "L2",
                "L3",
                "student_confidence",
                "is_fallback",
                "format_compliant",
                "latency_ms",
            ]
            if c in result_df.columns
        ]
        st.dataframe(result_df[show_cols_full], use_container_width=True, height=400)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Download results")
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇  Download predictions as CSV",
        data=csv_bytes,
        file_name=f"predictions_{model_cfg['key']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
