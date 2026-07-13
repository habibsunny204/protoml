from __future__ import annotations

import io
import os
from typing import Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")

# ── Shared theme constants ────────────────────────────────────────────────────
_BG_DARK  = "#121214"
_BG_CARD  = "#0A0A0B"
_BORDER   = "#2A2A2D"
_TEXT     = "#E8E8E8"
_MUTED    = "#8C8C91"
_BLUE     = "#5B9AFF"
_GREEN    = "#3FB950"
_PURPLE   = "#8A63FF"
_AMBER    = "#D29630"
_RED      = "#E05252"
_TEAL     = "#17becf"

def _dark_fig(w: float = 7, h: float = 3.6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_CARD)
    ax.tick_params(colors=_MUTED)
    ax.spines[:].set_color(_BORDER)
    return fig, ax


# ── Top-level metric cards ────────────────────────────────────────────────────

def render_top_metrics(
    best_acc="—", avg_acc="—", improvement="—", pvalue="—", metric_name="Macro F1"
):
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(
        f'<div class="metric-card cyan"><div class="metric-label">Best {metric_name}</div>'
        f'<div class="metric-value cyan">{best_acc}</div>'
        f'<div class="metric-delta">Highest across all models</div></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="metric-card teal"><div class="metric-label">Average {metric_name}</div>'
        f'<div class="metric-value teal">{avg_acc}</div>'
        f'<div class="metric-delta">Mean over all iterations</div></div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        f'<div class="metric-card indigo"><div class="metric-label">Improvement %</div>'
        f'<div class="metric-value indigo">{improvement}</div>'
        f'<div class="metric-delta">Optimization vs Baseline</div></div>',
        unsafe_allow_html=True,
    )
    m4.markdown(
        f'<div class="metric-card amber"><div class="metric-label">Status Check</div>'
        f'<div class="metric-value amber">{pvalue}</div>'
        f'<div class="metric-delta">Pipeline indicator</div></div>',
        unsafe_allow_html=True,
    )


# ── Data preview ──────────────────────────────────────────────────────────────

def render_data_preview(df: pd.DataFrame):
    """Show dataset head + summary statistics."""
    if df is None:
        return
    st.markdown("#### Sample Rows")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        st.markdown("#### Numeric Summary")
        desc = numeric_df.describe().round(4)
        st.dataframe(desc, use_container_width=True)

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        st.markdown("#### Categorical Columns")
        for col in cat_cols[:5]:
            vc = df[col].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(5, 2.2))
            fig.patch.set_facecolor(_BG_DARK)
            ax.set_facecolor(_BG_CARD)
            ax.barh(vc.index.astype(str), vc.values, color=_BLUE)
            ax.set_title(f"{col}", color=_TEXT, fontsize=10)
            ax.tick_params(colors=_MUTED, labelsize=8)
            ax.spines[:].set_color(_BORDER)
            ax.set_xlabel("Count", color=_MUTED, fontsize=8)
            ax.invert_yaxis()
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)


# ── Classification report ─────────────────────────────────────────────────────

def render_classification_report(raw_report: dict, label_classes: Optional[list]):
    # ── Accuracy metric card ─────────────────────────────────────────────────
    acc = raw_report.get("accuracy_score") or raw_report.get("accuracy")
    if acc is not None:
        acc_pct = f"{float(acc) * 100:.2f}%"
        macro_f1 = raw_report.get("macro avg", {}).get("f1-score")
        weighted_f1 = raw_report.get("weighted avg", {}).get("f1-score")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy",    acc_pct)
        c2.metric("Macro F1",    f"{macro_f1 * 100:.2f}%" if macro_f1 is not None else "—")
        c3.metric("Weighted F1", f"{weighted_f1 * 100:.2f}%" if weighted_f1 is not None else "—")
        st.markdown("---")

    per_class = {}
    for k, v in raw_report.items():
        if not isinstance(v, dict):
            continue
        label = (label_classes[int(k)]
                 if k.lstrip("-").isdigit() and label_classes is not None
                 else k)
        per_class[label] = v

    plot_classes = [c for c in per_class if c not in ("macro avg", "weighted avg")]
    if plot_classes:
        prec = [per_class[c]["precision"] for c in plot_classes]
        rec  = [per_class[c]["recall"]    for c in plot_classes]
        f1   = [per_class[c]["f1-score"]  for c in plot_classes]
        sup  = [int(per_class[c]["support"]) for c in plot_classes]

        x, w = np.arange(len(plot_classes)), 0.26
        fig, ax = _dark_fig(max(6, len(plot_classes) * 1.4), 3.6)
        ax.bar(x - w, prec, w, label="Precision", color=_BLUE,   zorder=3)
        ax.bar(x,     rec,  w, label="Recall",    color=_GREEN,  zorder=3)
        ax.bar(x + w, f1,   w, label="F1-Score",  color=_PURPLE, zorder=3)
        for i, n in enumerate(sup):
            ax.text(x[i], max(prec[i], rec[i], f1[i]) + 0.06,
                    f"n={n}", ha="center", va="bottom",
                    color=_MUTED, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_classes, color=_TEXT, fontsize=10)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score", color=_MUTED, fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.grid(True, color=_BORDER, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(
            handles=[mpatches.Patch(color=c, label=l) for c, l in [
                (_BLUE, "Precision"), (_GREEN, "Recall"), (_PURPLE, "F1-Score")]],
            loc="upper right", framealpha=0, labelcolor=_TEXT, fontsize=9)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    report_df = pd.DataFrame(per_class).transpose().reset_index()
    report_df.columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    report_df["Support"] = report_df["Support"].astype(int)

    def color_score(val):
        if isinstance(val, float):
            if val >= 0.85:
                return f"color: {_GREEN}; font-weight: 600"
            if val >= 0.70:
                return f"color: {_AMBER}; font-weight: 600"
            return f"color: {_RED}; font-weight: 600"
        return ""

    st.dataframe(
        report_df.style
            .map(color_score, subset=["Precision", "Recall", "F1-Score"])
            .format({"Precision": "{:.3f}", "Recall": "{:.3f}",
                     "F1-Score": "{:.3f}", "Support": "{:d}"})
            .set_properties(**{"background-color": _BG_CARD,
                                "color": _TEXT, "border-color": _BORDER}),
        use_container_width=True, hide_index=True,
    )


# ── Confusion matrix ──────────────────────────────────────────────────────────

def render_confusion_matrix(cm_data: dict):
    """Render a colour-coded confusion matrix heatmap."""
    if not cm_data or "matrix" not in cm_data:
        return
    matrix = np.array(cm_data["matrix"])
    labels = cm_data.get("labels") or [str(i) for i in range(matrix.shape[0])]

    n = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(4, n * 1.2), max(3.5, n * 1.0)))
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_CARD)

    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax).ax.tick_params(colors=_MUTED)

    thresh = matrix.max() / 2.0
    for r in range(n):
        for c in range(n):
            ax.text(c, r, str(matrix[r, c]),
                    ha="center", va="center", fontsize=9,
                    color="white" if matrix[r, c] > thresh else _BG_DARK)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", color=_TEXT, fontsize=8)
    ax.set_yticklabels(labels, color=_TEXT, fontsize=8)
    ax.set_xlabel("Predicted", color=_MUTED, fontsize=9)
    ax.set_ylabel("Actual",    color=_MUTED, fontsize=9)
    ax.set_title("Confusion Matrix", color=_TEXT, fontsize=11)
    ax.spines[:].set_color(_BORDER)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ── ROC curves ────────────────────────────────────────────────────────────────

def render_roc_curves(roc_data: dict):
    """Render binary or multiclass ROC curves."""
    if not roc_data:
        return
    fig, ax = _dark_fig(5.5, 4.0)

    if roc_data.get("type") == "binary":
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        auc = roc_data["auc"]
        ax.plot(fpr, tpr, color=_BLUE, lw=2,
                label=f"ROC (AUC = {auc:.3f})")
    else:
        colors = [_BLUE, _GREEN, _PURPLE, _AMBER, _RED, _TEAL,
                  "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
        for i, (cls, vals) in enumerate(roc_data.get("classes", {}).items()):
            c = colors[i % len(colors)]
            ax.plot(vals["fpr"], vals["tpr"], color=c, lw=1.5,
                    label=f"{cls} (AUC={vals['auc']:.3f})")

    ax.plot([0, 1], [0, 1], color=_BORDER, linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate", color=_MUTED, fontsize=9)
    ax.set_ylabel("True Positive Rate",  color=_MUTED, fontsize=9)
    ax.set_title("ROC Curve", color=_TEXT, fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER,
              labelcolor=_TEXT, fontsize=8,
              loc="lower right")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ── SHAP bar plot ─────────────────────────────────────────────────────────────

def render_shap_plot(shap_values: dict):
    """Bar chart of mean |SHAP| per feature."""
    if not shap_values:
        st.info("SHAP values not available for this model type.")
        return
    df_s = pd.DataFrame({"Feature": list(shap_values.keys()),
                          "Mean |SHAP|": list(shap_values.values())}) \
             .sort_values("Mean |SHAP|", ascending=True)

    fig, ax = _dark_fig(5.5, max(3.0, len(df_s) * 0.32))
    ax.barh(df_s["Feature"], df_s["Mean |SHAP|"], color=_PURPLE, height=0.7)
    ax.set_xlabel("Mean |SHAP value|", color=_MUTED, fontsize=9)
    ax.set_title("SHAP Feature Importance", color=_TEXT, fontsize=11)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ── Grad-CAM overlay ──────────────────────────────────────────────────────────

def render_gradcam(gradcam_data: dict):
    """Show original image + Grad-CAM overlay side-by-side."""
    if not gradcam_data or "heatmap" not in gradcam_data:
        return
    try:
        import matplotlib.cm as cm

        heatmap = np.array(gradcam_data["heatmap"], dtype=np.float32)
        raw_img = np.array(gradcam_data["image"], dtype=np.float32)

        # raw_img is C×H×W normalized; denorm for display
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std  = np.array([0.229, 0.224, 0.225])[:, None, None]
        img_display = np.clip(raw_img * std + mean, 0, 1).transpose(1, 2, 0)

        # Resize heatmap to image size
        from PIL import Image as PILImage
        h_pil  = PILImage.fromarray(np.uint8(heatmap * 255)).resize(
            (img_display.shape[1], img_display.shape[0]), PILImage.BILINEAR)
        heat_r = np.array(h_pil) / 255.0
        colormap = cm.get_cmap("jet")
        heat_rgb = colormap(heat_r)[..., :3]
        overlay  = np.clip(0.5 * img_display + 0.5 * heat_rgb, 0, 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
        fig.patch.set_facecolor(_BG_DARK)
        for a in (ax1, ax2):
            a.set_facecolor(_BG_CARD)
            a.axis("off")
        ax1.imshow(img_display)
        ax1.set_title("Original", color=_TEXT, fontsize=9)
        ax2.imshow(overlay)
        cn = gradcam_data.get("class_name", str(gradcam_data.get("pred_class", "")))
        ax2.set_title(f"Grad-CAM  |  {cn}", color=_TEXT, fontsize=9)
        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Grad-CAM display failed: {e}")


# ── Regression report ─────────────────────────────────────────────────────────

def render_regression_report(raw_report: dict):
    r2  = raw_report.get("R-Squared (R2)", "—")
    mse = raw_report.get("Mean Squared Error (MSE)", "—")
    mae = raw_report.get("Mean Absolute Error (MAE)", "—")

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card cyan"><div class="metric-label">R²</div>'
                f'<div class="metric-value cyan">{r2}</div>'
                f'<div class="metric-delta">Higher is better (Max 1.0)</div></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card amber"><div class="metric-label">MAE</div>'
                f'<div class="metric-value amber">{mae}</div>'
                f'<div class="metric-delta">Lower is better</div></div>',
                unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card indigo"><div class="metric-label">MSE</div>'
                f'<div class="metric-value indigo">{mse}</div>'
                f'<div class="metric-delta">Lower is better</div></div>',
                unsafe_allow_html=True)

    if "y_true" in raw_report and "y_pred" in raw_report:
        y_true = np.array(raw_report["y_true"])
        y_pred = np.array(raw_report["y_pred"])
        st.markdown("<br><h4 style='color:#E8E8E8;font-weight:600;font-size:1.1rem;'>"
                    "Actual vs. Predicted</h4>", unsafe_allow_html=True)
        fig, ax = _dark_fig(8, 3.8)
        ax.scatter(y_true, y_pred, color=_BLUE, alpha=0.5, edgecolors=_BG_DARK,
                   linewidth=0.5, s=40, label="Predictions")
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], color=_GREEN, linestyle="--", lw=2,
                label="Perfect Fit")
        ax.set_xlabel("Actual",    color=_MUTED, fontsize=10, fontweight="bold")
        ax.set_ylabel("Predicted", color=_MUTED, fontsize=10, fontweight="bold")
        ax.grid(True, color=_BORDER, linewidth=0.6, linestyle=":", zorder=0)
        ax.set_axisbelow(True)
        ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER, labelcolor=_TEXT,
                  loc="upper left", fontsize=9)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ── BO convergence ────────────────────────────────────────────────────────────

def render_convergence_plot(bo_scores: list):
    if not bo_scores:
        return
    st.markdown("#### Bayesian Optimization Convergence")
    st.caption("How the optimizer discovered improvements across successive trials.")
    best_so_far = np.maximum.accumulate(bo_scores)
    trials = range(1, len(bo_scores) + 1)
    fig, ax = _dark_fig(5, 3)
    ax.plot(trials, bo_scores, marker="x", linestyle="", color="#aec7e8",
            alpha=0.7, label="Trial Score")
    ax.plot(trials, best_so_far, marker="o", linestyle="-", color=_BLUE,
            linewidth=2, label="Best So Far")
    ax.set_xlabel("Optimization Trial", color=_MUTED, fontsize=9)
    ax.set_ylabel("Metric Score",       color=_MUTED, fontsize=9)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER, labelcolor=_TEXT, fontsize=8)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ── Training curve (vision) ───────────────────────────────────────────────────

def render_training_curve(train_losses: list, val_f1s: list):
    if not train_losses or not val_f1s:
        return
    st.markdown("#### Deep Learning Training Dynamics")
    st.caption("Epoch-by-epoch learning loss and validation generalisation.")
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    fig.patch.set_facecolor(_BG_DARK)
    for ax in (ax1, ax2):
        ax.set_facecolor(_BG_CARD)
        ax.tick_params(colors=_MUTED)
        ax.spines[:].set_color(_BORDER)
        ax.yaxis.grid(True, color=_BORDER, linewidth=0.6)
        ax.set_axisbelow(True)
    ax1.plot(epochs, train_losses, marker="o", color=_RED, lw=2)
    ax1.set_xlabel("Epoch", color=_MUTED, fontsize=9)
    ax1.set_ylabel("Training Loss", color=_MUTED, fontsize=9)
    ax1.set_title("Model Learning", color=_TEXT, fontsize=10)
    ax2.plot(epochs, val_f1s, marker="o", color=_GREEN, lw=2)
    ax2.set_xlabel("Epoch", color=_MUTED, fontsize=9)
    ax2.set_ylabel("Validation Macro F1", color=_MUTED, fontsize=9)
    ax2.set_title("Generalisation", color=_TEXT, fontsize=10)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ── Feature importance ────────────────────────────────────────────────────────

def render_feature_importance(importances_dict: dict):
    if not importances_dict:
        st.info("Feature importance not available for this algorithm.")
        return
    st.markdown("#### Top Feature Importances")
    st.caption("Data columns that most influenced the model's decision.")
    df_imp = pd.DataFrame({"Feature":    list(importances_dict.keys()),
                            "Importance": list(importances_dict.values())}) \
               .sort_values("Importance", ascending=True)
    fig, ax = _dark_fig(5.5, max(3.0, len(df_imp) * 0.32))
    ax.barh(df_imp["Feature"], df_imp["Importance"], color=_TEAL, height=0.7)
    ax.set_xlabel("Relative Importance / Weight", color=_MUTED, fontsize=9)
    ax.set_title("Drivers of Model Prediction",   color=_TEXT,  fontsize=11)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ── Export buttons ────────────────────────────────────────────────────────────

def render_export_buttons(results_df: Optional[pd.DataFrame],
                           html_content: str,
                           model_bytes: Optional[bytes] = None,
                           model_filename: str = "ProtoML_model.joblib"):
    st.markdown("### Export Results")
    st.caption("Download your optimization artifacts for publication and reproducibility.")
    st.markdown("""<style>
    div.stDownloadButton > button { color: #000 !important; }
    div.stDownloadButton > button * { color: #000 !important; }
    div.stDownloadButton > button:hover { color: #FFF !important; }
    div.stDownloadButton > button:hover * { color: #FFF !important; }
    </style>""", unsafe_allow_html=True)

    cols = st.columns(3 if model_bytes else 2)

    if results_df is not None:
        cols[0].download_button(
            label="Download Leaderboard (.csv)",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="ProtoML_Leaderboard.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if html_content:
        cols[1].download_button(
            label="Download Full Report (.html)",
            data=html_content.encode("utf-8"),
            file_name="ProtoML_Report.html",
            mime="text/html",
            use_container_width=True,
        )

    if model_bytes and len(cols) > 2:
        cols[2].download_button(
            label=f"Download Best Model ({model_filename.split('.')[-1].upper()})",
            data=model_bytes,
            file_name=model_filename,
            mime="application/octet-stream",
            use_container_width=True,
        )


def render_all_models_download(all_exported_paths: dict):
    """Show a download button for every trained model, 3 per row."""
    if not all_exported_paths:
        return
    st.markdown("#### Download Individual Models")
    st.caption(
        "Every model trained during this run is available for download "
        "with its hyperparameter metadata (.json sidecar)."
    )
    names  = list(all_exported_paths.keys())
    n_cols = 3
    for i in range(0, len(names), n_cols):
        row_names = names[i : i + n_cols]
        row_cols  = st.columns(n_cols)
        for col, name in zip(row_cols, row_names):
            path = all_exported_paths[name]
            if path and os.path.exists(path):
                with open(path, "rb") as fh:
                    data = fh.read()
                fname = os.path.basename(path)
                col.download_button(
                    label=f"{name}",
                    data=data,
                    file_name=fname,
                    mime="application/octet-stream",
                    use_container_width=True,
                    key=f"dl_model_{name}",
                )
            else:
                col.button(name, disabled=True, use_container_width=True,
                           key=f"dl_model_{name}_na")


# ── AI assistant ──────────────────────────────────────────────────────────────

def render_ai_chat(api_key: str):
    try:
        from google import genai
    except ImportError:
        from utils import show_ai_dependency_error
        show_ai_dependency_error()
        return

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hello! I am your ProtoML assistant. Ask me anything about your results."}
        ]

    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,#1A1A24,#121214);'
                    f'padding:15px 20px;border-radius:12px 12px 0 12px;border:1px solid {_BORDER};'
                    f'margin-bottom:12px;margin-left:15%;">'
                    f'<p style="color:{_BLUE};font-size:.75rem;font-weight:700;'
                    f'text-transform:uppercase;text-align:right;margin-bottom:5px;">You</p>'
                    f'<p style="color:{_TEXT};font-size:.95rem;line-height:1.5;'
                    f'text-align:right;margin:0;">{message["content"]}</p></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="background:{_BG_DARK};padding:15px 20px;'
                    f'border-radius:12px 12px 12px 0;border:1px solid {_BORDER};'
                    f'border-left:3px solid {_PURPLE};margin-bottom:12px;margin-right:15%;">'
                    f'<p style="color:{_PURPLE};font-size:.75rem;font-weight:700;'
                    f'text-transform:uppercase;margin-bottom:5px;">ProtoML Assistant</p>'
                    f'<p style="color:#D1D1D6;font-size:.95rem;line-height:1.6;'
                    f'margin:0;">{message["content"]}</p></div>',
                    unsafe_allow_html=True)

    if not api_key:
        st.chat_input("Enter API key in sidebar to unlock chat...", disabled=True)
    else:
        if prompt := st.chat_input("Ask about your dataset or results..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,#1A1A24,#121214);'
                    f'padding:15px 20px;border-radius:12px 12px 0 12px;border:1px solid {_BORDER};'
                    f'margin-bottom:12px;margin-left:15%;">'
                    f'<p style="color:{_BLUE};font-size:.75rem;font-weight:700;'
                    f'text-transform:uppercase;text-align:right;margin-bottom:5px;">You</p>'
                    f'<p style="color:{_TEXT};font-size:.95rem;line-height:1.5;'
                    f'text-align:right;margin:0;">{prompt}</p></div>',
                    unsafe_allow_html=True)
                with st.spinner("Analyzing..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        history = "\n".join(
                            f"{m['role'].upper()}: {m['content']}"
                            for m in st.session_state.messages)
                        lb_str = st.session_state.results_df.to_markdown(index=False) \
                                 if st.session_state.get("results_df") is not None else "N/A"
                        ctx = (
                            f"You are an expert AI data scientist.\n"
                            f"Leaderboard:\n{lb_str}\n"
                            f"Best score: {st.session_state.get('best_acc','N/A')}, "
                            f"Improvement: {st.session_state.get('imp','N/A')}\n"
                            f"Answer briefly and helpfully.\n\nHistory:\n{history}"
                        )
                        resp = client.models.generate_content(
                            model="gemini-2.5-flash", contents=ctx)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": resp.text})
                        st.markdown(
                            f'<div style="background:{_BG_DARK};padding:15px 20px;'
                            f'border-radius:12px 12px 12px 0;border:1px solid {_BORDER};'
                            f'border-left:3px solid {_PURPLE};margin-bottom:12px;margin-right:15%;">'
                            f'<p style="color:{_PURPLE};font-size:.75rem;font-weight:700;'
                            f'text-transform:uppercase;margin-bottom:5px;">ProtoML Assistant</p>'
                            f'<p style="color:#D1D1D6;font-size:.95rem;line-height:1.6;'
                            f'margin:0;">{resp.text}</p></div>',
                            unsafe_allow_html=True)
                    except Exception as e:
                        msg = str(e).lower()
                        if any(k in msg for k in
                               ("api_key_invalid", "api key not valid", "invalid_argument",
                                "401", "unauthenticated")):
                            st.error("Invalid API Key — check your Gemini key in the sidebar.")
                        elif any(k in msg for k in ("quota", "429")):
                            st.warning("API usage limit reached. Try again later.")
                        elif any(k in msg for k in ("connection", "timeout")):
                            st.warning("Cannot reach Gemini servers. Check your connection.")
                        else:
                            st.error(f"AI error: {e}")


# ── Training time bar chart ───────────────────────────────────────────────────

def render_timing_chart(timing_data: dict):
    """Horizontal bar chart of computation time per model."""
    if not timing_data:
        return
    models = list(timing_data.keys())
    times  = [timing_data[m] for m in models]
    pairs  = sorted(zip(times, models), reverse=True)
    times, models = zip(*pairs) if pairs else ([], [])

    fig, ax = _dark_fig(5.5, max(3.0, len(models) * 0.42))
    max_t   = max(times) if times else 1
    colors  = [_BLUE if t == max_t else "#4A6FA5" for t in times]
    bars    = ax.barh(models, times, color=colors, height=0.6)

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + max_t * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{t:.1f}s", va="center", ha="left", color=_TEXT, fontsize=8)

    ax.set_xlabel("Training Time (seconds)", color=_MUTED, fontsize=9)
    ax.set_title("Computation Time per Model", color=_TEXT, fontsize=11)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.set_xlim(0, max_t * 1.25)
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    st.caption("Faster is not always better — complex models spend more time but may generalise better.")


# ── CV fold box / violin plot ─────────────────────────────────────────────────

def render_cv_boxplot(fold_scores_per_model: dict):
    """Box plot of per-fold CV scores — shows variance across models."""
    if not fold_scores_per_model:
        return
    data   = {m: s for m, s in fold_scores_per_model.items() if s}
    if not data:
        st.info("No fold-level scores available.")
        return

    labels = list(data.keys())
    values = [data[m] for m in labels]

    fig, ax = _dark_fig(max(5.5, len(labels) * 1.6), 3.8)
    bp = ax.boxplot(
        values, patch_artist=True, labels=labels,
        medianprops=dict(color=_GREEN, linewidth=2),
        whiskerprops=dict(color=_MUTED, linewidth=1.2),
        capprops=dict(color=_MUTED, linewidth=1.2),
        flierprops=dict(marker="o", color=_AMBER, markersize=4, alpha=0.7),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(_BG_CARD)
        patch.set_edgecolor(_BLUE)
        patch.set_linewidth(1.5)

    ax.set_xticklabels(labels, rotation=35, ha="right", color=_TEXT, fontsize=8)
    ax.set_ylabel("CV Score per Fold", color=_MUTED, fontsize=9)
    ax.set_title("Cross-Validation Score Distribution", color=_TEXT, fontsize=11)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    st.caption("Green line = median. Box = 25th–75th percentile. Wider box = less stable model.")


# ── Calibration / reliability diagram ────────────────────────────────────────

def render_calibration_curve(calib_data: dict):
    """Reliability diagram — a perfectly calibrated model lies on the diagonal."""
    if not calib_data:
        st.info("Calibration data not available.")
        return

    fig, ax = _dark_fig(4.5, 4.2)
    ax.plot([0, 1], [0, 1], color=_BORDER, linestyle="--", lw=1.5,
            label="Perfect Calibration")

    if calib_data.get("type") == "binary":
        fop   = calib_data["fop"]
        mpv   = calib_data["mpv"]
        brier = calib_data.get("brier_score", "?")
        names = calib_data.get("class_names", ["neg", "pos"])
        ax.plot(mpv, fop, marker="o", color=_BLUE, lw=2, ms=6,
                label=f"{names[-1]} (Brier={brier:.3f})")
    else:
        colors = [_BLUE, _GREEN, _PURPLE, _AMBER, _RED, _TEAL,
                  "#ff7f0e", "#d62728", "#9467bd"]
        for i, (cn, vals) in enumerate(calib_data.get("classes", {}).items()):
            c     = colors[i % len(colors)]
            brier = vals.get("brier_score", "?")
            ax.plot(vals["mpv"], vals["fop"], marker="o", color=c, lw=1.5, ms=4,
                    label=f"{cn} (Brier={brier:.3f})")

    ax.set_xlabel("Mean Predicted Probability", color=_MUTED, fontsize=9)
    ax.set_ylabel("Fraction of Positives",      color=_MUTED, fontsize=9)
    ax.set_title("Reliability Diagram (Calibration)", color=_TEXT, fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER, labelcolor=_TEXT, fontsize=8)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    st.caption("A well-calibrated model follows the dashed diagonal. "
               "Brier score ↓ is better (0 = perfect). "
               "Curves above diagonal = under-confident; below = over-confident.")


# ── Correlation heatmap ───────────────────────────────────────────────────────

def render_correlation_heatmap(df: pd.DataFrame, feature_cols: list):
    """Seaborn correlation heatmap of numeric features."""
    try:
        import seaborn as sns
    except ImportError:
        st.info("Install seaborn (`pip install seaborn`) for the correlation heatmap.")
        return

    try:
        numeric = df[feature_cols].select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            st.info("Need ≥2 numeric features for a correlation heatmap.")
            return

        if numeric.shape[1] > 20:
            numeric = numeric.iloc[:, :20]
            st.caption("Showing first 20 numeric features.")

        corr = numeric.corr()
        n    = corr.shape[0]

        fig, ax = plt.subplots(figsize=(max(5, n * 0.85), max(4, n * 0.75)))
        fig.patch.set_facecolor(_BG_DARK)
        ax.set_facecolor(_BG_CARD)

        sns.heatmap(
            corr, ax=ax,
            annot=(n <= 14), fmt=".2f", cmap="coolwarm",
            linewidths=0.3, linecolor=_BG_DARK,
            vmin=-1, vmax=1, center=0,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 7, "color": _TEXT} if n <= 14 else {},
        )
        ax.set_title("Feature Correlation Matrix", color=_TEXT, fontsize=11, pad=10)
        ax.tick_params(colors=_MUTED, labelsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Values near ±1 indicate strong (multi)collinearity. "
                   "Highly correlated pairs may be redundant.")
    except Exception as e:
        st.warning(f"Correlation heatmap failed: {e}")


# ── Data quality report ───────────────────────────────────────────────────────

def render_data_quality_report(df: pd.DataFrame, feature_cols: list,
                                target_col: Optional[str] = None):
    """Show missing values, outliers, and feature skewness."""
    if df is None:
        return

    all_cols  = list(feature_cols) + ([target_col] if target_col else [])
    sub       = df[[c for c in all_cols if c in df.columns]]

    missing     = sub.isnull().sum()
    missing_pct = (missing / len(sub) * 100).round(1)
    miss_df     = pd.DataFrame({"Missing Count": missing,
                                  "Missing %": missing_pct})
    miss_df     = miss_df[miss_df["Missing Count"] > 0]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Missing Values")
        if miss_df.empty:
            st.success("No missing values detected.")
        else:
            st.warning(f"{len(miss_df)} column(s) have missing values.")
            st.dataframe(miss_df, use_container_width=True)

    with c2:
        st.markdown("##### Outliers (IQR method)")
        numeric_df = sub.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            outlier_rows = []
            for col in numeric_df.columns:
                s = numeric_df[col].dropna()
                if len(s) < 4:
                    continue
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr    = q3 - q1
                if iqr > 0:
                    n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
                    if n_out > 0:
                        outlier_rows.append(
                            {"Feature": col, "Outliers": n_out,
                             "% of rows": round(n_out / len(s) * 100, 1)})
            if outlier_rows:
                st.dataframe(pd.DataFrame(outlier_rows),
                              use_container_width=True, hide_index=True)
            else:
                st.success("No significant outliers detected.")
        else:
            st.info("No numeric features to check.")

    # Skewness bar
    numeric_df = sub.select_dtypes(include=[np.number])
    if not numeric_df.empty and numeric_df.shape[1] >= 2:
        from scipy.stats import skew as _skew
        sk_vals = {}
        for col in numeric_df.columns:
            s = numeric_df[col].dropna()
            if len(s) >= 3:
                try:
                    sk_vals[col] = round(float(_skew(s)), 3)
                except Exception:
                    pass
        if sk_vals:
            st.markdown("##### Feature Skewness")
            sk_df = (pd.DataFrame({"Feature": list(sk_vals.keys()),
                                    "Skewness": list(sk_vals.values())})
                       .sort_values("Skewness", key=abs, ascending=False)
                       .head(15))
            fig, ax = _dark_fig(5.5, max(2.5, len(sk_df) * 0.34))
            colors  = [_RED if abs(v) > 2 else (_AMBER if abs(v) > 1 else _GREEN)
                       for v in sk_df["Skewness"]]
            ax.barh(sk_df["Feature"], sk_df["Skewness"], color=colors, height=0.65)
            ax.axvline(0,  color=_BORDER, linewidth=1.0)
            ax.axvline(1,  color=_AMBER, linewidth=0.8, linestyle="--", alpha=0.6)
            ax.axvline(-1, color=_AMBER, linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_xlabel("Skewness", color=_MUTED, fontsize=9)
            ax.set_title("Feature Skewness  (|>1| skewed · |>2| highly skewed)",
                          color=_TEXT, fontsize=10)
            ax.tick_params(colors=_MUTED, labelsize=7)
            ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
            ax.set_axisbelow(True)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)


# ── Ablation study chart ──────────────────────────────────────────────────────

def render_ablation_chart(results: list):
    """Bar chart + table comparing ablation pipeline configurations."""
    if not results:
        st.info("No ablation results.")
        return

    df_a = pd.DataFrame(results)
    if "Config" not in df_a or "CV Score" not in df_a:
        st.dataframe(df_a, use_container_width=True)
        return

    colors = [_GREEN if ("Baseline" in str(c) or str(c).startswith("★"))
              else _BLUE for c in df_a["Config"]]

    fig, ax = _dark_fig(max(5.5, len(df_a) * 1.3), 3.5)
    bars = ax.bar(df_a["Config"], df_a["CV Score"].fillna(0),
                   color=colors, width=0.6, zorder=3)
    for bar, val in zip(bars, df_a["CV Score"].fillna(0)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                f"{val:.4f}", ha="center", va="bottom", color=_TEXT, fontsize=7.5)

    ax.set_xlabel("Pipeline Configuration", color=_MUTED, fontsize=9)
    ax.set_ylabel("CV Score", color=_MUTED, fontsize=9)
    ax.set_title("Ablation Study — Component Impact", color=_TEXT, fontsize=11)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.set_xticklabels(df_a["Config"], rotation=30, ha="right", fontsize=8)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.dataframe(df_a, use_container_width=True, hide_index=True)


# ── Inference prediction results ──────────────────────────────────────────────

def render_prediction_results(df_preds: pd.DataFrame):
    """Show inference results: predictions + confidence + distribution."""
    if df_preds is None or df_preds.empty:
        st.info("No predictions to display.")
        return

    st.markdown("#### Predictions")
    pred_cols = ["Prediction", "Confidence"]
    prob_cols = [c for c in df_preds.columns if c.startswith("P(")]
    disp_cols = [c for c in pred_cols + prob_cols if c in df_preds.columns]
    disp_df   = df_preds[disp_cols] if disp_cols else df_preds

    def _conf_color(val):
        if isinstance(val, (int, float)) and pd.notnull(val):
            if val >= 0.90:
                return f"color: {_GREEN}; font-weight: 600"
            if val >= 0.70:
                return f"color: {_AMBER}; font-weight: 600"
            return f"color: {_RED}; font-weight: 600"
        return ""

    if "Confidence" in disp_df.columns:
        styled = disp_df.style.map(_conf_color, subset=["Confidence"])
    else:
        styled = disp_df.style
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if "Prediction" in df_preds.columns:
        vc = df_preds["Prediction"].value_counts()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Class Distribution")
            fig, ax = _dark_fig(4, max(2.2, len(vc) * 0.65))
            ax.barh(vc.index.astype(str), vc.values, color=_BLUE, height=0.6)
            for i, v in enumerate(vc.values):
                ax.text(v + 0.1, i, str(v), va="center", color=_TEXT, fontsize=8)
            ax.set_xlabel("Count", color=_MUTED, fontsize=9)
            ax.tick_params(colors=_MUTED, labelsize=8)
            ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
            ax.set_axisbelow(True)
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
        with c2:
            st.markdown("#### Confidence Summary")
            if "Confidence" in df_preds.columns:
                conf = df_preds["Confidence"].dropna()
                st.metric("Mean Confidence",    f"{conf.mean():.3f}")
                st.metric("Min Confidence",     f"{conf.min():.3f}")
                low_n = int((conf < 0.7).sum())
                if low_n:
                    st.warning(f"{low_n} prediction(s) below 70% confidence — review carefully.")
                else:
                    st.success("All predictions ≥70% confidence.")

    # ── Confidence histogram ───────────────────────────────────────────────────
    if "Confidence" in df_preds.columns and len(df_preds) >= 5:
        render_confidence_histogram(df_preds)

    # ── Uncertain rows table ───────────────────────────────────────────────────
    if "Confidence" in df_preds.columns:
        low_conf = df_preds[df_preds["Confidence"] < 0.70].copy()
        if not low_conf.empty:
            with st.expander(f"Uncertain Predictions ({len(low_conf)} rows < 70% confidence)",
                             expanded=False):
                st.dataframe(low_conf.style.map(
                    lambda v: f"color: {_RED}; font-weight:600"
                    if isinstance(v, float) and v < 0.70 else "",
                    subset=["Confidence"],
                ), use_container_width=True, hide_index=True)


# ── Regression residual diagnostics ──────────────────────────────────────────

def render_regression_residuals(report_dict: dict):
    """4-panel residual diagnostic: Residuals vs Fitted, Q-Q, histogram, Scale-Location."""
    y_true = report_dict.get("y_true")
    y_pred = report_dict.get("y_pred")
    if y_true is None or y_pred is None:
        return
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred

    st.markdown("#### Residual Diagnostics")
    st.caption(
        "Healthy residuals should be randomly scattered around zero (no pattern), "
        "normally distributed, and variance-stable across fitted values."
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.patch.set_facecolor(_BG_DARK)
    for ax in axes.flat:
        ax.set_facecolor(_BG_CARD)
        ax.tick_params(colors=_MUTED, labelsize=8)
        ax.spines[:].set_color(_BORDER)

    # Panel 1 — Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, color=_BLUE, alpha=0.45, s=18,
               edgecolors=_BG_DARK, linewidth=0.4)
    ax.axhline(0, color=_GREEN, linewidth=1.4, linestyle="--")
    try:
        from numpy.polynomial.polynomial import polyfit as _pf
        coef = _pf(y_pred, residuals, 2)
        xs   = np.linspace(y_pred.min(), y_pred.max(), 200)
        ax.plot(xs, coef[0] + coef[1]*xs + coef[2]*xs**2,
                color=_AMBER, linewidth=1.2, linestyle=":", label="trend")
    except Exception:
        pass
    ax.set_xlabel("Fitted Values", color=_MUTED, fontsize=9)
    ax.set_ylabel("Residuals",     color=_MUTED, fontsize=9)
    ax.set_title("Residuals vs Fitted", color=_TEXT, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.4)

    # Panel 2 — Q-Q plot
    ax = axes[0, 1]
    try:
        from scipy import stats as _stats
        (osm, osr), (slope, intercept, _) = _stats.probplot(residuals, fit=True)
        ax.scatter(osm, osr, color=_PURPLE, alpha=0.5, s=16, edgecolors=_BG_DARK, linewidth=0.3)
        ref_x = np.array([osm[0], osm[-1]])
        ax.plot(ref_x, slope * ref_x + intercept, color=_GREEN, linewidth=1.4, linestyle="--")
    except Exception:
        ax.text(0.5, 0.5, "scipy not available", transform=ax.transAxes,
                color=_MUTED, ha="center", fontsize=9)
    ax.set_xlabel("Theoretical Quantiles", color=_MUTED, fontsize=9)
    ax.set_ylabel("Sample Quantiles",      color=_MUTED, fontsize=9)
    ax.set_title("Normal Q-Q Plot", color=_TEXT, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.4)

    # Panel 3 — Residual histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=min(40, max(10, len(residuals) // 20)),
            color=_BLUE, alpha=0.75, edgecolor=_BG_DARK, linewidth=0.4)
    try:
        from scipy.stats import norm as _norm
        mu, sigma = residuals.mean(), residuals.std()
        xs  = np.linspace(residuals.min(), residuals.max(), 200)
        pdf = _norm.pdf(xs, mu, sigma)
        ax2 = ax.twinx()
        ax2.plot(xs, pdf, color=_GREEN, linewidth=1.6, label="Normal PDF")
        ax2.set_facecolor(_BG_CARD)
        ax2.tick_params(colors=_MUTED, labelsize=7)
        ax2.spines[:].set_color(_BORDER)
        ax2.set_ylabel("Density", color=_MUTED, fontsize=8)
    except Exception:
        pass
    ax.axvline(0, color=_AMBER, linewidth=1.2, linestyle="--")
    ax.set_xlabel("Residual Value", color=_MUTED, fontsize=9)
    ax.set_ylabel("Frequency",      color=_MUTED, fontsize=9)
    ax.set_title("Residual Distribution", color=_TEXT, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)

    # Panel 4 — Scale-Location (√|residuals| vs fitted)
    ax = axes[1, 1]
    sqrt_abs_res = np.sqrt(np.abs(residuals))
    ax.scatter(y_pred, sqrt_abs_res, color=_TEAL, alpha=0.45, s=18,
               edgecolors=_BG_DARK, linewidth=0.4)
    try:
        coef2 = np.polyfit(y_pred, sqrt_abs_res, 1)
        xs2   = np.linspace(y_pred.min(), y_pred.max(), 200)
        ax.plot(xs2, np.polyval(coef2, xs2), color=_AMBER, linewidth=1.2, linestyle=":")
    except Exception:
        pass
    ax.set_xlabel("Fitted Values",          color=_MUTED, fontsize=9)
    ax.set_ylabel("√|Residuals|",           color=_MUTED, fontsize=9)
    ax.set_title("Scale-Location",          color=_TEXT, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.4)

    fig.suptitle("Residual Diagnostic Plots", color=_TEXT, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Normality test summary
    try:
        from scipy.stats import shapiro, kstest, normaltest
        n   = len(residuals)
        col1, col2, col3 = st.columns(3)
        if n <= 5000:
            stat_sw, p_sw = shapiro(residuals[:5000])
            col1.metric("Shapiro-Wilk p",
                        f"{p_sw:.4f}",
                        delta="Normal ✓" if p_sw > 0.05 else "Non-normal ⚠")
        stat_ks, p_ks = kstest(
            (residuals - residuals.mean()) / (residuals.std() or 1), "norm")
        col2.metric("KS Test p",
                    f"{p_ks:.4f}",
                    delta="Normal ✓" if p_ks > 0.05 else "Non-normal ⚠")
        stat_dp, p_dp = normaltest(residuals)
        col3.metric("D'Agostino p",
                    f"{p_dp:.4f}",
                    delta="Normal ✓" if p_dp > 0.05 else "Non-normal ⚠")
    except Exception:
        pass


# ── EDA: feature distributions (histograms + KDE) ────────────────────────────

def render_feature_distributions(
    df: pd.DataFrame,
    feature_cols: list,
    max_cols: int = 20,
):
    """Grid of per-column histogram + KDE overlays."""
    if df is None or not feature_cols:
        return
    numeric = [c for c in feature_cols if c in df.columns
               and pd.api.types.is_numeric_dtype(df[c])][:max_cols]
    if not numeric:
        st.info("No numeric features to plot.")
        return

    n_feat   = len(numeric)
    n_cols_g = min(3, n_feat)
    n_rows_g = (n_feat + n_cols_g - 1) // n_cols_g
    fig, axes = plt.subplots(n_rows_g, n_cols_g,
                              figsize=(n_cols_g * 3.8, n_rows_g * 2.8))
    fig.patch.set_facecolor(_BG_DARK)
    axes = np.array(axes).flatten() if n_feat > 1 else [axes]

    colors_cycle = [_BLUE, _PURPLE, _TEAL, _GREEN, _AMBER, _RED]
    for i, (col, ax) in enumerate(zip(numeric, axes)):
        ax.set_facecolor(_BG_CARD)
        ax.tick_params(colors=_MUTED, labelsize=7)
        ax.spines[:].set_color(_BORDER)
        s = df[col].dropna()
        if len(s) == 0:
            ax.text(0.5, 0.5, "all NaN", transform=ax.transAxes,
                    color=_MUTED, ha="center", fontsize=8)
            ax.set_title(col, color=_TEXT, fontsize=8)
            continue
        c = colors_cycle[i % len(colors_cycle)]
        n_bins = min(40, max(10, len(s) // 15))
        ax.hist(s, bins=n_bins, color=c, alpha=0.65,
                edgecolor=_BG_DARK, linewidth=0.3, density=True)
        try:
            from scipy.stats import gaussian_kde as _kde
            kde  = _kde(s)
            xs   = np.linspace(s.min(), s.max(), 300)
            ax.plot(xs, kde(xs), color=_TEXT, linewidth=1.2)
        except Exception:
            pass
        ax.axvline(s.mean(),   color=_GREEN, linewidth=1.0, linestyle="--",
                   label=f"μ={s.mean():.2f}")
        ax.axvline(s.median(), color=_AMBER, linewidth=1.0, linestyle=":",
                   label=f"med={s.median():.2f}")
        ax.set_title(col, color=_TEXT, fontsize=8, fontweight="bold")
        ax.yaxis.grid(True, color=_BORDER, linewidth=0.3)
        ax.set_axisbelow(True)

    for ax in axes[n_feat:]:
        ax.set_visible(False)

    plt.tight_layout(pad=0.6)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── EDA: feature box plots ────────────────────────────────────────────────────

def render_feature_boxplots(
    df: pd.DataFrame,
    feature_cols: list,
    max_cols: int = 20,
):
    """Side-by-side box plots showing spread and outliers per feature."""
    if df is None or not feature_cols:
        return
    numeric = [c for c in feature_cols if c in df.columns
               and pd.api.types.is_numeric_dtype(df[c])][:max_cols]
    if not numeric:
        st.info("No numeric features to plot.")
        return

    data_list = [df[c].dropna().values for c in numeric]
    n = len(numeric)

    fig_w = max(6, n * 0.9)
    fig, ax = _dark_fig(fig_w, 4.5)
    bp = ax.boxplot(
        data_list,
        patch_artist=True,
        vert=True,
        widths=0.55,
        showfliers=True,
        medianprops=dict(color=_GREEN, linewidth=1.6),
        whiskerprops=dict(color=_BORDER, linewidth=0.9),
        capprops=dict(color=_MUTED, linewidth=0.9),
        flierprops=dict(marker="o", color=_AMBER, markersize=3,
                         alpha=0.5, linestyle="none"),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(_BLUE)
        patch.set_alpha(0.45)
        patch.set_edgecolor(_BLUE)

    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(numeric, rotation=45, ha="right", fontsize=7, color=_MUTED)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_title("Feature Box Plots (IQR spreads + outliers)",
                  color=_TEXT, fontsize=10, fontweight="bold")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── EDA: target distribution ──────────────────────────────────────────────────

def render_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "classification",
):
    """Bar chart (classification) or histogram + KDE (regression) for target column."""
    if df is None or target_col not in df.columns:
        return
    s = df[target_col].dropna()

    st.markdown(f"##### Target Distribution: `{target_col}`")
    if task_type == "regression":
        fig, ax = _dark_fig(7, 3.2)
        n_bins = min(50, max(10, len(s) // 20))
        ax.hist(s, bins=n_bins, color=_BLUE, alpha=0.65,
                edgecolor=_BG_DARK, linewidth=0.3, density=True)
        try:
            from scipy.stats import gaussian_kde as _kde
            kde = _kde(s)
            xs  = np.linspace(s.min(), s.max(), 300)
            ax.plot(xs, kde(xs), color=_GREEN, linewidth=1.6, label="KDE")
        except Exception:
            pass
        ax.axvline(s.mean(),   color=_AMBER, linewidth=1.2, linestyle="--",
                   label=f"mean={s.mean():.2f}")
        ax.axvline(s.median(), color=_RED,   linewidth=1.2, linestyle=":",
                   label=f"median={s.median():.2f}")
        ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER, labelcolor=_TEXT, fontsize=8)
        ax.set_xlabel(target_col, color=_MUTED, fontsize=9)
        ax.set_ylabel("Density",  color=_MUTED, fontsize=9)
    else:
        vc = s.astype(str).value_counts()
        colors = [_BLUE, _PURPLE, _GREEN, _AMBER, _RED, _TEAL]
        fig, ax = _dark_fig(max(5, len(vc) * 0.8), 3.2)
        bars = ax.bar(range(len(vc)), vc.values,
                       color=[colors[i % len(colors)] for i in range(len(vc))],
                       alpha=0.75, edgecolor=_BG_DARK, linewidth=0.4, width=0.6)
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index, rotation=30, ha="right", fontsize=8, color=_MUTED)
        for bar, val in zip(bars, vc.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", color=_TEXT, fontsize=8,
                    fontweight="bold")
        ax.set_ylabel("Count",   color=_MUTED, fontsize=9)
        ax.yaxis.grid(True, color=_BORDER, linewidth=0.4)
        ax.set_axisbelow(True)
    ax.set_title(f"Target: {target_col}", color=_TEXT, fontsize=10, fontweight="bold")
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── EDA: statistical summary ──────────────────────────────────────────────────

def render_statistical_summary(df: pd.DataFrame, feature_cols: list):
    """Extended stats table: mean, median, std, IQR, min, max, skewness, kurtosis."""
    if df is None or not feature_cols:
        return
    numeric = [c for c in feature_cols if c in df.columns
               and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        st.info("No numeric features.")
        return

    try:
        from scipy.stats import skew as _skew, kurtosis as _kurt
    except ImportError:
        _skew = _kurt = None

    rows = []
    for col in numeric:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        row = {
            "Feature":  col,
            "Count":    len(s),
            "Missing":  int(df[col].isna().sum()),
            "Mean":     round(float(s.mean()), 4),
            "Median":   round(float(s.median()), 4),
            "Std":      round(float(s.std()), 4),
            "IQR":      round(float(q3 - q1), 4),
            "Min":      round(float(s.min()), 4),
            "Max":      round(float(s.max()), 4),
        }
        if _skew:
            try:
                row["Skewness"] = round(float(_skew(s)), 4)
                row["Kurtosis"] = round(float(_kurt(s)), 4)
            except Exception:
                pass
        rows.append(row)

    if rows:
        stat_df = pd.DataFrame(rows).set_index("Feature")
        st.dataframe(
            stat_df.style.background_gradient(
                cmap="Blues", subset=["Std"], axis=0),
            use_container_width=True,
        )


# ── EDA: missing value heatmap ────────────────────────────────────────────────

def render_missing_heatmap(df: pd.DataFrame, feature_cols: list):
    """Binary heatmap showing where missing values occur across rows/columns."""
    if df is None or not feature_cols:
        return
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        return

    sub = df[cols].isnull()
    if not sub.any().any():
        st.success("No missing values — heatmap skipped.")
        return

    # Limit rows for performance
    if len(sub) > 500:
        sub = sub.sample(500, random_state=42)

    fig_h = max(2.5, len(cols) * 0.28)
    fig, ax = _dark_fig(min(10, len(sub.columns) * 0.5 + 2), fig_h)
    ax.imshow(sub.T.values.astype(int), aspect="auto",
              cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=7, color=_MUTED)
    ax.set_xlabel("Row index (sample)", color=_MUTED, fontsize=8)
    ax.set_title("Missing Value Pattern (blue = missing)",
                  color=_TEXT, fontsize=9, fontweight="bold")
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Vision: class distribution ────────────────────────────────────────────────

def render_vision_class_distribution(class_counts: dict):
    """Horizontal bar chart of image counts per class."""
    if not class_counts:
        return
    labels = list(class_counts.keys())
    values = [class_counts[k] for k in labels]
    total  = sum(values)

    fig, ax = _dark_fig(6, max(2.5, len(labels) * 0.55))
    colors = [_BLUE, _PURPLE, _GREEN, _AMBER, _RED, _TEAL]
    bars   = ax.barh(labels, values,
                     color=[colors[i % len(colors)] for i in range(len(labels))],
                     height=0.6, alpha=0.80)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + total * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val}  ({val/total*100:.1f}%)",
                va="center", color=_TEXT, fontsize=8)
    ax.set_xlabel("Image Count", color=_MUTED, fontsize=9)
    ax.set_title(f"Class Distribution  (total: {total:,})",
                  color=_TEXT, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    imbalance = max(values) / (min(values) or 1)
    if imbalance > 3:
        st.warning(
            f"Class imbalance ratio {imbalance:.1f}× detected. "
            "Enable 'Handle class imbalance' in the Vision settings."
        )


# ── Vision: sample image grid ─────────────────────────────────────────────────

def render_vision_sample_grid(samples: dict, n_per_class: int = 4):
    """Show a grid of sample images: one row per class, n_per_class columns."""
    if not samples:
        return
    try:
        from PIL import Image as PILImage
    except ImportError:
        st.info("Pillow not available — skipping image grid.")
        return

    st.markdown("##### Sample Images per Class")
    for class_name, paths in samples.items():
        paths = paths[:n_per_class]
        st.caption(f"**{class_name}** ({len(paths)} samples shown)")
        cols = st.columns(len(paths))
        for col, path in zip(cols, paths):
            try:
                img = PILImage.open(path).convert("RGB")
                img.thumbnail((160, 160))
                col.image(img, use_container_width=True)
            except Exception:
                col.markdown(f"*Error*")


# ── Partial Dependence Plots ──────────────────────────────────────────────────

def render_pdp(pdp_data: dict):
    """Line plots of partial dependence for top features."""
    if not pdp_data:
        return
    features = list(pdp_data.keys())
    n = len(features)
    if n == 0:
        return

    st.markdown("#### Partial Dependence Plots")
    st.caption(
        "Shows how the model's average prediction changes as each feature varies, "
        "holding all other features at their mean values."
    )

    n_cols_g = min(3, n)
    n_rows_g = (n + n_cols_g - 1) // n_cols_g
    fig, axes = plt.subplots(n_rows_g, n_cols_g,
                              figsize=(n_cols_g * 3.8, n_rows_g * 2.8))
    fig.patch.set_facecolor(_BG_DARK)
    axes = np.array(axes).flatten() if n > 1 else [axes]

    colors = [_BLUE, _PURPLE, _GREEN, _AMBER, _TEAL, _RED]
    for i, (feat, (xs, ys)) in enumerate(pdp_data.items()):
        ax = axes[i]
        ax.set_facecolor(_BG_CARD)
        ax.tick_params(colors=_MUTED, labelsize=7)
        ax.spines[:].set_color(_BORDER)
        c = colors[i % len(colors)]
        ax.plot(xs, ys, color=c, linewidth=1.8)
        ax.fill_between(xs, ys, alpha=0.12, color=c)
        ax.axhline(np.mean(ys), color=_MUTED, linewidth=0.8, linestyle="--")
        ax.set_xlabel(feat,          color=_MUTED, fontsize=8)
        ax.set_ylabel("Avg. Prediction", color=_MUTED, fontsize=8)
        ax.set_title(feat,           color=_TEXT,  fontsize=9, fontweight="bold")
        ax.xaxis.grid(True, color=_BORDER, linewidth=0.3)
        ax.yaxis.grid(True, color=_BORDER, linewidth=0.3)
        ax.set_axisbelow(True)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout(pad=0.7)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── SHAP waterfall (single sample) ───────────────────────────────────────────

def render_shap_waterfall(waterfall_data: dict):
    """Horizontal waterfall bar for one sample's SHAP feature contributions."""
    if not waterfall_data:
        return
    features   = waterfall_data.get("features", [])
    values     = waterfall_data.get("shap_values", [])
    base_value = waterfall_data.get("base_value", 0.0)
    if not features or not values:
        return

    st.markdown("#### SHAP Waterfall — Single Sample Explanation")
    st.caption(
        "Shows how each feature pushes the prediction above or below the baseline "
        f"(baseline = {base_value:.4f})."
    )

    # Flatten any nested list/array values (can appear for multi-class SHAP)
    def _scalar(v):
        if isinstance(v, (list, np.ndarray)):
            arr = np.array(v).ravel()
            return float(np.mean(np.abs(arr)))
        return float(v)

    # Sort by absolute contribution
    pairs  = sorted(zip(features, values), key=lambda x: abs(_scalar(x[1])), reverse=True)[:15]
    feats  = [p[0] for p in pairs]
    vals   = [_scalar(p[1]) for p in pairs]

    fig, ax = _dark_fig(7, max(3, len(feats) * 0.42))
    colors  = [_GREEN if v >= 0 else _RED for v in vals]
    ax.barh(range(len(feats)), vals, color=colors, height=0.65, alpha=0.80,
            edgecolor=_BG_DARK, linewidth=0.4)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8, color=_MUTED)
    ax.axvline(0, color=_BORDER, linewidth=1.0)
    ax.set_xlabel("SHAP Value (feature contribution)", color=_MUTED, fontsize=9)
    ax.set_title(f"Waterfall  (base = {base_value:.4f})",
                  color=_TEXT, fontsize=10, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(v + (0.002 if v >= 0 else -0.002), i,
                f"{v:+.4f}", va="center",
                ha="left" if v >= 0 else "right",
                color=_TEXT, fontsize=7)
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── LIME explanation ──────────────────────────────────────────────────────────

def render_lime_explanation(lime_data: dict):
    """Horizontal bar showing LIME feature weights for one sample."""
    if not lime_data:
        return
    features = lime_data.get("features", [])
    weights  = lime_data.get("weights", [])
    label    = lime_data.get("label", "class")
    if not features or not weights:
        return

    st.markdown(f"#### LIME Explanation — Prediction: `{label}`")
    st.caption(
        "LIME fits a local linear model around this sample to explain why "
        "the model made this specific prediction."
    )
    pairs  = sorted(zip(features, weights), key=lambda x: abs(x[1]), reverse=True)[:15]
    feats  = [p[0] for p in pairs]
    vals   = [p[1] for p in pairs]

    fig, ax = _dark_fig(7, max(3, len(feats) * 0.42))
    colors  = [_GREEN if v >= 0 else _RED for v in vals]
    ax.barh(range(len(feats)), vals, color=colors, height=0.65, alpha=0.80,
            edgecolor=_BG_DARK, linewidth=0.4)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8, color=_MUTED)
    ax.axvline(0, color=_BORDER, linewidth=1.0)
    ax.set_xlabel("LIME Weight",  color=_MUTED, fontsize=9)
    ax.set_title("LIME Local Explanation", color=_TEXT, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Learning curve ────────────────────────────────────────────────────────────

def render_learning_curve(lc_data: dict):
    """Train size vs. CV score: shows if more data would help."""
    if not lc_data:
        return
    train_sizes = lc_data.get("train_sizes", [])
    train_means = lc_data.get("train_means", [])
    train_stds  = lc_data.get("train_stds",  [])
    val_means   = lc_data.get("val_means",   [])
    val_stds    = lc_data.get("val_stds",    [])
    metric_name = lc_data.get("metric_name", "Score")
    if not train_sizes:
        return

    ts = np.array(train_sizes)
    tm = np.array(train_means)
    ts_ = np.array(train_stds)
    vm  = np.array(val_means)
    vs  = np.array(val_stds)

    st.markdown("#### Learning Curve")
    st.caption(
        "If validation score keeps rising with more data, you could benefit from a "
        "larger dataset. A plateau means model capacity is the bottleneck."
    )
    fig, ax = _dark_fig(7, 3.6)
    ax.plot(ts, tm, color=_BLUE,   linewidth=1.8, label="Training score")
    ax.fill_between(ts, tm - ts_, tm + ts_, alpha=0.14, color=_BLUE)
    ax.plot(ts, vm, color=_GREEN,  linewidth=1.8, label="CV score")
    ax.fill_between(ts, vm - vs,  vm + vs,  alpha=0.14, color=_GREEN)
    ax.set_xlabel("Training samples", color=_MUTED, fontsize=9)
    ax.set_ylabel(metric_name,        color=_MUTED, fontsize=9)
    ax.set_title("Learning Curve",    color=_TEXT, fontsize=10, fontweight="bold")
    ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER, labelcolor=_TEXT, fontsize=9)
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Feature ablation chart ────────────────────────────────────────────────────

def render_feature_ablation_chart(ablation_results: list):
    """Bar chart of CV score delta when each feature is dropped."""
    if not ablation_results:
        return
    df = pd.DataFrame(ablation_results)
    if df.empty or "Feature Dropped" not in df.columns:
        return

    df = df.sort_values("Delta (%)", ascending=True)
    st.markdown("#### Feature Ablation")
    st.caption(
        "How much the CV score drops when each feature is removed individually. "
        "Large drops = high importance. Near-zero or positive = possibly noise."
    )

    colors = [_RED if v < -1 else (_AMBER if v < 0 else _GREEN)
              for v in df["Delta (%)"]]
    fig, ax = _dark_fig(7, max(3, len(df) * 0.45))
    ax.barh(df["Feature Dropped"], df["Delta (%)"],
            color=colors, height=0.65, alpha=0.80,
            edgecolor=_BG_DARK, linewidth=0.4)
    ax.axvline(0, color=_BORDER, linewidth=1.0)
    ax.set_xlabel("Score Delta (%)",     color=_MUTED, fontsize=9)
    ax.set_title("Feature Ablation — Drop-One Impact",
                  color=_TEXT, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── Confidence histogram ──────────────────────────────────────────────────────

def render_confidence_histogram(df_preds: pd.DataFrame):
    """Histogram of prediction confidence values with zone annotations."""
    if "Confidence" not in df_preds.columns:
        return
    conf = df_preds["Confidence"].dropna().values
    if len(conf) == 0:
        return

    st.markdown("#### Confidence Distribution")
    fig, ax = _dark_fig(7, 3.0)
    n_bins = min(30, max(10, len(conf) // 10))
    ax.hist(conf, bins=n_bins, color=_BLUE, alpha=0.75,
            edgecolor=_BG_DARK, linewidth=0.3)

    ax.axvspan(0.0, 0.70, alpha=0.08, color=_RED,   label="Low (<70%)")
    ax.axvspan(0.70, 0.90, alpha=0.08, color=_AMBER, label="Medium (70-90%)")
    ax.axvspan(0.90, 1.01, alpha=0.08, color=_GREEN, label="High (≥90%)")
    ax.axvline(0.70, color=_RED,   linewidth=1.0, linestyle="--")
    ax.axvline(0.90, color=_GREEN, linewidth=1.0, linestyle="--")

    ax.set_xlabel("Confidence", color=_MUTED, fontsize=9)
    ax.set_ylabel("Count",      color=_MUTED, fontsize=9)
    ax.set_title("Prediction Confidence Distribution",
                  color=_TEXT, fontsize=10, fontweight="bold")
    ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER, labelcolor=_TEXT, fontsize=8)
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Per-class accuracy table ──────────────────────────────────────────────────

def render_per_class_accuracy(report_dict: dict, class_names: Optional[list] = None):
    """Table + bar chart of per-class precision, recall, F1 and support."""
    if not report_dict:
        return
    rows = []
    for key, val in report_dict.items():
        if isinstance(val, dict) and "f1-score" in val:
            rows.append({
                "Class":     str(class_names[int(key)] if (
                                class_names and key.isdigit() and
                                int(key) < len(class_names)) else key),
                "Precision": round(val.get("precision", 0), 4),
                "Recall":    round(val.get("recall", 0), 4),
                "F1":        round(val.get("f1-score", 0), 4),
                "Support":   int(val.get("support", 0)),
            })
    if not rows:
        return

    df = pd.DataFrame(rows)
    st.markdown("#### Per-Class Accuracy")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with c2:
        fig, ax = _dark_fig(5, max(2.5, len(rows) * 0.55))
        y = range(len(rows))
        ax.barh([r["Class"] for r in rows], df["F1"].values,
                color=_BLUE, height=0.6, alpha=0.80)
        ax.axvline(df["F1"].mean(), color=_AMBER, linewidth=1.0,
                   linestyle="--", label=f"avg={df['F1'].mean():.3f}")
        ax.set_xlabel("F1 Score", color=_MUTED, fontsize=9)
        ax.set_title("Per-Class F1", color=_TEXT, fontsize=10, fontweight="bold")
        ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER, labelcolor=_TEXT, fontsize=8)
        ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ── LaTeX table export ────────────────────────────────────────────────────────

def render_latex_export(results_df: pd.DataFrame):
    """Generate a LaTeX booktabs table from the leaderboard DataFrame."""
    if results_df is None or results_df.empty:
        return
    try:
        latex_str = results_df.to_latex(
            index=False,
            float_format="%.4f",
            bold_rows=False,
            escape=True,
            caption="ProtoML Model Leaderboard",
            label="tab:protoCML_results",
        )
        # Wrap in booktabs-friendly form
        latex_str = latex_str.replace(r"\toprule", r"\hline\hline") \
                             .replace(r"\midrule", r"\hline") \
                             .replace(r"\bottomrule", r"\hline\hline")
    except Exception as e:
        latex_str = f"% LaTeX export failed: {e}"

    st.text_area("LaTeX Table (copy into your paper)", latex_str,
                  height=200, key="_latex_export")
    st.download_button(
        "Download LaTeX (.tex)",
        data=latex_str.encode("utf-8"),
        file_name="ProtoML_Leaderboard.tex",
        mime="text/plain",
    )


# ── Multi-label classification reporting ─────────────────────────────────────

def render_multilabel_report(report_dict: dict):
    """
    Show per-label precision / recall / F1 table plus summary metrics
    for multi-label classification results.
    report_dict must contain 'label_names', 'hamming_loss', 'micro_f1',
    'macro_f1', 'samples_f1' and per-label sub-dicts keyed by label name.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not report_dict or report_dict.get("_task") != "multilabel":
        st.info("No multi-label report available.")
        return

    label_names = report_dict.get("label_names", [])
    if not label_names:
        st.info("No labels found in report.")
        return

    # ── Summary bar ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Micro F1",    report_dict.get("micro_f1",    "—"))
    c2.metric("Macro F1",    report_dict.get("macro_f1",    "—"))
    c3.metric("Samples F1",  report_dict.get("samples_f1",  "—"))
    c4.metric("Hamming Loss", report_dict.get("hamming_loss", "—"))

    # ── Per-label table ──────────────────────────────────────────────────────
    rows = []
    for lbl in label_names:
        sub = report_dict.get(lbl, {})
        if isinstance(sub, dict):
            rows.append({
                "Label":     lbl,
                "Precision": round(sub.get("precision", 0.0), 4),
                "Recall":    round(sub.get("recall",    0.0), 4),
                "F1-Score":  round(sub.get("f1-score",  0.0), 4),
                "Support":   int(sub.get("support",     0)),
            })

    if rows:
        df_lbl = pd.DataFrame(rows)
        st.dataframe(df_lbl, use_container_width=True)

        # ── Per-label F1 bar chart ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.6), 4))
        colors = ["#2196F3" if f >= 0.7 else "#FF9800" if f >= 0.4 else "#F44336"
                  for f in df_lbl["F1-Score"]]
        ax.bar(df_lbl["Label"], df_lbl["F1-Score"], color=colors)
        ax.axhline(0.7, color="green",  linestyle="--", linewidth=0.8, label="0.70 threshold")
        ax.axhline(0.4, color="orange", linestyle="--", linewidth=0.8, label="0.40 threshold")
        ax.set_ylabel("F1-Score")
        ax.set_title("Per-label F1 Score")
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_multilabel_confusion_matrix(cm_data: dict):
    """
    Display per-label 2×2 confusion matrices from multilabel_confusion_matrix output.
    cm_data: {"matrix": [[[TN,FP],[FN,TP]], ...], "labels": [...], "multilabel": True}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if not cm_data or not cm_data.get("multilabel"):
        st.info("No multi-label confusion matrix available.")
        return

    matrices = cm_data.get("matrix", [])
    labels   = cm_data.get("labels", [])
    if not matrices or not labels:
        return

    n_labels = len(labels)
    n_cols   = min(4, n_labels)
    n_rows   = (n_labels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3, n_rows * 3))
    axes_flat = np.array(axes).ravel() if n_labels > 1 else [axes]

    for i, (mat, lbl) in enumerate(zip(matrices, labels)):
        ax = axes_flat[i]
        m  = np.array(mat)  # shape (2, 2): [[TN, FP], [FN, TP]]
        im = ax.imshow(m, cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(m[r, c]), ha="center", va="center",
                        fontsize=11, color="white" if m[r, c] > m.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"], fontsize=7)
        ax.set_title(lbl, fontsize=8, fontweight="bold")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── NLP-specific visualisations ───────────────────────────────────────────────

def render_nlp_lime_explanation(lime_data: dict):
    """Horizontal bar chart of LIME word-level explanations."""
    if not lime_data:
        return
    if "error" in lime_data:
        st.warning(f"LIME: {lime_data['error']}")
        return

    explanation = lime_data.get("explanation", [])
    pred_class  = lime_data.get("predicted_class", "?")
    text        = lime_data.get("text", "")

    if not explanation:
        st.info("No LIME explanation available for this sample.")
        return

    st.markdown(f"#### LIME Text Explanation — Prediction: `{pred_class}`")
    if text:
        st.caption(f"Sample: _{text[:200]}{'…' if len(text) > 200 else ''}_")
    st.caption(
        "Green bars indicate words that pushed the model toward this prediction; "
        "red bars indicate words that pushed it away."
    )

    pairs  = sorted(explanation, key=lambda x: abs(x[1]), reverse=True)[:15]
    words  = [p[0] for p in pairs]
    vals   = [p[1] for p in pairs]

    fig, ax = _dark_fig(7, max(3.5, len(words) * 0.44))
    colors  = [_GREEN if v >= 0 else _RED for v in vals]
    ax.barh(range(len(words)), vals, color=colors, height=0.65,
            alpha=0.82, edgecolor=_BG_DARK, linewidth=0.4)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=9, color=_TEXT)
    ax.axvline(0, color=_BORDER, linewidth=1.0)
    ax.set_xlabel("LIME Weight", color=_MUTED, fontsize=9)
    ax.set_title("Top Word Contributions (LIME)", color=_TEXT,
                 fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.6)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_nlp_top_features(top_features: dict):
    """Top TF-IDF coefficient words per class (linear models only)."""
    if not top_features:
        st.info("Top-feature extraction is only available for linear models "
                "(Logistic Regression, Linear SVC, SGD Classifier).")
        return

    st.markdown("#### Top TF-IDF Features per Class")
    st.caption("Words with the highest positive / negative classifier coefficients.")

    for label, sides in top_features.items():
        with st.expander(f"Class: {label}", expanded=False):
            pos = sides.get("positive", [])[:15]
            neg = sides.get("negative", [])[:15]

            if not pos and not neg:
                st.caption("No coefficients extracted.")
                continue

            # Two sub-columns: positive & negative
            c1, c2 = st.columns(2)

            def _bar(ax_local, items, color, title):
                words = [w for w, _ in items]
                scores = [s for _, s in items]
                ax_local.barh(range(len(words)), scores, color=color,
                              height=0.65, alpha=0.82,
                              edgecolor=_BG_DARK, linewidth=0.4)
                ax_local.set_yticks(range(len(words)))
                ax_local.set_yticklabels(words, fontsize=8, color=_TEXT)
                ax_local.set_title(title, color=_TEXT, fontsize=9, fontweight="bold")
                ax_local.tick_params(colors=_MUTED)
                ax_local.spines[:].set_color(_BORDER)
                ax_local.xaxis.grid(True, color=_BORDER, linewidth=0.4)
                ax_local.set_axisbelow(True)

            if pos:
                with c1:
                    fig, ax = plt.subplots(figsize=(3.5, max(2.5, len(pos) * 0.38)))
                    fig.patch.set_facecolor(_BG_DARK)
                    ax.set_facecolor(_BG_CARD)
                    _bar(ax, pos, _GREEN, "Positive (for class)")
                    plt.tight_layout(pad=0.5)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

            if neg:
                with c2:
                    fig, ax = plt.subplots(figsize=(3.5, max(2.5, len(neg) * 0.38)))
                    fig.patch.set_facecolor(_BG_DARK)
                    ax.set_facecolor(_BG_CARD)
                    _bar(ax, neg, _RED, "Negative (against class)")
                    plt.tight_layout(pad=0.5)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
