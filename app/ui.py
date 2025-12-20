from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

from .config import ensure_paths
from .services import BaselineService


def sidebar_controls() -> Tuple[str, str]:
    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox("Baseline model", ["LOGREG", "SVM"], index=0)
        dataset_name = st.text_input("HF dataset name", value="imdb")
        st.markdown("---")
        st.caption("If no checkpoint exists, a quick baseline will be trained on the dataset.")
    return model_name, dataset_name


def render_single_text(service: BaselineService, clf):
    st.subheader("Single Text Inference")
    default_text = "I absolutely loved this movie. The acting was superb and the story was touching!"
    text = st.text_area("Enter text", value=default_text, height=150)
    if st.button("Predict", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            pred, scores = service.predict_with_scores(clf, [text])
            label = "Positive" if int(pred[0]) == 1 else "Negative"
            if scores is not None:
                st.metric("Prediction", f"{label}", help=f"Positive score: {scores[0]:.3f}")
            else:
                st.metric("Prediction", label)


def render_batch_tab(service: BaselineService, clf):
    st.subheader("Batch Prediction from CSV")
    st.caption("Upload a CSV with a 'text' column. Optionally a 'label' column (0/1) for evaluation.")
    uploaded = st.file_uploader("CSV file", type=["csv"]) 
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding_errors="ignore")

        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            st.write("Preview:")
            st.dataframe(df.head(10))

            if st.button("Run batch prediction", key="batch"):
                preds, scores = service.predict_with_scores(clf, df["text"].astype(str).tolist())
                df_out = df.copy()
                df_out["prediction"] = preds
                if scores is not None:
                    df_out["positive_score"] = scores
                df_out["prediction_label"] = df_out["prediction"].map({0: "Negative", 1: "Positive"})

                st.subheader("Results")
                st.dataframe(df_out.head(20))

                # Metrics if label provided
                if "label" in df_out.columns:
                    try:
                        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                        y_true = df_out["label"].astype(int).values
                        y_pred = df_out["prediction"].astype(int).values
                        acc = accuracy_score(y_true, y_pred)
                        report = classification_report(y_true, y_pred, output_dict=True)
                        cm = confusion_matrix(y_true, y_pred)

                        st.metric("Accuracy", f"{acc:.4f}")
                        st.json(report)
                        st.write("Confusion Matrix:")
                        st.dataframe(pd.DataFrame(cm, index=["neg","pos"], columns=["pred_neg","pred_pos"]))
                    except Exception as e:
                        st.warning(f"Could not compute metrics: {e}")

                # Offer download
                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download predictions CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                )


def render_report_tab(model_name: str, reports_dir: Path):
    st.subheader("Model Report and Feature Importance")
    report_path = reports_dir / f"{model_name.lower()}_report.json"
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                rep = json.load(f)
            st.metric("Last Eval Accuracy", f"{rep.get('accuracy', float('nan')):.4f}")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Top Positive Words")
                pos_df = pd.DataFrame({
                    "word": rep.get("top_positive", {}).get("word", []),
                    "weight": rep.get("top_positive", {}).get("weight", []),
                })
                st.bar_chart(pos_df.set_index("word"))
            with col2:
                st.write("Top Negative Words")
                neg_df = pd.DataFrame({
                    "word": rep.get("top_negative", {}).get("word", []),
                    "weight": rep.get("top_negative", {}).get("weight", []),
                })
                st.bar_chart(neg_df.set_index("word"))

            st.caption("Report file: " + str(report_path))
        except Exception as e:
            st.warning(f"Failed to read report: {e}")
    else:
        st.info("Report not available yet. It will be created after the first training run.")


def run_app():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")
    st.title("💬 Sentiment Analysis")
    st.caption("Built on your Pytorch-Sentiment-Analysis project. ML Baseline via TF-IDF + Linear Model.")

    paths = ensure_paths()
    model_name, dataset_name = sidebar_controls()

    @st.cache_resource(show_spinner=True)
    def _load_model_cached(mn: str, dn: str):
        service = BaselineService(paths.checkpoints, paths.reports)
        try:
            model = service.get_model(
                mn,
                dn,
                on_info=st.info,
                on_success=st.success,
            )
        except Exception as e:
            st.error(f"Failed to load/train model: {e}")
            raise
        return service, model

    service, clf = _load_model_cached(model_name, dataset_name)

    tab_text, tab_file, tab_report = st.tabs(["Single Text", "Batch (CSV)", "Model Report"])
    with tab_text:
        render_single_text(service, clf)
    with tab_file:
        render_batch_tab(service, clf)
    with tab_report:
        render_report_tab(model_name, paths.reports)
