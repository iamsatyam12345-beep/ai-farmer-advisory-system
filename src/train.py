"""
Model training and evaluation pipeline for farmer query classification.
Uses TF-IDF Vectorization and Logistic Regression.
"""

import os
import argparse
import json
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import joblib

from src.preprocessing import clean_text


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Loads and validates the labeled farmer queries dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"query", "category"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {required_cols}. Found: {df.columns.tolist()}")

    df = df.dropna(subset=["query", "category"]).copy()
    df["cleaned_query"] = df["query"].apply(clean_text)
    return df


def build_pipeline() -> Pipeline:
    """Creates a scikit-learn Pipeline with TF-IDF Vectorizer and Logistic Regression."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                stop_words="english",
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=5.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def train_and_evaluate(
    data_path: str = "data/farmer_queries.csv",
    model_output_path: str = "models/farmer_intent_model.joblib",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains the model, evaluates it on a held-out test split,
    saves the trained model and metrics artifact, and returns the pipeline and metrics.
    """
    df = load_dataset(data_path)

    X = df["cleaned_query"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    metrics = {
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "num_classes": len(pipeline.classes_),
        "classes": [str(c) for c in pipeline.classes_],
        "accuracy": round(acc, 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "classification_report": report_dict,
    }

    # Save model artifact
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "classes": list(pipeline.classes_),
        "metrics": metrics,
    }
    joblib.dump(bundle, model_output_path)

    # Also save a JSON summary of metrics for transparency
    metrics_path = os.path.splitext(model_output_path)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("=================== MODEL TRAINING & EVALUATION ===================")
    print(f"Total queries: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Macro F1-Score:  {f1_macro:.4f}")
    print(f"Weighted F1:     {f1_weighted:.4f}")
    print("\nDetailed Classification Report:")
    print(report_text)
    print(f"Model successfully saved to: {model_output_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("====================================================================")

    return pipeline, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train farmer query classification model.")
    parser.add_argument("--data", default="data/farmer_queries.csv", help="Path to queries CSV")
    parser.add_argument("--output", default="models/farmer_intent_model.joblib", help="Output model path")
    args = parser.parse_args()

    train_and_evaluate(data_path=args.data, model_output_path=args.output)
