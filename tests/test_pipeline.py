"""
Unit tests for AI-based farmer query support and advisory system.
Tests preprocessing, dataset & knowledge base integrity, model training,
advisory retrieval, relevance checks, and fallback mechanisms.
"""

import os
import json
import pytest
import pandas as pd

from src.preprocessing import clean_text, is_meaningful_query
from src.train import train_and_evaluate, load_dataset, build_pipeline
from src.advisory_retrieval import FarmerAdvisoryEngine, SAFE_FALLBACK_GUIDANCE


EXPECTED_CATEGORIES = {
    "crop_disease",
    "pest_management",
    "soil_and_fertilizer",
    "irrigation_and_water",
    "weather_and_harvest",
}


class TestPreprocessing:
    def test_clean_text_normalizes_case_and_punctuation(self):
        raw = "My TOMATO leaves... have brown-spots & yellow rings!"
        cleaned = clean_text(raw)
        assert cleaned == "my tomato leaves have brown spots yellow rings"

    def test_clean_text_handles_non_string(self):
        assert clean_text(None) == ""
        assert clean_text(123) == ""

    def test_clean_text_collapses_whitespace(self):
        assert clean_text("  paddy   stem   borer  ") == "paddy stem borer"

    def test_is_meaningful_query(self):
        assert is_meaningful_query("brown spots on tomato") is True
        assert is_meaningful_query("pest attack") is True
        # Negative cases
        assert is_meaningful_query("") is False
        assert is_meaningful_query("   ") is False
        assert is_meaningful_query("a") is False
        assert is_meaningful_query("123456") is False
        assert is_meaningful_query("??? !!!") is False


class TestDatasetAndKnowledgeBase:
    def test_dataset_integrity(self):
        csv_path = "data/farmer_queries.csv"
        assert os.path.exists(csv_path), "Dataset CSV file must exist"
        df = pd.read_csv(csv_path)

        assert "query" in df.columns
        assert "category" in df.columns
        assert len(df) >= 50, f"Expected at least 50 samples, found {len(df)}"

        categories_in_data = set(df["category"].unique())
        assert categories_in_data == EXPECTED_CATEGORIES

        # Ensure no empty queries or categories
        assert df["query"].dropna().shape[0] == len(df)
        assert df["category"].dropna().shape[0] == len(df)

    def test_knowledge_base_integrity(self):
        kb_path = "data/advisory_knowledge_base.json"
        assert os.path.exists(kb_path), "Knowledge base JSON file must exist"
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)

        assert isinstance(kb, list)
        assert len(kb) >= 5, "Knowledge base must have multiple advisory entries"

        required_keys = {
            "id",
            "category",
            "topic",
            "keywords",
            "symptoms",
            "recommended_actions",
            "preventative_measures",
            "safety_precaution",
        }

        kb_categories = set()
        for item in kb:
            assert required_keys.issubset(item.keys()), f"Item {item.get('id')} missing keys"
            assert item["category"] in EXPECTED_CATEGORIES
            assert isinstance(item["keywords"], list)
            assert len(item["keywords"]) > 0
            assert len(item["recommended_actions"]) > 10
            kb_categories.add(item["category"])

        # Every expected category should be covered in the KB
        assert kb_categories == EXPECTED_CATEGORIES


class TestModelTraining:
    def test_training_and_metrics_generation(self, tmp_path):
        model_path = str(tmp_path / "test_model.joblib")
        pipeline, metrics = train_and_evaluate(
            data_path="data/farmer_queries.csv",
            model_output_path=model_path,
            test_size=0.25,
            random_state=42,
        )

        assert os.path.exists(model_path)
        assert hasattr(pipeline, "predict_proba")
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert metrics["num_classes"] == 5
        assert set(metrics["classes"]) == EXPECTED_CATEGORIES


class TestAdvisoryEngine:
    @pytest.fixture(scope="class")
    def engine(self, tmp_path_factory):
        # Train temporary model for testing engine
        tmp_dir = tmp_path_factory.mktemp("models")
        model_path = str(tmp_dir / "farmer_model.joblib")
        train_and_evaluate(
            data_path="data/farmer_queries.csv",
            model_output_path=model_path,
            random_state=42,
        )
        return FarmerAdvisoryEngine(
            model_path=model_path,
            kb_path="data/advisory_knowledge_base.json",
            default_confidence_threshold=0.30,
        )

    def test_engine_initialization(self, engine):
        assert engine.is_ready() is True
        assert len(engine.classes) == 5

    def test_successful_disease_retrieval(self, engine):
        query = "Tomato leaves have dark circular rings and yellow spots"
        res = engine.retrieve_advisory(query)

        assert res["status"] == "success"
        assert res["predicted_category"] == "crop_disease"
        assert res["confidence"] >= 0.30
        assert res["advisory"] is not None
        assert "blight" in res["advisory"]["topic"].lower() or "leaf spot" in res["advisory"]["topic"].lower()
        assert res["fallback_guidance"] is None

    def test_successful_pest_retrieval(self, engine):
        query = "Caterpillars boring holes into paddy stem causing dead hearts"
        res = engine.retrieve_advisory(query)

        assert res["status"] == "success"
        assert res["predicted_category"] == "pest_management"
        assert res["confidence"] >= 0.30
        assert res["advisory"] is not None
        assert "borer" in res["advisory"]["topic"].lower()

    def test_successful_fertilizer_retrieval(self, engine):
        query = "Older leaves turning pale yellow what dose of urea and DAP should I apply"
        res = engine.retrieve_advisory(query)

        assert res["status"] == "success"
        assert res["predicted_category"] == "soil_and_fertilizer"
        assert res["advisory"] is not None

    def test_fallback_on_empty_or_gibberish_query(self, engine):
        for bad_query in ["", "  ", "12345", "a", "??"]:
            res = engine.retrieve_advisory(bad_query)
            assert res["status"] == "fallback"
            assert res["advisory"] is None
            assert res["fallback_guidance"] == SAFE_FALLBACK_GUIDANCE
            assert "too brief" in res["fallback_reason"].lower() or "lacks" in res["fallback_reason"].lower()

    def test_fallback_on_unrelated_out_of_domain_query(self, engine):
        # Query unrelated to agriculture, should have low confidence or trigger fallback
        res = engine.retrieve_advisory(
            "What is quantum computing programming and cryptocurrency wallet trading?",
            confidence_threshold=0.50,
        )
        # Even if a category has the highest relative prob, threshold blocks it
        assert res["status"] == "fallback"
        assert res["fallback_guidance"] is not None
        assert res["advisory"] is None
