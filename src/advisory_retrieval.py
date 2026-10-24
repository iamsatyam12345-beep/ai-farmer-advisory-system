"""
Category-based agricultural advisory retrieval engine with relevance checks
and safe fallback mechanisms.
"""

import os
import json
from typing import Dict, Any, List, Optional
import joblib

from src.preprocessing import clean_text, is_meaningful_query


SAFE_FALLBACK_GUIDANCE = (
    "We could not confidently determine the specific agricultural category for your question. "
    "To help us provide accurate recommendations, please rephrase your question with key details such as:\n"
    "• Crop name (e.g., wheat, rice, tomato, cotton)\n"
    "• Observed symptoms (e.g., yellow spots, wilting, leaf holes, dry rot)\n"
    "• Specific management area (disease, insect pest, fertilizer dose, irrigation timing, or harvest/storage)\n\n"
    "For urgent or severe crop distress, please consult your nearest Krishi Vigyan Kendra (KVK) "
    "or call the National Kisan Call Center toll-free helpline at 1800-180-1551."
)


class FarmerAdvisoryEngine:
    """
    Advisory retrieval engine combining TF-IDF + Logistic Regression category classification
    with structured knowledge base lookup, confidence thresholds, and safe fallback.
    """

    def __init__(
        self,
        model_path: str = "models/farmer_intent_model.joblib",
        kb_path: str = "data/advisory_knowledge_base.json",
        default_confidence_threshold: float = 0.35,
    ):
        self.model_path = model_path
        self.kb_path = kb_path
        self.default_confidence_threshold = default_confidence_threshold
        self.pipeline = None
        self.classes: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.knowledge_base: List[Dict[str, Any]] = []

        self._load_resources()

    def _load_resources(self):
        """Loads the trained model pipeline and knowledge base records."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. Please run the training script first."
            )
        bundle = joblib.load(self.model_path)
        self.pipeline = bundle["pipeline"]
        self.classes = bundle.get("classes", list(self.pipeline.classes_))
        self.metrics = bundle.get("metrics", {})

        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base file not found at '{self.kb_path}'.")
        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.knowledge_base = json.load(f)

    def is_ready(self) -> bool:
        """Returns True if model and knowledge base are loaded."""
        return self.pipeline is not None and len(self.knowledge_base) > 0

    def predict_category(self, query: str) -> Dict[str, Any]:
        """
        Classifies query into an agricultural category and returns class probabilities.
        """
        cleaned = clean_text(query)
        if not cleaned:
            return {
                "category": None,
                "confidence": 0.0,
                "probabilities": {cls: 0.0 for cls in self.classes},
            }

        probs = self.pipeline.predict_proba([cleaned])[0]
        prob_dict = {
            cls: round(float(prob), 4)
            for cls, prob in zip(self.pipeline.classes_, probs)
        }
        best_category = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[best_category]

        return {
            "category": best_category,
            "confidence": confidence,
            "probabilities": prob_dict,
        }

    def _score_advisory_item(self, query_tokens: set, item: Dict[str, Any]) -> float:
        """
        Calculates lexical relevance score for an advisory item given user query tokens.
        """
        score = 0.0
        keywords = set(k.lower() for k in item.get("keywords", []))
        topic_tokens = set(clean_text(item.get("topic", "")).split())
        symptom_tokens = set(clean_text(item.get("symptoms", "")).split())

        # Exact keyword matches have highest weight
        keyword_overlap = query_tokens.intersection(keywords)
        score += len(keyword_overlap) * 3.0

        # Topic overlap
        topic_overlap = query_tokens.intersection(topic_tokens)
        score += len(topic_overlap) * 2.0

        # Symptom overlap
        symptom_overlap = query_tokens.intersection(symptom_tokens)
        score += len(symptom_overlap) * 1.0

        return score

    def retrieve_advisory(
        self,
        query: str,
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates the query with relevance validation, classifies intent,
        and retrieves relevant advice or safely triggers fallback.
        """
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold

        raw_query = query if isinstance(query, str) else ""
        cleaned = clean_text(raw_query)

        # 1. Relevance check: input validity
        if not is_meaningful_query(raw_query):
            return {
                "status": "fallback",
                "query": raw_query,
                "cleaned_query": cleaned,
                "predicted_category": None,
                "confidence": 0.0,
                "all_probabilities": {},
                "advisory": None,
                "fallback_reason": "Query is too brief, empty, or lacks meaningful agricultural context.",
                "fallback_guidance": SAFE_FALLBACK_GUIDANCE,
            }

        # 2. Category classification & confidence
        pred = self.predict_category(cleaned)
        category = pred["category"]
        confidence = pred["confidence"]
        probabilities = pred["probabilities"]

        # 3. Relevance check: confidence threshold
        if confidence < confidence_threshold:
            return {
                "status": "fallback",
                "query": raw_query,
                "cleaned_query": cleaned,
                "predicted_category": category,
                "confidence": confidence,
                "all_probabilities": probabilities,
                "advisory": None,
                "fallback_reason": (
                    f"Model confidence ({confidence:.1%}) is below the minimum relevance threshold "
                    f"({confidence_threshold:.1%}) for agricultural decision support."
                ),
                "fallback_guidance": SAFE_FALLBACK_GUIDANCE,
            }

        # 4. Category-based retrieval from knowledge base
        category_items = [
            item for item in self.knowledge_base
            if item.get("category") == category
        ]

        if not category_items:
            # Fallback if category has no KB entries
            return {
                "status": "fallback",
                "query": raw_query,
                "cleaned_query": cleaned,
                "predicted_category": category,
                "confidence": confidence,
                "all_probabilities": probabilities,
                "advisory": None,
                "fallback_reason": f"No active knowledge base entries available for category '{category}'.",
                "fallback_guidance": SAFE_FALLBACK_GUIDANCE,
            }

        # Select best matching advisory within the predicted category
        query_tokens = set(cleaned.split())
        scored_items = [
            (self._score_advisory_item(query_tokens, item), item)
            for item in category_items
        ]
        # Sort descending by score
        scored_items.sort(key=lambda x: x[0], reverse=True)
        best_match_score, best_advisory = scored_items[0]

        return {
            "status": "success",
            "query": raw_query,
            "cleaned_query": cleaned,
            "predicted_category": category,
            "confidence": confidence,
            "match_relevance_score": round(best_match_score, 2),
            "all_probabilities": probabilities,
            "advisory": best_advisory,
            "fallback_reason": None,
            "fallback_guidance": None,
        }
