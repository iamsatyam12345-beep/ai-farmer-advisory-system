"""
KisanSetu: AI-Based Farmer Query Support and Advisory System
Streamlit Web Application.
"""

import os
import json
import streamlit as st

from src.preprocessing import clean_text
from src.advisory_retrieval import FarmerAdvisoryEngine, SAFE_FALLBACK_GUIDANCE
from src.train import train_and_evaluate


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="KisanSetu - AI Farmer Advisory System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling (Agricultural Palette & Premium Modern UI)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #1b4d3e 0%, #0d2818 100%);
        color: #ffffff;
        padding: 2.2rem 2.4rem;
        border-radius: 18px;
        box-shadow: 0 10px 25px -5px rgba(27, 77, 62, 0.35);
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        color: #e8f5e9;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #c8e6c9;
        margin-top: 0.6rem;
        margin-bottom: 0;
        line-height: 1.5;
    }
    
    .badge-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-disease { background-color: #fee2e2; color: #991b1b; }
    .badge-pest { background-color: #fef3c7; color: #92400e; }
    .badge-soil { background-color: #ecfdf5; color: #065f46; }
    .badge-water { background-color: #e0f2fe; color: #075985; }
    .badge-harvest { background-color: #f3e8ff; color: #6b21a8; }
    
    .advisory-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin-top: 1.2rem;
    }
    
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .action-box {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
        color: #166534;
        line-height: 1.6;
    }
    
    .prevention-box {
        background: #f8fafc;
        border-left: 4px solid #64748b;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
        color: #334155;
        line-height: 1.6;
    }
    
    .safety-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
        color: #92400e;
        line-height: 1.6;
    }
    
    .fallback-box {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-left: 5px solid #ea580c;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1.2rem;
    }
    
    .stat-pill {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 10px 14px;
        text-align: center;
        margin-bottom: 10px;
    }
    .stat-val { font-size: 1.25rem; font-weight: 700; color: #0f172a; }
    .stat-lbl { font-size: 0.75rem; color: #64748b; text-transform: uppercase; }
    </style>
    """,
    unsafe_allow_html=True,
)


CATEGORY_NAMES = {
    "crop_disease": "Plant Disease Diagnosis",
    "pest_management": "Pest & Insect Control",
    "soil_and_fertilizer": "Soil Health & Nutrient Management",
    "irrigation_and_water": "Irrigation & Water Scheduling",
    "weather_and_harvest": "Weather, Harvest & Post-Harvest",
}

CATEGORY_BADGES = {
    "crop_disease": "badge-disease",
    "pest_management": "badge-pest",
    "soil_and_fertilizer": "badge-soil",
    "irrigation_and_water": "badge-water",
    "weather_and_harvest": "badge-harvest",
}


@st.cache_resource(show_spinner="Loading Farmer Advisory Engine...")
def get_advisory_engine():
    model_path = "models/farmer_intent_model.joblib"
    data_path = "data/farmer_queries.csv"
    kb_path = "data/advisory_knowledge_base.json"

    # Train model automatically if not already generated
    if not os.path.exists(model_path):
        train_and_evaluate(data_path=data_path, model_output_path=model_path)

    return FarmerAdvisoryEngine(
        model_path=model_path,
        kb_path=kb_path,
        default_confidence_threshold=0.35,
    )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌾 KisanSetu Control Panel")
    st.markdown("Automated Agricultural Decision Support & Extension")
    st.divider()

    engine = get_advisory_engine()

    # Model metrics display
    st.markdown("#### 📊 Model Performance")
    metrics = engine.metrics
    if metrics:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Test Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        with col_m2:
            st.metric("Macro F1", f"{metrics.get('macro_f1', 0):.3f}")

        st.caption(
            f"Dataset: {metrics.get('dataset_size', 0)} queries (Train: {metrics.get('train_size', 0)}, "
            f"Test: {metrics.get('test_size', 0)}) across {metrics.get('num_classes', 0)} categories."
        )
    else:
        st.info("Metrics not loaded.")

    st.divider()

    # Confidence Threshold Slider
    st.markdown("#### ⚙️ Relevance Threshold")
    user_threshold = st.slider(
        "Minimum Classification Confidence",
        min_value=0.20,
        max_value=0.80,
        value=0.35,
        step=0.05,
        help="Queries with top-class confidence below this threshold safely trigger the fallback advisory.",
    )

    st.divider()

    # Knowledge Base Directory
    st.markdown("#### 📚 Advisory Knowledge Base")
    with st.expander("Browse Advisory Topics"):
        kb_data = engine.knowledge_base
        selected_cat = st.selectbox(
            "Filter Category",
            options=list(CATEGORY_NAMES.keys()),
            format_func=lambda x: CATEGORY_NAMES.get(x, x),
        )
        filtered_items = [i for i in kb_data if i.get("category") == selected_cat]
        for itm in filtered_items:
            st.markdown(f"**{itm.get('topic')}**")
            st.caption(f"Symptoms: {itm.get('symptoms')[:120]}...")
            st.markdown("---")

    st.divider()
    st.markdown(
        "📞 **Kisan Call Center (Toll-Free)**:\n"
        "**1800-180-1551** (6:00 AM to 10:00 PM, All Days)"
    )
    st.caption("Disclaimer: Prototype AI system with illustrative synthetic data. Always verify critical decisions with certified agronomists.")


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1 class="hero-title">🌾 KisanSetu: Farmer Advisory Support</h1>
        <p class="hero-subtitle">
            Instant agricultural query classification and actionable advisory guidance for crop diseases, 
            pest outbreaks, nutrient dosing, irrigation scheduling, and post-harvest management.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sample Questions Carousel / Chips
# ---------------------------------------------------------
st.markdown("##### 💡 Try Sample Farmer Questions:")
sample_queries = [
    ("🍅 Tomato Blight", "My tomato leaves are turning yellow with dark brown spots and circular rings"),
    ("🐛 Rice Stem Borer", "Dead heart symptom in young paddy shoots with caterpillar inside the stem"),
    ("🧪 Fertilizer Dosing", "How much urea and DAP fertilizer should I apply per acre for paddy crop?"),
    ("💧 Irrigation Timing", "How many irrigations are required for wheat and what are critical watering stages?"),
    ("🌾 Grain Storage", "Safe grain moisture percentage for long term storage of wheat and paddy"),
]

sample_cols = st.columns(len(sample_queries))
selected_sample = None

for idx, (label, sample_text) in enumerate(sample_queries):
    with sample_cols[idx]:
        if st.button(label, use_container_width=True, key=f"sample_{idx}"):
            st.session_state["farmer_query_input"] = sample_text


# ---------------------------------------------------------
# Query Input Box
# ---------------------------------------------------------
query_input = st.text_area(
    "Enter Your Agricultural Question or Crop Concern:",
    value=st.session_state.get("farmer_query_input", ""),
    placeholder="e.g., My cotton leaves have white tiny flies underneath and sap is oozing...",
    height=100,
    key="farmer_query_input",
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    ask_button = st.button("🔍 Get Advisory", type="primary", use_container_width=True)
with col_btn2:
    if st.button("Clear", use_container_width=False):
        st.session_state["farmer_query_input"] = ""
        st.rerun()


# ---------------------------------------------------------
# Advisory Retrieval and Response Rendering
# ---------------------------------------------------------
if query_input:
    result = engine.retrieve_advisory(query_input, confidence_threshold=user_threshold)

    if result["status"] == "success":
        category = result["predicted_category"]
        confidence = result["confidence"]
        advisory = result["advisory"]
        probs = result["all_probabilities"]
        category_human = CATEGORY_NAMES.get(category, category)
        badge_class = CATEGORY_BADGES.get(category, "badge-soil")

        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 12px; margin-top: 1rem;">
                <span class="badge-tag {badge_class}">{category_human}</span>
                <span style="font-weight: 600; color: #15803d; font-size: 0.95rem;">
                    Confidence: {confidence:.1%}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Main Advisory Card
        st.markdown(
            f"""
            <div class="advisory-card">
                <h3 style="margin-top: 0; color: #0f172a; font-weight: 700;">
                    📌 {advisory.get('topic')}
                </h3>
                <p style="color: #475569; font-size: 0.95rem; margin-bottom: 1.2rem;">
                    <strong>Identified Symptoms / Context:</strong> {advisory.get('symptoms')}
                </p>
                <div class="section-header">⚡ Recommended Immediate Actions</div>
                <div class="action-box">{advisory.get('recommended_actions')}</div>
                <div class="section-header">🛡️ Long-Term Preventative Measures</div>
                <div class="prevention-box">{advisory.get('preventative_measures')}</div>
                <div class="section-header">⚠️ Chemical & Field Safety Precautions</div>
                <div class="safety-box">{advisory.get('safety_precaution')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Category Probabilities Visualizer
        with st.expander("📈 View Category Probability Distribution"):
            prob_cols = st.columns(len(probs))
            for i, (cat_key, prob_val) in enumerate(probs.items()):
                with prob_cols[i]:
                    st.markdown(
                        f"""
                        <div class="stat-pill">
                            <div class="stat-val">{prob_val:.1%}</div>
                            <div class="stat-lbl">{cat_key.replace('_', ' ')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    else:
        # Fallback Card
        st.markdown(
            f"""
            <div class="fallback-box">
                <h3 style="color: #c2410c; margin-top: 0; display: flex; align-items: center; gap: 8px;">
                    ⚠️ Need More Clarification (Low Confidence Fallback)
                </h3>
                <p style="color: #9a3412; font-weight: 600;">
                    {result.get('fallback_reason')}
                </p>
                <div style="color: #334155; line-height: 1.6; white-space: pre-line; background: #ffffff; padding: 1.2rem; border-radius: 8px; border: 1px solid #fed7aa;">
                    {result.get('fallback_guidance')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    st.info("👆 Type your farming query above or click any sample question to test the system.")
