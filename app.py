import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

from src.model import load_models, predict_one


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Intelligent Crop Disease Detection & Treatment Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM THEME
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --green: #176B45;
        --green-dark: #104B32;
        --green-soft: #EAF5EE;
        --ink: #17251D;
        --muted: #68776E;
        --surface: #FFFFFF;
        --line: #E4ECE6;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #F5F8F5;
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 248, 245, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #102F23;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #F3FAF5 !important;
    }

    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--ink);
        letter-spacing: -0.035em;
    }

    .hero {
        background: linear-gradient(120deg, #103D2B 0%, #176B45 58%, #2C8A5B 100%);
        padding: 34px 38px;
        border-radius: 22px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 12px 32px rgba(23,107,69,0.14);
    }

    .hero h1 {
        color: white !important;
        font-size: clamp(1.65rem, 3vw, 2.55rem);
        line-height: 1.2;
        margin: 8px 0 12px 0;
        max-width: 950px;
    }

    .hero p {
        color: #E1F2E7;
        font-size: 1rem;
        max-width: 820px;
        margin-bottom: 0;
    }

    .eyebrow {
        color: #BCE5CB;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .hero-tag {
        display: inline-block;
        margin-top: 20px;
        padding: 7px 12px;
        border: 1px solid rgba(255,255,255,0.25);
        background: rgba(255,255,255,0.10);
        border-radius: 30px;
        font-size: 0.78rem;
        color: #F3FFF6;
    }

    .section-kicker {
        color: var(--green);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .section-subtitle {
        color: var(--muted);
        margin-top: -8px;
        margin-bottom: 18px;
        font-size: 0.94rem;
    }

    .panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 5px 20px rgba(23, 55, 36, 0.035);
        height: 100%;
    }

    .panel-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .result-disease {
        color: var(--green-dark);
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.25;
        margin: 10px 0;
        overflow-wrap: anywhere;
    }

    .confidence-number {
        font-family: 'Manrope', sans-serif;
        color: var(--green);
        font-size: 2.35rem;
        font-weight: 800;
        margin: 5px 0;
    }

    .small-muted {
        color: var(--muted);
        font-size: 0.88rem;
    }

    .upload-note {
        background: #EDF6F0;
        border: 1px dashed #8BB99B;
        padding: 13px 16px;
        border-radius: 12px;
        color: #315E42;
        font-size: 0.88rem;
        margin: 12px 0 18px 0;
    }

    .step {
        background: #F7FAF7;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 9px;
        color: #294536;
        font-weight: 600;
    }

    .step-number {
        display: inline-flex;
        width: 25px;
        height: 25px;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: #DDEFE3;
        color: #176B45;
        font-size: 0.78rem;
        font-weight: 800;
        margin-right: 9px;
    }

    div[data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #A8C8B2;
        border-radius: 16px;
        padding: 12px;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 15px 17px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        color: var(--green-dark);
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }

    .stButton > button {
        background: var(--green);
        color: white;
        border: 0;
        border-radius: 10px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: var(--green-dark);
        color: white;
        border: 0;
    }

    [data-testid="stProgressBar"] > div > div {
        background-color: var(--green);
    }

    .footer {
        text-align: center;
        color: #7A887F;
        font-size: 0.82rem;
        padding: 16px 0 4px 0;
    }

    @media (max-width: 768px) {
        .block-container { padding: 1rem 1rem 2rem 1rem; }
        .hero { padding: 25px 22px; border-radius: 17px; }
        .panel { padding: 16px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER / HERO
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">🌱 Smart Agriculture • AI-Powered Assistance</div>
        <h1>Design and Development of Intelligent Crop Disease Detection and Treatment Assistant</h1>
        <p>
            An intelligent crop-health support system that analyzes soybean leaf
            images using hybrid deep learning and handcrafted feature analysis.
            Upload a leaf image to view the model's prediction and confidence.
        </p>
        <div class="hero-tag">HYBRID AI MODEL&nbsp;&nbsp; • &nbsp;&nbsp;SOYBEAN LEAF ANALYSIS</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🌿 CropCare AI")
    st.caption("Intelligent Crop Health Assistant")
    st.divider()

    st.markdown("### ⚙️ Model architecture")
    st.markdown("**Deep learning backbone**")
    st.write("EfficientNetV2-B0")

    st.markdown("**Feature selection**")
    st.write("mRMR + WOA-APSO")

    st.markdown("**Feature fusion**")
    st.write("Attention-based fusion")

    st.markdown("**Input resolution**")
    st.write("224 × 224 pixels")

    st.divider()
    st.markdown("### 📌 Project information")
    st.write("**Task:** Soybean leaf disease classification")
    st.write("**Classes:** 10")
    st.write("**Selected handcrafted features:** 24")
    st.write("**Deep feature dimension:** 1280")

    st.divider()
    st.info(
        "Predictions are generated by the trained Experiment E hybrid model. "
        "Always verify important crop-management decisions with a qualified "
        "agricultural expert."
    )


# =========================================================
# LOAD TRAINED MODELS
# =========================================================
@st.cache_resource
def get_models():
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    return load_models(artifacts_dir)


try:
    models = get_models()
except Exception as e:
    st.error("Model loading failed. Please verify the artifacts folder and model files.")
    with st.expander("Technical error details"):
        st.code(str(e))
    st.stop()


# =========================================================
# UPLOAD SECTION
# =========================================================
st.markdown('<div class="section-kicker">01 / Image input</div>', unsafe_allow_html=True)
st.header("Analyze a soybean leaf")
st.markdown(
    '<div class="section-subtitle">Upload a clear image of a soybean leaf. '
    'The model will analyze the image and return its most likely disease classes.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="upload-note">📷 <b>For better results:</b> Use a clear, well-lit '
    'image where the leaf is visible and occupies a good portion of the frame.</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Drag and drop a leaf image here, or browse files",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, JPEG, PNG, BMP and WEBP.",
)


# =========================================================
# PREDICTION
# =========================================================
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("The uploaded file could not be read as an image. Please try another file.")
        st.stop()

    image_rgb = np.array(image)

    st.divider()
    st.markdown('<div class="section-kicker">02 / AI diagnosis</div>', unsafe_allow_html=True)
    st.header("Analysis results")

    with st.spinner("🌿 The hybrid AI model is analyzing the leaf..."):
        try:
            result = predict_one(image_rgb, models)
        except Exception as e:
            st.error("Prediction failed. Please check the preprocessing and model artifacts.")
            with st.expander("Technical error details"):
                st.code(str(e))
            st.stop()

    disease = str(result["class_name"])
    confidence = float(result["confidence"])
    confidence = max(0.0, min(1.0, confidence))
    top_predictions = result.get("top_predictions", [])

    image_col, result_col = st.columns([1.02, 1], gap="large")

    with image_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**🖼️ Uploaded leaf image**")
        st.image(image, use_container_width=True)
        st.markdown(
            f'<p class="small-muted">Image dimensions: {image.width} × {image.height} px</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">Most likely prediction</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-disease">{disease.replace("_", " ")}</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">Model confidence</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-number">{confidence * 100:.2f}%</div>', unsafe_allow_html=True)
        st.progress(confidence)
        st.markdown(
            '<p class="small-muted">Confidence represents the model output for this '
            'prediction; it is not a guarantee of diagnosis.</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Summary metrics
    st.markdown("#### Prediction summary")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Predicted class", disease.replace("_", " "))
    with m2:
        st.metric("Confidence", f"{confidence * 100:.2f}%")
    with m3:
        st.metric("Classes evaluated", "10")

    # Top predictions
    st.divider()
    st.markdown('<div class="section-kicker">03 / Prediction comparison</div>', unsafe_allow_html=True)
    st.header("Top 3 predicted diseases")
    st.markdown(
        '<div class="section-subtitle">The three highest-scoring classes returned by the model.</div>',
        unsafe_allow_html=True,
    )

    if top_predictions:
        rows = []
        for rank, prediction in enumerate(top_predictions[:3], start=1):
            pred_name = str(prediction["class_name"])
            pred_conf = float(prediction["confidence"])
            rows.append(
                {
                    "Rank": rank,
                    "Disease": pred_name.replace("_", " "),
                    "Confidence (%)": round(pred_conf * 100, 2),
                }
            )

        df = pd.DataFrame(rows)
        chart_df = df.set_index("Disease")[["Confidence (%)"]]

        table_col, chart_col = st.columns([1, 1.15], gap="large")
        with table_col:
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "Confidence (%)": st.column_config.ProgressColumn(
                        "Confidence",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%",
                    ),
                },
            )
        with chart_col:
            st.bar_chart(chart_df, use_container_width=True)
    else:
        st.info("The model did not return a top-predictions list.")

    # Treatment assistant: safe, non-prescriptive guidance
    st.divider()
    st.markdown('<div class="section-kicker">04 / Crop-care support</div>', unsafe_allow_html=True)
    st.header("Treatment & next-step guidance")
    st.markdown(
        "The system's prediction can help identify what to investigate next. "
        "An image-based prediction alone should not be used to select or apply a pesticide."
    )

    with st.container(border=True):
        st.markdown(f"**Current model prediction:** `{disease.replace('_', ' ')}`")
        st.markdown(
            """
            **Recommended next steps**
            1. Inspect the whole plant and nearby plants for similar symptoms.
            2. Capture additional images in natural light, including the upper and lower leaf surfaces.
            3. Note the affected area, crop growth stage, recent weather and field conditions.
            4. Seek confirmation from a local agricultural extension officer or qualified plant pathologist.
            5. Follow only locally approved crop-protection guidance and product labels.
            """
        )
        st.caption(
            "Treatment recommendations are not generated by the classifier in this version. "
            "Do not apply chemicals based only on this prediction."
        )

    # Pipeline
    st.divider()
    st.markdown('<div class="section-kicker">05 / Under the hood</div>', unsafe_allow_html=True)
    st.header("Hybrid model workflow")
    st.markdown(
        '<div class="section-subtitle">The main processing stages used by the research model.</div>',
        unsafe_allow_html=True,
    )

    steps = [
        "Image upload and input preparation",
        "Leaf segmentation",
        "Handcrafted feature extraction",
        "mRMR feature selection",
        "WOA-APSO feature optimization",
        "EfficientNetV2-B0 deep feature extraction",
        "Attention-based feature fusion",
        "Disease classification and prediction",
    ]

    for idx, step in enumerate(steps, start=1):
        st.markdown(
            f'<div class="step"><span class="step-number">{idx}</span>{step}</div>',
            unsafe_allow_html=True,
        )

    # Technical model details
    with st.expander("📋 View technical model details"):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Input size", "224 × 224")
        d2.metric("Deep features", "1280")
        d3.metric("Selected features", "24")
        d4.metric("Disease classes", "10")
        st.caption(
            "Architecture details shown here describe the configured research pipeline. "
            "The actual inference behavior is determined by src/model.py and the loaded artifacts."
        )

    # Download prediction report
    if top_predictions:
        report_df = pd.DataFrame(
            [
                {
                    "Rank": i,
                    "Disease": str(p["class_name"]),
                    "Confidence": float(p["confidence"]),
                    "Confidence_Percent": float(p["confidence"]) * 100,
                }
                for i, p in enumerate(top_predictions[:3], start=1)
            ]
        )
        st.download_button(
            "⬇️ Download prediction report (CSV)",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="soybean_disease_prediction.csv",
            mime="text/csv",
        )

else:
    # Empty state
    st.divider()
    st.markdown('<div class="section-kicker">Ready when you are</div>', unsafe_allow_html=True)
    st.header("Start with a leaf image")
    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="panel">
                <h3>🌱 Your crop-health assistant</h3>
                <p class="small-muted">
                    Upload a soybean leaf image to receive a model-generated
                    disease prediction, confidence score and comparison of the
                    top three predicted classes.
                </p>
                <br>
                <b>Supported image formats</b>
                <p class="small-muted">JPG · JPEG · PNG · BMP · WEBP</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="panel">
                <h3>🧠 Hybrid AI pipeline</h3>
                <p class="small-muted">Combines two feature-learning approaches:</p>
                <ul>
                    <li>Handcrafted feature analysis</li>
                    <li>mRMR and WOA-APSO selection</li>
                    <li>EfficientNetV2-B0 deep features</li>
                    <li>Attention-based feature fusion</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        🌿 <b>Intelligent Crop Disease Detection and Treatment Assistant</b><br>
        Hybrid Deep Learning Research Project • Soybean Leaf Disease Analysis
    </div>
    """,
    unsafe_allow_html=True,
)
