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
    page_title="Soybean Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# TITLE
# =========================================================

st.title("🌱 Soybean Leaf Disease Detection")

st.subheader(
    "Hybrid Deep Learning + Handcrafted Feature Analysis"
)

st.write(
    "Upload a soybean leaf image and let the trained "
    "hybrid model identify the disease."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Architecture**")
    st.write("Hybrid Deep Learning Model")

    st.write("**Deep Model**")
    st.write("EfficientNetV2-B0")

    st.write("**Feature Selection**")
    st.write("mRMR + WOA-APSO")

    st.write("**Deep Features**")
    st.write("1280")

    st.write("**Selected Handcrafted Features**")
    st.write("24")

    st.write("**Image Size**")
    st.write("224 × 224")

    st.divider()

    st.info(
        "Prediction is generated using the trained "
        "Experiment E hybrid model."
    )


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def get_models():

    artifacts_dir = (
        Path(__file__).resolve().parent / "artifacts"
    )

    return load_models(artifacts_dir)


try:

    models = get_models()

except Exception as e:

    st.error("❌ Model loading failed.")

    st.code(str(e))

    st.stop()


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.header("📤 Upload Soybean Leaf")

uploaded_file = st.file_uploader(
    "Choose a soybean leaf image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ],
    help="Upload a clear soybean leaf image."
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # =====================================================
    # IMAGE + RESULT
    # =====================================================

    col1, col2 = st.columns(
        [1, 1],
        gap="large"
    )


    # =====================================================
    # IMAGE
    # =====================================================

    with col1:

        st.subheader("🖼️ Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )


    # =====================================================
    # PREDICTION
    # =====================================================

    with col2:

        st.subheader("🔍 Prediction Result")

        image_rgb = np.array(image)

        with st.spinner(
            "Analyzing soybean leaf..."
        ):

            result = predict_one(
                image_rgb,
                models
            )


        disease = result["class_name"]

        confidence = result["confidence"]


        # -------------------------------------------------
        # DISEASE
        # -------------------------------------------------

        st.write("### 🦠 Predicted Disease")

        st.success(disease)


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        st.write("### 🎯 Confidence")

        st.metric(
            "Model Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.progress(
            float(confidence)
        )


    # =====================================================
    # TOP 3 PREDICTIONS
    # =====================================================

    st.divider()

    st.header("📊 Top 3 Predictions")

    top_predictions = result[
        "top_predictions"
    ]


    rows = []

    for rank, prediction in enumerate(
        top_predictions,
        start=1
    ):

        rows.append(
            {
                "Rank": rank,
                "Disease": prediction[
                    "class_name"
                ],
                "Confidence": (
                    f"{prediction['confidence'] * 100:.2f}%"
                )
            }
        )


    df = pd.DataFrame(rows)


    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )


    # =====================================================
    # CONFIDENCE CHART
    # =====================================================

    st.subheader(
        "📈 Confidence Comparison"
    )

    chart_df = pd.DataFrame(
        {
            "Disease": [
                x["class_name"]
                for x in top_predictions
            ],

            "Confidence": [
                x["confidence"] * 100
                for x in top_predictions
            ]
        }
    )


    st.bar_chart(
        chart_df.set_index(
            "Disease"
        )
    )


    # =====================================================
    # MODEL PIPELINE
    # =====================================================

    st.divider()

    st.header("🧠 Model Pipeline")

    st.write(
        """
        **1. Image Upload**
        
        ↓
        
        **2. Leaf Segmentation**
        
        ↓
        
        **3. Handcrafted Feature Extraction**
        
        ↓
        
        **4. mRMR Feature Selection**
        
        ↓
        
        **5. WOA-APSO Feature Selection**
        
        ↓
        
        **6. EfficientNetV2-B0 Deep Features**
        
        ↓
        
        **7. Attention-Based Feature Fusion**
        
        ↓
        
        **8. Disease Classification**
        """
    )


    # =====================================================
    # MODEL DETAILS
    # =====================================================

    st.divider()

    st.header("📋 Model Details")


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Image Size",
            "224 × 224"
        )


    with c2:

        st.metric(
            "Deep Features",
            "1280"
        )


    with c3:

        st.metric(
            "Selected Features",
            "24"
        )


    with c4:

        st.metric(
            "Disease Classes",
            "10"
        )


else:

    st.info(
        " Upload a soybean leaf image above "
        "to start disease detection."
    )

    st.write(
        "**Supported formats:** "
        "JPG, JPEG, PNG, BMP, WEBP"
    )




st.divider()

st.caption(
    "🌱 Soybean Leaf Disease Detection System | "
    "Hybrid Deep Learning Research Project"
)
