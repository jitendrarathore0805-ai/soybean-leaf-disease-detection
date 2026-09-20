import json
import cv2
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0


IMG_SIZE = (224, 224)
BATCH_SIZE = 1


def load_models(artifacts_dir):
    artifacts_dir = Path(artifacts_dir)

    hybrid_model = tf.keras.models.load_model(
        artifacts_dir / "experiment_E_hybrid.keras",
        compile=False,
    )

    deep_model = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    deep_model.trainable = False

    mrmr_indices = np.load(
        artifacts_dir / "mrmr_indices.npy"
    )
    woa_indices = np.load(
        artifacts_dir / "woa_apso_indices.npy"
    )

    deep_scaler = _load_joblib(
        artifacts_dir / "deep_scaler.joblib"
    )
    hand_scaler = _load_joblib(
        artifacts_dir / "hand_scaler.joblib"
    )

    class_names = json.loads(
        (artifacts_dir / "class_names.json").read_text(
            encoding="utf-8"
        )
    )

    return {
        "hybrid_model": hybrid_model,
        "deep_model": deep_model,
        "mrmr_indices": mrmr_indices,
        "woa_indices": woa_indices,
        "deep_scaler": deep_scaler,
        "hand_scaler": hand_scaler,
        "class_names": class_names,
    }


def _load_joblib(path):
    import joblib
    return joblib.load(path)


def normalize_rows(x):
    return x / (
        np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    )


def predict_one(image_rgb, models):
    from .features import handcrafted_features_from_image

    # -----------------------------
    # 1. Handcrafted features
    # -----------------------------
    hand_all = handcrafted_features_from_image(image_rgb)

    hand_mrmr = hand_all[models["mrmr_indices"]]
    hand_selected = hand_mrmr[models["woa_indices"]]

    hand = models["hand_scaler"].transform(
        hand_selected.reshape(1, -1)
    )

    hand = normalize_rows(hand).astype(np.float32)

    # -----------------------------
    # 2. Deep features
    # -----------------------------
    # Training used 224x224 images
    deep_image = cv2.resize(
        image_rgb,
        IMG_SIZE,
        interpolation=cv2.INTER_AREA
    )

    deep_image = deep_image.astype(np.float32)
    deep_image = np.expand_dims(deep_image, axis=0)

    deep = models["deep_model"].predict(
        deep_image,
        verbose=0
    )

    deep = models["deep_scaler"].transform(deep)
    deep = normalize_rows(deep).astype(np.float32)

    # -----------------------------
    # 3. Hybrid model prediction
    # -----------------------------
    probabilities = models["hybrid_model"].predict(
        [deep, hand],
        verbose=0
    )[0]

    order = np.argsort(probabilities)[::-1]
    top = order[:3]

    return {
        "class_name": models["class_names"][int(order[0])],
        "confidence": float(probabilities[order[0]]),
        "top_predictions": [
            {
                "class_name": models["class_names"][int(i)],
                "confidence": float(probabilities[i]),
            }
            for i in top
        ],
    }