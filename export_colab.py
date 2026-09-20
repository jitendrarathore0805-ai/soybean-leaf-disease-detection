# RUN THIS AT THE VERY END OF YOUR EXISTING COLAB NOTEBOOK
# after the current Experiment E training/evaluation code.
#
# This does NOT retrain the model. It only exports the artifacts
# required by the Streamlit inference application.

import os
import json
import joblib
import shutil

DEPLOY_DIR = "/content/soybean_streamlit_artifacts"
os.makedirs(DEPLOY_DIR, exist_ok=True)

# 1. Already saved by your notebook
shutil.copy(
    "/content/experiment_E_outputs/experiment_E_hybrid.keras",
    os.path.join(DEPLOY_DIR, "experiment_E_hybrid.keras")
)

shutil.copy(
    "/content/experiment_E_outputs/mrmr_indices.npy",
    os.path.join(DEPLOY_DIR, "mrmr_indices.npy")
)

shutil.copy(
    "/content/experiment_E_outputs/woa_apso_indices.npy",
    os.path.join(DEPLOY_DIR, "woa_apso_indices.npy")
)

# 2. These two scalers are currently created in the notebook
# but are NOT saved by the original code.
joblib.dump(
    deep_scaler,
    os.path.join(DEPLOY_DIR, "deep_scaler.joblib")
)

joblib.dump(
    hand_scaler,
    os.path.join(DEPLOY_DIR, "hand_scaler.joblib")
)

# 3. Class order is needed for prediction labels.
with open(
    os.path.join(DEPLOY_DIR, "class_names.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(class_names, f, indent=2)

print("Deployment artifacts created:")
for name in sorted(os.listdir(DEPLOY_DIR)):
    path = os.path.join(DEPLOY_DIR, name)
    print(f"{name:35s} {os.path.getsize(path) / (1024**2):.2f} MB")

# Optional: create a zip for download from Colab
shutil.make_archive(
    "/content/soybean_streamlit_artifacts",
    "zip",
    DEPLOY_DIR
)

print("\nZIP:")
print("/content/soybean_streamlit_artifacts.zip")
