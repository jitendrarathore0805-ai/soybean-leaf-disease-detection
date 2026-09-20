# app url - https://soybean-leaf-disease-detection-6fnb4vjx9wr72xedyylvhv.streamlit.app/
# Soybean Disease Detection — Streamlit

This project converts the Colab **Experiment E** pipeline into a Streamlit
inference application.

## Model pipeline

```text
Uploaded Leaf Image
        |
        v
Leaf Segmentation
        |
        +-----------------------------+
        |                             |
        v                             v
Handcrafted Features           EfficientNetV2-B0
        |                             |
        v                             v
mRMR (48)                    Deep Features (1280)
        |
        v
WOA-APSO (24)
        |                             |
        +-------------+---------------+
                      |
                      v
          StandardScaler + L2
                      |
                      v
             Attention Fusion
                      |
                      v
             10-class Softmax
                      |
                      v
             Disease Prediction
```

## Dataset

The Colab run used 557 training images, 144 validation images and 10
classes. See the supplied Colab output for the exact class distribution.

## Step 1 — Export artifacts from Colab

At the very end of the original Colab notebook, run:

```python
exec(open("/content/export_colab.py").read())
```

Or paste the code from `export_colab.py` into a final Colab cell.

This creates:

```text
experiment_E_hybrid.keras
mrmr_indices.npy
woa_apso_indices.npy
deep_scaler.joblib
hand_scaler.joblib
class_names.json
```

Download these files and put them into this project's `artifacts/` folder.

## Step 2 — Local test

Windows PowerShell:

```powershell
cd soybean_streamlit_app

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

streamlit run app.py
```

The browser will open the Streamlit application.

## Step 3 — GitHub + Streamlit Community Cloud

Push this folder to GitHub.

Do NOT commit the large `.keras` model if GitHub rejects the file-size limit.
Use Git LFS or another model-storage method for large artifacts.

Then create a Streamlit Community Cloud app and set:

```text
Main file: app.py
```

The app needs access to all six artifact files.

## Important deployment note

EfficientNetV2-B0 with `weights="imagenet"` downloads the ImageNet backbone
weights the first time the app loads the model. Therefore the deployment
environment needs internet access during first model initialization.

## Current model result from the supplied Colab run

Validation set:

- Accuracy: 97.92%
- Macro Precision: 97.01%
- Macro Recall: 97.19%
- Macro F1: 97.08%
- Balanced Accuracy: 97.19%

These are the results of the supplied validation run, not a guarantee for
new uploaded images.

## Troubleshooting

### Missing artifacts

The app will list the exact missing files.

### TensorFlow/Keras model loading error

Use a TensorFlow version compatible with the environment that saved the
`.keras` model. If necessary, pin the exact TensorFlow version shown in
your Colab environment.

### Slow first prediction

The first run loads:
- the hybrid Keras model
- EfficientNetV2-B0 ImageNet weights

Later predictions are cached in the Streamlit process.

### Wrong predictions

Make sure the scaler files and feature-index files came from the same
training run as `experiment_E_hybrid.keras`.
