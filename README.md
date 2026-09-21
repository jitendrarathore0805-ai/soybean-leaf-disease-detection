 App url - https://soybean-leaf-disease-detection-6fnb4vjx9wr72xedyylvhv.streamlit.app/
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

