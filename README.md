 App url - https://soybean-leaf-disease-detection-6fnb4vjx9wr72xedyylvhv.streamlit.app/
# Soybean Disease Detection — Streamlit

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

## Current model result from the supplied Colab run

Validation set:

- Accuracy: 97.92%
- Macro Precision: 97.01%
- Macro Recall: 97.19%
- Macro F1: 97.08%
- Balanced Accuracy: 97.19%



