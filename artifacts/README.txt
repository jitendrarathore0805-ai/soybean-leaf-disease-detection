PUT THESE FILES HERE:

1. experiment_E_hybrid.keras
2. mrmr_indices.npy
3. woa_apso_indices.npy
4. deep_scaler.joblib
5. hand_scaler.joblib
6. class_names.json

Use export_colab.py in your original Colab notebook to create them.

IMPORTANT:
The original Colab code saves the model and feature indices, but it does
not save the two StandardScaler objects. The export block saves them
without retraining.

Expected class order comes from the original dataset folder ordering.
