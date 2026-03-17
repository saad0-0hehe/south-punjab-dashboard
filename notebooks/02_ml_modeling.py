"""
02_ml_modeling.py — Machine Learning Analysis
South Punjab Development Dashboard

This script:
1. Prepares features from the cleaned data
2. Trains Linear and Ridge regression models
3. Evaluates model performance
4. Generates prediction and feature importance plots
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data, clean_data
from src.ml_model import (
    prepare_features,
    train_linear,
    train_ridge,
    find_best_alpha,
    evaluate_model,
    plot_predictions,
    plot_feature_importance,
    plot_residuals,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Step 1: Load & Prepare Data ─────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading and Preparing Data")
print("=" * 60)

df = load_data()
df = clean_data(df)

data = prepare_features(df)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
feature_names = data["feature_names"]

print(f"Features used: {feature_names}")

# ─── Step 2: Train Models ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Training Models")
print("=" * 60)

# Linear Regression
lr_model = train_linear(X_train, y_train)

# Ridge — find best alpha first
best_alpha, alpha_results = find_best_alpha(X_train, y_train)
print("\nAlpha tuning results:")
print(alpha_results.to_string(index=False))

# Train Ridge with best alpha
ridge_model = train_ridge(X_train, y_train, alpha=best_alpha)

# ─── Step 3: Evaluate Models ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Evaluating Models")
print("=" * 60)

lr_metrics = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
ridge_metrics = evaluate_model(ridge_model, X_test, y_test, f"Ridge (α={best_alpha})")

print("\n📋 Model Comparison:")
print(f"{'Metric':<12} {'Linear':>10} {'Ridge':>10}")
print("-" * 34)
for metric in ["r2", "mae", "rmse"]:
    print(f"{metric.upper():<12} {lr_metrics[metric]:>10.3f} {ridge_metrics[metric]:>10.3f}")

# ─── Step 4: Generate Plots ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Generating ML Visualizations")
print("=" * 60)

output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(output_dir, exist_ok=True)

# Predictions plot — Linear
y_pred_lr = lr_model.predict(X_test)
plot_predictions(y_test, y_pred_lr, "Linear Regression",
                 save_path=os.path.join(output_dir, "10_lr_predictions.png"))
print("  ✅ Linear Regression predictions plot")

# Predictions plot — Ridge
y_pred_ridge = ridge_model.predict(X_test)
plot_predictions(y_test, y_pred_ridge, f"Ridge (α={best_alpha})",
                 save_path=os.path.join(output_dir, "11_ridge_predictions.png"))
print("  ✅ Ridge predictions plot")

# Feature importance — Linear
plot_feature_importance(lr_model, feature_names, "Linear Regression",
                        save_path=os.path.join(output_dir, "12_lr_feature_importance.png"))
print("  ✅ Linear Regression feature importance")

# Feature importance — Ridge
plot_feature_importance(ridge_model, feature_names, f"Ridge (α={best_alpha})",
                        save_path=os.path.join(output_dir, "13_ridge_feature_importance.png"))
print("  ✅ Ridge feature importance")

# Residuals — Ridge
plot_residuals(y_test, y_pred_ridge, f"Ridge (α={best_alpha})",
               save_path=os.path.join(output_dir, "14_ridge_residuals.png"))
print("  ✅ Ridge residual plot")

plt.close("all")

print(f"\n✅ All ML plots saved to: {output_dir}")
print("=" * 60)
print("ML MODELING COMPLETE!")
print("=" * 60)
