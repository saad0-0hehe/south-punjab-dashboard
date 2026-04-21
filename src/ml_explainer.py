"""
SHAP Explainability Module
South Punjab Development Dashboard

Provides SHAP-based model interpretability for the poverty prediction models.
"""

import shap
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def compute_shap_values(model, X_train, X_test, feature_names):
    """
    Compute SHAP values using LinearExplainer.

    Returns
    -------
    shap.Explanation
        SHAP values object for the test set.
    """
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer(X_test)
    shap_values.feature_names = list(feature_names)
    return shap_values


def plot_shap_summary(shap_values):
    """
    Generate a SHAP beeswarm summary plot as a matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, show=False, plot_size=None)
    plt.tight_layout()
    return plt.gcf()


def plot_shap_waterfall(shap_values, index, district_name=""):
    """
    Generate a SHAP waterfall plot for a single observation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_values[index], show=False)
    if district_name:
        plt.title(f"SHAP Explanation: {district_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return plt.gcf()
