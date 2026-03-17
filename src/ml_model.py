"""
Machine Learning Module
South Punjab Development Dashboard

Linear and Ridge regression models to predict poverty scores
from literacy and health indicators.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─── Feature Preparation ─────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "literacy_rate", "female_literacy", "gender_literacy_gap",
    "primary_enrollment_rate", "middle_enrollment_rate",
    "immunization_coverage", "clean_water_access",
    "electricity_access", "hospitals_per_100k"
]

TARGET_COLUMN = "poverty_headcount"


def prepare_features(df, features=None, target=None, test_size=0.25, random_state=42):
    """
    Prepare features and target for modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe
    features : list, optional
        Feature column names (default: FEATURE_COLUMNS)
    target : str, optional
        Target column name (default: TARGET_COLUMN)
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    features = features or FEATURE_COLUMNS
    target = target or TARGET_COLUMN
    
    # Filter to available features
    available = [f for f in features if f in df.columns]
    missing = set(features) - set(available)
    if missing:
        print(f"⚠️  Missing features (skipped): {missing}")
    
    X = df[available].copy()
    y = df[target].copy()
    
    # Drop any rows with NaN
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=available, index=X.index)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✅ Prepared features: {len(available)} features, {len(X_train)} train, {len(X_test)} test samples")
    
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler, "feature_names": available
    }


# ─── Model Training ──────────────────────────────────────────────────────────

def train_linear(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"✅ Linear Regression trained | CV R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return model


def train_ridge(X_train, y_train, alpha=1.0):
    """
    Train a Ridge Regression model.
    
    Parameters
    ----------
    alpha : float
        Regularization strength
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"✅ Ridge Regression (α={alpha}) trained | CV R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return model


def find_best_alpha(X_train, y_train, alphas=None):
    """
    Find the best Ridge alpha using cross-validation.
    
    Returns
    -------
    float
        Best alpha value
    """
    if alphas is None:
        alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    best_alpha, best_score = None, -np.inf
    results = []
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        mean_score = scores.mean()
        results.append({"alpha": alpha, "cv_r2_mean": mean_score, "cv_r2_std": scores.std()})
        
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha
    
    print(f"✅ Best alpha: {best_alpha} (CV R² = {best_score:.3f})")
    return best_alpha, pd.DataFrame(results)


# ─── Model Evaluation ────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model on test data.
    
    Returns
    -------
    dict
        Evaluation metrics: r2, mae, rmse
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        "model": model_name,
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    
    print(f"\n📊 {model_name} Evaluation:")
    print(f"   R² Score  : {metrics['r2']:.3f}")
    print(f"   MAE       : {metrics['mae']:.2f}%")
    print(f"   RMSE      : {metrics['rmse']:.2f}%")
    
    return metrics


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_predictions(y_test, y_pred, model_name="Model", save_path=None):
    """
    Scatter plot: Actual vs Predicted poverty headcount.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_test, y_pred, c="#E63946", alpha=0.7, s=80, edgecolors="white", linewidth=0.5)
    
    # Perfect prediction line
    lims = [min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "--", color="#457B9D", linewidth=2, alpha=0.8, label="Perfect Prediction")
    
    ax.set_xlabel("Actual Poverty Headcount (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Poverty Headcount (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name}: Actual vs Predicted", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11)
    
    # Add R² annotation
    r2 = r2_score(y_test, y_pred)
    ax.annotate(f"R² = {r2:.3f}", xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=13, fontweight="bold", color="#1D3557",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F1FAEE", alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_feature_importance(model, feature_names, model_name="Model", save_path=None):
    """
    Horizontal bar chart of model coefficients (feature importance).
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    coefs = pd.Series(model.coef_, index=feature_names)
    coefs = coefs.sort_values()
    
    colors = ["#E63946" if c < 0 else "#2A9D8F" for c in coefs]
    
    ax.barh(coefs.index, coefs.values, color=colors, edgecolor="white", linewidth=0.5, height=0.6)
    ax.axvline(x=0, color="black", linewidth=0.8)
    
    ax.set_xlabel("Coefficient Value (standardized)", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name}: Feature Importance (Coefficients)",
                 fontsize=14, fontweight="bold", pad=15)
    
    # Format tick labels (set ticks first to avoid warning)
    ax.set_yticks(range(len(coefs)))
    ax.set_yticklabels([name.replace("_", " ").title() for name in coefs.index])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_residuals(y_test, y_pred, model_name="Model", save_path=None):
    """
    Residual plot to check model assumptions.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    residuals = y_test - y_pred
    
    ax.scatter(y_pred, residuals, c="#457B9D", alpha=0.7, s=60, edgecolors="white")
    ax.axhline(y=0, color="#E63946", linewidth=2, linestyle="--")
    
    ax.set_xlabel("Predicted Values", fontsize=12, fontweight="bold")
    ax.set_ylabel("Residuals", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name}: Residual Plot", fontsize=14, fontweight="bold", pad=15)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
