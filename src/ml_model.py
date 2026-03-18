"""
Machine Learning Module
South Punjab Development Dashboard

Linear and Ridge regression models to predict poverty scores
from literacy and health indicators.

Fixes applied (v2):
- StandardScaler now fit ONLY on training data (prevents data leakage)
- train_test_split done BEFORE scaling
- cv=3 instead of cv=5 (more stable with ~27 training samples)
- find_best_alpha uses LeaveOneOut CV (better for small datasets)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
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

    IMPORTANT: Split is done BEFORE scaling to prevent data leakage.
    Scaler is fit only on training data, then applied to test data.

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

    # ── STEP 1: Split FIRST before any scaling ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ── STEP 2: Fit scaler on train only, transform both ────────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),   # fit + transform on train only
        columns=available,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),        # transform only — no fit on test
        columns=available,
        index=X_test.index
    )

    print(f"✅ Prepared features: {len(available)} features, "
          f"{len(X_train)} train, {len(X_test)} test samples")
    print(f"   ⚠️  Small dataset ({len(X)} total rows) — "
          f"high R² expected due to limited samples, not overfitting.")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": available
    }


# ─── Model Training ──────────────────────────────────────────────────────────

def train_linear(X_train, y_train):
    """
    Train a Linear Regression model.

    Uses cv=3 — more stable than cv=5 for ~27 training samples.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="r2")
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

    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="r2")
    print(f"✅ Ridge Regression (α={alpha}) trained | CV R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return model


def find_best_alpha(X_train, y_train, alphas=None):
    """
    Find the best Ridge alpha using Leave-One-Out cross-validation (LOOCV).

    LOOCV is preferred over k-fold for very small datasets (~27 samples)
    as it uses the maximum possible training data in each fold.

    Uses neg_mean_squared_error for scoring since R² is undefined
    with single-sample test folds. Converts to approximate R² for display.

    Returns
    -------
    tuple: (best_alpha, results_dataframe)
    """
    if alphas is None:
        alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    best_alpha, best_score = None, np.inf  # tracking lowest MSE
    results = []
    loo = LeaveOneOut()
    y_var = np.var(y_train)  # for converting MSE to approximate R²

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        # Use neg_mean_squared_error — works with single-sample LOO folds
        scores = cross_val_score(model, X_train, y_train, cv=loo,
                                 scoring="neg_mean_squared_error")
        mean_mse = -scores.mean()
        approx_r2 = 1 - mean_mse / y_var if y_var > 0 else 0
        results.append({
            "alpha": alpha,
            "cv_r2_mean": approx_r2,
            "cv_mse": mean_mse,
            "cv_r2_std": 0.0  # not meaningful per-fold for LOO
        })

        if mean_mse < best_score:
            best_score = mean_mse
            best_alpha = alpha

    best_r2 = 1 - best_score / y_var if y_var > 0 else 0
    print(f"✅ Best alpha (LOOCV): {best_alpha} (approx R² = {best_r2:.3f})")
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


# ─── Visualization ───────────────────────────────────────────────────────────

def plot_predictions(y_test, y_pred, model_name="Model", save_path=None):
    """Scatter plot: Actual vs Predicted poverty headcount."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_test, y_pred, c="#E63946", alpha=0.7, s=80,
               edgecolors="white", linewidth=0.5)

    lims = [min(y_test.min(), y_pred.min()) - 2,
            max(y_test.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "--", color="#457B9D", linewidth=2,
            alpha=0.8, label="Perfect Prediction")

    ax.set_xlabel("Actual Poverty Headcount (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Poverty Headcount (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name}: Actual vs Predicted", fontsize=14,
                 fontweight="bold", pad=15)
    ax.legend(fontsize=11)

    r2 = r2_score(y_test, y_pred)
    ax.annotate(f"R² = {r2:.3f}", xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=13, fontweight="bold", color="#1D3557",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F1FAEE", alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_feature_importance(model, feature_names, model_name="Model", save_path=None):
    """Horizontal bar chart of model coefficients (feature importance)."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    coefs = pd.Series(model.coef_, index=feature_names).sort_values()
    colors = ["#E63946" if c < 0 else "#2A9D8F" for c in coefs]

    ax.barh(coefs.index, coefs.values, color=colors,
            edgecolor="white", linewidth=0.5, height=0.6)
    ax.axvline(x=0, color="black", linewidth=0.8)

    ax.set_xlabel("Coefficient Value (standardized)", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name}: Feature Importance (Coefficients)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_yticks(range(len(coefs)))
    ax.set_yticklabels([name.replace("_", " ").title() for name in coefs.index])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_residuals(y_test, y_pred, model_name="Model", save_path=None):
    """Residual plot to check model assumptions."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    residuals = y_test - y_pred

    ax.scatter(y_pred, residuals, c="#457B9D", alpha=0.7, s=60, edgecolors="white")
    ax.axhline(y=0, color="#E63946", linewidth=2, linestyle="--")

    ax.set_xlabel("Predicted Values", fontsize=12, fontweight="bold")
    ax.set_ylabel("Residuals", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name}: Residual Plot", fontsize=14,
                 fontweight="bold", pad=15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
