"""
Polynomial pipeline and coefficient extraction.

This script implements a pipeline to investigate how hull geometry
modifies the dominant Froude number signal in residuary resistance prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from helperfnc import load_data


def run_ridge_polynomial_search(X, y):
    """Execute grid search over polynomial degrees and regularization."""
    # Scale after expansion to ensure interaction terms are regularized fairly
    pipeline = Pipeline([
        ('poly_features', PolynomialFeatures()),
        ('scaler', StandardScaler()),
        ('regressor', Ridge())
    ])

    param_grid = {
        'poly_features__degree': list(range(1, 6)),  # Balanced for complexity
        'regressor__alpha': np.logspace(-3, 3, 7)
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring='r2',
        n_jobs=-1
    )

    grid_search.fit(X, y)
    return grid_search


def inspect_coefficients(best_model):
    """Extract coefficients to validate the physical sense of the modifiers."""
    coefficients = best_model.named_steps['regressor'].coef_
    features = best_model.named_steps['poly_features'].get_feature_names_out()

    summary = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', ascending=False)

    return summary


def plot_model_performance(model, X, y):
    """Visualise model accuracy and alignment with hydrodynamic physics."""
    y_pred = model.predict(X)
    fn_col = 'Fr' if 'Fr' in X.columns else X.columns[-1] # Identify Froude number location (could be last column)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Regression Accuracy: Predicted vs. Actual
    ax1.scatter(y, y_pred, alpha=0.5, color='teal', label='Predictions')
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal Fit')
    ax1.set_title("Standard Accuracy: Predicted vs. Actual")
    ax1.set_xlabel("Actual Residuary Resistance")
    ax1.set_ylabel("Predicted Residuary Resistance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Physics Check: Resistance vs. Froude Number
    ax2.scatter(X[fn_col], y, label="Actual Data", alpha=0.3, color='gray')
    ax2.scatter(X[fn_col], y_pred, label="Model Prediction", marker='x', alpha=0.6, color='darkorange')
    ax2.set_title(f"Physics Validation: Resistance vs. {fn_col}")
    ax2.set_xlabel("Froude Number (Fn)")
    ax2.set_ylabel("Residuary Resistance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assets/prr_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Load and prepare hydrodynamic data
    X, y = load_data('yacht_hydro.csv')

    # Execute search for optimal hyperparameters
    search = run_ridge_polynomial_search(X, y)
    best_pipeline = search.best_estimator_

    print(f"Best Parameters: {search.best_params_}")
    print(f"Cross-Validated R^2: {search.best_score_:.4f}")

    # Interpret the coefficients and interactions
    coef_summary = inspect_coefficients(best_pipeline)
    print("\nTop 10 Feature Interactions (Physical Modifiers):")
    print(coef_summary.head(10))

    # Final Visualisation
    plot_model_performance(best_pipeline, X, y)
    