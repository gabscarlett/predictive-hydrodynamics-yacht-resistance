"""
Final object-oriented Decision Tree architecture.

This module provides a framework for modelling the Residuary Resistance (Rr)
of sailing yachts. It employs Scikit-learn pipeline architecture to capture the 
non-linear interactions between hull geometry and the Froude Number (Fr). 
The model integrates standardisation and Decision Tree Regression, providing tools for 
cross-validation, hyperparameter inspection, and feature importance analysis.
"""

from typing import Optional, Tuple, List, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


class YachtResistanceModel:
    """
    A modular predictive model for yacht hydrodynamics.

    This class predicts Residuary Resistance based on hull geometry and 
    hydrodynamic parameters, specifically targeting the non-linear transition 
    between displacement and planing regimes.
    """

    def __init__(
        self, 
        max_depth: Optional[int] = None, 
        min_samples_leaf: int = 1
    ) -> None:
        """
        Initialise the model with a pipeline of standardisation and regression.

        Parameters
        ----------
        max_depth : int, default=None
            The maximum depth of the decision tree. Used to control over-fitting.
        min_samples_leaf : int, default=1
            The minimum number of samples required to be at a leaf node.

        Notes
        -----
        Even though tree-based models are naturally scale-invariant, 
        StandardScaler is included here as a best practice for pipeline 
        modularity. This ensures the architecture remains robust if the 
        regressor is seamlessly swapped for a scale-sensitive estimator 
        (e.g., Ridge or SVM) in future iterations.
        """
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            ))
        ])

    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray]
    ) -> 'YachtResistanceModel':
        """
        Fit the model to the hydrodynamic dataset.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The training input samples. Features include hull geometry 
            parameters (LC, PC, L/D, B/Dr, L/B) and the Froude Number (Fr).
        y : numpy.ndarray of shape (n_samples,)
            The target values representing Residuary Resistance (Rr).

        Returns
        -------
        self : object
            Returns the instance itself.

        Notes
        -----
        The model is inherently sensitive to the transition at Fr ≈ 0.3–0.4, 
        reflecting the onset of the high-wave-making resistance regime where 
        geometry modifiers become critical.
        """
        self.pipeline.fit(X, y)
        return self

    def evaluate_cross_validation(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray], 
        cv: int = 5
    ) -> Tuple[float, float]:
        """
        Perform cross-validation to assess model robustness.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input features (geometry and Fr).
        y : numpy.ndarray of shape (n_samples,)
            Target values (Rr).
        cv : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        mean_score : float
            Mean R^2 score across folds.
        std_score : float
            Standard deviation of scores across folds.

        Notes
        -----
        This evaluation moves beyond simple R^2 metrics on a single split to 
        ensure the model's "regime-based" logic, where resistance behaviour 
        changes significantly with Fr, is robust across different subsets 
        of the underlying dataset.
        """
        scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='r2')
        return scores.mean(), scores.std()

    def plot_validation_curve(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray], 
        param_name: str = 'regressor__max_depth',
        param_range: np.ndarray = np.arange(1, 11)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualise the impact of a hyperparameter on training and test scores.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input features.
        y : numpy.ndarraye of shape (n_samples,)
            Target values (Rr).
        param_name : str, default='regressor__max_depth'
            The name of the parameter to vary (using pipeline naming syntax).
        param_range : numpy.ndarray, default=np.arange(1, 11)
            The values of the parameter that will be evaluated.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure object.
        ax : matplotlib.axes.Axes
            The generated axes object.
        """
        train_scores, test_scores = validation_curve(
            self.pipeline, X, y, param_name=param_name,
            param_range=param_range, cv=5, scoring='r2'
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(param_range, train_mean, label='Training Score', color='navy')
        ax.fill_between(param_range, train_mean - train_std,
                        train_mean + train_std, alpha=0.15, color='navy')
        ax.plot(param_range, test_mean, label='Cross-validation Score', color='crimson')
        ax.fill_between(param_range, test_mean - test_std,
                        test_mean + test_std, alpha=0.15, color='crimson')

        # This plot identifies the "Bias-Variance Tradeoff". As tree depth increases,
        # training error decreases but the gap with validation error may grow,
        # indicating where tree pruning is necessary to prevent over-fitting.
        ax.set_title('Validation Curve (Bias-Variance Tradeoff Analysis)')
        ax.set_xlabel(param_name.split('__')[-1].replace('_', ' ').title())
        ax.set_ylabel('R^2 Score')
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig, ax

    def plot_feature_importances(
        self, 
        feature_names: Union[List[str], pd.Index]
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Identify and visualise the primary drivers of residuary resistance.

        Parameters
        ----------
        feature_names : list of str or pandas.Index
            The labels for the input features (LC, PC, L/D, B/Dr, L/B, Fr).

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Notes
        -----
        This method identifies which geometry "modifiers" most significantly 
        influence resistance alongside the primary Froude Number signal, 
        validating the multivariate nature of the fluid dynamics.
        """
        importances = self.pipeline.named_steps['regressor'].feature_importances_
        indices = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices], align='center', color='teal')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Relative Gini Importance')
        ax.set_title('Feature Importances: Primary Signal vs. Geometry Modifiers')
        
        return fig, ax

    def plot_partial_dependence(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        features: List[Union[Tuple[str, str], str, int]]
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualise the interaction between geometry and hydrodynamics.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Input data used for calculating partial dependence.
        features : list of tuples, str, or int
            The features to target for interaction (e.g., [('PC', 'Fr')]).

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Notes
        -----
        This method validates the engineering hypothesis that hull geometry
        acts as a non-linear modifier to the hydrodynamic lift and drag regimes.
        It is particularly useful for observing the sensitivity shift at Fr > 0.3.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(
            self.pipeline, X, features, ax=ax, kind='average'
        )
        fig.suptitle('Partial Dependence: Interaction of Geometry and Froude Number')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        return fig, ax


if __name__ == "__main__":

    from helperfnc import load_data

    # 0. Load the data 
    X_train, y_train = load_data('yacht_hydro.csv')
    feature_columns = X_train.columns

    # 1. Model Initialisation and Training
    # max_depth=5 provides a balanced starting point for the Bias-Variance tradeoff
    model = YachtResistanceModel(max_depth=5)
    model.fit(X_train, y_train)

    # 2. Performance Evaluation
    mean_r2, std_r2 = model.evaluate_cross_validation(X_train, y_train)
    print(f"Cross-Validation R^2 Score: {mean_r2:.3f} (+/- {std_r2:.3f})")

    # 3. Visualisation: Validation Curve
    fig_val, ax_val = model.plot_validation_curve(X_train, y_train)
    plt.show()

    # 4. Visualisation: Feature Importances
    fig_feat, ax_feat = model.plot_feature_importances(feature_columns)
    plt.show()

    # 5. Visualisation: Partial Dependence
    fig_pd, ax_pd = model.plot_partial_dependence(X_train, features=[('Prismatic_coeff', 'Froude_number')])
    plt.show()