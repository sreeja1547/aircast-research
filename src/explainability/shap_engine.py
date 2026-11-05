"""Explainable AI (XAI) engine using exact TreeSHAP attributions.

Extracts feature importance rankings, interaction effects, and local attributions
for severe pollution episodes vs clean background conditions using XGBoost's C++ TreeSHAP.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from src.models import XGBoostForecaster


class ShapExplanationEngine:
    """TreeSHAP Explanation Engine for tree-based forecasters.

    Leverages XGBoost's native exact TreeSHAP algorithm to compute additive
    feature contributions satisfying efficiency, symmetry, and dummy properties.

    Parameters
    ----------
    model : XGBoostForecaster
        Fitted XGBoost model instance.
    """

    def __init__(self, model: XGBoostForecaster) -> None:
        if not model.is_fitted:
            raise RuntimeError("Model must be fitted before initializing ShapExplanationEngine.")
        self.model = model
        self.feature_names = model.feature_names

    def explain(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Compute exact TreeSHAP values for input samples.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Evaluation feature matrix.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'shap_values': (n_samples, n_features) np.ndarray
            - 'base_value': float expected model value
            - 'mean_abs_shap': pd.Series of global feature importance
            - 'feature_names': list of features
        """
        shap_res = self.model.compute_shap_values(X)
        shap_vals = shap_res["shap_values"]
        base_val = shap_res["base_value"]

        # Global feature importance = mean absolute SHAP value
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        mean_abs_series = pd.Series(
            mean_abs, index=self.feature_names, name="mean_abs_shap"
        ).sort_values(ascending=False)

        return {
            "shap_values": shap_vals,
            "base_value": base_val,
            "mean_abs_shap": mean_abs_series,
            "feature_names": self.feature_names,
        }

    def generate_summary_plot(
        self,
        X_sample: pd.DataFrame,
        output_path: str = "artifacts/plots/shap_summary.png",
        top_k: int = 15,
    ) -> None:
        """Generate high-resolution SHAP global feature attribution summary plot."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        res = self.explain(X_sample)
        shap_vals = res["shap_values"]
        mean_abs = res["mean_abs_shap"].iloc[:top_k]

        sns.set_theme(style="whitegrid", font="sans-serif")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

        # Panel A: Mean Absolute SHAP Bar Plot (Global Importance)
        sns.barplot(
            x=mean_abs.values,
            y=mean_abs.index,
            ax=axes[0],
            palette="crest",
        )
        axes[0].set_title(f"(a) Top {top_k} Features by Mean |SHAP| (Global Impact)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel(r"Mean Absolute Attribution ($|\phi_i|$, $\mu g/m^3$)", fontsize=11)
        axes[0].set_ylabel("Feature Name", fontsize=11)

        # Panel B: SHAP Value Distribution (Beeswarm-style distribution boxplot)
        top_features = list(mean_abs.index[:8])
        plot_records = []
        for feat in top_features:
            f_idx = self.feature_names.index(feat)
            for s_val in shap_vals[:, f_idx]:
                plot_records.append({"feature": feat, "shap_value": s_val})

        df_dist = pd.DataFrame(plot_records)
        sns.boxplot(
            data=df_dist,
            y="feature",
            x="shap_value",
            ax=axes[1],
            palette="coolwarm",
            fliersize=2,
        )
        axes[1].axvline(0, color="gray", linestyle="--", linewidth=1.2)
        axes[1].set_title(r"(b) Attribution Directionality (Top 8 Features)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel(r"SHAP Value ($\phi_i$, $\mu g/m^3$ impact on prediction)", fontsize=11)
        axes[1].set_ylabel("")

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
