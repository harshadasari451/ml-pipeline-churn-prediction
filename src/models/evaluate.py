"""
Model evaluation module with metrics calculation and visualization.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.helpers import ensure_dir, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Model evaluation class for calculating metrics and creating visualizations.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize ModelEvaluator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.metrics_config = self.config.get('metrics', {})
        
        # Create output directory for plots
        self.plots_dir = ensure_dir("reports/figures")
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        }
        
        # Add ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn']
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.savefig(self.plots_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            save_path: Path to save the plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        else:
            plt.savefig(self.plots_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: list,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importance.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            top_n: Number of top features to show
            title: Plot title
            save_path: Path to save the plot
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.savefig(self.plots_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[list] = None,
        plot: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            plot: Whether to create plots
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Log metrics
        logger.info("Test Metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Check against thresholds
        self._check_thresholds(metrics)
        
        # Create plots
        if plot:
            self.plot_confusion_matrix(y_test, y_pred)
            
            if y_pred_proba is not None:
                self.plot_roc_curve(y_test, y_pred_proba)
            
            if feature_names and hasattr(model, 'feature_importances_'):
                self.plot_feature_importance(model, feature_names)
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(
            y_test, y_pred,
            target_names=['No Churn', 'Churn']
        ))
        
        return metrics
    
    def _check_thresholds(self, metrics: Dict[str, float]) -> None:
        """
        Check if metrics meet minimum thresholds.
        
        Args:
            metrics: Dictionary of metrics
        """
        thresholds = {
            'accuracy': self.metrics_config.get('min_accuracy', 0.75),
            'f1': self.metrics_config.get('min_f1_score', 0.70),
            'roc_auc': self.metrics_config.get('min_roc_auc', 0.80),
            'precision': self.metrics_config.get('min_precision', 0.65),
            'recall': self.metrics_config.get('min_recall', 0.65),
        }
        
        logger.info("\nMetrics vs Thresholds:")
        for metric_name, threshold in thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                status = "✓" if value >= threshold else "✗"
                logger.info(f"  {status} {metric_name}: {value:.4f} (threshold: {threshold:.4f})")
    
    def compare_models(
        self,
        models_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a comparison plot of multiple models.
        
        Args:
            models_results: Dictionary of model names to metrics
            save_path: Path to save the plot
        """
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Prepare data
        model_names = list(models_results.keys())
        metrics_data = {metric: [] for metric in metrics_to_plot}
        
        for model_name in model_names:
            for metric in metrics_to_plot:
                value = models_results[model_name].get(metric, 0)
                metrics_data[metric].append(value)
        
        # Create plot
        x = np.arange(len(model_names))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics_to_plot):
            ax.bar(x + i * width, metrics_data[metric], width, label=metric)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        else:
            plt.savefig(self.plots_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        
        plt.close()


def main():
    """Main function for model evaluation."""
    from src.utils.logger import setup_logging
    from src.utils.helpers import load_pickle
    
    setup_logging()
    
    # This would typically be called from train.py
    logger.info("Model evaluator module loaded successfully!")


if __name__ == "__main__":
    main()
