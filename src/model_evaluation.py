import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
import plotly.graph_objects as go
import plotly.express as px
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        model: Trained model (pipeline).
        X_test: Test features.
        y_test: Test labels.
        model_name: Name for reporting.
        
    Returns:
        dict: Metrics dictionary.
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
    
    logger.info(f"{model_name} Performance:")
    logger.info(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc if auc != 'N/A' else 'N/A'}")
    
    return {
        'model_name': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def plot_confusion_matrix(y_test, y_pred, model_name="Model"):
    """
    Plot Confusion Matrix using Seaborn.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_roc_curve(y_test, y_prob, model_name="Model"):
    """
    Plot ROC Curve using Matplotlib.
    """
    if y_prob is None:
        logger.warning("No probabilities provided for ROC curve.")
        return
        
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, y_prob):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.show()

def plot_feature_importance(model, feature_names, model_name="Model", top_n=20):
    """
    Plot Feature Importance for tree-based models.
    """
    # Extract feature importances
    # If model is a pipeline, we need to access the classifier step
    # Specifically, our pipeline structure is: SimpleImputer (maybe) -> SMOTE (internal) -> Classifier
    # Wait, our pipeline in model_training.py is: [('smote', SMOTE), ('classifier', clf)]
    # We need to access 'classifier'.
    
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps['classifier']
    else:
        classifier = model
        
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        if len(feature_names) != len(importances):
            logger.warning("Feature names and importances length mismatch. Adjusting if possible.")
            # This happens if OHE changed feature count, but feature_names passed should match X_train columns.
            # If mismatch, we can't label correctly.
            return
            
        feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feat_imp = feat_imp.sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feat_imp, palette='viridis')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.tight_layout()
        plt.show()
    else:
        logger.info(f"{model_name} does not provide feature importances.")

def calculate_business_metrics(evaluation, y_test, avg_monthly_charges=70, avg_tenure_months=24):
    """
    Estimate business impact.
    
    Assumption:
    - True Positive (Correctly identified churner): We can offer incentive to retain.
      Let's say success rate of retention is 50%, cost of incentive is $50.
      Value saved = (Average LTV - Incentive Cost) * 0.5
      Where LTV = Avg Monthly Charges * Avg Tenure Remaining (say 12 months)
    - False Positive (Incorrectly flagged): We waste incentive cost.
    - False Negative (Missed churner): We lose LTV.
    - True Negative (Correctly happy): No action.
    
    This is a simplified view defined in the prompt as:
    "estimated revenue saved"
    """
    cm = confusion_matrix(y_test, evaluation['y_pred'])
    tn, fp, fn, tp = cm.ravel()
    
    # Assumptions
    ltv = avg_monthly_charges * 12 # Value of retaining a customer for a year
    # marketing_cost = 0 # Cost to replace?
    retention_cost = 100 # Cost of retention offer
    success_rate = 0.5 # Probability offer is accepted
    
    # Revenue Saved
    # We identify TP churners. We try to save them.
    # Saved = TP * success_rate * (LTV - retention_cost)
    revenue_saved = tp * success_rate * (ltv - retention_cost)
    
    # Costs Incurred (Incentives given to FP)
    # We offer retention to everyone we predict as churn (TP + FP)
    # But only TP are actual churners. FP are happy but take the free money? 
    # Or maybe we only lose the retention cost on FP.
    wasted_budget = fp * retention_cost
    
    net_benefit = revenue_saved - wasted_budget
    
    return {
        'revenue_saved': revenue_saved,
        'wasted_budget': wasted_budget,
        'net_benefit': net_benefit,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

if __name__ == "__main__":
    pass
