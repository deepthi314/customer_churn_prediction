import pandas as pd
import numpy as np
import logging
import joblib
import os

# FORCE SINGLE THREADED EXECUTION TO PREVENT DEADLOCKS
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_models():
    """
    Define the models to be trained.
    
    Returns:
        dict: Dictionary of model instances.
    """
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', n_jobs=1),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=1),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=1),
        # 'SVM': SVC(probability=True, random_state=42, class_weight='balanced') # Optional, often slow
    }
    return models

def get_hyperparameters():
    """
    Define hyperparameters for grid/random search.
    
    Returns:
        dict: Dictionary of parameters for each model.
    """
    params = {
        'LogisticRegression': {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l2']
        },
        'RandomForest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1]
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.1],
            'classifier__subsample': [0.8],
            'classifier__colsample_bytree': [0.8]
        }
    }
    return params

def train_model(model_name, model, X_train, y_train, use_smote=True, tune_hyperparameters=True):
    """
    Train a single model with optional SMOTE and hyperparameter tuning.
    
    Args:
        model_name (str): Name of the model.
        model (sklearn/xgboost model): Model instance.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        use_smote (bool): Whether to use SMOTE for oversampling.
        tune_hyperparameters (bool): Whether to perform GridSearchCV.
        
    Returns:
        dict: Dictionary containing best model, best params, and training metrics.
    """
    logger.info(f"Training {model_name}...")
    
    steps = []
    if use_smote:
        steps.append(('smote', SMOTE(random_state=42)))
    
    steps.append(('classifier', model))
    
    pipeline = ImbPipeline(steps)
    
    if tune_hyperparameters:
        param_grid = get_hyperparameters().get(model_name, {})
        if param_grid:
            logger.info(f"Tuning hyperparameters for {model_name}...")
            # Use StratifiedKFold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Using RandomizedSearchCV for speed, or GridSearch for thoroughness
            # Let's use RandomizedSearchCV for RF and XGB to save time in example, 
            # but Prompt asked for GridSearchCV or RandomizedSearchCV.
            # Using GridSearch for LR, Randomized for others if params are large.
            # For simplicity, let's stick to GridSearch but with limited params as defined.
            
            search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=cv, 
                scoring='roc_auc', 
                n_jobs=1,  # Use 1 core to prevent Windows deadlock
                verbose=3  # Higher verbosity to see progress
            )
            search.fit(X_train, y_train)
            
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_score = search.best_score_
            logger.info(f"Best params for {model_name}: {best_params}")
            logger.info(f"Best CV ROC-AUC for {model_name}: {best_score:.4f}")
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = "Default"
            best_score = "N/A"
    else:
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        best_params = "Default"
        best_score = "N/A"
        
    return {
        'model': best_model,
        'best_params': best_params,
        'best_score': best_score
    }

def train_all_models(X_train, y_train, use_smote=True, tune_hyperparameters=False):
    """
    Train all defined models.
    
    Args:
        X_train, y_train: Training data.
        use_smote (bool): Use SMOTE.
        tune_hyperparameters (bool): Tune params.
        
    Returns:
        dict: Results for all models.
    """
    models = get_models()
    results = {}
    
    best_overall_score = -1
    best_model_name = None
    
    for name, model in models.items():
        try:
            res = train_model(name, model, X_train, y_train, use_smote, tune_hyperparameters)
            results[name] = res
            
            # Track best model based on CV score or just store all
            # If no tuning, we don't have CV score readily available unless we calculate it. 
            # Assuming tuning is OFF by default for speed in prototype, but ON for final.
            if res['best_score'] != "N/A" and isinstance(res['best_score'], float):
                 if res['best_score'] > best_overall_score:
                     best_overall_score = res['best_score']
                     best_model_name = name
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
            pass
            
    results['best_model_name'] = best_model_name
    return results

def save_model(model, filepath, feature_names=None):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model object (pipeline).
        filepath (str): Path to save.
        feature_names (list): List of feature names to ensure alignment during inference.
    """
    try:
        data_to_save = {'model': model, 'feature_names': feature_names}
        joblib.dump(data_to_save, filepath)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

if __name__ == "__main__":
    pass
