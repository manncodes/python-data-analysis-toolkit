import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

def train_test_split_stratified(X, y, test_size=0.2, random_state=None, stratify=True):
    """
    Split data into train and test sets with optional stratification.
    
    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target variable
    test_size : float, optional
        Proportion of the dataset to include in the test split
    random_state : int, optional
        Random seed for reproducibility
    stratify : bool, optional
        Whether to use stratified sampling (for classification tasks)
        
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    # Determine if stratification should be used
    stratify_param = y if stratify else None
    
    # Handle categorical target for classification
    if stratify and isinstance(y, pd.Series) and y.dtype == 'object':
        # Check if unique values are too many for effective stratification
        if y.nunique() > len(y) * 0.2:  # If more than 20% unique values
            stratify_param = None
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    
    return X_train, X_test, y_train, y_test

def cross_validation_metrics(model, X, y, cv=5, random_state=None, task='auto', scoring=None):
    """
    Perform cross-validation and return comprehensive metrics.
    
    Parameters
    ----------
    model : estimator object
        The model to evaluate
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target variable
    cv : int, optional
        Number of cross-validation folds
    random_state : int, optional
        Random seed for reproducibility
    task : str, optional
        'classification', 'regression', or 'auto' (detect automatically)
    scoring : str, list, or dict, optional
        Metrics to use for scoring (if None, use default based on task)
        
    Returns
    -------
    dict
        Dictionary containing cross-validation metrics
    """
    # Convert to numpy arrays if necessary
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # Detect the task type if set to 'auto'
    if task == 'auto':
        if hasattr(model, '_estimator_type'):
            task = model._estimator_type
        else:
            # Try to infer from target variable
            unique_values = np.unique(y)
            if len(unique_values) < 10 or (isinstance(unique_values[0], (str, bool)) and len(unique_values) < 100):
                task = 'classification'
            else:
                task = 'regression'
    
    # Set up scoring metrics based on the task
    if scoring is None:
        if task == 'classification':
            scoring = {
                'accuracy': 'accuracy',
                'precision_macro': 'precision_macro',
                'recall_macro': 'recall_macro',
                'f1_macro': 'f1_macro'
            }
            
            # Check if binary classification (for ROC AUC)
            if len(np.unique(y)) == 2:
                scoring['roc_auc'] = 'roc_auc'
                
        elif task == 'regression':
            scoring = {
                'neg_mean_squared_error': 'neg_mean_squared_error',
                'neg_mean_absolute_error': 'neg_mean_absolute_error',
                'r2': 'r2'
            }
        else:
            raise ValueError("Task must be 'classification', 'regression', or 'auto'")
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=True
    )
    
    # Process results
    results = {}
    for metric in scoring:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        # For negative metrics, convert back to positive
        if metric.startswith('neg_'):
            test_scores = -test_scores
            train_scores = -train_scores
            metric_name = metric[4:]  # Remove 'neg_' prefix
        else:
            metric_name = metric
        
        results[metric_name] = {
            'test_mean': np.mean(test_scores),
            'test_std': np.std(test_scores),
            'train_mean': np.mean(train_scores),
            'train_std': np.std(train_scores),
            'test_values': test_scores.tolist(),
            'train_values': train_scores.tolist()
        }
    
    # Add fit times
    results['fit_time'] = {
        'mean': np.mean(cv_results['fit_time']),
        'std': np.std(cv_results['fit_time']),
        'values': cv_results['fit_time'].tolist()
    }
    
    results['score_time'] = {
        'mean': np.mean(cv_results['score_time']),
        'std': np.std(cv_results['score_time']),
        'values': cv_results['score_time'].tolist()
    }
    
    return results

def feature_importance_analysis(model, feature_names, top_n=None):
    """
    Extract and format feature importances from a trained model.
    
    Parameters
    ----------
    model : estimator object
        Trained model with feature_importances_ or coef_ attribute
    feature_names : list
        List of feature names
    top_n : int, optional
        Number of top features to return (if None, return all)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with feature names and importance scores
    """
    # Check if model has feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        # Handle multi-class case
        if importances.ndim > 1:
            importances = np.mean(importances, axis=0)
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute")
    
    # Create DataFrame with feature names and importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Filter to top N if specified
    if top_n is not None:
        importance_df = importance_df.head(top_n)
    
    return importance_df

def encode_categorical_features(data, columns=None, drop_original=True):
    """
    Encode categorical features using Label Encoding.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing categorical columns
    columns : list, optional
        List of column names to encode (if None, encode all object columns)
    drop_original : bool, optional
        Whether to drop original categorical columns
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with encoded features
    dict
        Dictionary mapping column names to their LabelEncoder objects
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    # Make a copy of the DataFrame
    result = data.copy()
    
    # If columns not specified, use all object dtype columns
    if columns is None:
        columns = result.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create a dictionary to store encoders
    encoders = {}
    
    # Encode each specified column
    for col in columns:
        if col not in result.columns:
            continue
            
        # Create and fit encoder
        le = LabelEncoder()
        result[f'{col}_encoded'] = le.fit_transform(result[col].astype(str))
        
        # Store encoder for potential inverse transform
        encoders[col] = le
        
        # Drop original column if specified
        if drop_original:
            result = result.drop(columns=[col])
    
    return result, encoders