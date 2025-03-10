import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def select_features_by_score(X, y, method='f_test', task='auto', k=None, percentile=None):
    """
    Select top features based on statistical tests or mutual information.
    
    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target variable
    method : str, optional
        Method to use: 'f_test', 'mutual_info'
    task : str, optional
        'classification', 'regression', or 'auto' (detect automatically)
    k : int, optional
        Number of top features to select (if None, use percentile)
    percentile : int, optional
        Percentage of top features to select (if k is None)
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        Selected features
    list
        Names of selected features (if X is a DataFrame)
    dict
        Scores for each feature
    """
    # Convert to numpy arrays if necessary
    is_pandas = isinstance(X, pd.DataFrame)
    
    if is_pandas:
        feature_names = X.columns.tolist()
        X_values = X.values
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_values = X
        
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y
    
    # Detect the task type if set to 'auto'
    if task == 'auto':
        unique_values = np.unique(y_values)
        if len(unique_values) < 10 or (isinstance(unique_values[0], (str, bool)) and len(unique_values) < 100):
            task = 'classification'
        else:
            task = 'regression'
    
    # Set up the feature selection method
    if method == 'f_test':
        if task == 'classification':
            score_func = f_classif
        elif task == 'regression':
            score_func = f_regression
        else:
            raise ValueError("Task must be 'classification', 'regression', or 'auto'")
    elif method == 'mutual_info':
        if task == 'classification':
            score_func = mutual_info_classif
        elif task == 'regression':
            score_func = mutual_info_regression
        else:
            raise ValueError("Task must be 'classification', 'regression', or 'auto'")
    else:
        raise ValueError("Method must be 'f_test' or 'mutual_info'")
    
    # Determine k based on percentile if not provided
    if k is None:
        if percentile is None:
            percentile = 50  # Default to top 50%
        k = max(1, int(X_values.shape[1] * percentile / 100))
    
    # Apply feature selection
    selector = SelectKBest(score_func=score_func, k=k)
    X_new = selector.fit_transform(X_values, y_values)
    
    # Get the scores and selected feature indices
    scores = selector.scores_
    selected_indices = np.argsort(scores)[-k:][::-1]  # Indices of top k features
    selected_features = [feature_names[i] for i in selected_indices]
    
    # Create a dictionary of feature scores
    feature_scores = {feature_names[i]: scores[i] for i in range(len(feature_names))}
    
    # If input was a DataFrame, return a DataFrame with selected features
    if is_pandas:
        X_selected = X.iloc[:, selected_indices]
    else:
        X_selected = X_new
    
    return X_selected, selected_features, feature_scores

def select_features_rfe(X, y, estimator=None, n_features_to_select=None, step=1, task='auto'):
    """
    Select features using Recursive Feature Elimination (RFE).
    
    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target variable
    estimator : estimator object, optional
        A supervised learning estimator with a fit method
    n_features_to_select : int, optional
        Number of features to select
    step : int or float, optional
        Number/percentage of features to remove at each iteration
    task : str, optional
        'classification', 'regression', or 'auto' (detect automatically)
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        Selected features
    list
        Names of selected features (if X is a DataFrame)
    dict
        Feature rankings (lower = better)
    """
    # Convert to numpy arrays if necessary
    is_pandas = isinstance(X, pd.DataFrame)
    
    if is_pandas:
        feature_names = X.columns.tolist()
        X_values = X.values
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_values = X
        
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y
    
    # Detect the task type if set to 'auto'
    if task == 'auto':
        unique_values = np.unique(y_values)
        if len(unique_values) < 10 or (isinstance(unique_values[0], (str, bool)) and len(unique_values) < 100):
            task = 'classification'
        else:
            task = 'regression'
    
    # Set default estimator if none provided
    if estimator is None:
        if task == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        elif task == 'regression':
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Task must be 'classification', 'regression', or 'auto'")
    
    # Set default n_features_to_select if not provided
    if n_features_to_select is None:
        n_features_to_select = max(1, X_values.shape[1] // 2)  # Default to half
    
    # Apply RFE
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
    X_new = rfe.fit_transform(X_values, y_values)
    
    # Get the feature rankings and selected feature indices
    rankings = rfe.ranking_
    selected_indices = np.where(rankings == 1)[0]
    selected_features = [feature_names[i] for i in selected_indices]
    
    # Create a dictionary of feature rankings
    feature_rankings = {feature_names[i]: rankings[i] for i in range(len(feature_names))}
    
    # If input was a DataFrame, return a DataFrame with selected features
    if is_pandas:
        X_selected = X.iloc[:, selected_indices]
    else:
        X_selected = X_new
    
    return X_selected, selected_features, feature_rankings

def remove_low_variance_features(X, threshold=0.0):
    """
    Remove features with low variance.
    
    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    threshold : float, optional
        Threshold for variance (features with lower variance will be removed)
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        Features after removing low variance ones
    list
        Names of retained features (if X is a DataFrame)
    dict
        Variance of each feature
    """
    # Convert to numpy arrays if necessary
    is_pandas = isinstance(X, pd.DataFrame)
    
    if is_pandas:
        feature_names = X.columns.tolist()
        X_values = X.values
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_values = X
    
    # Calculate variance for each feature
    variances = np.var(X_values, axis=0)
    
    # Create a dictionary of feature variances
    feature_variances = {feature_names[i]: variances[i] for i in range(len(feature_names))}
    
    # Apply variance thresholding
    selector = VarianceThreshold(threshold=threshold)
    X_new = selector.fit_transform(X_values)
    
    # Get indices of retained features
    retained_indices = selector.get_support(indices=True)
    retained_features = [feature_names[i] for i in retained_indices]
    
    # If input was a DataFrame, return a DataFrame with retained features
    if is_pandas:
        X_selected = X.iloc[:, retained_indices]
    else:
        X_selected = X_new
    
    return X_selected, retained_features, feature_variances

def detect_multicollinearity(X, threshold=0.8):
    """
    Detect multicollinearity among features using correlation analysis.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Features
    threshold : float, optional
        Correlation threshold to identify multicollinearity
        
    Returns
    -------
    list
        Groups of highly correlated features
    pandas.DataFrame
        Correlation matrix
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Set diagonal to zero to exclude self-correlation
    np.fill_diagonal(corr_matrix.values, 0)
    
    # Find groups of correlated features
    correlated_groups = []
    processed_features = set()
    
    for feature in corr_matrix.columns:
        if feature in processed_features:
            continue
            
        # Find features highly correlated with the current feature
        correlated = corr_matrix.index[corr_matrix[feature] > threshold].tolist()
        
        if correlated:
            # Add the current feature to the group
            group = [feature] + correlated
            correlated_groups.append(group)
            
            # Mark all features in this group as processed
            processed_features.update(group)
    
    return correlated_groups, corr_matrix