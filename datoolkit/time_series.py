import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

def check_stationarity(time_series, significance=0.05):
    """
    Test for stationarity using the Augmented Dickey-Fuller test.
    
    Parameters
    ----------
    time_series : pandas.Series
        Time series data
    significance : float, optional
        Significance level for the test
        
    Returns
    -------
    bool
        True if the series is stationary, False otherwise
    dict
        Test statistics and critical values
    """
    # Perform Augmented Dickey-Fuller test
    result = adfuller(time_series.dropna())
    
    # Extract and format the results
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    # Check if the series is stationary at the given significance level
    is_stationary = p_value < significance
    
    # Prepare the results dictionary
    test_results = {
        'ADF_Statistic': adf_stat,
        'p-value': p_value,
        'Critical_Values': critical_values,
        'is_stationary': is_stationary
    }
    
    return is_stationary, test_results

def decompose_time_series(time_series, period=None, model='additive'):
    """
    Decompose time series into trend, seasonal, and residual components.
    
    Parameters
    ----------
    time_series : pandas.Series
        Time series data
    period : int, optional
        Number of periods in a seasonal cycle (e.g., 12 for monthly data with yearly cycle)
    model : str, optional
        Type of decomposition: 'additive' or 'multiplicative'
        
    Returns
    -------
    statsmodels.tsa.seasonal.DecomposeResult
        Decomposition result object with trend, seasonal, and residual components
    """
    # Infer period if not provided
    if period is None:
        # Check if the index is a DatetimeIndex
        if isinstance(time_series.index, pd.DatetimeIndex):
            # Get the frequency string
            freq = pd.infer_freq(time_series.index)
            
            if freq:
                if 'D' in freq:  # Daily data
                    period = 7  # Weekly cycle
                elif 'M' in freq:  # Monthly data
                    period = 12  # Yearly cycle
                elif 'Q' in freq:  # Quarterly data
                    period = 4  # Yearly cycle
                elif 'Y' in freq:  # Yearly data
                    period = 1  # No seasonality
                else:
                    period = 10  # Default
            else:
                # Try to infer from the length of the series
                if len(time_series) >= 365:
                    period = 7  # Assume daily data with weekly cycle
                elif len(time_series) >= 24:
                    period = 12  # Assume monthly data with yearly cycle
                else:
                    period = 4  # Default
        else:
            # For non-datetime index, use a default
            period = 10
    
    # Perform decomposition
    decomposition = seasonal_decompose(time_series.dropna(), model=model, period=period)
    
    return decomposition

def calculate_acf_pacf(time_series, nlags=40, alpha=0.05):
    """
    Calculate autocorrelation (ACF) and partial autocorrelation (PACF) functions.
    
    Parameters
    ----------
    time_series : pandas.Series
        Time series data
    nlags : int, optional
        Number of lags to include
    alpha : float, optional
        Significance level for confidence intervals
        
    Returns
    -------
    tuple
        (acf_values, pacf_values, confidence_interval)
    """
    # Calculate ACF and PACF
    acf_values = acf(time_series.dropna(), nlags=nlags)
    pacf_values = pacf(time_series.dropna(), nlags=nlags)
    
    # Calculate confidence interval
    confidence = stats.norm.ppf(1 - alpha/2) / np.sqrt(len(time_series.dropna()))
    
    return acf_values, pacf_values, confidence

def difference_series(time_series, order=1):
    """
    Apply differencing to make a time series stationary.
    
    Parameters
    ----------
    time_series : pandas.Series
        Time series data
    order : int, optional
        Number of times to difference the series
        
    Returns
    -------
    pandas.Series
        Differenced time series
    """
    # Make a copy of the original series
    diff_series = time_series.copy()
    
    # Apply differencing
    for i in range(order):
        diff_series = diff_series.diff()
    
    # Drop NaN values created by differencing
    diff_series = diff_series.dropna()
    
    return diff_series

def exponential_smoothing_forecast(time_series, periods=10, seasonal_periods=None, 
                                   trend=None, seasonal=None, damped=False):
    """
    Forecast future values using Exponential Smoothing.
    
    Parameters
    ----------
    time_series : pandas.Series
        Time series data
    periods : int, optional
        Number of periods to forecast
    seasonal_periods : int, optional
        Number of periods in a seasonal cycle
    trend : str, optional
        Type of trend: None, 'add', or 'mul'
    seasonal : str, optional
        Type of seasonality: None, 'add', or 'mul'
    damped : bool, optional
        Whether to use damped trend
        
    Returns
    -------
    pandas.Series
        Forecasted values
    dict
        Model information and metrics
    """
    # Make a copy of the series and ensure it's a Series object
    ts = time_series.copy()
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts)
    
    # Handle missing values
    ts = ts.dropna()
    
    # Try to infer seasonal_periods if not provided
    if seasonal_periods is None and seasonal is not None:
        if isinstance(ts.index, pd.DatetimeIndex):
            freq = pd.infer_freq(ts.index)
            if freq:
                if 'D' in freq:  # Daily data
                    seasonal_periods = 7  # Weekly cycle
                elif 'M' in freq:  # Monthly data
                    seasonal_periods = 12  # Yearly cycle
                elif 'Q' in freq:  # Quarterly data
                    seasonal_periods = 4  # Yearly cycle
                elif 'H' in freq:  # Hourly data
                    seasonal_periods = 24  # Daily cycle
                else:
                    seasonal_periods = 4  # Default
        else:
            # Default value
            seasonal_periods = 4
    
    # Fit exponential smoothing model
    model = ExponentialSmoothing(
        ts,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        damped=damped
    )
    
    # Fit the model
    fit = model.fit()
    
    # Make forecast
    forecast = fit.forecast(periods)
    
    # Calculate error metrics for the in-sample predictions
    predictions = fit.fittedvalues
    
    # Handle case where predictions might have different index than original series
    common_index = ts.index.intersection(predictions.index)
    actual = ts.loc[common_index]
    pred = predictions.loc[common_index]
    
    # Calculate metrics
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, pred)
    
    # Prepare results
    model_info = {
        'model_params': {
            'trend': trend,
            'seasonal': seasonal,
            'seasonal_periods': seasonal_periods,
            'damped': damped
        },
        'metrics': {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae
        },
        'model': fit
    }
    
    return forecast, model_info

def detect_outliers(time_series, method='zscore', threshold=3):
    """
    Detect outliers in time series data.
    
    Parameters
    ----------
    time_series : pandas.Series
        Time series data
    method : str, optional
        Method for outlier detection: 'zscore', 'iqr', or 'rolling'
    threshold : float, optional
        Threshold for outlier detection
        
    Returns
    -------
    pandas.Series
        Boolean mask where True indicates outliers
    """
    # Make a copy and ensure it's a Series
    ts = time_series.copy()
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts)
    
    # Handle missing values
    ts = ts.dropna()
    
    if method == 'zscore':
        # Z-score method
        z_scores = np.abs(stats.zscore(ts))
        outliers = z_scores > threshold
        
    elif method == 'iqr':
        # IQR method
        q1 = ts.quantile(0.25)
        q3 = ts.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (ts < lower_bound) | (ts > upper_bound)
        
    elif method == 'rolling':
        # Rolling mean and standard deviation
        window = min(20, len(ts) // 5)  # Use a reasonable window size
        rolling_mean = ts.rolling(window=window, center=True).mean()
        rolling_std = ts.rolling(window=window, center=True).std()
        
        # For the beginning and end of the series where rolling stats might be NaN
        rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
        rolling_std = rolling_std.fillna(method='bfill').fillna(method='ffill')
        
        # Detect outliers
        outliers = np.abs(ts - rolling_mean) > threshold * rolling_std
        
    else:
        raise ValueError("Method must be 'zscore', 'iqr', or 'rolling'")
    
    # Convert to pandas Series with the same index as the original
    outliers = pd.Series(outliers, index=ts.index)
    
    return outliers

def find_best_parameters(time_series, test_size=0.2, 
                        trends=['add', 'mul', None], 
                        seasonals=['add', 'mul', None],
                        seasonal_periods=[4, 7, 12],
                        damped_options=[True, False]):
    """
    Find the best parameters for exponential smoothing by trying different combinations.
    
    Parameters
    ----------
    time_series : pandas.Series
        Time series data
    test_size : float, optional
        Proportion of data to use for testing
    trends : list, optional
        List of trend options to try
    seasonals : list, optional
        List of seasonal options to try
    seasonal_periods : list, optional
        List of seasonal periods to try
    damped_options : list, optional
        List of damped options to try
        
    Returns
    -------
    dict
        Best parameters and their corresponding metrics
    """
    # Make a copy and ensure it's a Series
    ts = time_series.copy()
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts)
    
    # Handle missing values
    ts = ts.dropna()
    
    # Split into train and test
    train_size = int(len(ts) * (1 - test_size))
    train = ts.iloc[:train_size]
    test = ts.iloc[train_size:]
    
    # Initialize best parameters
    best_params = None
    best_rmse = float('inf')
    
    # Try different combinations
    for trend in trends:
        for seasonal in seasonals:
            # Skip combinations that don't make sense
            if seasonal is None and seasonal_periods != [None]:
                seasonal_period_options = [None]
            else:
                seasonal_period_options = seasonal_periods
                
            for seasonal_period in seasonal_period_options:
                # Skip if we don't need seasonal_period
                if seasonal is None and seasonal_period is not None:
                    continue
                    
                for damped in damped_options:
                    # Skip damped if no trend
                    if trend is None and damped:
                        continue
                        
                    try:
                        # Fit model
                        model = ExponentialSmoothing(
                            train,
                            trend=trend,
                            seasonal=seasonal,
                            seasonal_periods=seasonal_period,
                            damped=damped
                        )
                        
                        fit = model.fit()
                        
                        # Forecast for test period
                        forecast = fit.forecast(len(test))
                        
                        # Calculate RMSE
                        rmse = np.sqrt(mean_squared_error(test, forecast))
                        
                        # Update best parameters if this is better
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = {
                                'trend': trend,
                                'seasonal': seasonal,
                                'seasonal_periods': seasonal_period,
                                'damped': damped,
                                'rmse': rmse
                            }
                    except:
                        # Skip if the combination causes an error
                        continue
    
    return best_params