# Python Data Analysis Toolkit 📊

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

**A modern Python library for data analysis, visualization, and machine learning.**

</div>

## 🌟 Overview

Python Data Analysis Toolkit (datoolkit) is a comprehensive library designed to streamline the data science workflow. It provides intuitive interfaces for statistical analysis, visualization, preprocessing, feature selection, time series analysis, and machine learning.

## ✨ Features

- **📈 Statistical Analysis** - Descriptive statistics and correlation analysis
- **🎨 Data Visualization** - Beautiful plots with sensible defaults
- **🧹 Data Preprocessing** - Handle missing values and normalize data
- **⏱️ Time Series Analysis** - Tools for analyzing and forecasting temporal data
- **🔍 Feature Selection** - Multiple methods to select optimal features
- **🧠 Machine Learning Utilities** - Cross-validation, feature importance, and more

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/manncodes/python-data-analysis-toolkit.git

# Install the package
cd python-data-analysis-toolkit
pip install -e .
```

## 🚀 Quick Start

```python
import pandas as pd
from datoolkit.stats import descriptive_stats
from datoolkit.visualization import plot_histogram
from datoolkit.preprocessing import normalization

# Load your data
data = pd.read_csv('your_data.csv')

# Get comprehensive statistics
stats = descriptive_stats(data['column_name'])
print(stats)

# Create an insightful visualization
fig, ax = plot_histogram(data['column_name'], show_stats=True)

# Preprocess your data
normalized_data, scaler = normalization(data, method='zscore')
```

## 📚 Documentation

### Statistical Analysis

```python
from datoolkit.stats import descriptive_stats, correlation_analysis

# Calculate descriptive statistics
stats = descriptive_stats(data['your_column'])

# Analyze correlations with p-values
corr_matrix, p_values = correlation_analysis(data)
```

### Data Visualization

```python
from datoolkit.visualization import plot_histogram, plot_correlation_matrix, plot_scatter

# Create a histogram with statistics
fig, ax = plot_histogram(data['column_name'], 
                        title='Distribution Analysis',
                        kde=True, 
                        show_stats=True)

# Visualize correlations with significance filtering
fig, ax = plot_correlation_matrix(corr_matrix, 
                                 p_values=p_values, 
                                 p_threshold=0.05)

# Create an enhanced scatter plot
fig, ax = plot_scatter(data['x_column'], data['y_column'],
                     add_reg_line=True)
```

### Data Preprocessing

```python
from datoolkit.preprocessing import normalization, handle_missing_values

# Normalize data with various methods
normalized_data, scaler = normalization(data, method='zscore')
# Available methods: 'zscore', 'minmax', 'robust', 'log', 'quantile'

# Handle missing values intelligently
cleaned_data, imputation_info = handle_missing_values(
    data, 
    strategy='mean',
    max_missing_threshold=0.3
)
```

### Feature Selection

```python
from datoolkit.feature_selection import select_features_by_score, remove_low_variance_features

# Select top features based on statistical tests
X_selected, selected_features, scores = select_features_by_score(
    X, y, method='f_test', k=10
)

# Remove features with low variance
X_reduced, kept_features, variances = remove_low_variance_features(
    X, threshold=0.01
)
```

### Time Series Analysis

```python
from datoolkit.time_series import check_stationarity, decompose_time_series, exponential_smoothing_forecast

# Test if your time series is stationary
is_stationary, test_results = check_stationarity(time_series_data)

# Decompose time series into trend, seasonal, and residual components
decomposition = decompose_time_series(time_series_data, period=12)

# Forecast future values
forecast, model_info = exponential_smoothing_forecast(
    time_series_data, 
    periods=10, 
    seasonal_periods=12
)
```

### Machine Learning Utilities

```python
from datoolkit.ml import train_test_split_stratified, cross_validation_metrics, feature_importance_analysis

# Split data with appropriate stratification
X_train, X_test, y_train, y_test = train_test_split_stratified(
    X, y, test_size=0.2
)

# Get comprehensive cross-validation metrics
cv_results = cross_validation_metrics(model, X, y, cv=5)

# Analyze feature importance
importance_df = feature_importance_analysis(model, X.columns)
```

## 📊 Example Notebooks

Check out the [`examples/`](./examples) directory for Jupyter notebooks demonstrating real-world usage:

- [`basic_usage.ipynb`](./examples/basic_usage.ipynb) - Getting started with basic functionality

## 🧮 Mathematical Background

The toolkit implements various statistical methods and algorithms:

### Normalization Methods

- **Z-score Normalization**:
  $z = \frac{x - \mu}{\sigma}$
  
- **Min-Max Scaling**:
  $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$
  
- **Robust Scaling**:
  $x_{robust} = \frac{x - \text{median}(x)}{\text{IQR}(x)}$

### Time Series Analysis

- **Augmented Dickey-Fuller Test** for stationarity
- **Seasonal Decomposition** into trend, seasonal, and residual components
- **Exponential Smoothing** for forecasting

### Feature Selection

- **F-test** and **Mutual Information** for feature ranking
- **Recursive Feature Elimination** for iterative feature selection
- **Variance Threshold** for removing low-variance features

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://scipy.org/)
- [statsmodels](https://www.statsmodels.org/)
