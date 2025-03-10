# Python Data Analysis Toolkit

A comprehensive Python library for data analysis, visualization, and machine learning tasks. This toolkit provides a collection of utility functions to streamline common data science workflows.

## Features

- **Statistical Analysis**: Calculate descriptive statistics and correlation analysis
- **Data Visualization**: Create informative plots with sensible defaults
- **Data Preprocessing**: Handle missing values and normalize data
- **Machine Learning Utilities**: Cross-validation, feature importance, and more

## Installation

```bash
# Clone the repository
git clone https://github.com/manncodes/python-data-analysis-toolkit.git

# Install the package
cd python-data-analysis-toolkit
pip install -e .
```

## Usage Examples

### Statistical Analysis

```python
import pandas as pd
from datoolkit.stats import descriptive_stats, correlation_analysis

# Load your data
data = pd.read_csv('your_data.csv')

# Calculate descriptive statistics
stats = descriptive_stats(data['your_column'])
print(stats)

# Correlation analysis
corr_matrix, p_values = correlation_analysis(data)
```

### Data Visualization

```python
import pandas as pd
from datoolkit.visualization import plot_histogram, plot_correlation_matrix, plot_scatter

# Load your data
data = pd.read_csv('your_data.csv')

# Create a histogram
fig, ax = plot_histogram(data['column_name'], bins=20, 
                         title='Distribution of Values')

# Create a correlation matrix visualization
corr_matrix = data.corr()
fig, ax = plot_correlation_matrix(corr_matrix, title='Feature Correlations')

# Create a scatter plot with regression line
fig, ax = plot_scatter(data['x_column'], data['y_column'], 
                      title='Relationship between X and Y')
```

### Data Preprocessing

```python
import pandas as pd
from datoolkit.preprocessing import normalization, handle_missing_values

# Load your data
data = pd.read_csv('your_data.csv')

# Normalize data
normalized_data, scaler = normalization(data, method='zscore')

# Handle missing values
cleaned_data, imputation_info = handle_missing_values(
    data, strategy='mean', max_missing_threshold=0.3
)
```

### Machine Learning Utilities

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datoolkit.ml import train_test_split_stratified, cross_validation_metrics, feature_importance_analysis

# Load your data
data = pd.read_csv('your_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split_stratified(
    X, y, test_size=0.2, random_state=42
)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate with cross-validation
cv_results = cross_validation_metrics(model, X, y, cv=5)
print(cv_results)

# Analyze feature importance
importance_df = feature_importance_analysis(model, X.columns)
print(importance_df)
```

## Mathematical Background

### Normalization Methods

The toolkit implements several normalization techniques:

1. **Z-score Normalization**:
   
   $$z = \frac{x - \mu}{\sigma}$$
   
   Where $\mu$ is the mean and $\sigma$ is the standard deviation.

2. **Min-Max Scaling**:
   
   $$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

3. **Robust Scaling**:
   
   $$x_{robust} = \frac{x - median(x)}{IQR(x)}$$
   
   Where $IQR$ is the interquartile range.

4. **Log Transformation**:
   
   $$x_{log} = \log(x + shift)$$
   
   Where $shift$ is added to handle zeros or negative values.

### Correlation Analysis

The correlation coefficient $r$ between two variables $X$ and $Y$ is calculated as:

$$r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

### Cross-Validation Metrics

For classification tasks, the following metrics are computed:

- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1 Score**: $2 \cdot \frac{precision \cdot recall}{precision + recall}$

For regression tasks:

- **Mean Squared Error**: $\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
- **Mean Absolute Error**: $\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
- **R-squared**: $1 - \frac{\sum_{i} (y_i - \hat{y}_i)^2}{\sum_{i} (y_i - \bar{y})^2}$

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
