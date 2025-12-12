# Product Performance Analysis - Machine Learning Assignment

A comprehensive machine learning project analyzing supermarket product sales data using K-means clustering and regression techniques.

## Project Overview

This project implements:
- **Data Preprocessing**: Missing value handling, outlier detection, feature normalization
- **K-means Clustering**: From-scratch implementation to discover product groupings
- **Regression Analysis**: Linear and Polynomial regression to predict product profit

## Project Structure

```
assignment_data_mining/
├── source_code/
│   ├── main.py                    # Main analysis pipeline
│   ├── preprocessing.py           # Data preprocessing module
│   ├── kmeans.py                  # K-means clustering implementation
│   └── regression.py              # Regression analysis module
├── data/
│   ├── product_sales.csv          # Original dataset
│   ├── cleaned_data.csv           # Cleaned dataset (generated)
│   └── normalized_data.csv        # Normalized dataset (generated)
├── results/
│   └── (generated visualizations and statistics)
├── README.md                      # This file
└── assignment_description.pdf     # Assignment requirements
```

## Requirements

### Python Version
- Python 3.8 or higher

### Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

## Installation & Setup

### 1. Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd assignment_data_mining
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Verify Project Structure

Ensure the following directories exist:
- `source_code/`
- `data/`
- `results/`

If `results/` doesn't exist, create it:

```bash
mkdir results
```

## Usage

### Running the Complete Analysis

To run the entire analysis pipeline:

```bash
cd source_code
python main.py
```

This will:
1. Preprocess the data
2. Perform K-means clustering with elbow method
3. Train and compare regression models
4. Generate all visualizations
5. Save results to the `results/` directory

### Running Individual Components

#### Data Preprocessing Only

```bash
cd source_code
python preprocessing.py
```

#### K-means Clustering Only

```bash
cd source_code
python kmeans.py
```

#### Regression Analysis Only

```bash
cd source_code
python regression.py
```

## Dataset Information

**File**: `product_sales.csv`

**Features**:
- `product_id`: Unique product identifier
- `product_name`: Name of the product
- `category`: Product category (Dairy, Bakery, Produce, Meat, Beverages, Snacks)
- `price`: Retail price in dollars
- `cost`: Cost per unit in dollars
- `units_sold`: Number of units sold per month
- `promotion_frequency`: Number of promotions per month (0-4)
- `shelf_level`: Shelf position (1=bottom, 3=eye-level, 5=top)
- `profit`: Monthly profit in dollars **(TARGET VARIABLE)**

**Data Quality Issues** (intentional):
- Missing values: ~3-5 records
- Outliers: High-value specialty products
- Scale differences: Features have different ranges

## Key Features

### 1. Data Preprocessing
- **Missing Value Analysis**: Identifies and reports all missing values
- **Missing Value Handling**: Drops rows with missing product names, imputes numerical values with median
- **Outlier Detection**: Uses IQR method to detect outliers
- **Outlier Treatment**: Keeps outliers (justified for premium products)
- **Normalization**: Min-Max scaling (0-1) for K-means clustering

### 2. K-means Clustering (FROM SCRATCH)
- **Custom Implementation**: No sklearn.cluster.KMeans.fit() used
- **Initialization**: K-means++ for better initial centroids
- **Elbow Method**: Tests k=2 through k=8 to find optimal clusters
- **Cluster Analysis**: Comprehensive statistics for each cluster
- **Business Insights**: Meaningful cluster names and recommendations

### 3. Regression Analysis
- **Linear Regression**: Baseline model
- **Polynomial Regression**: Degree 2 and 3 for capturing non-linear relationships
- **Evaluation Metrics**: MSE, MAE, R² Score
- **Overfitting Detection**: Compares training vs testing performance
- **Visualizations**: Actual vs Predicted plots, Residual analysis

## Output Files

After running the analysis, the following files will be generated:

### Data Files (in `data/`)
- `cleaned_data.csv`: Preprocessed data
- `normalized_data.csv`: Normalized features

### Results Files (in `results/`)
- `preprocessing_analysis.png`: Data preprocessing visualizations
- `elbow_curve.png`: Elbow method for optimal k
- `cluster_visualization.png`: 2D scatter plot of clusters
- `cluster_statistics.csv`: Detailed cluster statistics
- `actual_vs_predicted_linear.png`: Linear regression predictions
- `actual_vs_predicted_polynomial_2.png`: Polynomial regression predictions
- `residuals_linear.png`: Residual analysis
- `model_comparison.png`: Comprehensive model comparison
- `model_comparison.csv`: Model performance metrics

## Understanding the Results

### Clustering Results
Look for the optimal k in the elbow curve where the WCSS decrease rate slows down significantly.

Each cluster will have:
- **Name**: Based on characteristics (e.g., "Budget Best-Sellers")
- **Statistics**: Average price, units sold, profit, etc.
- **Business Insights**: Actionable recommendations

### Regression Results
The model comparison shows:
- **MSE/MAE**: Lower is better (prediction error)
- **R² Score**: Higher is better (0-1 scale, measures fit quality)
- **Overfitting Gap**: Train R² - Test R² (should be < 0.1)

## Customization

### Changing Preprocessing Methods

Edit `main.py` line 27-36:

```python
# Use Z-score instead of IQR for outlier detection
outlier_info = preprocessor.detect_outliers(method='zscore', threshold=3)

# Remove outliers instead of keeping them
preprocessor.handle_outliers(method='remove', outlier_info=outlier_info)

# Use Z-score normalization instead of Min-Max
normalized_data, scaler_params = preprocessor.normalize_features(method='zscore')
```

### Changing Number of Clusters

Edit `main.py` line 64:

```python
optimal_k = 4  # Set your preferred k value
```

### Changing Regression Models

Edit `main.py` to add more polynomial degrees:

```python
poly_model_4, poly_features_4 = regression_analyzer.train_polynomial_regression(degree=4)
```

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Install missing packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Issue: "FileNotFoundError"
**Solution**: Make sure you're running the script from the `source_code/` directory:
```bash
cd source_code
python main.py
```

### Issue: "results/ directory not found"
**Solution**: Create the directory:
```bash
mkdir results
```

### Issue: Plots not displaying
**Solution**: The plots are saved to the `results/` directory. Open them manually or add `plt.show()` in the code.

## Algorithm Explanations

### K-means Algorithm Steps

1. **Initialize**: Choose k random points as initial centroids (or use K-means++)
2. **Assign**: Assign each point to the nearest centroid
3. **Update**: Calculate new centroids as the mean of assigned points
4. **Repeat**: Continue steps 2-3 until convergence (centroids don't move)
5. **Output**: Final cluster labels and centroids

### Elbow Method

The elbow method helps find optimal k by:
1. Running K-means for different k values (2, 3, 4, ..., 8)
2. Calculating WCSS (Within-Cluster Sum of Squares) for each k
3. Plotting k vs WCSS
4. Finding the "elbow" where WCSS decrease slows significantly

### Why Normalization is Necessary

K-means uses Euclidean distance. Without normalization:
- Features with large ranges (e.g., units_sold: 15-1450) dominate
- Features with small ranges (e.g., shelf_level: 1-5) have little impact
- Clusters are biased toward high-range features

Normalization ensures all features contribute equally.

## Performance Metrics Explained

### Mean Squared Error (MSE)
- Average of squared prediction errors
- Penalizes large errors more
- Lower is better

### Mean Absolute Error (MAE)
- Average of absolute prediction errors
- More interpretable than MSE
- Lower is better

### R² Score (Coefficient of Determination)
- Proportion of variance explained by the model
- Range: 0 to 1 (higher is better)
- 1.0 = perfect predictions
- 0.0 = model is no better than predicting the mean

## Future Improvements

- Add more regression models (Ridge, Lasso, Random Forest)
- Implement cross-validation for more robust evaluation
- Add feature importance analysis
- Create interactive dashboard with Streamlit
- Add time-series analysis if temporal data available

## Authors 

Iker Gonzalez Torre (6402702)
Francisco Ortiz Diaz (6438537)
