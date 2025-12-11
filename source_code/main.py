"""
Main Analysis Pipeline
This script runs the complete analysis pipeline:
1. Data Preprocessing
2. K-means Clustering
3. Regression Analysis
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from preprocessing import DataPreprocessor
from kmeans import KMeans, KMeansAnalyzer
from regression import RegressionAnalyzer

def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80 + "\n")

def main():
    """Run the complete analysis pipeline."""

    print_header("PRODUCT PERFORMANCE ANALYSIS - MACHINE LEARNING PROJECT")

    # ===================================================================
    # STEP 1: DATA PREPROCESSING
    # ===================================================================
    print_header("STEP 1: DATA PREPROCESSING")

    preprocessor = DataPreprocessor('../data/product_sales.csv')

    # Analyze missing values
    print("\n[1.1] Analyzing Missing Values...")
    missing_info = preprocessor.analyze_missing_values()

    # Handle missing values
    print("\n[1.2] Handling Missing Values...")
    preprocessor.handle_missing_values()

    # Detect outliers
    print("\n[1.3] Detecting Outliers...")
    outlier_info = preprocessor.detect_outliers(method='iqr', threshold=1.5)

    # Handle outliers (keeping them for this dataset)
    print("\n[1.4] Handling Outliers...")
    preprocessor.handle_outliers(method='keep', outlier_info=outlier_info)

    # Normalize features
    print("\n[1.5] Normalizing Features...")
    normalized_data, scaler_params = preprocessor.normalize_features(method='minmax')

    # Generate preprocessing visualizations
    print("\n[1.6] Generating Preprocessing Visualizations...")
    preprocessor.visualize_preprocessing()

    # Save preprocessed data
    preprocessor.data.to_csv('../data/cleaned_data.csv', index=False)
    normalized_data.to_csv('../data/normalized_data.csv', index=False)

    print("\n✓ Data Preprocessing Complete!")
    print(f"  - Cleaned data saved to: ../data/cleaned_data.csv")
    print(f"  - Normalized data saved to: ../data/normalized_data.csv")

    # ===================================================================
    # STEP 2: K-MEANS CLUSTERING
    # ===================================================================
    print_header("STEP 2: K-MEANS CLUSTERING ANALYSIS")

    # Select features for clustering
    feature_cols = ['price', 'cost', 'units_sold', 'promotion_frequency', 'shelf_level']

    # Initialize analyzer
    print("\n[2.1] Initializing K-means Analyzer...")
    kmeans_analyzer = KMeansAnalyzer(
        data=normalized_data,
        feature_columns=feature_cols,
        original_data=preprocessor.data
    )

    # Perform elbow method
    print("\n[2.2] Performing Elbow Method Analysis...")
    elbow_results = kmeans_analyzer.elbow_method(k_range=range(2, 9))
    kmeans_analyzer.plot_elbow_curve(elbow_results)

    # Based on elbow curve, choose optimal k
    print("\n[2.3] Selecting Optimal k...")
    wcss_values = elbow_results['wcss_values']
    k_values = elbow_results['k_values']

    # Calculate rate of decrease
    decreases = [wcss_values[i] - wcss_values[i+1] for i in range(len(wcss_values)-1)]
    rate_of_decrease = [decreases[i] - decreases[i+1] for i in range(len(decreases)-1)]

    # Find elbow (maximum rate of decrease change)
    optimal_k = k_values[rate_of_decrease.index(max(rate_of_decrease)) + 2]
    print(f"\n✓ Optimal k determined: {optimal_k}")

    # Fit K-means with optimal k
    print(f"\n[2.4] Fitting K-means with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(kmeans_analyzer.X)

    # Analyze clusters
    print("\n[2.5] Analyzing Clusters...")

    # Generate cluster names based on characteristics
    temp_stats = []
    for k in range(optimal_k):
        cluster_data = preprocessor.data[kmeans.labels == k]
        temp_stats.append({
            'avg_price': cluster_data['price'].mean(),
            'avg_units': cluster_data['units_sold'].mean(),
            'avg_profit': cluster_data['profit'].mean()
        })

    # Create meaningful names
    cluster_names = []
    for i, stats in enumerate(temp_stats):
        if stats['avg_price'] < 4 and stats['avg_units'] > 600:
            cluster_names.append("Budget Best-Sellers")
        elif stats['avg_price'] > 10:
            cluster_names.append("Premium Specialty")
        elif stats['avg_units'] > 600:
            cluster_names.append("High-Volume Products")
        elif stats['avg_profit'] > 700:
            cluster_names.append("High-Profit Items")
        else:
            cluster_names.append(f"Mid-Range Products")

    stats_df = kmeans_analyzer.analyze_clusters(kmeans, cluster_names=cluster_names)

    # Visualize clusters
    print("\n[2.6] Generating Cluster Visualizations...")
    kmeans_analyzer.visualize_clusters(
        kmeans,
        feature_x='price',
        feature_y='units_sold',
        cluster_names=cluster_names
    )

    # Save results
    stats_df.to_csv('../results/cluster_statistics.csv', index=False)
    print("\n✓ K-means Clustering Complete!")
    print(f"  - Cluster statistics saved to: ../results/cluster_statistics.csv")

    # ===================================================================
    # STEP 3: REGRESSION ANALYSIS
    # ===================================================================
    print_header("STEP 3: REGRESSION ANALYSIS")

    # Initialize regression analyzer
    print("\n[3.1] Initializing Regression Analyzer...")
    target_col = 'profit'
    regression_analyzer = RegressionAnalyzer(
        data=preprocessor.data,
        feature_columns=feature_cols,
        target_column=target_col
    )

    # Split data
    print("\n[3.2] Splitting Data...")
    regression_analyzer.split_data(test_size=0.3, random_state=42)

    # Train Linear Regression
    print("\n[3.3] Training Linear Regression...")
    linear_model = regression_analyzer.train_linear_regression()

    # Train Polynomial Regression (degree 2)
    print("\n[3.4] Training Polynomial Regression (degree 2)...")
    poly_model_2, poly_features_2 = regression_analyzer.train_polynomial_regression(degree=2)

    # Optional: Train Polynomial Regression (degree 3)
    print("\n[3.5] Training Polynomial Regression (degree 3)...")
    poly_model_3, poly_features_3 = regression_analyzer.train_polynomial_regression(degree=3)

    # Compare models
    print("\n[3.6] Comparing Models...")
    comparison_df = regression_analyzer.compare_models()

    # Generate visualizations
    print("\n[3.7] Generating Regression Visualizations...")
    regression_analyzer.plot_actual_vs_predicted('linear')
    regression_analyzer.plot_actual_vs_predicted('polynomial_2')
    regression_analyzer.plot_residuals('linear')
    regression_analyzer.plot_model_comparison(comparison_df)

    # Save results
    comparison_df.to_csv('../results/model_comparison.csv', index=False)
    print("\n✓ Regression Analysis Complete!")
    print(f"  - Model comparison saved to: ../results/model_comparison.csv")

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print_header("ANALYSIS COMPLETE - SUMMARY")

    print("Data Preprocessing:")
    print(f"  ✓ Original records: {len(preprocessor.raw_data)}")
    print(f"  ✓ Final records: {len(preprocessor.data)}")
    print(f"  ✓ Features normalized: {len(feature_cols)}")

    print("\nK-means Clustering:")
    print(f"  ✓ Optimal clusters: {optimal_k}")
    print(f"  ✓ WCSS: {kmeans.wcss:.2f}")
    print(f"  ✓ Iterations: {kmeans.n_iterations}")

    print("\nRegression Analysis:")
    best_model_idx = comparison_df['Test MSE'].idxmin()
    best_model = comparison_df.loc[best_model_idx, 'Model']
    best_mse = comparison_df.loc[best_model_idx, 'Test MSE']
    best_r2 = comparison_df.loc[best_model_idx, 'Test R²']
    print(f"  ✓ Best Model: {best_model}")
    print(f"  ✓ Test MSE: {best_mse:.2f}")
    print(f"  ✓ Test R²: {best_r2:.4f}")

    print("\nAll results saved to: ../results/")
    print("\n" + "=" * 80)
    print("Thank you for using the Product Performance Analysis tool!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
