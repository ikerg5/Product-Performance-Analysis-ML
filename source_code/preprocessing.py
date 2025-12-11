"""
Data Preprocessing Module
This module handles data cleaning, missing value imputation, outlier detection, and normalization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    """
    A class to handle all data preprocessing tasks including:
    - Missing value analysis and handling
    - Outlier detection and treatment
    - Feature normalization/standardization
    """

    def __init__(self, file_path):
        """
        Initialize the preprocessor with data from CSV file.

        Parameters:
        file_path (str): Path to the CSV file
        """
        self.raw_data = pd.read_csv(file_path)
        self.data = self.raw_data.copy()
        self.preprocessing_report = {}

    def analyze_missing_values(self):
        """
        Analyze and report missing values in the dataset.

        Returns:
        dict: Summary of missing values per column
        """
        print("=" * 60)
        print("MISSING VALUE ANALYSIS")
        print("=" * 60)

        missing_summary = {}
        total_rows = len(self.data)

        for column in self.data.columns:
            missing_count = self.data[column].isnull().sum()
            missing_percentage = (missing_count / total_rows) * 100

            if missing_count > 0:
                missing_summary[column] = {
                    'count': missing_count,
                    'percentage': missing_percentage
                }
                print(f"\n{column}:")
                print(f"  Missing: {missing_count} ({missing_percentage:.2f}%)")

        if not missing_summary:
            print("\nNo missing values found!")

        self.preprocessing_report['missing_values'] = missing_summary
        return missing_summary

    def handle_missing_values(self, strategy='auto'):
        """
        Handle missing values based on the specified strategy.

        Parameters:
        strategy (str): 'auto', 'drop', 'mean', 'median', 'mode', or 'forward_fill'

        Returns:
        pd.DataFrame: Data with missing values handled
        """
        print("\n" + "=" * 60)
        print("HANDLING MISSING VALUES")
        print("=" * 60)

        initial_rows = len(self.data)

        # For product_name, we'll drop rows with missing names
        if self.data['product_name'].isnull().any():
            missing_names = self.data['product_name'].isnull().sum()
            print(f"\nDropping {missing_names} rows with missing product names...")
            self.data = self.data.dropna(subset=['product_name'])

        # For numerical columns, impute with median (more robust to outliers)
        numerical_cols = ['price', 'cost', 'units_sold', 'promotion_frequency',
                         'shelf_level', 'profit']

        for col in numerical_cols:
            if self.data[col].isnull().any():
                median_value = self.data[col].median()
                missing_count = self.data[col].isnull().sum()
                print(f"\nImputing {missing_count} missing values in '{col}' with median: {median_value:.2f}")
                self.data[col].fillna(median_value, inplace=True)

        final_rows = len(self.data)
        print(f"\nRows before: {initial_rows}")
        print(f"Rows after: {final_rows}")
        print(f"Rows dropped: {initial_rows - final_rows}")

        return self.data

    def detect_outliers(self, method='iqr', threshold=1.5):
        """
        Detect outliers using IQR or Z-score method.

        Parameters:
        method (str): 'iqr' or 'zscore'
        threshold (float): IQR multiplier (default 1.5) or Z-score threshold (default 3)

        Returns:
        dict: Dictionary containing outlier information for each numerical column
        """
        print("\n" + "=" * 60)
        print(f"OUTLIER DETECTION ({method.upper()} method)")
        print("=" * 60)

        numerical_cols = ['price', 'cost', 'units_sold', 'promotion_frequency',
                         'shelf_level', 'profit']

        outlier_info = {}

        for col in numerical_cols:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers = self.data[(self.data[col] < lower_bound) |
                                    (self.data[col] > upper_bound)]

                outlier_info[col] = {
                    'method': 'IQR',
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'count': len(outliers),
                    'indices': outliers.index.tolist()
                }

                print(f"\n{col}:")
                print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
                print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                print(f"  Outliers: {len(outliers)}")

            elif method == 'zscore':
                mean = self.data[col].mean()
                std = self.data[col].std()
                z_scores = np.abs((self.data[col] - mean) / std)

                outliers = self.data[z_scores > threshold]

                outlier_info[col] = {
                    'method': 'Z-score',
                    'threshold': threshold,
                    'count': len(outliers),
                    'indices': outliers.index.tolist()
                }

                print(f"\n{col}:")
                print(f"  Mean: {mean:.2f}, Std: {std:.2f}")
                print(f"  Z-score threshold: {threshold}")
                print(f"  Outliers: {len(outliers)}")

        self.preprocessing_report['outliers'] = outlier_info
        return outlier_info

    def handle_outliers(self, method='keep', outlier_info=None):
        """
        Handle outliers based on specified method.

        Parameters:
        method (str): 'keep', 'remove', or 'cap'
        outlier_info (dict): Output from detect_outliers()

        Returns:
        pd.DataFrame: Data with outliers handled
        """
        print("\n" + "=" * 60)
        print(f"HANDLING OUTLIERS (Method: {method})")
        print("=" * 60)

        if method == 'keep':
            print("\nKeeping all outliers - they may represent legitimate high-value products")
            return self.data

        if outlier_info is None:
            print("No outlier information provided. Run detect_outliers() first.")
            return self.data

        initial_rows = len(self.data)

        if method == 'remove':
            # Collect all outlier indices
            all_outlier_indices = set()
            for col, info in outlier_info.items():
                all_outlier_indices.update(info['indices'])

            print(f"\nRemoving {len(all_outlier_indices)} rows with outliers...")
            self.data = self.data.drop(index=list(all_outlier_indices))
            self.data = self.data.reset_index(drop=True)

        elif method == 'cap':
            for col, info in outlier_info.items():
                if info['method'] == 'IQR':
                    lower = info['lower_bound']
                    upper = info['upper_bound']

                    capped_lower = (self.data[col] < lower).sum()
                    capped_upper = (self.data[col] > upper).sum()

                    self.data[col] = self.data[col].clip(lower=lower, upper=upper)

                    print(f"\n{col}: Capped {capped_lower} low outliers and {capped_upper} high outliers")

        final_rows = len(self.data)
        print(f"\nRows before: {initial_rows}")
        print(f"Rows after: {final_rows}")

        return self.data

    def normalize_features(self, method='minmax', exclude_cols=None):
        """
        Normalize numerical features.

        Parameters:
        method (str): 'minmax' (0-1 scaling) or 'zscore' (standardization)
        exclude_cols (list): Columns to exclude from normalization

        Returns:
        tuple: (normalized_data, scaler_params)
        """
        print("\n" + "=" * 60)
        print(f"FEATURE NORMALIZATION ({method.upper()})")
        print("=" * 60)

        if exclude_cols is None:
            exclude_cols = ['product_id']

        # Select numerical columns to normalize
        numerical_cols = ['price', 'cost', 'units_sold', 'promotion_frequency',
                         'shelf_level', 'profit']

        cols_to_normalize = [col for col in numerical_cols if col not in exclude_cols]

        self.normalized_data = self.data.copy()
        scaler_params = {}

        for col in cols_to_normalize:
            if method == 'minmax':
                min_val = self.data[col].min()
                max_val = self.data[col].max()

                self.normalized_data[col] = (self.data[col] - min_val) / (max_val - min_val)

                scaler_params[col] = {
                    'method': 'minmax',
                    'min': min_val,
                    'max': max_val
                }

                print(f"\n{col}:")
                print(f"  Original range: [{min_val:.2f}, {max_val:.2f}]")
                print(f"  Normalized range: [0.00, 1.00]")

            elif method == 'zscore':
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()

                self.normalized_data[col] = (self.data[col] - mean_val) / std_val

                scaler_params[col] = {
                    'method': 'zscore',
                    'mean': mean_val,
                    'std': std_val
                }

                print(f"\n{col}:")
                print(f"  Mean: {mean_val:.2f}, Std: {std_val:.2f}")
                print(f"  Standardized: mean=0, std=1")

        print(f"\n✓ Normalized {len(cols_to_normalize)} features using {method} method")

        self.scaler_params = scaler_params
        return self.normalized_data, scaler_params

    def get_preprocessing_summary(self):
        """
        Generate a comprehensive preprocessing summary.

        Returns:
        dict: Summary of all preprocessing steps
        """
        summary = {
            'original_rows': len(self.raw_data),
            'final_rows': len(self.data),
            'original_columns': len(self.raw_data.columns),
            'final_columns': len(self.data.columns),
            'preprocessing_steps': self.preprocessing_report
        }

        return summary

    def visualize_preprocessing(self, save_path='../results/'):
        """
        Create visualizations showing the preprocessing effects.

        Parameters:
        save_path (str): Path to save visualization plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Missing values heatmap
        ax1 = axes[0, 0]
        sns.heatmap(self.raw_data.isnull(), cbar=False, yticklabels=False,
                   cmap='viridis', ax=ax1)
        ax1.set_title('Missing Values in Original Data', fontsize=14, fontweight='bold')

        # 2. Distribution comparison (before/after)
        ax2 = axes[0, 1]
        ax2.hist(self.raw_data['profit'].dropna(), bins=30, alpha=0.5, label='Original', color='blue')
        ax2.hist(self.data['profit'], bins=30, alpha=0.5, label='Cleaned', color='green')
        ax2.set_title('Profit Distribution: Before vs After', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Profit ($)')
        ax2.set_ylabel('Frequency')
        ax2.legend()

        # 3. Outlier detection box plots
        ax3 = axes[1, 0]
        self.data[['price', 'units_sold', 'profit']].boxplot(ax=ax3)
        ax3.set_title('Box Plots for Outlier Detection', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Value')

        # 4. Correlation heatmap
        ax4 = axes[1, 1]
        numerical_data = self.data[['price', 'cost', 'units_sold',
                                    'promotion_frequency', 'shelf_level', 'profit']]
        correlation = numerical_data.corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=ax4, square=True)
        ax4.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{save_path}preprocessing_analysis.png", dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to {save_path}preprocessing_analysis.png")

        return fig


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor('../data/product_sales.csv')

    # Step 1: Analyze missing values
    missing_info = preprocessor.analyze_missing_values()

    # Step 2: Handle missing values
    preprocessor.handle_missing_values()

    # Step 3: Detect outliers
    outlier_info = preprocessor.detect_outliers(method='iqr', threshold=1.5)

    # Step 4: Handle outliers (keeping them as they may represent premium products)
    preprocessor.handle_outliers(method='keep', outlier_info=outlier_info)

    # Step 5: Normalize features
    normalized_data, scaler_params = preprocessor.normalize_features(method='minmax')

    # Step 6: Get summary
    summary = preprocessor.get_preprocessing_summary()
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Original rows: {summary['original_rows']}")
    print(f"Final rows: {summary['final_rows']}")
    print(f"Rows removed: {summary['original_rows'] - summary['final_rows']}")

    # Step 7: Visualize
    preprocessor.visualize_preprocessing()

    # Save cleaned data
    preprocessor.data.to_csv('../data/cleaned_data.csv', index=False)
    normalized_data.to_csv('../data/normalized_data.csv', index=False)
    print("\n✓ Cleaned data saved to '../data/cleaned_data.csv'")
    print("✓ Normalized data saved to '../data/normalized_data.csv'")
