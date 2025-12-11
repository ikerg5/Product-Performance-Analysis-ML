"""
Regression Analysis Module
This module implements regression models to predict product profit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RegressionAnalyzer:
    """
    A class to perform regression analysis on product data.
    Implements Linear and Polynomial Regression models.
    """

    def __init__(self, data, feature_columns, target_column='profit'):
        """
        Initialize the regression analyzer.

        Parameters:
        data (pd.DataFrame): Cleaned data
        feature_columns (list): List of feature column names
        target_column (str): Target variable name
        """
        self.data = data
        self.feature_columns = feature_columns
        self.target_column = target_column

        self.X = data[feature_columns]
        self.y = data[target_column]

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.models = {}
        self.predictions = {}
        self.metrics = {}

    def split_data(self, test_size=0.3, random_state=42):
        """
        Split data into training and testing sets.

        Parameters:
        test_size (float): Proportion of data for testing
        random_state (int): Random seed

        Returns:
        tuple: X_train, X_test, y_train, y_test
        """
        print("=" * 60)
        print("DATA SPLITTING")
        print("=" * 60)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        print(f"\nTotal samples: {len(self.X)}")
        print(f"Training samples: {len(self.X_train)} ({(1-test_size)*100:.0f}%)")
        print(f"Testing samples: {len(self.X_test)} ({test_size*100:.0f}%)")
        print(f"\nFeatures used: {', '.join(self.feature_columns)}")
        print(f"Target variable: {self.target_column}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_linear_regression(self):
        """
        Train a Linear Regression model.

        Returns:
        LinearRegression: Trained model
        """
        print("\n" + "=" * 60)
        print("LINEAR REGRESSION")
        print("=" * 60)

        # Create and train model
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        # Store model and predictions
        self.models['linear'] = model
        self.predictions['linear'] = {
            'train': y_pred_train,
            'test': y_pred_test
        }

        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, y_pred_train)
        test_mse = mean_squared_error(self.y_test, y_pred_test)
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)

        self.metrics['linear'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': np.sqrt(train_mse),
            'test_rmse': np.sqrt(test_mse)
        }

        # Print results
        print("\nModel Coefficients:")
        for feature, coef in zip(self.feature_columns, model.coef_):
            print(f"  {feature}: {coef:.4f}")
        print(f"  Intercept: {model.intercept_:.4f}")

        print("\nTraining Performance:")
        print(f"  MSE: {train_mse:.2f}")
        print(f"  RMSE: {np.sqrt(train_mse):.2f}")
        print(f"  MAE: {train_mae:.2f}")
        print(f"  R² Score: {train_r2:.4f}")

        print("\nTesting Performance:")
        print(f"  MSE: {test_mse:.2f}")
        print(f"  RMSE: {np.sqrt(test_mse):.2f}")
        print(f"  MAE: {test_mae:.2f}")
        print(f"  R² Score: {test_r2:.4f}")

        return model

    def train_polynomial_regression(self, degree=2):
        """
        Train a Polynomial Regression model.

        Parameters:
        degree (int): Degree of polynomial features

        Returns:
        tuple: (model, poly_features)
        """
        print("\n" + "=" * 60)
        print(f"POLYNOMIAL REGRESSION (degree={degree})")
        print("=" * 60)

        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(self.X_train)
        X_test_poly = poly.transform(self.X_test)

        print(f"\nOriginal features: {self.X_train.shape[1]}")
        print(f"Polynomial features: {X_train_poly.shape[1]}")

        # Create and train model
        model = LinearRegression()
        model.fit(X_train_poly, self.y_train)

        # Make predictions
        y_pred_train = model.predict(X_train_poly)
        y_pred_test = model.predict(X_test_poly)

        # Store model and predictions
        self.models[f'polynomial_{degree}'] = {
            'model': model,
            'poly_features': poly
        }
        self.predictions[f'polynomial_{degree}'] = {
            'train': y_pred_train,
            'test': y_pred_test
        }

        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, y_pred_train)
        test_mse = mean_squared_error(self.y_test, y_pred_test)
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)

        self.metrics[f'polynomial_{degree}'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': np.sqrt(train_mse),
            'test_rmse': np.sqrt(test_mse),
            'degree': degree
        }

        # Print results
        print("\nTraining Performance:")
        print(f"  MSE: {train_mse:.2f}")
        print(f"  RMSE: {np.sqrt(train_mse):.2f}")
        print(f"  MAE: {train_mae:.2f}")
        print(f"  R² Score: {train_r2:.4f}")

        print("\nTesting Performance:")
        print(f"  MSE: {test_mse:.2f}")
        print(f"  RMSE: {np.sqrt(test_mse):.2f}")
        print(f"  MAE: {test_mae:.2f}")
        print(f"  R² Score: {test_r2:.4f}")

        # Check for overfitting
        if train_r2 - test_r2 > 0.1:
            print("\n⚠️  Warning: Possible overfitting detected!")
            print(f"   Training R² ({train_r2:.4f}) is significantly higher than Testing R² ({test_r2:.4f})")

        return model, poly

    def compare_models(self):
        """
        Compare performance of all trained models.

        Returns:
        pd.DataFrame: Comparison table
        """
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        comparison_data = []

        for model_name, metrics in self.metrics.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Train MSE': metrics['train_mse'],
                'Test MSE': metrics['test_mse'],
                'Train MAE': metrics['train_mae'],
                'Test MAE': metrics['test_mae'],
                'Train R²': metrics['train_r2'],
                'Test R²': metrics['test_r2'],
                'Overfitting Gap': metrics['train_r2'] - metrics['test_r2']
            })

        comparison_df = pd.DataFrame(comparison_data)

        print("\n", comparison_df.to_string(index=False))

        # Determine best model
        best_model_idx = comparison_df['Test MSE'].idxmin()
        best_model = comparison_df.loc[best_model_idx, 'Model']

        print(f"\n✓ Best Model: {best_model}")
        print(f"  Test MSE: {comparison_df.loc[best_model_idx, 'Test MSE']:.2f}")
        print(f"  Test MAE: {comparison_df.loc[best_model_idx, 'Test MAE']:.2f}")
        print(f"  Test R²: {comparison_df.loc[best_model_idx, 'Test R²']:.4f}")

        return comparison_df

    def plot_actual_vs_predicted(self, model_name='linear', save_path='../results/'):
        """
        Plot actual vs predicted values.

        Parameters:
        model_name (str): Name of model to plot
        save_path (str): Path to save plot
        """
        if model_name not in self.predictions:
            print(f"Model '{model_name}' not found. Available models: {list(self.predictions.keys())}")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Training data plot
        y_pred_train = self.predictions[model_name]['train']
        ax1.scatter(self.y_train, y_pred_train, alpha=0.6, s=50, edgecolors='black')
        ax1.plot([self.y_train.min(), self.y_train.max()],
                [self.y_train.min(), self.y_train.max()],
                'r--', linewidth=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Profit ($)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted Profit ($)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Training Set: Actual vs Predicted\n{model_name.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add R² score annotation
        train_r2 = self.metrics[model_name]['train_r2']
        ax1.text(0.05, 0.95, f'R² = {train_r2:.4f}',
                transform=ax1.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Testing data plot
        y_pred_test = self.predictions[model_name]['test']
        ax2.scatter(self.y_test, y_pred_test, alpha=0.6, s=50, edgecolors='black', c='green')
        ax2.plot([self.y_test.min(), self.y_test.max()],
                [self.y_test.min(), self.y_test.max()],
                'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Profit ($)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Predicted Profit ($)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Testing Set: Actual vs Predicted\n{model_name.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add R² score annotation
        test_r2 = self.metrics[model_name]['test_r2']
        ax2.text(0.05, 0.95, f'R² = {test_r2:.4f}',
                transform=ax2.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f"{save_path}actual_vs_predicted_{model_name}.png", dpi=300, bbox_inches='tight')
        print(f"\n✓ Actual vs Predicted plot saved to {save_path}actual_vs_predicted_{model_name}.png")

        return fig

    def plot_residuals(self, model_name='linear', save_path='../results/'):
        """
        Plot residual analysis.

        Parameters:
        model_name (str): Name of model to analyze
        save_path (str): Path to save plot
        """
        if model_name not in self.predictions:
            print(f"Model '{model_name}' not found.")
            return

        y_pred_test = self.predictions[model_name]['test']
        residuals = self.y_test - y_pred_test

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Residual scatter plot
        ax1.scatter(y_pred_test, residuals, alpha=0.6, s=50, edgecolors='black')
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Profit ($)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Residuals ($)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Residual Plot\n{model_name.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Residual histogram
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Residuals ($)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title(f'Residual Distribution\n{model_name.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{save_path}residuals_{model_name}.png", dpi=300, bbox_inches='tight')
        print(f"✓ Residual plot saved to {save_path}residuals_{model_name}.png")

        return fig

    def plot_model_comparison(self, comparison_df, save_path='../results/'):
        """
        Create visualization comparing all models.

        Parameters:
        comparison_df (pd.DataFrame): Comparison dataframe from compare_models()
        save_path (str): Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # MSE Comparison
        ax1 = axes[0, 0]
        x = np.arange(len(comparison_df))
        width = 0.35
        ax1.bar(x - width/2, comparison_df['Train MSE'], width, label='Train MSE', alpha=0.8)
        ax1.bar(x + width/2, comparison_df['Test MSE'], width, label='Test MSE', alpha=0.8)
        ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax1.set_ylabel('MSE', fontsize=12, fontweight='bold')
        ax1.set_title('Mean Squared Error Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # MAE Comparison
        ax2 = axes[0, 1]
        ax2.bar(x - width/2, comparison_df['Train MAE'], width, label='Train MAE', alpha=0.8)
        ax2.bar(x + width/2, comparison_df['Test MAE'], width, label='Test MAE', alpha=0.8)
        ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
        ax2.set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # R² Comparison
        ax3 = axes[1, 0]
        ax3.bar(x - width/2, comparison_df['Train R²'], width, label='Train R²', alpha=0.8)
        ax3.bar(x + width/2, comparison_df['Test R²'], width, label='Test R²', alpha=0.8)
        ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax3.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax3.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Overfitting Gap
        ax4 = axes[1, 1]
        colors = ['green' if gap < 0.1 else 'orange' if gap < 0.2 else 'red'
                 for gap in comparison_df['Overfitting Gap']]
        ax4.bar(x, comparison_df['Overfitting Gap'], color=colors, alpha=0.8)
        ax4.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, label='Warning Threshold')
        ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Overfitting Gap (Train R² - Test R²)', fontsize=12, fontweight='bold')
        ax4.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{save_path}model_comparison.png", dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison plot saved to {save_path}model_comparison.png")

        return fig


# Example usage
if __name__ == "__main__":
    # Load cleaned data
    data = pd.read_csv('../data/cleaned_data.csv')

    # Define features (exclude target variable)
    feature_cols = ['price', 'cost', 'units_sold', 'promotion_frequency', 'shelf_level']
    target_col = 'profit'

    # Initialize analyzer
    analyzer = RegressionAnalyzer(
        data=data,
        feature_columns=feature_cols,
        target_column=target_col
    )

    # Split data
    analyzer.split_data(test_size=0.3, random_state=42)

    # Train Linear Regression
    linear_model = analyzer.train_linear_regression()

    # Train Polynomial Regression (degree 2)
    poly_model_2, poly_features_2 = analyzer.train_polynomial_regression(degree=2)

    # Train Polynomial Regression (degree 3) - optional
    poly_model_3, poly_features_3 = analyzer.train_polynomial_regression(degree=3)

    # Compare models
    comparison_df = analyzer.compare_models()

    # Visualizations
    analyzer.plot_actual_vs_predicted('linear')
    analyzer.plot_actual_vs_predicted('polynomial_2')
    analyzer.plot_residuals('linear')
    analyzer.plot_model_comparison(comparison_df)

    # Save comparison results
    comparison_df.to_csv('../results/model_comparison.csv', index=False)
    print("\n✓ Model comparison saved to '../results/model_comparison.csv'")
