"""
K-means Clustering Implementation
This module implements K-means clustering algorithm from scratch.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class KMeans:
    """
    K-means clustering algorithm implemented from scratch.

    This implementation includes:
    - Random and K-means++ initialization
    - Iterative cluster assignment and centroid update
    - Convergence detection
    - WCSS calculation for elbow method
    """

    def __init__(self, n_clusters=3, max_iterations=300, tolerance=1e-4,
                 init_method='kmeans++', random_state=42):
        """
        Initialize K-means clustering algorithm.

        Parameters:
        n_clusters (int): Number of clusters (k)
        max_iterations (int): Maximum number of iterations
        tolerance (float): Convergence threshold
        init_method (str): 'random' or 'kmeans++'
        random_state (int): Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.init_method = init_method
        self.random_state = random_state

        self.centroids = None
        self.labels = None
        self.wcss = None
        self.n_iterations = 0

    def _initialize_centroids(self, X):
        """
        Initialize centroids using specified method.

        Parameters:
        X (np.array): Data matrix (n_samples, n_features)

        Returns:
        np.array: Initial centroids (n_clusters, n_features)
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        if self.init_method == 'random':
            # Randomly select k data points as initial centroids
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[indices]

        elif self.init_method == 'kmeans++':
            # K-means++ initialization for better initial centroids
            centroids = np.zeros((self.n_clusters, n_features))

            # Choose first centroid randomly
            centroids[0] = X[np.random.choice(n_samples)]

            # Choose remaining centroids
            for i in range(1, self.n_clusters):
                # Calculate distances to nearest centroid
                distances = np.array([
                    min([np.linalg.norm(x - c) ** 2 for c in centroids[:i]])
                    for x in X
                ])

                # Choose next centroid with probability proportional to distance squared
                probabilities = distances / distances.sum()
                cumulative_probs = probabilities.cumsum()
                r = np.random.rand()

                for j, cum_prob in enumerate(cumulative_probs):
                    if r < cum_prob:
                        centroids[i] = X[j]
                        break

        return centroids

    def _assign_clusters(self, X, centroids):
        """
        Assign each data point to the nearest centroid.

        Parameters:
        X (np.array): Data matrix (n_samples, n_features)
        centroids (np.array): Current centroids (n_clusters, n_features)

        Returns:
        np.array: Cluster labels for each data point
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)

        for i, point in enumerate(X):
            # Calculate Euclidean distance to each centroid
            distances = np.linalg.norm(centroids - point, axis=1)
            # Assign to nearest centroid
            labels[i] = np.argmin(distances)

        return labels

    def _update_centroids(self, X, labels):
        """
        Update centroids based on mean of assigned points.

        Parameters:
        X (np.array): Data matrix (n_samples, n_features)
        labels (np.array): Current cluster assignments

        Returns:
        np.array: Updated centroids
        """
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))

        for k in range(self.n_clusters):
            # Get all points assigned to cluster k
            cluster_points = X[labels == k]

            if len(cluster_points) > 0:
                # Update centroid to mean of assigned points
                centroids[k] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                centroids[k] = X[np.random.choice(X.shape[0])]

        return centroids

    def _calculate_wcss(self, X, labels, centroids):
        """
        Calculate Within-Cluster Sum of Squares (WCSS).

        Parameters:
        X (np.array): Data matrix
        labels (np.array): Cluster assignments
        centroids (np.array): Cluster centroids

        Returns:
        float: WCSS value
        """
        wcss = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - centroids[k]) ** 2)
        return wcss

    def fit(self, X):
        """
        Fit K-means clustering to data.

        Parameters:
        X (np.array or pd.DataFrame): Data matrix

        Returns:
        self: Fitted K-means object
        """
        # Convert to numpy array if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Initialize centroids
        self.centroids = self._initialize_centroids(X)

        # Iterative optimization
        for iteration in range(self.max_iterations):
            # Assign clusters
            old_labels = self.labels
            self.labels = self._assign_clusters(X, self.centroids)

            # Update centroids
            new_centroids = self._update_centroids(X, self.labels)

            # Check convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)

            self.centroids = new_centroids
            self.n_iterations = iteration + 1

            if centroid_shift < self.tolerance:
                print(f"Converged after {self.n_iterations} iterations")
                break

        # Calculate final WCSS
        self.wcss = self._calculate_wcss(X, self.labels, self.centroids)

        return self

    def predict(self, X):
        """
        Predict cluster labels for new data.

        Parameters:
        X (np.array or pd.DataFrame): Data to predict

        Returns:
        np.array: Predicted cluster labels
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        return self._assign_clusters(X, self.centroids)

    def fit_predict(self, X):
        """
        Fit K-means and return cluster labels.

        Parameters:
        X (np.array or pd.DataFrame): Data matrix

        Returns:
        np.array: Cluster labels
        """
        self.fit(X)
        return self.labels


class KMeansAnalyzer:
    """
    Helper class for K-means analysis including elbow method and cluster interpretation.
    """

    def __init__(self, data, feature_columns, original_data=None):
        """
        Initialize analyzer.

        Parameters:
        data (pd.DataFrame): Normalized/preprocessed data for clustering
        feature_columns (list): List of feature column names to use
        original_data (pd.DataFrame): Original (unnormalized) data for interpretation
        """
        self.data = data
        self.feature_columns = feature_columns
        self.original_data = original_data if original_data is not None else data
        self.X = data[feature_columns].values

    def elbow_method(self, k_range=range(2, 9), random_state=42):
        """
        Perform elbow method to find optimal number of clusters.

        Parameters:
        k_range (range): Range of k values to test
        random_state (int): Random seed

        Returns:
        dict: Dictionary with k values and corresponding WCSS
        """
        print("=" * 60)
        print("ELBOW METHOD ANALYSIS")
        print("=" * 60)

        wcss_values = []
        k_values = list(k_range)

        for k in k_values:
            print(f"\nTesting k={k}...")
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            kmeans.fit(self.X)
            wcss_values.append(kmeans.wcss)
            print(f"  WCSS: {kmeans.wcss:.2f}")

        results = {
            'k_values': k_values,
            'wcss_values': wcss_values
        }

        return results

    def plot_elbow_curve(self, elbow_results, save_path='../results/'):
        """
        Plot elbow curve.

        Parameters:
        elbow_results (dict): Results from elbow_method()
        save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))

        k_values = elbow_results['k_values']
        wcss_values = elbow_results['wcss_values']

        plt.plot(k_values, wcss_values, 'bo-', linewidth=2, markersize=10)
        plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
        plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)

        # Annotate each point with WCSS value
        for k, wcss in zip(k_values, wcss_values):
            plt.annotate(f'{wcss:.0f}', (k, wcss), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{save_path}elbow_curve.png", dpi=300, bbox_inches='tight')
        print(f"\n✓ Elbow curve saved to {save_path}elbow_curve.png")

        return plt.gcf()

    def analyze_clusters(self, kmeans_model, cluster_names=None):
        """
        Analyze and interpret clusters.

        Parameters:
        kmeans_model (KMeans): Fitted K-means model
        cluster_names (list): Optional names for clusters

        Returns:
        pd.DataFrame: Cluster statistics
        """
        print("\n" + "=" * 60)
        print("CLUSTER ANALYSIS")
        print("=" * 60)

        # Add cluster labels to original data
        self.original_data['cluster'] = kmeans_model.labels

        cluster_stats = []

        for k in range(kmeans_model.n_clusters):
            cluster_data = self.original_data[self.original_data['cluster'] == k]

            stats = {
                'Cluster': k,
                'Name': cluster_names[k] if cluster_names else f"Cluster {k}",
                'Count': len(cluster_data),
                'Avg_Price': cluster_data['price'].mean(),
                'Avg_Cost': cluster_data['cost'].mean(),
                'Avg_Units_Sold': cluster_data['units_sold'].mean(),
                'Avg_Profit': cluster_data['profit'].mean(),
                'Avg_Promotion_Freq': cluster_data['promotion_frequency'].mean(),
                'Avg_Shelf_Level': cluster_data['shelf_level'].mean(),
                'Total_Profit': cluster_data['profit'].sum()
            }

            cluster_stats.append(stats)

        stats_df = pd.DataFrame(cluster_stats)

        # Print formatted output
        for idx, row in stats_df.iterrows():
            print(f"\n{row['Name']} (Cluster {row['Cluster']}):")
            print(f"  Products: {row['Count']}")
            print(f"  Avg Price: ${row['Avg_Price']:.2f}")
            print(f"  Avg Units Sold: {row['Avg_Units_Sold']:.0f}")
            print(f"  Avg Profit: ${row['Avg_Profit']:.2f}")
            print(f"  Total Profit: ${row['Total_Profit']:.2f}")
            print(f"  Avg Promotions/month: {row['Avg_Promotion_Freq']:.1f}")
            print(f"  Avg Shelf Level: {row['Avg_Shelf_Level']:.1f}")

        return stats_df

    def visualize_clusters(self, kmeans_model, feature_x='price', feature_y='units_sold',
                          cluster_names=None, save_path='../results/'):
        """
        Create 2D scatter plot of clusters.

        Parameters:
        kmeans_model (KMeans): Fitted K-means model
        feature_x (str): Feature for x-axis
        feature_y (str): Feature for y-axis
        cluster_names (list): Optional cluster names
        save_path (str): Path to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Get feature indices
        x_idx = self.feature_columns.index(feature_x)
        y_idx = self.feature_columns.index(feature_y)

        # Plot each cluster
        colors = plt.cm.Set3(np.linspace(0, 1, kmeans_model.n_clusters))

        for k in range(kmeans_model.n_clusters):
            cluster_points = self.X[kmeans_model.labels == k]
            label = cluster_names[k] if cluster_names else f'Cluster {k}'

            ax.scatter(cluster_points[:, x_idx], cluster_points[:, y_idx],
                      c=[colors[k]], label=label, s=100, alpha=0.6, edgecolors='black')

        # Plot centroids
        centroids = kmeans_model.centroids
        ax.scatter(centroids[:, x_idx], centroids[:, y_idx],
                  c='red', marker='X', s=500, edgecolors='black',
                  linewidths=2, label='Centroids', zorder=10)

        ax.set_xlabel(feature_x.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(feature_y.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(f'K-means Clustering Results (k={kmeans_model.n_clusters})',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_path}cluster_visualization.png", dpi=300, bbox_inches='tight')
        print(f"\n✓ Cluster visualization saved to {save_path}cluster_visualization.png")

        return fig


# Example usage
if __name__ == "__main__":
    # Load normalized data
    normalized_data = pd.read_csv('../data/normalized_data.csv')
    cleaned_data = pd.read_csv('../data/cleaned_data.csv')

    # Select features for clustering (excluding target variable 'profit')
    feature_cols = ['price', 'cost', 'units_sold', 'promotion_frequency', 'shelf_level']

    # Initialize analyzer
    analyzer = KMeansAnalyzer(
        data=normalized_data,
        feature_columns=feature_cols,
        original_data=cleaned_data
    )

    # Perform elbow method
    elbow_results = analyzer.elbow_method(k_range=range(2, 9))
    analyzer.plot_elbow_curve(elbow_results)

    # Based on elbow curve, choose optimal k (e.g., k=4)
    optimal_k = 4
    print(f"\n{'=' * 60}")
    print(f"FITTING K-MEANS WITH OPTIMAL k={optimal_k}")
    print('=' * 60)

    # Fit K-means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(analyzer.X)

    # Analyze clusters
    cluster_names = [
        "Budget Best-Sellers",
        "Premium Low-Volume",
        "Mid-Range Steady",
        "High-Volume Promotions"
    ]
    stats_df = analyzer.analyze_clusters(kmeans, cluster_names=cluster_names)

    # Visualize clusters
    analyzer.visualize_clusters(kmeans, feature_x='price', feature_y='units_sold',
                               cluster_names=cluster_names)

    # Save results
    stats_df.to_csv('../results/cluster_statistics.csv', index=False)
    print("\n✓ Cluster statistics saved to '../results/cluster_statistics.csv'")
