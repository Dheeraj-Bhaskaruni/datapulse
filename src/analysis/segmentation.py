"""User and entity segmentation module."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

logger = logging.getLogger(__name__)


class SegmentationAnalyzer:
    """Clustering-based segmentation analysis."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.labels = None
        self.feature_names: List[str] = []

    def find_optimal_k(self, X: pd.DataFrame, k_range: range = range(2, 11)) -> Dict[str, Any]:
        """Find optimal number of clusters using elbow method and silhouette scores."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_clean = X[numeric_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X_clean)

        results: Dict[str, Any] = {'k_values': [], 'inertias': [], 'silhouette_scores': []}

        for k in k_range:
            if k >= len(X_scaled):
                break
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            results['k_values'].append(k)
            results['inertias'].append(float(kmeans.inertia_))
            results['silhouette_scores'].append(float(silhouette_score(X_scaled, labels)))

        best_k = results['k_values'][np.argmax(results['silhouette_scores'])]
        results['optimal_k'] = best_k
        return results

    def segment_kmeans(self, X: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform K-Means segmentation."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = numeric_cols
        X_clean = X[numeric_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X_clean)

        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = self.model.fit_predict(X_scaled)
        self.labels = labels

        result_df = X.copy()
        result_df['segment'] = labels

        metrics: Dict[str, Any] = {
            'n_clusters': n_clusters,
            'silhouette_score': float(silhouette_score(X_scaled, labels)),
            'calinski_harabasz': float(calinski_harabasz_score(X_scaled, labels)),
            'cluster_sizes': pd.Series(labels).value_counts().to_dict(),
        }

        # Cluster profiles
        profiles = {}
        for cluster in range(n_clusters):
            mask = labels == cluster
            profiles[f'cluster_{cluster}'] = {
                col: {'mean': float(X_clean[col][mask].mean()), 'std': float(X_clean[col][mask].std())}
                for col in numeric_cols
            }
        metrics['profiles'] = profiles

        logger.info(f"K-Means segmentation: {n_clusters} clusters, silhouette={metrics['silhouette_score']:.4f}")
        return result_df, metrics

    def segment_dbscan(
        self, X: pd.DataFrame, eps: float = 0.5, min_samples: int = 5
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform DBSCAN segmentation."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_clean = X[numeric_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X_clean)

        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.model.fit_predict(X_scaled)

        result_df = X.copy()
        result_df['segment'] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()

        metrics: Dict[str, Any] = {
            'n_clusters': n_clusters,
            'n_noise_points': int(n_noise),
            'noise_pct': float(n_noise / len(labels) * 100),
            'cluster_sizes': (
                pd.Series(labels[labels != -1]).value_counts().to_dict() if n_clusters > 0 else {}
            ),
        }

        if n_clusters > 1:
            non_noise = labels != -1
            if non_noise.sum() > n_clusters:
                metrics['silhouette_score'] = float(silhouette_score(X_scaled[non_noise], labels[non_noise]))

        return result_df, metrics
