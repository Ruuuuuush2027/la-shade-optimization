"""Baseline methods for comparison."""

import numpy as np
import pandas as pd
from typing import List


def random_baseline(data: pd.DataFrame, k: int, n_trials: int = 5, seed: int = 42) -> List[int]:
    """
    Random baseline with multiple trials.

    Args:
        data: DataFrame with grid points
        k: Number of shades to place
        n_trials: Number of random trials
        seed: Random seed

    Returns:
        Best random solution
    """
    np.random.seed(seed)
    valid_indices = data.index.tolist()

    if len(valid_indices) == 0:
        return []

    if k >= len(valid_indices):
        # Not enough unique locations; return all points
        return valid_indices.copy()

    best_placements = None
    best_score = -np.inf

    for trial in range(n_trials):
        placements = list(np.random.choice(valid_indices, size=k, replace=False))
        score = data.loc[placements, 'land_surface_temp_c'].sum() if 'land_surface_temp_c' in data.columns else 0

        if score > best_score:
            best_score = score
            best_placements = placements

    return best_placements


def greedy_by_feature(data: pd.DataFrame,
                     k: int,
                     feature: str = 'land_surface_temp_c',
                     ascending: bool = False) -> List[int]:
    """
    Greedy selection by single feature.

    Args:
        data: DataFrame with grid points
        k: Number of shades to place
        feature: Feature to sort by
        ascending: Sort ascending (True) or descending (False)

    Returns:
        Top-k indices by feature
    """
    if feature not in data.columns:
        print(f"  Warning: Feature '{feature}' not found. Using random.")
        return random_baseline(data, k)

    # Sort by feature
    sorted_data = data.sort_values(by=feature, ascending=ascending)

    # Select top-k
    top_k = sorted_data.head(k).index.tolist()

    return top_k


def kmeans_clustering(data: pd.DataFrame,
                     k: int,
                     features: List[str] = None) -> List[int]:
    """
    K-means clustering baseline.

    Clusters high-need locations and places shade at centroids.

    Args:
        data: DataFrame with grid points
        k: Number of shades (= number of clusters)
        features: Features to use for clustering

    Returns:
        Indices closest to cluster centroids
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("  Warning: scikit-learn not installed. Falling back to greedy by feature.")
        return greedy_by_feature(data, k)

    # Default features for clustering
    if features is None:
        features = ['land_surface_temp_c', 'cva_sovi_score', 'cva_population']

    # Filter available features
    available_features = [f for f in features if f in data.columns]

    if not available_features:
        print("  Warning: No clustering features available. Using random.")
        return random_baseline(data, k)

    # Prepare data for clustering
    X = data[available_features].fillna(data[available_features].median())

    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Find closest point to each centroid
    selected_indices = []
    for i in range(k):
        centroid = kmeans.cluster_centers_[i]

        # Points in this cluster
        cluster_mask = kmeans.labels_ == i
        cluster_points = X_scaled[cluster_mask]
        cluster_indices = data.index[cluster_mask].tolist()

        if len(cluster_indices) == 0:
            continue

        # Find closest to centroid
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]

        selected_indices.append(closest_idx)

    return selected_indices


def expert_heuristic(data: pd.DataFrame, k: int) -> List[int]:
    """
    Expert heuristic baseline.

    Rule-based approach mimicking urban planning guidelines:
    - High priority: Near transit + high poverty + high heat
    - Medium priority: Near Olympic venues + moderate heat
    - Low priority: Other locations with some need

    Args:
        data: DataFrame with grid points
        k: Number of shades to place

    Returns:
        Indices selected by heuristic rules
    """
    # Calculate priority scores based on rules
    priority_scores = np.zeros(len(data))

    for i, idx in enumerate(data.index):
        row = data.loc[idx]
        score = 0

        # Rule 1: High heat (>70th percentile) → +30 points
        if 'land_surface_temp_c' in data.columns:
            temp = row['land_surface_temp_c']
            temp_threshold = data['land_surface_temp_c'].quantile(0.7)
            if temp >= temp_threshold:
                score += 30

        # Rule 2: High poverty (>30%) → +25 points
        if 'lashade_poverty' in data.columns:
            poverty = row['lashade_poverty']
            if pd.notna(poverty) and poverty >= 0.30:
                score += 25

        # Rule 3: Near transit (<400m) → +20 points
        transit_cols = ['dist_to_bus_stop_1', 'dist_to_metro_stop_1']
        for col in transit_cols:
            if col in data.columns:
                dist = row[col]
                if pd.notna(dist) and dist < 0.4:  # 400m
                    score += 20
                    break

        # Rule 4: High SOVI (>0.5) → +15 points
        if 'cva_sovi_score' in data.columns:
            sovi = row['cva_sovi_score']
            if pd.notna(sovi) and sovi >= 0.5:
                score += 15

        # Rule 5: Near Olympic venue (<1km) → +15 points
        if 'dist_to_venue1' in data.columns:
            venue_dist = row['dist_to_venue1']
            if pd.notna(venue_dist) and venue_dist < 1.0:
                score += 15

        # Rule 6: High population (>75th percentile) → +10 points
        if 'cva_population' in data.columns:
            pop = row['cva_population']
            pop_threshold = data['cva_population'].quantile(0.75)
            if pop >= pop_threshold:
                score += 10

        # Rule 7: Low existing shade (<25%) → +10 points
        shade_cols = ['lashade_tot1500']
        for col in shade_cols:
            if col in data.columns:
                shade = row[col]
                if pd.notna(shade) and shade < 0.25:
                    score += 10
                    break

        priority_scores[i] = score

    # Select top-k by priority score
    sorted_indices = data.index[np.argsort(-priority_scores)].tolist()
    selected = sorted_indices[:k]

    return selected
