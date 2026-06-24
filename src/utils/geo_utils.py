"""Geospatial utilities for TERRAPYGE."""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

def calculate_centroid(gdf):
    """Calculate centroid for each geometry."""
    return gdf.geometry.centroid

def spatial_join(target_gdf, source_gdf, how='inner'):
    """Perform spatial join between GeoDataFrames."""
    return gpd.sjoin(target_gdf, source_gdf, how=how, predicate='intersects')


def normalize_features(features):
    """Normalize features using StandardScaler (z-score normalization).

    Handles NaN values gracefully. Returns (normalized_features, scaler).
    Fit on non-NaN values only.
    """
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    scaler = StandardScaler()
    mask = ~np.isnan(features).any(axis=1)
    if mask.sum() == 0:
        return features, scaler
    scaler.fit(features[mask])
    normalized = features.copy()
    normalized[mask] = scaler.transform(features[mask])
    return normalized, scaler
