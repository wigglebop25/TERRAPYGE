"""Map visualization utilities."""

import folium
import geopandas as gpd
import matplotlib.pyplot as plt

def create_folium_map(gdf, center=None):
    """Create interactive Folium map."""
    if center is None:
        center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=12)
    return m

def plot_slope_units(gdf, ax=None):
    """Plot slope units."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, edgecolor='black', linewidth=0.5)
    return ax
