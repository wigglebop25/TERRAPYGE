"""Data acquisition utilities for TERRAPYGE."""

import geopandas as gpd
import rasterio
from pathlib import Path

def load_dem(dem_path):
    """Load Digital Elevation Model."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        profile = src.profile
    return dem, profile

def load_shapefile(shp_path):
    """Load shapefile with proper CRS handling."""
    gdf = gpd.read_file(shp_path)
    return gdf

def validate_crs(gdf, target_crs='EPSG:32651'):
    """Validate and reproject if needed."""
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf
