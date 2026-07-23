# TERRAPYGE Data Schema

## Overview
This document defines all data inputs, intermediate formats, and outputs for the TERRAPYGE pipeline. All rasters are in **EPSG:32651 (UTM Zone 51N)** at **30m resolution** aligned to the conditioned DEM grid (334×301 pixels).

---

## 1. Raw Inputs (GEE Exports)

| File | Path | CRS | Resolution | Bands | Description |
|------|------|-----|------------|-------|-------------|
| DEM | `data/raw/buhisan/dem/Buhisan_DEM_SRTM_30m.tif` | EPSG:4326 | 30m | 1 (elevation) | SRTM 30m, clipped to Buhisan bbox |
| SoilGrids | `data/raw/buhisan/soil/Buhisan_SoilGrids_250m.tif` | EPSG:4326 | 250m | 6 (clay, sand, silt, pH, BD, SOC) | ISRIC SoilGrids 0-5cm mean |
| WorldCover | `data/raw/buhisan/landcover/Buhisan_WorldCover_10m.tif` | EPSG:4326 | 10m | 1 (class) | ESA WorldCover 2020 |
| DynamicWorld | `data/raw/buhisan/landcover/Buhisan_DynamicWorld.tif` | EPSG:4326 | 10m | 5 (water, trees, grass, bare, built) | Google DW probabilities |
| CHIRPS | `data/raw/buhisan/climate/Buhisan_CHIRPS_AnnualRainfall.tif` | EPSG:4326 | 5km | 1 (mm/yr) | 2020-2023 mean annual |
| WorldClim | `data/raw/buhisan/climate/Buhisan_WorldClim_Annual.tif` | EPSG:4326 | 1km | 2 (tavg, prec) | 30-yr normals |
| ERA5-Land | `data/raw/buhisan/climate/Buhisan_ERA5Land.tif` | EPSG:4326 | 10km | 2 (soil moisture, runoff) | 2020-2023 monthly mean |
| JRC Water | `data/raw/buhisan/water/Buhisan_JRC_Water.tif` | EPSG:4326 | 30m | 2 (occurrence, seasonality) | Global Surface Water |

---

## 2. Processed Rasters (EPSG:32651, 30m, 334×301)

All rasters in `data/processed/buhisan/` are reprojected, resampled, and aligned to the conditioned DEM grid.

| File | Bands | Dtype | Nodata | Description |
|------|-------|-------|--------|-------------|
| `dem_utm.tif` | 1 | float32 | NaN | Raw DEM reprojected to UTM |
| `dem_conditioned.tif` | 1 | float32 | NaN | Whitebox: fill depressions + pits |
| `slope.tif` | 1 | float32 | NaN | Degrees (Whitebox `slope`) |
| `aspect.tif` | 1 | float32 | NaN | Degrees 0-360 (Whitebox `aspect`) |
| `curv_plan.tif` | 1 | float32 | NaN | Plan curvature (Whitebox) |
| `curv_profile.tif` | 1 | float32 | NaN | Profile curvature (Whitebox) |
| `twi.tif` | 1 | float32 | NaN | Topographic Wetness Index (Whitebox `wetness_index`) |
| `spi.tif` | 1 | float32 | NaN | Stream Power Index (Whitebox `stream_power_index`) |
| `soil_utm.tif` | 6 | float32 | NaN | clay, sand, silt, pH, BD, SOC |
| `worldcover_utm.tif` | 1 | uint8 | 255 | ESA class (10,20,30,40,50,60,80,90,95) |
| `dynamicworld_utm.tif` | 5 | float32 | NaN | water, trees, grass, bare, built probs |
| `chirps_utm.tif` | 1 | float32 | NaN | Annual rainfall (mm) |
| `worldclim_utm.tif` | 2 | float32 | NaN | tavg (°C×10), prec (mm) |
| `era5_utm.tif` | 2 | float32 | NaN | soil moisture (m³/m³), runoff (m) |
| `jrc_water_utm.tif` | 2 | float32 | NaN | occurrence (%), seasonality |
| `su.tif` | 1 | uint16 | 65535 | Slope unit IDs (1-13297) |

---

## 3. Slope Units (Vector)

### `slope_units.gpkg` / `slope_units.shp`
| Field | Type | Description |
|-------|------|-------------|
| `su_id` | int32 | Unique slope unit ID (1-13297) |
| `geometry` | Polygon | SU boundary |
| `area_m2` | float64 | Area in square meters |
| `perimeter` | float64 | Perimeter in meters |

**Post-aggregation** (after feature join):
| Feature Column | Source Raster | Aggregation |
|----------------|---------------|-------------|
| `dem_mean` | dem_conditioned | mean |
| `slope_mean` | slope | mean |
| `aspect_mean` | aspect | circular mean |
| `curv_plan_mean` | curv_plan | mean |
| `curv_profile_mean` | curv_profile | mean |
| `twi_mean` | twi | mean |
| `spi_mean` | spi | mean |
| `soil_b0_mean` ... `soil_b5_mean` | soil_utm (6 bands) | mean |
| `worldcover_mean` | worldcover_utm | mode (majority class) |
| `dynamicworld_b0_mean` ... `dynamicworld_b4_mean` | dynamicworld_utm (5 bands) | mean |
| `chirps_mean` | chirps_utm | mean |
| `worldclim_b0_mean`, `worldclim_b1_mean` | worldclim_utm | mean |
| `era5_b0_mean`, `era5_b1_mean` | era5_utm | mean |
| `jrc_water_b0_mean`, `jrc_water_b1_mean` | jrc_water_utm | mean |

---

## 4. Graph Structure (PyTorch Geometric HeteroData)

### `buhisan_hetero.pt`
```python
HeteroData(
    # Node: Slope Unit
    su = Node(
        x: FloatTensor[13297, 15],      # Standardized features
        y: LongTensor[13297],            # Labels (0/1 synthetic)
        num_nodes: 13297,
        feat_names: List[str],           # 15 feature names
        scaler_mean: List[float],        # StandardScaler mean
        scaler_scale: List[float],       # StandardScaler scale
        crs: "EPSG:32651",
        transform: List[float],          # Affine transform [a,b,c,d,e,f]
    ),
    
    # Edge: Spatial (undirected, Queen contiguity)
    ('su', 'spatial', 'su') = Edge(
        edge_index: LongTensor[2, 88772],   # COO format
        # edge_attr: None (could add: shared_boundary_length)
    ),
    
    # Edge: Hydrological (directed, D8 flow) — PLACEHOLDER
    ('su', 'hydro', 'su') = Edge(
        edge_index: LongTensor[2, 0],       # Empty for now
        # edge_attr: None (could add: flow_accum, distance)
    ),
)
```

### Feature Vector (15 dims, order in `feat_names`)
| Index | Name | Source | Original Unit |
|-------|------|--------|---------------|
| 0 | `value` | SU attribute | — |
| 1 | `label` | SU attribute | — |
| 2 | `dem_mean` | DEM | m |
| 3 | `slope_mean` | Slope | degrees |
| 4 | `aspect_mean` | Aspect | degrees |
| 5 | `curv_plan_mean` | Plan curvature | 1/m |
| 6 | `curv_profile_mean` | Profile curvature | 1/m |
| 7 | `twi_mean` | TWI | ln(m) |
| 8 | `spi_mean` | SPI | m² |
| 9 | `soil_b0_mean` | Clay % | % |
| 10 | `soil_b1_mean` | Sand % | % |
| 11 | `soil_b2_mean` | Silt % | % |
| 12 | `soil_b3_mean` | pH | pH×10 |
| 13 | `soil_b4_mean` | Bulk density | g/cm³×100 |
| 14 | `soil_b5_mean` | SOC | g/kg |

> **Note**: Additional features (worldcover, dynamicworld, chirps, worldclim, era5, jrc_water) are aggregated but may be reduced during feature selection.

---

## 5. Labels

### Synthetic (Current)
| File | Format | Description |
|------|--------|-------------|
| `data['su'].y` in graph | LongTensor[13297] | Binary (0/1), 15% positive |

**Generation heuristic**: High slope + high TWI + high SPI + high curvature → landslide-prone.

### Real (Future — PHIVOLCS)
| Source | Format | Integration |
|--------|--------|-------------|
| PHIVOLCS HazardHunter polygons | GeoJSON/Shapefile | `gpd.sjoin(su, hazard, predicate='intersects')` |
| NASA GLC points | CSV | Buffer 30m → point-in-polygon |
| Historical events (news) | CSV | Manual digitization → point-in-polygon |

**Target schema**:
```python
# Multi-class (5 classes matching hazard map)
y: LongTensor[13297]  # 0=Very Low, 1=Low, 2=Moderate, 3=High, 4=Very High
```

---

## 6. Outputs

### Hazard Map Vector
| File | Format | Features | Properties |
|------|--------|----------|------------|
| `buhisan_hazard_map.gpkg` | GeoPackage | 13,297 | `su_id`, `landslide_prob`, `hazard_class` |
| `buhisan_hazard_map.geojson` | GeoJSON | 13,297 | Same as GPKG |
| `buhisan_hazard_map.html` | HTML | — | Interactive Folium map |

### Hazard Classes (Quantile Bins)
| Class | Probability Range | Count | Color |
|-------|-------------------|-------|-------|
| Very Low | [0.00, 0.20) | 9,912 | #2E7D32 |
| Low | [0.20, 0.40) | 1,585 | #8BC34A |
| Moderate | [0.40, 0.60) | 906 | #FFEB3B |
| High | [0.60, 0.80) | 599 | #FF9800 |
| Very High | [0.80, 1.00] | 295 | #F44336 |

### Model Artifacts
| File | Format | Contents |
|------|--------|----------|
| `best_model.pt` | PyTorch state_dict | GCN weights (3 layers, 64 hidden) |
| `metrics.json` | JSON | AUC, AP, F1, hyperparams |
| `graph_metadata.json` | JSON | Feature names, scaler params, CRS |

---

## 6b. Results Directory

### `results/` — Analysis Outputs (gitignored)
| File | Format | Contents |
|------|--------|----------|
| `baseline_comparison.json` | JSON | All model metrics (LR, RF, XGBoost, GNN) |
| `model_comparison.json` | JSON | GNN spatial-only vs dual-edge comparison |

### `models/` — Trained Model Weights (gitignored)
| File | Format | Contents |
|------|--------|----------|
| `gnn_dual_edge.pt` | PyTorch state_dict | Dual-edge GNN model weights |
| `baselines/*.pkl` | Pickle | Baseline model objects |

### Baseline Comparison Schema (`results/baseline_comparison.json`)
```json
{
  "Logistic Regression": {"model": "...", "auc": 0.XX, "ap": 0.XX, "f1": 0.XX},
  "Random Forest": {"model": "...", "auc": 0.XX, "ap": 0.XX, "f1": 0.XX},
  "XGBoost": {"model": "...", "auc": 0.XX, "ap": 0.XX, "f1": 0.XX},
  "GNN (dual-edge)": {"model": "...", "auc": 0.XX, "ap": 0.XX, "f1": 0.XX},
  "GNN (spatial-only)": {"model": "...", "auc": 0.XX, "ap": 0.XX, "f1": 0.XX}
}
```

---

## 7. Coordinate Reference Systems

| Stage | CRS | EPSG | Units |
|-------|-----|------|-------|
| GEE Exports | WGS84 | 4326 | degrees |
| Processing / Graph | UTM 51N | 32651 | meters |
| Final Outputs | UTM 51N / WGS84 | 32651 / 4326 | meters / degrees |

**Transform (UTM → WGS84)**: `pyproj.Transformer.from_crs(32651, 4326, always_xy=True)`

---

## 8. Data Quality Checks

| Check | Expected | Tool |
|-------|----------|------|
| DEM no negative (after mask) | `min >= 0` | `np.nanmin(dem) >= 0` |
| SU raster unique count | 13,297 | `len(np.unique(su_raster))` |
| SU vector feature count | 13,297 | `len(gdf)` |
| Graph node count | 13,297 | `data['su'].num_nodes` |
| Spatial edge symmetry | Undirected | `(edge_index == edge_index.flip(0)).all()` |
| Feature NaN ratio | < 5% | `np.isnan(X).mean() < 0.05` |
| Label balance | ~15% positive | `y.float().mean() ≈ 0.15` |

---

## 9. Versioning

| Dataset | Version | Date | Notes |
|---------|---------|------|-------|
| SRTM DEM | v3 (2020) | 2024 | NASA/USGS |
| SoilGrids | 2019 | 2024 | ISRIC |
| WorldCover | v100 (2020) | 2024 | ESA |
| DynamicWorld | V1 | 2024 | Google |
| CHIRPS | v2.0 | 2024 | UCSB |
| WorldClim | v2.1 | 2024 | Fick & Hijmans |
| ERA5-Land | 2024 | 2024 | ECMWF |
| JRC Water | v1.3 | 2024 | EC JRC |

---

## 10. File Size Summary

| Category | Total Size | Files |
|----------|------------|-------|
| Raw (GEE) | ~150 MB | 8 |
| Processed Rasters | ~50 MB | 16 |
| Slope Units (vector) | ~5 MB | 2 |
| Graph | ~2 MB | 1 |
| Model | ~500 KB | 1 |
| Outputs | ~40 MB | 5 |
| **Total** | **~250 MB** | **33** |
