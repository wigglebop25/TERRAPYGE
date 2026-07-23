"""Phase 1: DEM Preprocessing Pipeline.

Reprojects DEM to EPSG:32651, masks offshore pixels, conditions with
WhiteboxTools, and derives slope/aspect.

Usage:
    python phase1_dem_preprocessing.py
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from pathlib import Path
from whitebox import WhiteboxTools


def load_raw_dem(dem_path):
    """Load raw DEM and return data + profile."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        src_crs = src.crs
        src_bounds = src.bounds
    return dem, profile, src_crs, src_bounds


def reproject_dem_to_utm(dem, src_profile, src_crs, dst_crs='EPSG:32651'):
    """Reproject DEM from source CRS to UTM Zone 51N."""
    src_transform = src_profile['transform']
    src_height, src_width = dem.shape

    # Calculate the transform and dimensions for the destination CRS
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, src_width, src_height,
        *rasterio.transform.array_bounds(src_height, src_width, src_transform)
    )

    # Create destination array
    dem_utm = np.zeros((dst_height, dst_width), dtype=np.float64)

    # Reproject
    reproject(
        source=dem,
        destination=dem_utm,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=None,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear
    )

    # Build new profile
    dst_profile = src_profile.copy()
    dst_profile.update(
        crs=dst_crs,
        transform=dst_transform,
        width=dst_width,
        height=dst_height,
        dtype='float64',
        nodata=np.nan
    )

    return dem_utm, dst_profile


def mask_offshore(dem, threshold=0.0):
    """Mask offshore/negative pixels as NaN."""
    dem_masked = dem.copy()
    dem_masked[dem_masked < threshold] = np.nan
    return dem_masked


def save_raster(data, profile, output_path):
    """Save a raster to disk."""
    out_profile = profile.copy()
    out_profile.update(dtype='float64', nodata=np.nan)

    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(data, 1)


def run_whitebox_conditioning(dem_path, output_dir, wbt):
    """Run WhiteboxTools DEM conditioning pipeline."""
    temp_filled = output_dir / 'temp_filled.tif'
    temp_pits = output_dir / 'temp_pits_filled.tif'

    # Step 1: Fill depressions (fix_flats=True applies small gradient to flat areas)
    print("  Filling depressions (fix_flats=True)...")
    wbt.fill_depressions(
        str(dem_path),
        str(temp_filled),
        fix_flats=True
    )

    # Step 2: Fill single-cell pits
    print("  Filling single-cell pits...")
    wbt.fill_single_cell_pits(
        str(temp_filled),
        str(temp_pits)
    )

    # Load result
    with rasterio.open(temp_pits) as src:
        conditioned = src.read(1).astype(np.float64)
        cond_profile = src.profile.copy()

    # WhiteboxTools uses -32768 as internal nodata — mask it as NaN
    conditioned[conditioned <= -32767] = np.nan

    # Clean up temp files
    for f in [temp_filled, temp_pits]:
        f.unlink(missing_ok=True)

    return conditioned, cond_profile


def derive_slope_aspect(dem_path, output_dir, wbt):
    """Derive slope and aspect from conditioned DEM using WhiteboxTools."""
    slope_path = output_dir / 'CebuCity_Slope_utm.tif'
    aspect_path = output_dir / 'CebuCity_Aspect_utm.tif'

    print("  Calculating slope...")
    wbt.slope(
        str(dem_path),
        str(slope_path),
        units='degrees'
    )

    print("  Calculating aspect...")
    wbt.aspect(
        str(dem_path),
        str(aspect_path)
    )

    # Load results — mask WhiteboxTools internal nodata (-32768)
    with rasterio.open(slope_path) as src:
        slope = src.read(1).astype(np.float64)
        slope_profile = src.profile.copy()
    slope[slope <= -32767] = np.nan

    with rasterio.open(aspect_path) as src:
        aspect = src.read(1).astype(np.float64)
        aspect_profile = src.profile.copy()
    aspect[aspect <= -32767] = np.nan

    return slope, slope_profile, aspect, aspect_profile


def compute_stats(data, name):
    """Compute and return descriptive statistics."""
    valid = data[~np.isnan(data)]
    return {
        'name': name,
        'shape': data.shape,
        'dtype': str(data.dtype),
        'n_total': data.size,
        'n_valid': valid.size,
        'n_nan': int(np.isnan(data).sum()),
        'min': float(np.nanmin(data)),
        'max': float(np.nanmax(data)),
        'mean': float(np.nanmean(data)),
        'std': float(np.nanstd(data)),
        'median': float(np.nanmedian(data)),
    }


def main():
    # Paths
    raw_dem_path = Path.cwd() / 'data' / 'raw' / 'dem' / 'CebuCity_DEM.tif'
    output_dir = Path.cwd() / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    dem_utm_path = output_dir / 'CebuCity_DEM_utm.tif'
    dem_conditioned_path = output_dir / 'CebuCity_DEM_conditioned_utm.tif'

    # Initialize WhiteboxTools
    wbt = WhiteboxTools()
    wbt.set_verbose_mode(False)

    # === Step 1: Load raw DEM ===
    print("=" * 60)
    print("STEP 1: Loading raw DEM")
    print("=" * 60)
    dem_raw, raw_profile, raw_crs, raw_bounds = load_raw_dem(raw_dem_path)
    print(f"  Shape: {dem_raw.shape}")
    print(f"  CRS: {raw_crs}")
    print(f"  Resolution: {raw_profile['transform']}")
    print(f"  Elevation range: {np.nanmin(dem_raw):.1f} to {np.nanmax(dem_raw):.1f} m")
    print(f"  Negative pixels: {int((dem_raw < 0).sum())}")

    # === Step 2: Reproject to UTM ===
    print("\n" + "=" * 60)
    print("STEP 2: Reprojecting to EPSG:32651 (UTM Zone 51N)")
    print("=" * 60)
    dem_utm, utm_profile = reproject_dem_to_utm(dem_raw, raw_profile, raw_crs)
    print(f"  New shape: {dem_utm.shape}")
    print(f"  New CRS: {utm_profile['crs']}")
    pixel_res = abs(utm_profile['transform'].a), abs(utm_profile['transform'].e)
    print(f"  New resolution: {pixel_res[0]:.2f} x {pixel_res[1]:.2f} meters")
    print(f"  Elevation range: {np.nanmin(dem_utm):.1f} to {np.nanmax(dem_utm):.1f} m")

    # === Step 3: Mask offshore pixels ===
    print("\n" + "=" * 60)
    print("STEP 3: Masking offshore (negative) pixels")
    print("=" * 60)
    n_neg_before = int(np.sum(dem_utm[~np.isnan(dem_utm)] < 0))
    dem_utm = mask_offshore(dem_utm, threshold=0.0)
    n_nan_after = int(np.isnan(dem_utm).sum())
    print(f"  Pixels masked as nodata: {n_nan_after} (were negative: {n_neg_before})")
    print(f"  Elevation range: {np.nanmin(dem_utm):.1f} to {np.nanmax(dem_utm):.1f} m")

    # Save pre-masked UTM DEM
    save_raster(dem_utm, utm_profile, dem_utm_path)
    print(f"  Saved: {dem_utm_path}")

    # === Step 4: WhiteboxTools conditioning ===
    print("\n" + "=" * 60)
    print("STEP 4: WhiteboxTools DEM conditioning")
    print("=" * 60)
    conditioned, cond_profile = run_whitebox_conditioning(
        dem_utm_path, output_dir, wbt
    )

    # Validate conditioning
    diff = conditioned - dem_utm
    n_changed = int(np.sum(diff[~np.isnan(diff)] != 0))
    print(f"  Pixels changed by conditioning: {n_changed}")
    print(f"  Conditioned range: {np.nanmin(conditioned):.1f} to {np.nanmax(conditioned):.1f} m")

    # Save conditioned DEM
    save_raster(conditioned, cond_profile, dem_conditioned_path)
    print(f"  Saved: {dem_conditioned_path}")

    # === Step 5: Derive slope and aspect ===
    print("\n" + "=" * 60)
    print("STEP 5: Deriving slope and aspect (WhiteboxTools)")
    print("=" * 60)
    slope, slope_prof, aspect, aspect_prof = derive_slope_aspect(
        dem_conditioned_path, output_dir, wbt
    )

    # Validate slope
    slope_valid = slope[~np.isnan(slope)]
    print(f"  Slope range: {slope_valid.min():.2f} to {slope_valid.max():.2f} degrees")
    print(f"  Slope mean: {slope_valid.mean():.2f} degrees")
    print(f"  Slope std: {slope_valid.std():.2f} degrees")

    # Validate aspect
    aspect_valid = aspect[~np.isnan(aspect)]
    print(f"  Aspect range: {aspect_valid.min():.1f} to {aspect_valid.max():.1f} degrees")
    print(f"  Aspect mean: {aspect_valid.mean():.1f} degrees")

    # Re-save slope/aspect with proper NaN nodata (WhiteboxTools uses -32768 internally)
    slope_out_path = output_dir / 'CebuCity_Slope_utm.tif'
    aspect_out_path = output_dir / 'CebuCity_Aspect_utm.tif'
    save_raster(slope, slope_prof, slope_out_path)
    save_raster(aspect, aspect_prof, aspect_out_path)
    print(f"  Re-saved slope: {slope_out_path}")
    print(f"  Re-saved aspect: {aspect_out_path}")

    # === Step 6: Compute stats and write QC report ===
    print("\n" + "=" * 60)
    print("STEP 6: Writing QC report")
    print("=" * 60)

    stats = {
        'raw_dem': compute_stats(dem_raw, 'Raw DEM (EPSG:4326)'),
        'dem_utm': compute_stats(dem_utm, 'DEM UTM (EPSG:32651, masked)'),
        'dem_conditioned': compute_stats(conditioned, 'Conditioned DEM (WhiteboxTools)'),
        'slope': compute_stats(slope, 'Slope (WhiteboxTools)'),
        'aspect': compute_stats(aspect, 'Aspect (WhiteboxTools)'),
    }

    qc_path = Path.cwd() / 'results' / 'phase1_qc.txt'
    qc_path.parent.mkdir(parents=True, exist_ok=True)

    with open(qc_path, 'w', encoding='utf-8') as f:
        f.write("TERRAPYGE Phase 1: DEM Preprocessing QC Report\n")
        f.write("=" * 60 + "\n\n")

        for key, s in stats.items():
            f.write(f"--- {s['name']} ---\n")
            f.write(f"  Shape:    {s['shape']}\n")
            f.write(f"  Dtype:    {s['dtype']}\n")
            f.write(f"  Total:    {s['n_total']}\n")
            f.write(f"  Valid:    {s['n_valid']}\n")
            f.write(f"  NaN:      {s['n_nan']}\n")
            f.write(f"  Min:      {s['min']:.4f}\n")
            f.write(f"  Max:      {s['max']:.4f}\n")
            f.write(f"  Mean:     {s['mean']:.4f}\n")
            f.write(f"  Std:      {s['std']:.4f}\n")
            f.write(f"  Median:   {s['median']:.4f}\n")
            f.write("\n")

        # Validation checks
        f.write("VALIDATION CHECKS\n")
        f.write("-" * 40 + "\n")

        # Slope sanity check
        slope_mean = slope_valid.mean()
        if 5 <= slope_mean <= 30:
            f.write(f"  [PASS] Slope mean {slope_mean:.2f} degrees (expected 5-30 for urban terrain)\n")
        else:
            f.write(f"  [FAIL] Slope mean {slope_mean:.2f} degrees (outside expected 5-30 range)\n")

        # No negative values in conditioned DEM
        n_neg = int(np.sum(conditioned[~np.isnan(conditioned)] < 0))
        if n_neg == 0:
            f.write("  [PASS] No negative elevation values in conditioned DEM\n")
        else:
            f.write(f"  [FAIL] {n_neg} negative elevation values in conditioned DEM\n")

        # Nodata handling
        if np.isnan(conditioned).sum() > 0:
            f.write(f"  [PASS] Conditioned DEM has {int(np.isnan(conditioned).sum())} nodata pixels\n")
        else:
            f.write("  [WARN] Conditioned DEM has no nodata pixels (expected some for offshore)\n")

        # Slope max sanity
        if slope_valid.max() < 90:
            f.write(f"  [PASS] Slope max {slope_valid.max():.2f} < 90 degrees\n")
        else:
            f.write(f"  [WARN] Slope max {slope_valid.max():.2f} = 90 degrees (possible flat/pit artifact)\n")

    print(f"  QC report saved: {qc_path}")

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"  UTM DEM:          {dem_utm_path}")
    print(f"  Conditioned DEM:  {dem_conditioned_path}")
    print(f"  Slope:            {output_dir / 'CebuCity_Slope_utm.tif'}")
    print(f"  Aspect:           {output_dir / 'CebuCity_Aspect_utm.tif'}")
    print(f"  QC report:        {qc_path}")


if __name__ == '__main__':
    main()
