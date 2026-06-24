"""Phase 1, Step 6: Slope Unit Extraction using r.slopeunits (GRASS GIS).

Extracts geomorphologically accurate slope units from the conditioned DEM
using the r.slopeunits algorithm via GRASS GIS.

Prerequisites:
    - GRASS GIS 8.x+ installed and on PATH
    - r.slopeunits add-on installed: grass --tmp-mapset --exec g.extension extension=r.slopeunits

Usage:
    python phase1_slope_units.py
"""

import subprocess
import sys
from pathlib import Path


# r.slopeunits parameters (from config.yaml)
PARAMS = {
    'flow_accumulation_threshold': 10000,  # square meters
    'reduction_factor': 2,
    'min_area': 300000,  # square meters
    'min_circular_variance': 0.1,
}


def find_grass():
    """Find GRASS GIS executable. Returns the command path or exits."""
    grass_cmd = 'grass'
    try:
        subprocess.run([grass_cmd, '--version'], capture_output=True, timeout=10)
        return grass_cmd
    except FileNotFoundError:
        for candidate in [
            r'C:\OSGeo4W\bin\grass84.bat',
            r'C:\OSGeo4W64\bin\grass84.bat',
            r'C:\Program Files\GRASS GIS 8.4\grass84.bat',
            r'C:\Program Files\GRASS GIS 8.3\grass83.bat',
        ]:
            if Path(candidate).exists():
                return candidate
        print("ERROR: GRASS GIS not found. Install from https://grass.osgeo.org/download/windows/")
        sys.exit(1)


def check_grass(grass_cmd):
    """Verify GRASS GIS works and r.slopeunits is available."""
    try:
        result = subprocess.run(
            [grass_cmd, '--version'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print("ERROR: GRASS GIS returned non-zero exit code.")
            print(result.stderr)
            return False
        # Print just the version line (strip GRASS banner noise)
        for line in result.stdout.splitlines():
            if 'GRASS GIS' in line and 'version' in line.lower():
                print(f"  {line.strip()}")
                break
        else:
            print(f"  GRASS GIS: OK")

        # Check r.slopeunits
        result = subprocess.run(
            [grass_cmd, '--tmp-mapset', '--exec', 'r.slopeunits', '--help'],
            capture_output=True, text=True, timeout=30
        )
        if 'r.slopeunits' in result.stdout.lower() or 'r.slopeunits' in result.stderr.lower():
            print("  r.slopeunits: available")
            return True
        else:
            print("ERROR: r.slopeunits not found. Install with:")
            print(f"  {grass_cmd} --tmp-mapset --exec g.extension extension=r.slopeunits")
            return False

    except FileNotFoundError:
        print("ERROR: GRASS GIS not found.")
        print("Install from: https://grass.osgeo.org/download/windows/")
        return False
    except subprocess.TimeoutExpired:
        print("ERROR: GRASS GIS check timed out.")
        return False


def run_slopeunits(dem_path, output_dir, grass_cmd):
    """Run r.slopeunits on the conditioned DEM via GRASS GIS.

    Uses a single GRASS session (via Python API) so the imported DEM
    persists across all commands. This avoids the --tmp-mapset per-command
    isolation issue on Windows.

    Parameters
    ----------
    dem_path : Path
        Path to the conditioned DEM (EPSG:32651, GeoTIFF).
    output_dir : Path
        Directory to write slope_units.shp and slope_units.geojson.
    grass_cmd : str
        Path to the GRASS executable.

    Returns
    -------
    bool
        True if extraction succeeded.
    """
    output_shp = output_dir / 'slope_units.shp'
    output_geojson = output_dir / 'slope_units.geojson'

    # r.slopeunits parameters
    t = PARAMS['flow_accumulation_threshold']
    r = PARAMS['reduction_factor']
    a = PARAMS['min_area']
    c = PARAMS['min_circular_variance']

    print(f"  Parameters: t={t}, r={r}, a={a}, c={c}")

    # Write a Python script that runs all GRASS commands in a single session.
    # GRASS needs a project with matching CRS (EPSG:32651). The grass84.bat -c
    # flag creates it from the DEM, then we import and run r.slopeunits.create.
    # The GRASS_ADDON_BASE env var must be set so the addon is discoverable.
    grass_py = output_dir / '_grass_slopeunits_run.py'
    grass_py.write_text(f'''
import os
os.environ["GRASS_ADDON_BASE"] = os.path.join(os.environ.get("APPDATA", ""), "GRASS8", "addons")

import grass.script as gs

dem = r"{dem_path}"
out_shp = r"{output_shp}"
out_geo = r"{output_geojson}"

print("Step 1: Importing DEM...")
gs.run_command("r.in.gdal", input=dem, output="dem", overwrite=True)
gs.run_command("g.region", raster="dem")
print("  Region set.")

print("Step 2: Running r.slopeunits.create...")
gs.run_command("r.slopeunits.create",
    demmap="dem",
    slumap="slope_units",
    slumapvect="slope_units_vec",
    thresh={t},
    rf={r},
    areamin={a},
    cvmin={c},
    maxiteration=100,
    overwrite=True
)

print("Step 3: Counting slope units...")
stats = gs.read_command("r.stats", flags="cn", input="slope_units")
lines = [l.strip() for l in stats.strip().splitlines() if l.strip() and l.strip() != "*"]
print(f"  Slope unit categories: {{len(lines)}}")

print("Step 4: Exporting shapefile...")
gs.run_command("v.out.ogr", input="slope_units_vec", type="area",
    output=out_shp, format="ESRI_Shapefile", overwrite=True)
print("Step 5: Exporting GeoJSON...")
gs.run_command("v.out.ogr", input="slope_units_vec", type="area",
    output=out_geo, format="GeoJSON", overwrite=True)

shp_exists = os.path.exists(out_shp)
geo_exists = os.path.exists(out_geo)
print(f"Shapefile exists: {{shp_exists}}")
print(f"GeoJSON exists: {{geo_exists}}")
if shp_exists:
    print(f"Shapefile size: {{os.path.getsize(out_shp)}} bytes")
print("DONE")
''', encoding='utf-8')

    # Run via GRASS with a CRS-matched project.
    # The -c flag creates a new project from the DEM's CRS.
    gisdbase = output_dir.parent / 'grassdb'
    project = gisdbase / 'cebu_utm'

    print("  Creating GRASS project with matching CRS...")
    try:
        result = subprocess.run(
            [grass_cmd, '-c', 'EPSG:32651', str(project),
             '--exec', 'python', str(grass_py)],
            capture_output=True, text=True, timeout=600
        )

        # Print stdout (contains our progress messages)
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                if line.strip():
                    print(f"    {line}")

        if result.returncode != 0:
            print(f"  ERROR: GRASS session failed (exit code {result.returncode})")
            if result.stderr.strip():
                err_lines = result.stderr.strip().splitlines()
                for line in err_lines[-10:]:
                    print(f"    {line}")
            return False

    except subprocess.TimeoutExpired:
        print("  ERROR: r.slopeunits timed out (600s limit).")
        return False
    finally:
        grass_py.unlink(missing_ok=True)

    # Verify outputs
    if output_shp.exists():
        print(f"  Output shapefile: {output_shp}")
    else:
        print(f"  WARNING: Shapefile not found at {output_shp}")

    if output_geojson.exists():
        print(f"  Output GeoJSON: {output_geojson}")
    else:
        print(f"  WARNING: GeoJSON not found at {output_geojson}")

    return output_shp.exists()


def main():
    # Paths
    dem_path = Path.cwd() / 'data' / 'processed' / 'CebuCity_DEM_conditioned_utm.tif'
    output_dir = Path.cwd() / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 1, STEP 6: Slope Unit Extraction (r.slopeunits)")
    print("=" * 60)

    # Find GRASS GIS
    print("\nChecking prerequisites...")
    grass_cmd = find_grass()
    if not check_grass(grass_cmd):
        print("\nCannot proceed without GRASS GIS. Please install it first.")
        print("See: https://grass.osgeo.org/download/windows/")
        sys.exit(1)

    # Verify DEM exists
    if not dem_path.exists():
        print(f"\nERROR: Conditioned DEM not found: {dem_path}")
        print("Run phase1_dem_preprocessing.py first.")
        sys.exit(1)

    print(f"\nInput DEM: {dem_path}")
    print(f"Output dir: {output_dir}")

    # Run extraction
    success = run_slopeunits(dem_path, output_dir, grass_cmd)

    if success:
        print("\n" + "=" * 60)
        print("SLOPE UNIT EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"  Shapefile: {output_dir / 'slope_units.shp'}")
        print(f"  GeoJSON:   {output_dir / 'slope_units.geojson'}")
    else:
        print("\nSlope unit extraction failed. See errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
