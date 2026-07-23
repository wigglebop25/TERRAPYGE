# Google Earth Engine for TERRAPYGE

## Overview
Google Earth Engine (GEE) is used to acquire Digital Elevation Model (DEM) data for Cebu City, Philippines. This document describes the setup, script, and data acquisition process.

## Registration

### Step 1: Access Earth Engine
**URL:** https://code.earthengine.google.com

### Step 2: Register for Noncommercial Access
1. **Organization Type:** Public or private academic institution
2. **Plan:** Community (free, no credit card required)
3. **Use Case:** Scientific research
4. **Geographic Scope:** Regional (Cebu City, Philippines)

### Step 3: Approval
- **Timeline:** Usually instant to 24 hours
- **Cost:** Free for academic use
- **Quota:** 150 EECU-hours (sufficient for project)

## DEM Extraction Script

### Complete Script
```javascript
// ============================================
// TERRAPYGE: Cebu City DEM Extraction Script
// ============================================

// Define Cebu City bounding box
var cebuCity = ee.Geometry.Rectangle([123.8, 10.2, 124.0, 10.4]);

// Load SRTM DEM data (30m resolution)
var dem = ee.Image('USGS/SRTMGL1_003');

// Clip to Cebu City area
var cebuDEM = dem.clip(cebuCity);

// Add to map for visualization
Map.centerObject(cebuCity, 10);
Map.addLayer(cebuDEM, {
  min: 0, 
  max: 1000, 
  palette: ['blue', 'green', 'yellow', 'red']
}, 'Cebu City DEM');

// Export DEM to Google Drive
Export.image.toDrive({
  image: cebuDEM,
  description: 'CebuCity_DEM_30m',
  region: cebuCity,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

print('DEM export task submitted. Check Tasks tab.');
```

### Script Explanation
- **`ee.Geometry.Rectangle`**: Defines Cebu City bounding box
- **`ee.Image('USGS/SRTMGL1_003')`**: Loads SRTM DEM data
- **`dem.clip(cebuCity)`**: Crops to Cebu City area
- **`Export.image.toDrive`**: Sends file to Google Drive
- **`scale: 30`**: 30-meter resolution

## Data Acquisition Process

### Step 1: Run Script
1. **Open:** https://code.earthengine.google.com
2. **Copy** the script above
3. **Paste** into code editor
4. **Click "Run"** button

### Step 2: Export to Google Drive
1. **Check "Tasks"** tab (right side)
2. **Click "Run"** on the export task
3. **Wait** for completion (1-5 minutes)
4. **Status changes** to "Completed"

### Step 3: Download to Local Storage
1. **Go to:** https://drive.google.com
2. **Check** your Google Drive folder
3. **Download** `CebuCity_DEM_30m.tif`
4. **Save to:** `D:\TERRAPYGE\data\raw\dem\`

## Data Specifications

### DEM File Details
- **Name:** `CebuCity_DEM_30m.tif`
- **Format:** GeoTIFF
- **Resolution:** 30 meters
- **Coordinate System:** WGS84 (EPSG:4326)
- **Coverage:** Cebu City area
- **File Size:** ~10-50 MB

### Additional Outputs
- **Slope:** Calculated from DEM
- **Aspect:** Calculated from DEM
- **Both exported** to Google Drive

## Troubleshooting

### Common Issues

#### Issue 1: "Project not authorized"
**Solution:**
1. **Wait** for registration approval
2. **Or create new project** in Google Cloud Console
3. **Enable Earth Engine API**

#### Issue 2: "Quota exceeded"
**Solution:**
1. **Community tier:** 150 EECU-hour limit
2. **Reduce region size** (make rectangle smaller)
3. **Wait** for monthly quota reset

#### Issue 3: "Task failed"
**Solution:**
1. **Check "Tasks"** tab for error message
2. **Simplify script** (use smaller region)
3. **Try again**

## Integration with TERRAPYGE

### File Location
- **Save DEM:** `D:\TERRAPYGE\data\raw\dem\CebuCity_DEM_30m.tif`
- **Save Slope:** `D:\TERRAPYGE\data\raw\dem\CebuCity_Slope.tif`
- **Save Aspect:** `D:\TERRAPYGE\data\raw\dem\CebuCity_Aspect.tif`

### Python Loading Example
```python
import rasterio
import matplotlib.pyplot as plt

# Load DEM
with rasterio.open('D:/TERRAPYGE/data/raw/dem/CebuCity_DEM_30m.tif') as src:
    dem = src.read(1)
    profile = src.profile

# Visualize
plt.figure(figsize=(10, 8))
plt.imshow(dem, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.title('Cebu City DEM')
plt.show()
```

## Status

### Completed
- [x] Google Earth Engine registration
- [x] DEM extraction script created
- [x] DEM data downloaded
- [x] Data saved to project directory

### Pending
- [ ] Government data (MGB, BSWM) - eFOI requests submitted

## References
- **Google Earth Engine:** https://code.earthengine.google.com
- **SRTM Data:** USGS/SRTMGL1_003
- **Documentation:** https://developers.google.com/earth-engine