"""TERRAPYGE: Generate hazard maps for SAGE hydro-only and SAGE dual-edge models.

Generates GeoPackage, GeoJSON, CSV, and interactive Folium maps for both models.
Includes comparison plot of probability distributions.
"""

import sys
sys.path.insert(0, r'D:\TERRAPYGE')

import torch
import numpy as np
import json
import geopandas as gpd
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import folium
from folium.plugins import Fullscreen, MiniMap, MeasureControl, Search

from src.terrapyge.data.graph import load_graph
from src.terrapyge.models.gnn import get_model

# Paths
PROC_DIR = Path(r'D:\TERRAPYGE\data\processed\buhisan')
RESULTS_DIR = Path(r'D:\TERRAPYGE\results')
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load config
import yaml
with open(r'D:\TERRAPYGE\config.yaml') as f:
    config = yaml.safe_load(f)
gnn_cfg = config.get('gnn', {})


def load_sage_model(model_path, in_channels):
    """Load SAGE model with weights."""
    model = get_model(
        model_type='HeteroSAGE',
        in_channels=in_channels,
        hidden_channels=gnn_cfg.get('hidden_channels', 64),
        out_channels=2,
        num_layers=gnn_cfg.get('num_layers', 3),
        dropout=gnn_cfg.get('dropout', 0.2),
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def run_inference(model, data, edge_index_dict):
    """Run model inference and return probabilities."""
    x_dict = {'su': data['su'].x}
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    return probs


def get_hazard_class(prob):
    """Convert probability to 5-class hazard label."""
    if prob < 0.2:
        return 'Very Low'
    elif prob < 0.4:
        return 'Low'
    elif prob < 0.6:
        return 'Moderate'
    elif prob < 0.8:
        return 'High'
    else:
        return 'Very High'


def get_color(prob):
    """Get color for hazard class."""
    if prob < 0.2:
        return '#2E7D32'  # Dark green
    elif prob < 0.4:
        return '#8BC34A'  # Light green
    elif prob < 0.6:
        return '#FFEB3B'  # Yellow
    elif prob < 0.8:
        return '#FF9800'  # Orange
    else:
        return '#F44336'  # Red


def generate_hazard_map(gdf, probs, model_name, edge_config, features_df=None):
    """Generate hazard map files for a model."""
    # Add predictions to GeoDataFrame
    gdf_copy = gdf.copy()
    gdf_copy['landslide_prob'] = probs
    gdf_copy['hazard_class'] = [get_hazard_class(p) for p in probs]

    # Export GeoPackage
    gpkg_path = PROC_DIR / f'hazard_{model_name}.gpkg'
    gdf_copy.to_file(gpkg_path, driver='GPKG')
    print(f'  GeoPackage: {gpkg_path}')

    # Export GeoJSON
    geojson_path = PROC_DIR / f'hazard_{model_name}.geojson'
    gdf_copy.to_file(geojson_path, driver='GeoJSON')
    print(f'  GeoJSON: {geojson_path}')

    # Export CSV (SU_ID + probability + class only)
    csv_path = PROC_DIR / f'hazard_{model_name}.csv'
    csv_df = pd.DataFrame({
        'su_id': gdf_copy.index,
        'landslide_prob': probs,
        'hazard_class': [get_hazard_class(p) for p in probs],
    })
    csv_df.to_csv(csv_path, index=False)
    print(f'  CSV: {csv_path}')

    # Generate Folium map
    html_path = PROC_DIR / f'hazard_{model_name}.html'
    generate_folium_map(gdf_copy, probs, model_name, edge_config, html_path, features_df)
    print(f'  HTML: {html_path}')

    # Hazard class distribution
    class_counts = {}
    for cls in ['Very Low', 'Low', 'Moderate', 'High', 'Very High']:
        class_counts[cls] = int(sum(1 for p in probs if get_hazard_class(p) == cls))

    return {
        'model_name': model_name,
        'edge_config': edge_config,
        'prob_min': float(probs.min()),
        'prob_max': float(probs.max()),
        'prob_mean': float(probs.mean()),
        'prob_std': float(probs.std()),
        'hazard_distribution': class_counts,
    }


def generate_folium_map(gdf, probs, model_name, edge_config, output_path, features_df=None):
    """Generate interactive Folium map with satellite, toggle, title, search, download."""
    # Reproject to lat/lon for Folium
    gdf_4326 = gdf.to_crs(epsg=4326)
    centroid = gdf_4326.geometry.centroid
    center = [centroid.y.mean(), centroid.x.mean()]

    # Satellite basemap (Esri World Imagery)
    m = folium.Map(
        location=center,
        zoom_start=14,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        zoom_control=True,
    )

    # OpenStreetMap toggle
    folium.TileLayer(tiles='OpenStreetMap', name='OpenStreetMap', attr='OpenStreetMap').add_to(m)

    # Title with AUC score
    auc_value = 0.9500 if 'hydro_only' in model_name else 0.9491
    model_label = model_name.replace('_', ' ').title()
    title_html = f'''<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9999;background:white;padding:10px 20px;border:2px solid grey;border-radius:5px;font-size:16px;font-weight:bold;box-shadow:2px 2px 6px rgba(0,0,0,0.3);">TERRAPYGE: {model_label} (AUC: {auc_value:.4f}) — Landslide Susceptibility</div>'''
    m.get_root().html.add_child(folium.Element(title_html))

    # Create groups for each hazard class
    class_groups = {}
    for cls in ['Very Low', 'Low', 'Moderate', 'High', 'Very High']:
        class_groups[cls] = folium.FeatureGroup(name=f'Class: {cls}')

    # Add polygons with popup and tooltip
    for idx, row in gdf_4326.iterrows():
        prob = row['landslide_prob']
        color = get_color(prob)
        hazard_class = get_hazard_class(prob)
        area_m2 = row.geometry.area * 111319.5 * 111319.5

        popup_html = f'<b>Slope Unit: {idx}</b><br><b>Probability:</b> {prob:.3f}<br><b>Class:</b> {hazard_class}<br><b>Area:</b> {area_m2:.0f} m²<br>'

        # Add detailed features if available
        if features_df is not None:
            su_row = features_df[features_df['su_id'] == idx]
            if not su_row.empty:
                su_data = su_row.iloc[0]
                if 'slope_mean' in su_data:
                    popup_html += f"<b>Slope:</b> {su_data.get('slope_mean', 'N/A'):.1f}°<br>"
                if 'dem_mean' in su_data:
                    popup_html += f"<b>Elevation:</b> {su_data.get('dem_mean', 'N/A'):.1f} m<br>"
                if 'twi_mean' in su_data:
                    popup_html += f"<b>TWI:</b> {su_data.get('twi_mean', 'N/A'):.2f}<br>"
                if 'spi_mean' in su_data:
                    popup_html += f"<b>SPI:</b> {su_data.get('spi_mean', 'N/A'):.2f}<br>"
                if 'worldcover_mean' in su_data:
                    popup_html += f"<b>Land Cover:</b> {su_data.get('worldcover_mean', 'N/A')}<br>"

        geo_json = folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x, p=prob, c=color: {'fillColor': c, 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.7},
            tooltip=f"SU: {idx} | {hazard_class} | {prob:.3f}",
            popup=folium.Popup(popup_html, max_width=300),
            name=f'{idx} {hazard_class}'
        )
        geo_json.add_to(class_groups[hazard_class])

    # Add all class groups to map
    for cls, group in class_groups.items():
        group.add_to(m)

    # Create unified group for search
    all_groups = folium.FeatureGroup(name='All Slope Units')
    for cls, group in class_groups.items():
        for child in group._children.values():
            if isinstance(child, folium.GeoJson):
                child.add_to(all_groups)
    all_groups.add_to(m)

    # Search bar (search by SU ID and hazard class)
    Search(layer=all_groups, geom_type='Polygon', placeholder='Search SU ID or class...', collapsed=False, search_label='name', weight=3).add_to(m)

    # Layer control
    folium.LayerControl().add_to(m)

    # Fullscreen button
    Fullscreen().add_to(m)

    # Minimap
    m.add_child(MiniMap(toggle_display=True, position='bottomright'))

    # Measurement tool
    MeasureControl(position='topleft').add_to(m)

    # Legend
    legend_html = '''<div style="position:fixed;bottom:50px;left:50px;width:220px;height:220px;background:white;border:2px solid grey;z-index:9999;font-size:14px;padding:10px;border-radius:5px;"><b>Landslide Susceptibility</b><br><i style="background:#2E7D32;width:20px;height:20px;display:inline-block;"></i> Very Low (&lt;0.2)<br><i style="background:#8BC34A;width:20px;height:20px;display:inline-block;"></i> Low (0.2-0.4)<br><i style="background:#FFEB3B;width:20px;height:20px;display:inline-block;"></i> Moderate (0.4-0.6)<br><i style="background:#FF9800;width:20px;height:20px;display:inline-block;"></i> High (0.6-0.8)<br><i style="background:#F44336;width:20px;height:20px;display:inline-block;"></i> Very High (&gt;0.8)<br><hr style="margin:5px 0;"><small>Toggle basemap: top-right layer control</small></div>'''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Reset view button
    reset_html = '<div style="position:fixed;top:10px;right:10px;z-index:9999;"><button onclick="map.setView(center, 14)" style="background:white;border:2px solid grey;border-radius:5px;padding:8px 12px;font-size:12px;cursor:pointer;box-shadow:2px 2px 6px rgba(0,0,0,0.3);">Reset View</button></div>'
    m.get_root().html.add_child(folium.Element(reset_html))

    # Download data button (exports ALL slope units as GeoJSON)
    geojson_data = gdf_4326.to_json()
    download_html = f'''<div style="position:fixed;top:10px;right:120px;z-index:9999;"><button onclick="downloadGeoJSON()" style="background:white;border:2px solid grey;border-radius:5px;padding:8px 12px;font-size:12px;cursor:pointer;box-shadow:2px 2px 6px rgba(0,0,0,0.3);">Download GeoJSON</button></div><script>function downloadGeoJSON(){{var data={geojson_data};var blob=new Blob([JSON.stringify(data)],{{type:'application/json'}});var url=URL.createObjectURL(blob);var a=document.createElement('a');a.href=url;a.download='{model_name}_hazard_map.geojson';document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url);}}</script>'''
    m.get_root().html.add_child(folium.Element(download_html))

    m.save(output_path)


def plot_comparison(probs_hydro, probs_dual):
    """Generate comparison plot of probability distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hydro-only
    axes[0].hist(probs_hydro, bins=50, color='#2196F3', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0.2, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0.4, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0.6, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Predicted Probability', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('SAGE Hydro-Only', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Dual-edge
    axes[1].hist(probs_dual, bins=50, color='#FF9800', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0.2, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0.4, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0.6, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Predicted Probability', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('SAGE Dual-Edge', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    fig.suptitle('Hazard Map Comparison: Probability Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'hazard_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Comparison plot: {FIGURES_DIR / "hazard_comparison.png"}')


def main():
    print('=' * 70)
    print('TERRAPYGE: GENERATE HAZARD MAPS')
    print('=' * 70)

    # Load graph
    print('\nLoading graph...')
    data = load_graph(PROC_DIR / 'buhisan_hetero.pt')
    print(f'  Nodes: {data["su"].num_nodes}')
    print(f'  Features: {data["su"].x.shape[1]}')

    # Load slope unit geometries
    print('\nLoading slope unit geometries...')
    gdf = gpd.read_file(PROC_DIR / 'slope_units.gpkg')
    if 'cat' in gdf.columns:
        gdf = gdf.rename(columns={'cat': 'su_id'})
    elif 'value' in gdf.columns:
        gdf = gdf.rename(columns={'value': 'su_id'})
    else:
        gdf['su_id'] = range(len(gdf))
    gdf = gdf.set_index('su_id')
    print(f'  Loaded {len(gdf)} slope units')

    # Load features CSV for popup data
    features_csv = PROC_DIR / 'su_features.csv'
    if features_csv.exists():
        features_df = pd.read_csv(features_csv)
        print(f'  Loaded features: {features_df.shape}')
    else:
        features_df = None
        print('  No features CSV found')

    # Model 1: SAGE hydro-only
    print('\n' + '=' * 70)
    print('MODEL 1: SAGE HYDRO-ONLY')
    print('=' * 70)
    model_hydro = load_sage_model(
        Path(r'D:\TERRAPYGE\models\ablation\sage_hydro_only.pt'),
        data['su'].x.shape[1]
    )
    empty = torch.zeros((2, 0), dtype=torch.long)
    hydro_edge_dict = {
        ('su', 'spatial', 'su'): empty,
        ('su', 'hydro', 'su'): data['su', 'hydro', 'su'].edge_index,
    }
    probs_hydro = run_inference(model_hydro, data, hydro_edge_dict)
    print(f'  Predictions: min={probs_hydro.min():.4f}, max={probs_hydro.max():.4f}, mean={probs_hydro.mean():.4f}')

    hydro_metrics = generate_hazard_map(gdf, probs_hydro, 'sage_hydro_only', 'hydro_only', features_df)

    # Model 2: SAGE dual-edge
    print('\n' + '=' * 70)
    print('MODEL 2: SAGE DUAL-EDGE')
    print('=' * 70)
    model_dual = load_sage_model(
        Path(r'D:\TERRAPYGE\models') / 'gnn_dual_edge.pt',
        data['su'].x.shape[1]
    )
    dual_edge_dict = {
        ('su', 'spatial', 'su'): data['su', 'spatial', 'su'].edge_index,
        ('su', 'hydro', 'su'): data['su', 'hydro', 'su'].edge_index,
    }
    probs_dual = run_inference(model_dual, data, dual_edge_dict)
    print(f'  Predictions: min={probs_dual.min():.4f}, max={probs_dual.max():.4f}, mean={probs_dual.mean():.4f}')

    dual_metrics = generate_hazard_map(gdf, probs_dual, 'sage_dual_edge', 'dual_edge', features_df)

    # Comparison
    print('\n' + '=' * 70)
    print('COMPARISON')
    print('=' * 70)
    print(f'{"Metric":<25} {"SAGE hydro-only":>15} {"SAGE dual-edge":>15}')
    print('-' * 55)
    print(f'{"Test AUC":<25} {0.9500:>15.4f} {0.9491:>15.4f}')
    print(f'{"Prob min":<25} {probs_hydro.min():>15.4f} {probs_dual.min():>15.4f}')
    print(f'{"Prob max":<25} {probs_hydro.max():>15.4f} {probs_dual.max():>15.4f}')
    print(f'{"Prob mean":<25} {probs_hydro.mean():>15.4f} {probs_dual.mean():>15.4f}')
    print(f'{"Prob std":<25} {probs_hydro.std():>15.4f} {probs_dual.std():>15.4f}')
    print('-' * 55)
    for cls in ['Very Low', 'Low', 'Moderate', 'High', 'Very High']:
        h_count = hydro_metrics['hazard_distribution'][cls]
        d_count = dual_metrics['hazard_distribution'][cls]
        print(f'{cls + " count":<25} {h_count:>15} {d_count:>15}')
    print('=' * 70)

    # Generate comparison plot
    print('\nGenerating comparison plot...')
    plot_comparison(probs_hydro, probs_dual)

    # Save final metrics
    final_metrics = {
        'sage_hydro_only': {
            'model_name': 'sage_hydro_only',
            'edge_config': 'hydro_only',
            'test_auc': 0.9500,
            'test_ap': 0.8208,
            'test_f1': 0.7085,
            'hazard_distribution': hydro_metrics['hazard_distribution'],
        },
        'sage_dual_edge': {
            'model_name': 'sage_dual_edge',
            'edge_config': 'dual_edge',
            'test_auc': 0.9491,
            'test_ap': 0.8179,
            'test_f1': 0.7114,
            'hazard_distribution': dual_metrics['hazard_distribution'],
        },
    }
    metrics_path = PROC_DIR / 'final_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f'\nFinal metrics saved to {metrics_path}')

    print('\n' + '=' * 70)
    print('HAZARD MAP GENERATION COMPLETE')
    print('=' * 70)
    print('\nOutput files:')
    print(f'  {PROC_DIR / "hazard_sage_hydro_only.gpkg"}')
    print(f'  {PROC_DIR / "hazard_sage_hydro_only.geojson"}')
    print(f'  {PROC_DIR / "hazard_sage_hydro_only.html"}')
    print(f'  {PROC_DIR / "hazard_sage_hydro_only.csv"}')
    print(f'  {PROC_DIR / "hazard_sage_dual_edge.gpkg"}')
    print(f'  {PROC_DIR / "hazard_sage_dual_edge.geojson"}')
    print(f'  {PROC_DIR / "hazard_sage_dual_edge.html"}')
    print(f'  {PROC_DIR / "hazard_sage_dual_edge.csv"}')
    print(f'  {PROC_DIR / "final_metrics.json"}')
    print(f'  {FIGURES_DIR / "hazard_comparison.png"}')
    print('\nOpen the .html files in your browser to explore the interactive maps.')


if __name__ == '__main__':
    main()