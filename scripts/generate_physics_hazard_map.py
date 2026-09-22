"""TERRAPYGE: generate the physics-informed hazard map.

Loads the physics-informed graph and a trained GNN model selected via the
``inference:`` block in config.yaml, runs inference, and exports a 4-class
susceptibility map (Low / Moderate / High / Very High) plus an interactive
HTML map.

Outputs (data/processed/buhisan/):
  physics_hazard_map.gpkg
  physics_hazard_map.geojson
  physics_hazard_map.csv
  physics_hazard_map.html
  physics_final_metrics.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import folium
from folium.plugins import Fullscreen, MiniMap, MeasureControl, Search

from src.terrapyge.data.graph import load_graph
from src.terrapyge.models.gnn import get_model
from src.terrapyge.models.experiments import get_edge_configs, N_FEATURES
from src.terrapyge.models.training import load_config
from src.terrapyge.utils.paths import (
    ABLATION_MODELS,
    PROCESSED_BUHISAN as PROC_DIR,
    ROOT,
)

CONFIG = load_config(str(ROOT / 'config.yaml'))
INFER_CFG = CONFIG.get('inference', {})
GNN_CFG = CONFIG.get('gnn', {})
HAZARD_CFG = CONFIG.get('hazard', {})

GRAPH = PROC_DIR / 'buhisan_hetero_physics.pt'
MODEL_NAME = INFER_CFG.get('model', 'with_physics__heterogcn__dual_edge')
MODEL_PATH = ABLATION_MODELS / f'{MODEL_NAME}.pt'
EDGE_CONFIG = INFER_CFG.get('edge_config', 'dual_edge')
PHYS_CSV = PROC_DIR / 'physics_features.csv'
SU_GPKG = PROC_DIR / 'slope_units.gpkg'

CLASSES = HAZARD_CFG.get('class_labels', ['Low', 'Moderate', 'High', 'Very High'])

CLASS_COLORS = {
    'Low': '#2E7D32',
    'Moderate': '#8BC34A',
    'High': '#FF9800',
    'Very High': '#F44336',
}


def quantile_thresholds(probs, phys_classes):
    """Derive probability thresholds matching the physics class proportions.

    Physics classes: 0=Low, 1=Moderate, 2=High, 3=Very High.
    Returns thresholds [t_moderate, t_high, t_very_high].
    """
    counts = pd.Series(phys_classes).value_counts(normalize=True)
    p_low = float(counts.get(0, 0.85))
    p_mod = float(counts.get(1, 0.12))
    p_high = float(counts.get(2, 0.02))
    p_vh = float(counts.get(3, 0.01))
    t_mod = float(np.quantile(probs, p_low))
    t_high = float(np.quantile(probs, p_low + p_mod))
    t_vh = float(np.quantile(probs, p_low + p_mod + p_high))
    return [t_mod, t_high, t_vh]


def hazard_class(p, thresholds):
    if p < thresholds[0]:
        return CLASSES[0]
    elif p < thresholds[1]:
        return CLASSES[1]
    elif p < thresholds[2]:
        return CLASSES[2]
    else:
        return CLASSES[3]


def run_inference():
    data = load_graph(GRAPH)
    data['su'].x = data['su'].x[:, :N_FEATURES].contiguous()

    model = get_model(
        model_type=GNN_CFG.get('model_type', 'HeteroGCN'),
        in_channels=N_FEATURES,
        hidden_channels=GNN_CFG.get('hidden_channels', 64),
        out_channels=2,
        num_layers=GNN_CFG.get('num_layers', 3),
        dropout=GNN_CFG.get('dropout', 0.2),
    )
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    edge_index_dict = get_edge_configs(data)[EDGE_CONFIG]
    x_dict = {'su': data['su'].x}
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    return probs


def build_folium(gdf, probs, thresholds, out_path):
    gdf_4326 = gdf.to_crs(epsg=4326)
    b = gdf_4326.total_bounds  # [minx, miny, maxx, maxy]
    center = [(b[1] + b[3]) / 2.0, (b[0] + b[2]) / 2.0]

    m = folium.Map(
        location=center,
        zoom_start=14,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        zoom_control=True,
    )
    folium.TileLayer(tiles='OpenStreetMap', name='OpenStreetMap',
                     attr='OpenStreetMap').add_to(m)

    title = ('<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
             'z-index:9999;background:white;padding:10px 20px;border:2px solid grey;'
             'border-radius:5px;font-size:15px;font-weight:bold;'
             'box-shadow:2px 2px 6px rgba(0,0,0,0.3);">'
             f'TERRAPYGE: Physics-Informed Landslide Susceptibility '
             f'({MODEL_NAME}) \u2014 Buhisan Watershed</div>')
    m.get_root().html.add_child(folium.Element(title))

    groups = {c: folium.FeatureGroup(name=f'Class: {c}') for c in CLASSES}

    for idx, row in gdf_4326.iterrows():
        p = float(row['landslide_prob'])
        cls = row['hazard_class']
        color = CLASS_COLORS[cls]
        fs = row.get('static_fs', float('nan'))
        dn = row.get('newmark_dn_cm', float('nan'))
        slope = row.get('slope_mean', float('nan'))
        popup = (f"<b>Slope Unit: {idx}</b><br>"
                 f"<b>Susceptibility prob.:</b> {p:.3f}<br>"
                 f"<b>Class:</b> {cls}<br>"
                 f"<b>Slope:</b> {slope:.1f}\u00b0<br>"
                 f"<b>Static FS:</b> {fs:.2f}<br>"
                 f"<b>Newmark Dn:</b> {dn:.2f} cm")
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x, c=color: {'fillColor': c, 'color': 'black',
                                               'weight': 0.5, 'fillOpacity': 0.7},
            tooltip=f"SU {idx} | {cls} | {p:.3f}",
            popup=folium.Popup(popup, max_width=300),
            name=f'{idx} {cls}',
        ).add_to(groups[cls])

    for c, g in groups.items():
        g.add_to(m)

    allg = folium.FeatureGroup(name='All Slope Units')
    for c, g in groups.items():
        for child in g._children.values():
            if isinstance(child, folium.GeoJson):
                child.add_to(allg)
    allg.add_to(m)

    Search(layer=allg, geom_type='Polygon', placeholder='Search SU ID or class...',
           collapsed=False, search_label='name', weight=3).add_to(m)
    folium.LayerControl().add_to(m)
    Fullscreen().add_to(m)
    m.add_child(MiniMap(toggle_display=True, position='bottomright'))
    MeasureControl(position='topleft').add_to(m)

    t_mod, t_high, t_vh = thresholds
    legend_rows = [
        (CLASSES[0], CLASS_COLORS[CLASSES[0]], f'&lt;{t_mod:.2f}'),
        (CLASSES[1], CLASS_COLORS[CLASSES[1]], f'{t_mod:.2f}-{t_high:.2f}'),
        (CLASSES[2], CLASS_COLORS[CLASSES[2]], f'{t_high:.2f}-{t_vh:.2f}'),
        (CLASSES[3], CLASS_COLORS[CLASSES[3]], f'&gt;{t_vh:.2f}'),
    ]
    legend_body = ''.join(
        f'<i style="background:{c};width:16px;height:16px;display:inline-block;"></i> {lbl} ({rng})<br>'
        for lbl, c, rng in legend_rows
    )
    legend = ('<div style="position:fixed;bottom:50px;left:50px;width:230px;'
              'background:white;border:2px solid grey;z-index:9999;font-size:13px;'
              'padding:10px;border-radius:5px;">'
              '<b>Physics-Informed Susceptibility</b><br>'
              f'{legend_body}'
              '<small>Quantile thresholds from physics class proportions</small><br>'
              '<hr style="margin:5px 0;"><small>Toggle basemap: top-right</small></div>')
    m.get_root().html.add_child(folium.Element(legend))

    m.save(str(out_path))


def main():
    print('=' * 70)
    print('TERRAPYGE: PHYSICS-INFORMED HAZARD MAP')
    print('=' * 70)
    print(f'Model: {MODEL_NAME}  |  Edge config: {EDGE_CONFIG}')

    print('Running inference...')
    probs = run_inference()
    print(f'  Prob min={probs.min():.4f} max={probs.max():.4f} mean={probs.mean():.4f}')

    print('Loading slope units...')
    gdf = gpd.read_file(SU_GPKG)
    if 'cat' in gdf.columns:
        gdf = gdf.rename(columns={'cat': 'su_id'})
    elif 'value' in gdf.columns:
        gdf = gdf.rename(columns={'value': 'su_id'})
    gdf = gdf.set_index('su_id').sort_index()

    phys = pd.read_csv(PHYS_CSV).set_index('su_id').sort_index()
    thresholds = quantile_thresholds(probs, phys['physics_class'].values)
    print(f'  Quantile thresholds (Moderate/High/VeryHigh): '
          f'{thresholds[0]:.4f} / {thresholds[1]:.4f} / {thresholds[2]:.4f}')

    gdf['landslide_prob'] = probs
    gdf['hazard_class'] = [hazard_class(p, thresholds) for p in probs]
    gdf['static_fs'] = phys['static_fs'].values
    gdf['newmark_dn_cm'] = phys['newmark_dn_cm'].values

    # feature slope for popup
    feats = pd.read_csv(PROC_DIR / 'su_features.csv').set_index('su_id').sort_index()
    gdf['slope_mean'] = feats['slope_mean'].values

    dist = (pd.Series([hazard_class(p, thresholds) for p in probs])
            .value_counts().reindex(CLASSES).fillna(0).astype(int))
    print('\nHazard class distribution:')
    print(dist.to_string())

    gdf.drop(columns=['slope_mean']).to_file(PROC_DIR / 'physics_hazard_map.gpkg', driver='GPKG')
    gdf.drop(columns=['slope_mean']).to_file(PROC_DIR / 'physics_hazard_map.geojson', driver='GeoJSON')
    pd.DataFrame({
        'su_id': gdf.index,
        'landslide_prob': probs,
        'hazard_class': gdf['hazard_class'].values,
    }).to_csv(PROC_DIR / 'physics_hazard_map.csv', index=False)
    print('\nSaved GPKG / GeoJSON / CSV')

    print('Building interactive map...')
    build_folium(gdf, probs, thresholds, PROC_DIR / 'physics_hazard_map.html')
    print(f'Saved: {PROC_DIR / "physics_hazard_map.html"}')

    metrics = {
        'model': MODEL_NAME,
        'edge_config': EDGE_CONFIG,
        'n_slope_units': int(len(gdf)),
        'prob_min': float(probs.min()),
        'prob_max': float(probs.max()),
        'prob_mean': float(probs.mean()),
        'prob_std': float(probs.std()),
        'hazard_distribution': {k: int(v) for k, v in dist.items()},
        'thresholds_moderate_high_veryhigh': thresholds,
    }
    with open(PROC_DIR / 'physics_final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Saved: {PROC_DIR / "physics_final_metrics.json"}')
    print('\nDONE')


if __name__ == '__main__':
    main()
