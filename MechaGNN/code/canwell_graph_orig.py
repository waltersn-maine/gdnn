"""
canwell_graph_orig.py
----------------
07Nov26: revising the original from March2026
in order to reproduce or question early results
of improved performance from included glacier basin
----------------
Builds a NetworkX graph over the Canwell north slope + basin domain.

Step 1: Load rasters and create nodes
Step 2: Assign node attributes (elevation, slope, aspect, doubslope, DoD)
Step 3: Assign node type flags (is_slope, is_basin)
Step 4: Connect 8-neighbors with edge weights
Step 5: Save graph as pickle

Usage:
folder: MechaGNN/code
    python3 canwell_graph_orig.py

Requires: rasterio, networkx, numpy, geopandas, shapely, pickle
"""

import os
import numpy as np
import rasterio
from rasterio.transform import xy
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import pickle
import psutil

# ── USER INPUT ────────────────────────────────────────────────────────────────
DATA_DIR = "/home/samuelnwalters/Geodyne/gdnn/MechaGNN/data/"  # fill in your path
# ─────────────────────────────────────────────────────────────────────────────

DEM = "elevs/nwdomain2000_elev5m.tif"
SLOPE_FEAT = "elevs/nwdomain2000_slope5m.tif"
ASPECT = "elevs/nwdomain2000_aspect5m.tif"
DOUBSLP = "elevs/nwdomain2000_doubslope5m.tif"
DDEM = "elevs/nwdomain2025_00_ddem5m.tif"
SLOPE_GPKG = "outlines/nwslope_utm.gpkg"
BASIN = "outlines/wbasin_utm.gpkg"
OUTPUT = "constructs/canwell_graphNov26.pkl"


DEM_PATH      = os.path.join(DATA_DIR, DEM)
SLOPE_PATH    = os.path.join(DATA_DIR, SLOPE_FEAT)
ASPECT_PATH   = os.path.join(DATA_DIR, ASPECT)
DOUBSLP_PATH  = os.path.join(DATA_DIR, DOUBSLP)
DIFF_PATH     = os.path.join(DATA_DIR, DDEM)
SLOPE_POLY    = os.path.join(DATA_DIR, SLOPE_GPKG)
BASIN_POLY    = os.path.join(DATA_DIR, BASIN)
OUTPUT_PATH   = os.path.join(DATA_DIR, OUTPUT)

NODATA = -9999

# ── STEP 1: LOAD RASTERS ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading rasters")
print("=" * 60)

def load_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        transform = src.transform
        crs = src.crs
    return data, transform, crs

dem,     transform, crs = load_raster(DEM_PATH)
slope,   _,         _   = load_raster(SLOPE_PATH)
aspect,  _,         _   = load_raster(ASPECT_PATH)
doubslp, _,         _   = load_raster(DOUBSLP_PATH)
diff,    _,         _   = load_raster(DIFF_PATH)

height, width = dem.shape
cell_size = 5 # Fill in cell size (m resolution) from raster
print(f"Raster shape: {height} rows x {width} cols")
print(f"Total pixels: {height * width:,}")
print(f"Cell size (Verify this!): {cell_size}")

# valid mask: pixel must have a DEM value
valid_mask = ~np.isnan(dem)
valid_count = np.sum(valid_mask)
print(f"Valid pixels: {valid_count:,}")
print(f"RAM usage: {psutil.virtual_memory().percent:.1f}%")


# ── STEP 2: LOAD POLYGONS AND BUILD POINT-IN-POLYGON MASKS ───────────────────
print("\n" + "=" * 60)
print("STEP 2: Building slope/basin masks from polygons")
print("=" * 60)

slope_poly = gpd.read_file(SLOPE_POLY).geometry.union_all()
basin_poly = gpd.read_file(BASIN_POLY).geometry.union_all()

# Build coordinate arrays for valid pixels only
rows, cols = np.where(valid_mask)
xs, ys = xy(transform, rows, cols)  # real-world coordinates
xs = np.array(xs)
ys = np.array(ys)

print(f"Classifying {valid_count:,} pixels into slope/basin...")

# Vectorized point-in-polygon using shapely prepared geometry
from shapely import prepare, contains_xy
prepare(slope_poly)
prepare(basin_poly)

in_slope = contains_xy(slope_poly, xs, ys)
in_basin = contains_xy(basin_poly, xs, ys)

print(f"  Slope nodes: {np.sum(in_slope):,}")
print(f"  Basin nodes: {np.sum(in_basin):,}")
print(f"  Overlap:     {np.sum(in_slope & in_basin):,}")
print(f"RAM usage: {psutil.virtual_memory().percent:.1f}%")

# ── STEP 3: BUILD GRAPH ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Building graph nodes and edges")
print("=" * 60)

G = nx.Graph()

# Build a lookup: (row, col) -> node_id for edge construction
node_id_map = {}
node_id = 0

print("  Adding nodes...")
for i, (r, c) in enumerate(zip(rows, cols)):
    if i % 500000 == 0:
        ram = psutil.virtual_memory().percent
        print(f"    Node {i:,}/{valid_count:,}, RAM: {ram:.1f}%")

    G.add_node(node_id,
        dem_idx   = (int(r), int(c)),
        elev      = float(dem[r, c]),
        slope     = float(slope[r, c])    if not np.isnan(slope[r, c])   else None,
        aspect    = float(aspect[r, c])   if not np.isnan(aspect[r, c])  else None,
        doubslope = float(doubslp[r, c])  if not np.isnan(doubslp[r, c]) else None,
        diff      = float(diff[r, c]) if (r < diff.shape[0] and c < diff.shape[1] and not np.isnan(diff[r, c])) else None,
        is_slope  = bool(in_slope[i]),
        is_basin  = bool(in_basin[i]),
    )
    node_id_map[(int(r), int(c))] = node_id
    node_id += 1

print(f"  Total nodes added: {G.number_of_nodes():,}")

# ── STEP 4: ADD EDGES ─────────────────────────────────────────────────────────
print("\n  Adding edges (8-connectivity)...")

neighbors = [(-1,0),(1,0),(0,1),(0,-1),
             (-1,-1),(-1,1),(1,-1),(1,1)]

edge_count = 0
for i, (r, c) in enumerate(zip(rows, cols)):
    if i % 500000 == 0:
        ram = psutil.virtual_memory().percent
        print(f"    Edge progress {i:,}/{valid_count:,}, RAM: {ram:.1f}%")

    current_id = node_id_map[(int(r), int(c))]
    current_elev = dem[r, c]

    for dr, dc in neighbors:
        nr, nc = r + dr, c + dc
        if (int(nr), int(nc)) in node_id_map:
            neighbor_id = node_id_map[(int(nr), int(nc))]
            # only add edge once (undirected)
            if neighbor_id > current_id:
                dist = cell_size * np.sqrt(dr**2 + dc**2) 
                neigh_slope = abs(current_elev - dem[nr, nc])/dist
                G.add_edge(current_id, neighbor_id, weight=float(neigh_slope))
                edge_count += 1

print(f"  Total edges added: {G.number_of_edges():,}")
print(f"RAM usage: {psutil.virtual_memory().percent:.1f}%")

# ── STEP 5: SAVE ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Saving graph")
print("=" * 60)

save_dict = {
    'G':          G,
    'transform':  transform,
    'crs':        crs,
    'dem_data':   dem,
    'valid_mask': valid_mask,
    'shape':      (height, width),
}

with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(save_dict, f)

print(f"Graph saved to:\n  {OUTPUT_PATH}")
print(f"Nodes: {G.number_of_nodes():,}")
print(f"Edges: {G.number_of_edges():,}")
print(f"RAM usage: {psutil.virtual_memory().percent:.1f}%")
print("\nDone.")

# ── STEP 6: metadata sidecar ──────────────────────────────────────
import json, datetime

meta = {
    "file": OUTPUT_PATH,
    "created": datetime.date.today().isoformat(),
    "built_by": "canwell_graph_orig.py",
    "description": "8-neighbour graph over Canwell NW slope + W basin, 5 m grid.",
    "grid": {
        "shape_rows_cols": [int(height), int(width)],
        "resolution_m": cell_size,
        "epsg": crs.to_epsg(),
        "transform_abcdef": [float(v) for v in list(transform)[:6]],
    },
    "sources": {
        "elev": DEM, "ddem": DDEM, "slope": SLOPE_FEAT,
        "aspect": ASPECT, "doubslope": DOUBSLP,
        "basin_mask": BASIN, "slope_mask": SLOPE_GPKG,
        "epoch_pair": "2025-08-04 LiDAR minus 2000 Isabel Pass IfSAR (DTMs)",
    },
    "counts": {
        "nodes": int(G.number_of_nodes()),
        "basin_nodes": int(np.sum(in_basin)),
        "slope_nodes": int(np.sum(in_slope)),
        "edges": int(G.number_of_edges()),
        
 
    },
    "objects": {
        "G": "the networkx graph",
        "crs, transform, shape": "as expected",
        "valid_mask": "as expected",
        "dem_data": "the original DEM",


    },
    "conventions": {
        "ddem_sign": "negative = surface lowering (later minus earlier)",
        "aspect": "raw degrees 0-360, 0 for flat; encode sin/cos at prep",
        "edge_weights": "none (SAGEConv mean aggregation ignores them)",
        "deferred_to_prep": "aspect sin/cos, basin terrain neutral-fill, scaling, train/val/test split",
    },
}

META_PATH   = os.path.join(DATA_DIR, "constructs/canwell_graphNov26.meta.json")
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)
print("wrote ../data/constructs/canwell_graphNov26.meta.json")
