"""
build_graph.py 
-- builds graph for slope + basin domain 
-- clean rebuild of canwell_graph.py
Nodes: domain cells with valid 2010 elevation
Edges: 8-neighbor
Stores RAW per-node attributes + edge index
scale/fill/mask at GNN stage
"""

import numpy as np
import rasterio

# ----------- paths ------------
ELEV   = "../data/elevs/nwclip_elev2010.tif"         # 2010 elevation, whole domain
DDEM   = "../data/elevs/nwclip_ddem_2025_2010.tif"   # 2025-2010 dDEM, whole domain
SLOPE  = "../data/elevs/nwslope_slope2010.tif"        # slope-only (basin = nodata)
ASPECT = "../data/elevs/nwslope_aspect2010.tif"       # slope-only
DOUBS  = "../data/elevs/nwslope_doubslope2010.tif"    # slope-only
BASIN  = "../data/outlines/mask_wbasin.tif"           # basin mask
SLOPEM = "../data/outlines/mask_nwslope.tif"          # slope mask


def load(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        return a, src.transform, src.crs

elev, transform, crs = load(ELEV)
ddem, _, _ = load(DDEM)
slope, _, _ = load(SLOPE)
aspect, _, _ = load(ASPECT)
doubs, _, _ = load(DOUBS)
basin  = load(BASIN)[0]  == 1  # True inside basin
slopem = load(SLOPEM)[0] == 1  # True inside slope

H, W = elev.shape
print("shape:", (H,W),
      "| basin:", int(basin.sum()),
      "| slope:", int(slopem.sum()),
      "| overlap:", int((basin & slopem).sum()),
      "| basin + slope:", int(basin.sum()+slopem.sum()),
      "| domain total (should match basin + slope):", int(np.count_nonzero(~np.isnan(elev))))

# ---------- step 2 : node set + index grid ---------------

domain    = basin | slopem             # total domain is basin OR slope
node_mask = domain & ~np.isnan(elev)  # require valid elevation

n_dropped = int(domain.sum() - node_mask.sum())
print("domain cells without valid elev (dropped):", n_dropped)

rows, cols = np.where(node_mask) # valid row/col indices; row-major = deterministic node order
N = rows.size
idx_grid = np.full((H,W), -1, dtype=np.int64)   # initialize
idx_grid[rows, cols] = np.arange(N)              # assign node index -> (row, col); -1 = not a node

is_basin = basin[rows, cols]                    # basin mask for valid nodes
is_slope = slopem[rows, cols]                   # slope mask for valid nodes
assert not (is_basin & is_slope).any(), "a node is both basin and slope" # verify no overlap
assert (is_basin ^ is_slope).all(),     "a node is neither basin nor slope" # verify none excluded
print("nodes: N = ", N, "| basin:", int(is_basin.sum()), "| slope:", int(is_slope.sum()))

# ------------ step 3: edges: 8-neighbor, vectorized through idx_grid -----------------

offsets = [(0,1), (1,0), (1,1), (1, -1)]  # cover each undirected edge once
src_list, dst_list = [], []

# connects a row- or column-reduced grid in a given direction
for dr, dc in offsets:
    # a -- the base cell grid minus first or final row and/or column
    a = idx_grid[max(0,-dr):H-max(0,dr), max(0,-dc):W-max(0,dc)]
    # b -- the same grid shifted by the offset
    b = idx_grid[max(0, dr):H-max(0,-dr), max(0, dc):W-max(0,-dc)]
    # m -- keep only pairs where both ends are real nodes (i.e. node number > -1)
    m = (a >= 0) & (b >= 0)
    # -- accumulate
    src_list.append(a[m]); dst_list.append(b[m])

src = np.concatenate(src_list)
dst = np.concatenate(dst_list)

# add pairs in both directions for an undirected graph
edge_index = np.vstack([np.concatenate([src, dst]), # both directions
                        np.concatenate([dst, src])]).astype(np.int64)

print("undirected edges:", src.size,
      "| directed:", edge_index.shape[1],
      "| mean degree:", round(edge_index.shape[1] / N, 2))

# --------- step 4: raw node attributes ------------

def at_nodes(a): return a[rows, cols].astype(np.float32)

elev_n   = at_nodes(elev)
slope_n  = at_nodes(slope)     # NaN on basin nodes (neutral fill later)
aspect_n = at_nodes(aspect)    # NaN on basin nodes
doubs_n  = at_nodes(doubs)     # NaN on basin nodes
ddem_n   = at_nodes(ddem)      # NaN where ddem is missing

print("slope (feature) NaN:", int(np.isnan(slope_n).sum()), "(expect basin count = 327106)")
print("ddem NaN total:", int(np.isnan(ddem_n).sum()))
print("  basin ddem NaN:", int(np.isnan(ddem_n[is_basin]).sum()),
      "| slope ddem NaN:", int(np.isnan(ddem_n[is_slope]).sum()))

# ------- step 5: save ----------------

np.savez_compressed(
    "../data/constructs/canwell_graph.npz",
    rows=rows.astype(np.int32), cols=cols.astype(np.int32),
    elev=elev_n, slope=slope_n, aspect=aspect_n, doubslope=doubs_n, ddem=ddem_n,
    is_basin=is_basin, is_slope=is_slope,
    edge_index=edge_index,
    shape=np.array([H,W], dtype=np.int64),
    transform=np.array(list(transform)[:6], dtype=np.float64),
    crs_wkt=crs.to_wkt(),
)

print("saved ../data/constructs/canwell_graph.npz  |  nodes:", N, "edges:", edge_index.shape[1])

# ── STEP 6: metadata sidecar ──────────────────────────────────────
import json, datetime

meta = {
    "file": "canwell_graph.npz",
    "created": datetime.date.today().isoformat(),
    "built_by": "build_graph.py",
    "description": "8-neighbour graph over Canwell NW slope + W basin, 5 m grid.",
    "grid": {
        "shape_rows_cols": [int(H), int(W)],
        "resolution_m": 5,
        "epsg": crs.to_epsg(),
        "transform_abcdef": [float(v) for v in list(transform)[:6]],
    },
    "sources": {
        "elev": ELEV, "ddem": DDEM, "slope": SLOPE,
        "aspect": ASPECT, "doubslope": DOUBS,
        "basin_mask": BASIN, "slope_mask": SLOPEM,
        "epoch_pair": "2025-08-04 LiDAR minus 2010 IfSAR (DTMs)",
    },
    "counts": {
        "nodes": int(N),
        "basin_nodes": int(is_basin.sum()),
        "slope_nodes": int(is_slope.sum()),
        "edges_undirected": int(src.size),
        "edges_directed": int(edge_index.shape[1]),
        "mean_degree": round(edge_index.shape[1] / N, 3),
    },
    "audit": {
        "domain_cells_without_elev_dropped": int(n_dropped),
        "slope_feature_nan": int(np.isnan(slope_n).sum()),
        "ddem_nan_basin": int(np.isnan(ddem_n[is_basin]).sum()),
        "ddem_nan_slope": int(np.isnan(ddem_n[is_slope]).sum()),
        "usable_slope_targets": int((is_slope & ~np.isnan(ddem_n)).sum()),
    },
    "arrays": {
        "rows,cols": "int32 (N,) raster indices, row-major node order",
        "elev": "float32 (N,) real for all nodes",
        "slope,aspect,doubslope": "float32 (N,) real on slope, NaN on basin (neutral-fill at prep)",
        "ddem": "float32 (N,) TARGET on slope / SIGNAL on basin; NaN where missing",
        "is_basin,is_slope": "bool (N,) mutually exclusive",
        "edge_index": "int64 (2,E) directed, both orientations",
        "shape,transform,crs_wkt": "grid georeferencing for writing predictions back",
    },
    "conventions": {
        "ddem_sign": "negative = surface lowering (later minus earlier)",
        "aspect": "raw degrees 0-360, 0 for flat; encode sin/cos at prep",
        "edge_weights": "none (SAGEConv mean aggregation ignores them)",
        "deferred_to_prep": "aspect sin/cos, basin terrain neutral-fill, scaling, train/val/test split",
    },
}

with open("../data/constructs/canwell_graph.meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("wrote ../data/constructs/canwell_graph.meta.json")