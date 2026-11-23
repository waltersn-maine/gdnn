import numpy as np, torch
from torch_geometric.utils import to_undirected

def _z(X):
    m = torch.nanmean(X, 0); v = torch.nanmean((X - m)**2, 0)
    return torch.nan_to_num((X - m) / torch.sqrt(v + 1e-6))

def load_artifacts(slope, basin, links):
    sl, bg, pl = (np.load(p, allow_pickle=True) for p in (slope, basin, links))
    A = dict(N_s=int(sl["X"].shape[0]), N_b=int(bg["X"].shape[0]))
    A["x_s"] = _z(torch.tensor(sl["X"], dtype=torch.float32))
    A["x_b"] = _z(torch.tensor(bg["X"], dtype=torch.float32))
    A["X_raw"] = np.asarray(sl["X"], np.float32)              # GBT eats raw + NaN
    A["y"] = torch.tensor(sl["y"], dtype=torch.float32)
    A["node_rc"] = np.asarray(sl["node_rc"])
    for k in ("e_tree", "e_pitskip", "e_pitpit", "pit_torch"):
        A[k] = torch.tensor(sl[k], dtype=torch.long)
    A["e_basin"] = torch.tensor(bg["edge_index"], dtype=torch.long)
    A["link_pit"] = torch.tensor(pl["link_pit"], dtype=torch.long)
    A["link_basin"] = torch.tensor(pl["link_basin"], dtype=torch.long)
    return A

def build_edges(A, mode):
    N_s, N_b = A["N_s"], A["N_b"]
    e = [A["e_tree"], A["e_pitskip"], A["e_pitpit"]]
    if mode == "none":
        return to_undirected(torch.cat(e, 1), num_nodes=N_s)
    e.append(A["e_basin"] + N_s)
    e.append(torch.stack([A["pit_torch"][A["link_pit"]], A["link_basin"] + N_s]))
    return to_undirected(torch.cat(e, 1), num_nodes=N_s + N_b)

def basin_feats(A, mode, perm):
    if mode == "none": return None
    return A["x_b"][perm] if mode == "shuffle" else A["x_b"]

def make_split(A, seed, fr=(0.70, 0.15)):
    g = torch.Generator().manual_seed(seed)
    valid = torch.where(torch.isfinite(A["y"]))[0]
    perm = valid[torch.randperm(valid.numel(), generator=g)]
    a, b = int(fr[0]*perm.numel()), int((fr[0]+fr[1])*perm.numel())
    return perm[:a], perm[a:b], perm[b:]