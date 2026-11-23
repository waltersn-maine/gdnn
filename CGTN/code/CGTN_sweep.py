import os, csv, time, torch, torch.nn.functional as F, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import CGTN_torchdata as D
from CGTN_torchmodels import HeteroSAGE, HeteroGT

SLOPE = "../data/constructs/sledG_torch_pitedge_pitpit_2010.npz"
BASIN = "../data/constructs/basin_mushgraph2010.npz"
LINKS = "../data/constructs/pit_basin_links2010.npz"
OUT = "../results"; CK = f"{OUT}/ckpt"; os.makedirs(CK, exist_ok=True)
CSV = f"{OUT}/sweep_results.csv"
SEEDS, MODES, EPOCHS = [0, 13, 42], ["full", "shuffle", "none"], 600
dev = "cuda" if torch.cuda.is_available() else "cpu"

def r2(p, y, idx):
    t = y[idx]; return (1 - ((p[idx]-t)**2).sum() / ((t-t.mean())**2).sum()).item()

def log(row):
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["ts","arch","mode","seed","best_ep","val_r2","test_r2","sec"])
        w.writerow(row)

def vram(tag=""):
    if not torch.cuda.is_available(): return
    free, total = torch.cuda.mem_get_info()
    print(f"[{tag}] alloc {torch.cuda.memory_allocated()/1e9:.2f} GB | "
          f"reserved {torch.cuda.memory_reserved()/1e9:.2f} GB | "
          f"free {free/1e9:.2f} / {total/1e9:.2f} GB")

def train(make, A, edges, x_b, split, tag, seed):
    torch.manual_seed(seed)
    tr, va, te = [s.to(dev) for s in split]
    x_s, y = A["x_s"].to(dev), A["y"].to(dev)
    m = make().to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    best = (-1e9, 0, 0, None)
    for ep in range(1, EPOCHS+1):
        m.train(); opt.zero_grad()
        out = m(x_s, x_b, edges); F.mse_loss(out[tr], y[tr]).backward(); opt.step()
        if ep % 25 == 0:
            m.eval()
            with torch.no_grad(): out = m(x_s, x_b, edges); vr, tr2 = r2(out, y, va), r2(out, y, te)
            if vr > best[0]: best = (vr, tr2, ep, out.detach().cpu())
    torch.save({"state": m.state_dict(), "pred": best[3], "val": best[0], "test": best[1],
                "split": [s.cpu() for s in split]}, f"{CK}/{tag}.pt")
    return best

def run():
    A = D.load_artifacts(SLOPE, BASIN, LINKS)
    in_s, in_b = A["x_s"].shape[1], A["x_b"].shape[1]
    y = A["y"].numpy()
    edge_cache = {m: D.build_edges(A, m).to(dev) for m in MODES}
    vram("edges cached")

    # ---- Phase 1: GBT (fast, same split/test nodes as the nets) ----
    for seed in SEEDS:
        tr, va, te = D.make_split(A, seed); t0 = time.time()
        g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, l2_regularization=1.0, random_state=seed)
        g.fit(A["X_raw"][tr.numpy()], y[tr.numpy()])
        gt = r2_score(y[te.numpy()], g.predict(A["X_raw"][te.numpy()]))
        log([time.strftime("%H:%M:%S"), "GBT", "-", seed, "-", "-", f"{gt:.4f}", f"{time.time()-t0:.0f}"])
        print(f"seed {seed} | GBT test R² {gt:.4f}")

    # ---- Phase 2: GNN / GTN sweep (overnight) ----
    for seed in SEEDS:
        split = D.make_split(A, seed)
        perm = torch.randperm(A["N_b"], generator=torch.Generator().manual_seed(seed))
        for mode in MODES:
            x_b = D.basin_feats(A, mode, perm)
            x_b = x_b.to(dev) if x_b is not None else None
            for arch, make in [("GNN", lambda: HeteroSAGE(in_s, in_b, layers=5)),
                               ("GTN", lambda: HeteroGT(in_s, in_b, heads=4, layers=4))]:
                tag = f"{arch}_{mode}_s{seed}"; t0 = time.time()
                vr, tr2, ep, _ = train(make, A, edge_cache[mode], x_b, split, tag, seed)
                log([time.strftime("%H:%M:%S"), arch, mode, seed, ep, f"{vr:.4f}", f"{tr2:.4f}", f"{time.time()-t0:.0f}"])
                print(f"seed {seed} | {arch} {mode}: val {vr:.3f} test {tr2:.3f} ({ep}ep)")

if __name__ == "__main__":
    run()