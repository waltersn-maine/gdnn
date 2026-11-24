import time, csv, torch, torch.nn.functional as F, numpy as np
from torch_geometric.utils import to_undirected
import CGTN_torchdata as D
from CGTN_torchmodels import HeteroGT

SLOPE="../data/constructs/sledG_torch_pitedge_pitpit_2010.npz"
BASIN="../data/constructs/basin_mushgraph2010.npz"
LINKS="../data/constructs/pit_basin_links2010.npz"
CSV="../results/sweep_results.csv"; CK="../results/ckpt"
SEEDS=[0,13,42]; EPOCHS=600
dev="cuda" if torch.cuda.is_available() else "cpu"

def r2(p,y,i): t=y[i]; return (1-((p[i]-t)**2).sum()/((t-t.mean())**2).sum()).item()
def log(row):
    with open(CSV,"a",newline="") as f: csv.writer(f).writerow(row)

def train(x_s, edges, in_b, split, y, tag, seed):
    torch.manual_seed(seed)
    tr,va,te=[s.to(dev) for s in split]
    xs,yv,ei=x_s.to(dev),y.to(dev),edges.to(dev)
    m=HeteroGT(x_s.shape[1], in_b, heads=4, layers=4).to(dev)
    opt=torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    best=(-1e9,0,0,None)
    for ep in range(1,EPOCHS+1):
        m.train(); opt.zero_grad()
        out=m(xs,None,ei); F.mse_loss(out[tr],yv[tr]).backward(); opt.step()
        if ep%25==0:
            m.eval()
            with torch.no_grad(): out=m(xs,None,ei); vr,t2=r2(out,yv,va),r2(out,yv,te)
            if vr>best[0]: best=(vr,t2,ep,out.detach().cpu())
    torch.save({"state":m.state_dict(),"pred":best[3],"val":best[0],"test":best[1],
                "split":[s.cpu() for s in split]}, f"{CK}/{tag}.pt")
    return best

def run():
    A=D.load_artifacts(SLOPE,BASIN,LINKS); N_s=A["N_s"]; y=A["y"]; in_b=A["x_b"].shape[1]
    feat=list(np.load(SLOPE,allow_pickle=True)["feat_names"])
    keep=[i for i,f in enumerate(feat) if f!="path_dist_pit"]        # -> 11 features
    x11=A["x_s"][:,keep]
    drain=to_undirected(torch.cat([A["e_tree"],A["e_pitskip"],A["e_pitpit"]],1), num_nodes=N_s)
    E=drain.size(1)
    print("11 feats:", [feat[i] for i in keep]); print(f"drainage edges (directed): {E:,}")
    for seed in SEEDS:
        split=D.make_split(A,seed)
        g=torch.Generator().manual_seed(seed)                        # fresh random graph per seed
        s=torch.randint(0,N_s,(E//2,),generator=g); d=torch.randint(0,N_s,(E//2,),generator=g)
        mk=s!=d
        rand=to_undirected(torch.stack([s[mk],d[mk]]), num_nodes=N_s)
        np.savez_compressed(f"../results/random_graph_s{seed}.npz",
                            edge_index=rand.to(torch.int32).numpy(), n_nodes=N_s)
        for tag,edges in [("none11",drain),("random",rand)]:
            t0=time.time()
            vr,t2,ep,_=train(x11,edges,in_b,split,y,f"GTN_{tag}_s{seed}",seed)
            log([time.strftime("%H:%M:%S"),"GTN",tag,seed,ep,f"{vr:.4f}",f"{t2:.4f}",f"{time.time()-t0:.0f}"])
            print(f"seed{seed} GTN {tag}: val {vr:.3f} test {t2:.3f} edges {edges.size(1):,} ({ep}ep)")

if __name__=="__main__": run()
