import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TransformerConv

class HeteroSAGE(nn.Module):
    def __init__(self, in_s, in_b, hid=64, layers=4):
        super().__init__()
        self.enc_s, self.enc_b = nn.Linear(in_s, hid), nn.Linear(in_b, hid)
        self.convs = nn.ModuleList([SAGEConv(hid, hid) for _ in range(layers)])
        self.head = nn.Linear(hid, 1)
    def forward(self, x_s, x_b, ei):
        h = self.enc_s(x_s) if x_b is None else torch.cat([self.enc_s(x_s), self.enc_b(x_b)], 0)
        for c in self.convs: h = F.relu(c(h, ei))
        return self.head(h).squeeze(-1)

class HeteroGT(nn.Module):
    def __init__(self, in_s, in_b, hid=64, heads=4, layers=4):
        super().__init__()
        self.enc_s, self.enc_b = nn.Linear(in_s, hid), nn.Linear(in_b, hid)
        self.convs = nn.ModuleList([TransformerConv(hid, hid//heads, heads=heads) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.head = nn.Linear(hid, 1)
    def forward(self, x_s, x_b, ei, return_attn=False):
        h = self.enc_s(x_s) if x_b is None else torch.cat([self.enc_s(x_s), self.enc_b(x_b)], 0)
        attn = None
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if return_attn and i == len(self.convs)-1:
                out, attn = conv(h, ei, return_attention_weights=True)   # attn = (edge_index, alpha[E,heads])
                h = norm(h + F.relu(out))
            else:
                h = norm(h + F.relu(conv(h, ei)))
        out = self.head(h).squeeze(-1)
        return (out, attn) if return_attn else out