from torch_geometric.nn import GATv2Conv
import torch.nn as nn
from torch import cat

class GATv2Styler(nn.Module):
    def __init__(self, D, H=256):
        super().__init__()
        self.lin_in = nn.Linear(D, H)
        self.c1 = GATv2Conv(H, H//4, heads=4, concat=True, dropout=0.1, edge_dim=1)
        self.c2 = GATv2Conv(H, H//4, heads=4, concat=True, dropout=0.1, edge_dim=1)
        self.n1 = nn.LayerNorm(H)
        self.n2 = nn.LayerNorm(H)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.lin_out = nn.Linear(H, D)
    def forward(self,x,ei,ea):
        h = self.drop(self.act(self.lin_in(x)))
        h = self.drop(self.n1(self.act(self.c1(h,ei,ea))))
        h = self.drop(self.n2(self.act(self.c2(h,ei,ea))))
        return self.lin_out(h)+x

class SimpleUNet(nn.Module):
    def __init__(self,in_ch=64,base=64):
        super().__init__()
        def blk(ci,co): return nn.Sequential(nn.Conv2d(ci,co,3,padding=1),nn.ReLU(),nn.Conv2d(co,co,3,padding=1),nn.ReLU())
        self.e1 = blk(in_ch,base)
        self.p = nn.MaxPool2d(2)
        self.e2 = blk(base,base*2)
        self.e3 = blk(base*2,base*4)
        self.u2 = nn.ConvTranspose2d(base*4,base*2,2,stride=2)
        self.d2 = blk(base*4,base*2)
        self.u1 = nn.ConvTranspose2d(base*2,base,2,stride=2)
        self.d1 = blk(base*2,base)
        self.out = nn.Conv2d(base,3,1)
    def forward(self,x):
        e1=self.e1(x); p1=self.p(e1)
        e2=self.e2(p1); p2=self.p(e2)
        e3=self.e3(p2)
        u2=self.u2(e3)
        d2=self.d2(cat([u2,e2],1))
        u1=self.u1(d2)
        d1=self.d1(cat([u1,e1],1))
        return self.out(d1)
