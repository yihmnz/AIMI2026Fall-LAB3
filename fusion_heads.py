"""Feature-fusion heads. All models return (four_class_logits, optional_covid_logits)."""
import torch
from torch import nn
from torch.nn import functional as F

MODES = ('concat', 'projected_concat', 'static_weighted', 'dynamic_gated', 'gated_concat')

class ConcatHead(nn.Module):
    def __init__(self, dim, dropout, hidden, aux):
        super().__init__()
        self.body=nn.Sequential(nn.LayerNorm(dim),nn.Dropout(dropout),nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(dropout))
        self.classifier=nn.Linear(hidden,4)
        self.aux=nn.Linear(hidden,1) if aux>0 else None
    def forward(self,x):
        h=self.body(x)
        return self.classifier(h), self.aux(h).flatten() if self.aux is not None else None
    def gate_weights(self,x):
        return None

class FusionHead(nn.Module):
    def __init__(self, feature_dims, mode='dynamic_gated', hidden=256, project_dim=128, dropout=.35, aux=0., gate_hidden=64):
        super().__init__()
        if mode not in MODES or mode=='concat':raise ValueError(mode)
        self.mode=mode
        self.feature_dims=list(feature_dims)
        self.n=len(feature_dims)
        self.projections=nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d),nn.Linear(d,project_dim),nn.GELU()) for d in feature_dims
        ])
        if mode=='static_weighted':
            self.gate_logits=nn.Parameter(torch.zeros(self.n))
        elif mode in ('dynamic_gated','gated_concat'):
            self.gate=nn.Sequential(nn.LayerNorm(project_dim*self.n),
                                    nn.Linear(project_dim*self.n,gate_hidden),nn.GELU(),
                                    nn.Linear(gate_hidden,self.n))
        out_dim=project_dim*self.n if mode in ('projected_concat','gated_concat') else project_dim
        self.body=nn.Sequential(nn.LayerNorm(out_dim),nn.Dropout(dropout),nn.Linear(out_dim,hidden),nn.GELU(),nn.Dropout(dropout))
        self.classifier=nn.Linear(hidden,4)
        self.aux=nn.Linear(hidden,1) if aux>0 else None
    def _features(self,x):
        return torch.stack([layer(t) for layer,t in zip(self.projections,torch.split(x,self.feature_dims,dim=1))],dim=1)
    def gate_weights(self,x):
        if self.mode=='static_weighted':
            return self.gate_logits.softmax(0).expand(x.size(0),-1)
        if self.mode in ('dynamic_gated','gated_concat'):
            z=self._features(x)
            return self.gate(z.flatten(1)).softmax(-1)
        return None
    def forward(self,x):
        z=self._features(x)
        if self.mode=='projected_concat':
            fused=z.flatten(1)
        else:
            weights=self.gate_weights(x)
            if self.mode=='gated_concat':
                fused=(z*weights.unsqueeze(-1)).flatten(1)
            else:
                fused=(z*weights.unsqueeze(-1)).sum(1)
        h=self.body(fused)
        return self.classifier(h), self.aux(h).flatten() if self.aux is not None else None

def make_head(mode, dims, hidden, dropout, aux, project_dim=128, gate_hidden=64):
    if mode=='concat':return ConcatHead(sum(dims),dropout,hidden,aux)
    return FusionHead(dims,mode,hidden,project_dim,dropout,aux,gate_hidden)
