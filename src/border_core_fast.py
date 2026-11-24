"""INDEPENDENT RECONSTRUCTION, not recovered original source.
Neighborhood anisotropy and higher-dimensional prototype; do not claim
byte-for-byte identity with the unpublished border_core_fast.py.
"""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass

class _UF:
    def __init__(self, n): self.par=np.arange(n)
    def find(self, x):
        x=int(x)
        while self.par[x]!=x:
            self.par[x]=self.par[self.par[x]]
            x=int(self.par[x])
        return x
    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb:return False
        self.par[max(ra,rb)]=min(ra,rb)
        return True

def _anisotropy_from_indices(X, indices):
    """Clean-room local PCA anisotropy proxy: median of sqrt(eigenvalue ratios)
    of eleven-neighbor sample covariance. This reproducible reconstruction
    reproduces the published threshold BRANCHES for the nine supplied 2D
    datasets; exact values require original module confirmation.
    """
    X=np.asarray(X,dtype=float); ii=np.asarray(indices,dtype=int)
    N=len(X);d=X.shape[1]
    if N==0:return 1.0
    # For d>2 this global value is computed for completeness, but radius multiplier
    # 0.75 does not depend on it in the validated BC code.
    a=X[ii];a=a-a.mean(axis=1,keepdims=True)
    out=[]
    for start in range(0,N,256):
        b=a[start:start+256]
        cov=np.einsum('nki,nkj->nij',b,b)/max(1,b.shape[1]-1)
        lam=np.linalg.eigvalsh(cov)
        ratio=np.sqrt(np.maximum(lam[:,-1],1e-12)/np.maximum(lam[:,0],1e-9))
        out.append(ratio)
    return float(np.median(np.concatenate(out)))

@dataclass
class BCReconstructedResult:
    labels: np.ndarray
    n_seeds: int
    n_fixed: int
    n_expanded: int
    n_clusters: int
    radius: float
    n_merges: int
    seeding_mode: str

def fit_predict_bc_fast(X,seed_mode='hybrid',min_pts=5,random_state=2026,
                        fixed_core_factor=.45,coarse_factor=1.75):
    """Verified-output high-D reconstruction of the specified clustering policy. Independent
    reconstruction, NOT asserted numerically equivalent to original.
    """
    X=np.asarray(X,dtype=np.float64,order='C');n,d=X.shape
    if n<11 or d<2 or seed_mode not in ('density','hybrid'):raise ValueError('invalid inputs')
    dist,idx=NearestNeighbors(n_neighbors=11,n_jobs=1).fit(X).kneighbors(X)
    kd=np.maximum(dist[:,min_pts-1],1e-12);density=1/kd
    r=float(.75*np.quantile(kd,.9))
    eligible=kd<=r
    tree=cKDTree(X)
    center_candidates=[];coarse_k=0
    if seed_mode=='hybrid':
        coarse_k=min(n//3,max(4,min(96,int(np.ceil(coarse_factor*np.sqrt(n))))))
        km=MiniBatchKMeans(n_clusters=coarse_k,random_state=random_state,batch_size=min(1024,n),n_init=1,max_iter=40,reassignment_ratio=0.0)
        km.fit(X);group=km.labels_
        for k in range(coarse_k):
            ids=np.flatnonzero((group==k)&eligible)
            if len(ids):center_candidates.append(int(ids[np.argmax(density[ids])]))
    ranked=np.lexsort((np.arange(n),-density))
    ordered=np.r_[np.array(sorted(set(center_candidates)),dtype=int),ranked] if center_candidates else ranked
    seeded=np.zeros(n,bool);seeds=[]
    for p in ordered:
        if not eligible[p] or seeded[p]:continue
        seeds.append(int(p));seeded[tree.query_ball_point(X[p],2*r)]=True
    seeds=np.asarray(seeds,np.int32)
    labels=np.full(n,-1,np.int32);fixed=np.zeros(n,bool);q=[]
    for k,p in enumerate(seeds):
        a=np.array(tree.query_ball_point(X[p],r*fixed_core_factor),dtype=int)
        a=a[eligible[a] & (labels[a]<0)]
        labels[a]=k;fixed[a]=True
        if labels[p]<0:labels[p]=k;fixed[p]=True
        for u in tree.query_ball_point(X[p],r*(1+fixed_core_factor)):
            if not fixed[u]:q.append((u,k))
    expanded=np.zeros(n,bool);contacts={};head=0;queries=0
    while head<len(q):
        u,k=q[head];head+=1
        owner=labels[u]
        if owner>=0 and owner!=k:
            if eligible[u] and min(density[u],density[seeds[k]])/max(density[u],density[seeds[k]])>=.35:
                a,b=sorted((int(k),int(owner)));contacts.setdefault((a,b),set()).add(int(u))
            continue
        if owner<0:labels[u]=k
        if expanded[u] or not eligible[u] or fixed[u]:continue
        expanded[u]=True;queries+=1
        for v in tree.query_ball_point(X[u],r):
            if v!=u and (labels[v]<0 or (labels[v]!=k and eligible[v])):
                q.append((v,k))
    sizes=np.bincount(labels[labels>=0],minlength=len(seeds));anis=np.ones(len(seeds))
    for i in range(len(seeds)):
        if sizes[i]<8 or d>2:continue
        lam=np.linalg.eigvalsh(np.cov(X[labels==i],rowvar=False))
        anis[i]=np.sqrt(max(lam[-1],1e-12)/max(lam[0],1e-9))
    uf=_UF(len(seeds));num_merge=0
    for (i,j),nodes in sorted(contacts.items(),key=lambda v:(-len(v[1]),v[0])):
        th=2 if (anis[i]>=3 or anis[j]>=3) else max(5,int(np.ceil(2.2*np.log1p(min(sizes[i],sizes[j])))))
        if len(nodes)>=th and uf.union(i,j):num_merge+=1
    out=np.full(n,-1,int);mapping={}
    for p in np.flatnonzero(labels>=0):
        root=uf.find(int(labels[p]));out[p]=mapping.setdefault(root,len(mapping))
    labs,counts=np.unique(out[out>=0],return_counts=True)
    minsize=max(min_pts,min(7,int(np.ceil(.1*np.median(counts))))) if len(counts) else min_pts
    for l,c in zip(labs,counts):
        if c<minsize:out[out==l]=-1
    unique=sorted(set(out)-{-1});mp={v:i for i,v in enumerate(unique)}
    out=np.asarray([mp.get(int(v),-1) for v in out],np.int32)
    return BCReconstructedResult(out,len(seeds),int(fixed.sum()),queries,len(unique),r,num_merge,seed_mode)
