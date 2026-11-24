"""Controlled factorial interventions on the reconstructed 2-D Border Core routine.
P=0: no protected initialization, each seed labels only itself and is queued.
F=0: allow any point to propagate after assignment, regardless of radius eligibility.
M=0: replace anisotropy-dependent contact threshold by the size-only branch.
I=1: minibatch initialization; I=0: density-ranked initialization.
These interventions define counterfactual algorithms; they are not historical BC modes.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import cKDTree
from numba import njit
from border_core_fast import _anisotropy_from_indices, _UF
from border_core_grid import _expand_grid

@dataclass
class AblationResult:
    labels: np.ndarray
    n_clusters: int
    coverage: float
    n_queries: int
    n_merges: int
    n_seeds: int
    frac_eligible: float


def fit_ablation(X, I=0, P=1, F=1, M=1, random_state=2026):
    X=np.asarray(X,dtype=np.float64,order='C')
    n,d=X.shape
    if d!=2 or n<11: raise ValueError('Ablation supports 2D with n>=11')
    dist,idx=NearestNeighbors(n_neighbors=11,algorithm='kd_tree',n_jobs=1).fit(X).kneighbors(X)
    kd=np.maximum(dist[:,4],1e-12)
    anis_global=_anisotropy_from_indices(X,idx)
    r=float((1.25 if anis_global>=4.0 else .95)*np.quantile(kd,.9))
    density=1/kd;elig=kd<=r
    tree=cKDTree(X)
    starts=[];coarse_k=0
    if I:
        coarse_k=min(n//3,max(4,min(96,int(np.ceil(1.75*np.sqrt(n))))))
        km=MiniBatchKMeans(n_clusters=coarse_k,random_state=random_state,
            batch_size=min(1024,n),n_init=1,max_iter=40,reassignment_ratio=0.0)
        km.fit(X)
        for k in range(coarse_k):
            inds=np.flatnonzero((km.labels_==k)&elig)
            if len(inds):starts.append(int(inds[np.argmax(density[inds])]))
    ranked=np.lexsort((np.arange(n),-density))
    ordered=np.r_[np.array(sorted(set(starts)),dtype=int),ranked] if starts else ranked
    seeded=np.zeros(n,bool);seeds=[]
    for p in ordered:
        if not elig[p] or seeded[p]:continue
        seeds.append(int(p));seeded[tree.query_ball_point(X[p],2*r)]=True
    seeds=np.asarray(seeds,np.int32)
    ix=np.floor((X[:,0]-X[:,0].min())/r).astype(np.int64)
    iy=np.floor((X[:,1]-X[:,1].min())/r).astype(np.int64)
    stride=int(iy.max())+3;keys=ix*stride+iy
    order=np.argsort(keys,kind='stable').astype(np.int32)
    uniq,st=np.unique(keys[order],return_index=True)
    ed=np.r_[st[1:],len(X)]
    labels=np.full(n,-1,np.int32);fixed=np.zeros(n,np.uint8)
    q_u=[];q_k=[]
    for k,p in enumerate(seeds):
        if P:
            inside=np.asarray(tree.query_ball_point(X[p],r*.45),dtype=int)
            a=inside[elig[inside]&(labels[inside]<0)]
            labels[a]=k;fixed[a]=1
            if labels[p]<0:labels[p]=k;fixed[p]=1
            for u in tree.query_ball_point(X[p],1.45*r):
                if not fixed[u]:q_u.append(u);q_k.append(k)
        else:
            # Only seed owners are assigned initially. No point is marked protected.
            if labels[p]<0: labels[p]=k
            q_u.append(int(p));q_k.append(k)
            for u in tree.query_ball_point(X[p],1.45*r):
                if u!=p:q_u.append(u);q_k.append(k)
    prop_elig=elig.astype(bool) if F else np.ones(n,bool)
    labels,c_u,c_k,c_o,queries,events,checks=_expand_grid(X,prop_elig,density,seeds,labels,fixed,
        ix,iy,stride,uniq,st,ed,order,np.asarray(q_u,np.int32),np.asarray(q_k,np.int32),r)
    contacts={}
    for u,k,owner in zip(c_u,c_k,c_o):
        pair=tuple(sorted((int(k),int(owner))))
        contacts.setdefault(pair,set()).add(int(u))
    sizes=np.bincount(labels[labels>=0],minlength=len(seeds))
    anis=np.ones(len(seeds))
    valid=np.flatnonzero(labels>=0);sort=valid[np.argsort(labels[valid],kind='stable')]
    cut=np.r_[0,np.cumsum(sizes)]
    for k in range(len(seeds)):
        if sizes[k]<8:continue
        pts=X[sort[cut[k]:cut[k+1]]]
        eig=np.linalg.eigvalsh(np.cov(pts,rowvar=False))
        anis[k]=np.sqrt(max(eig[-1],1e-12)/max(eig[0],1e-9))
    uf=_UF(len(seeds));nmerge=0
    for (i,j),nodes in sorted(contacts.items(),key=lambda v:(-len(v[1]),v[0])):
        size_th=max(5,int(np.ceil(2.2*np.log1p(min(sizes[i],sizes[j])))))
        threshold=(2 if anis[i]>=3 or anis[j]>=3 else size_th) if M else size_th
        if len(nodes)>=threshold and uf.union(i,j):nmerge+=1
    out=np.full(n,-1,int);mapped={}
    for p in np.flatnonzero(labels>=0):
        root=uf.find(int(labels[p]));out[p]=mapped.setdefault(root,len(mapped))
    labs,counts=np.unique(out[out>=0],return_counts=True)
    minsize=max(5,min(7,int(np.ceil(.1*np.median(counts))))) if len(counts) else 5
    for l,count in zip(labs,counts):
        if count<minsize:out[out==l]=-1
    uniq_labs=sorted(set(out)-{-1});remap={v:i for i,v in enumerate(uniq_labs)}
    out=np.asarray([remap.get(int(v),-1) for v in out],dtype=np.int32)
    return AblationResult(out,len(uniq_labs),float((out!=-1).mean()),int(queries),nmerge,len(seeds),float(elig.mean()))
