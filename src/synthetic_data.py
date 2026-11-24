"""Pre-specified, newly generated 2D stress tests (not external independent datasets).
Configuration locked before outcome inspection: no BC parameter fitting, no oracle K.
Reproducibility manifest & script sha256 are included in package.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib,time
import numpy as np,pandas as pd,scipy
from sklearn.datasets import make_blobs,make_moons,make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN,HDBSCAN,OPTICS
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from border_core_ablation import fit_ablation
from border_core_fast import _anisotropy_from_indices
NAMES=(['blobs_K3','blobs_K6','blobs_K10','blobs_K28','moons_n035','moons_n070','moons_n110','moons_n160',
    'circles_n020','circles_n050','circles_n090','circles_n140','aniso_low','aniso_high','imbalance','overlap'])
SEED=6100
MANIFEST={
    'study_scope':'researcher-generated synthetic stress tests; not independent externally curated dataset validation',
    'datasets':NAMES,'seed_rule':'6100 + position',
    'BC':'reconstruction default m=5, r=alpha*Q90(delta), I=0,P=1,F=1,M=1',
    'DBSCAN':'m in [3,5,8] x eps scale in [.70,.90,1.,1.15,1.40], total 15',
    'HDBSCAN':'min_cluster_size [5,10,20,40,80] x min_samples [3,5,8], total 15',
    'selection':'S_nonnoise*C-0.2*(1-C); at least 2 nonnoise groups, coverage >=0.5; no cluster-count penalty; no reference label use',
    'silhouette':'deterministic forced 2 per predicted cluster plus random subsample up to 280, RNG 2026',
    'controls':'DBSCAN-Single uses eps=alpha Q90 k-distance, m=5; OPTICS fixed min_samples=5, xi=0.05',
    'time_comparison':'wall-clock descriptive only; Python+Numba BC versus scikit-learn library implementations, first JIT warmup excluded',
    'excluded':'No external independent benchmark, no blinded pre-registration, no multi-machine timing',
}
# manifest retained from archived study

def generated(i):
    rng=np.random.default_rng(SEED+i)
    if i<=3:
        k=[3,6,10,28][i]
        n=[600,900,1200,1680][i]
        X,y=make_blobs(n_samples=n,centers=k,cluster_std=[.55,.48,.55,.22][i],center_box=(-9,9),random_state=SEED+i)
    elif i<=7:
        noise=[.035,.07,.11,.16][i-4]
        X,y=make_moons(n_samples=800,noise=noise,random_state=SEED+i)
    elif i<=11:
        noise=[.02,.05,.09,.14][i-8]
        X,y=make_circles(n_samples=800,factor=.50,noise=noise,random_state=SEED+i)
    elif i==12:
        X,y=make_blobs(n_samples=850,centers=5,cluster_std=.45,random_state=SEED+i)
        X=X@np.array([[2.2,.85],[.04,.30]])
    elif i==13:
        X,y=make_blobs(n_samples=850,centers=5,cluster_std=.65,random_state=SEED+i)
        X=X@np.array([[3.,1.2],[.04,.18]])
    elif i==14:
        X,y=make_blobs(n_samples=[600,100,65,35],centers=[[-4,-4],[4,4],[-3,5],[5,-5]],
            cluster_std=[.58,.55,.50,.43],random_state=SEED+i)
    else:
        X,y=make_blobs(n_samples=900,centers=6,cluster_std=1.6,center_box=(-5,5),random_state=SEED+i)
    return StandardScaler().fit_transform(X),np.asarray(y)

def sil(X,y,cap=280):
    ids=np.flatnonzero(y!=-1);kinds=np.unique(y[ids]);n=len(ids)
    if len(kinds)<2 or n<2 or len(kinds)>=min(n,cap)//2: return np.nan
    if n>cap:
        rng=np.random.default_rng(2026)
        forced=np.concatenate([rng.choice(np.flatnonzero(y==v),min(2,int(sum(y==v))),replace=False) for v in kinds])
        if len(forced)>cap: return np.nan
        other=np.setdiff1d(ids,forced)
        ids=np.r_[forced,rng.choice(other,cap-len(forced),replace=False)]
    if len(np.unique(y[ids]))<2:return np.nan
    return float(silhouette_score(X[ids],y[ids]))

def J(X,y,penalty=False,cap=280):
    C=float(np.mean(y!=-1));K=len(set(y)-{-1})
    if K<2 or C<.5:return -float('inf'),np.nan,C,K
    s=sil(X,y,cap=cap)
    return (s*C-.2*(1-C)-(.015*max(K-25,0)/25 if penalty else 0),s,C,K) if np.isfinite(s) else (-float('inf'),s,C,K)

def dbscan_candidates(X):
    dist,idx=NearestNeighbors(n_neighbors=11,n_jobs=1).fit(X).kneighbors(X)
    ani=_anisotropy_from_indices(X,idx)
    alpha=1.25 if ani>=4 else .95
    cand=[]
    for m in [3,5,8]:
        base=alpha*np.quantile(np.maximum(dist[:,m-1],1e-12),.9)
        for factor in [.7,.9,1.,1.15,1.4]:
            eps=max(1e-12,float(base*factor))
            y=DBSCAN(eps=eps,min_samples=m,n_jobs=1).fit_predict(X)
            cand.append((f'{m}:{factor:g}',y))
    single=DBSCAN(eps=alpha*np.quantile(np.maximum(dist[:,4],1e-12),.9),min_samples=5,n_jobs=1).fit_predict(X)
    return cand,single

def hdbscan_candidates(X):
    return [(f'{a}:{b}',HDBSCAN(min_cluster_size=a,min_samples=b,n_jobs=1,copy=False).fit_predict(X))
            for a in [5,10,20,40,80] for b in [3,5,8]]

def select(X,cands,penalty=False,cap=280):
    options=[]
    for order,(param,y) in enumerate(cands):
        score,s,c,k=J(X,y,penalty=penalty,cap=cap)
        options.append(dict(score=float(score),sil=float(s) if np.isfinite(s) else np.nan,coverage=c,k=k,y=y,param=param,order=order))
    best=sorted(options,key=lambda x:(-x['score'],-x['coverage'],x['order']))[0]
    return best

