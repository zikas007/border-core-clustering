"""Reference implementations of post-merging controls.
Identical generated inputs and label-blind selector. Comparisons made after inspecting
exploratory results; exploratory controls, not independent confirmation.
"""
import os, sys, time, argparse
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'): os.environ[k]='1'
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'src'))
import synthetic_data as h
from border_core_fast import _anisotropy_from_indices
FAM=['blobs','moons','circles','other']

def variants(X,y):
    y=np.asarray(y,dtype=np.int32)
    un=np.unique(y[y>=0]); ncl=len(un)
    if ncl<2: return {nm:(y.copy(),0) for nm in ('unmodified','geometric_contact','size_only_contact','min_distance')}
    dist,idx=NearestNeighbors(n_neighbors=11,algorithm='kd_tree').fit(X).kneighbors(X)
    kd=np.maximum(dist[:,4],1e-12);r=(1.25 if _anisotropy_from_indices(X,idx)>=4 else .95)*np.quantile(kd,.9)
    eligible=kd<=r;rho=1/kd
    ids={int(l):i for i,l in enumerate(un)}
    sizes=np.bincount(np.searchsorted(un,y[y>=0]),minlength=ncl)
    anis=np.ones(ncl)
    for a,l in enumerate(un):
        pts=X[y==l]
        if len(pts)>=8:
            eig=np.linalg.eigvalsh(np.cov(pts,rowvar=False));anis[a]=np.sqrt(max(eig[-1],1e-12)/max(eig[0],1e-9))
    pairs=cKDTree(X).query_pairs(r,output_type='ndarray')
    contact={};prox=set()
    for ii,jj in pairs:
        a,b=int(y[ii]),int(y[jj]);
        if a<0 or b<0 or a==b:continue
        key=tuple(sorted((ids[a],ids[b])))
        prox.add(key)
        if not(eligible[ii] and eligible[jj]): continue
        if min(rho[ii],rho[jj])/max(rho[ii],rho[jj]) <.35:continue
        contact.setdefault(key,set()).update((int(ii),int(jj)))
    def combine(edges):
        par=np.arange(ncl); count=0
        def find(v):
            while par[v]!=v:
                par[v]=par[par[v]];v=par[v]
            return v
        for a,b in edges:
            aa,bb=find(a),find(b)
            if aa!=bb:par[aa]=bb;count+=1
        labmap=np.array([find(i) for i in range(ncl)])
        out=np.full(len(y),-1,dtype=np.int32)
        out[y>=0]=labmap[np.searchsorted(un,y[y>=0])]
        _,out_nonneg=np.unique(out[out>=0],return_inverse=True);out[out>=0]=out_nonneg
        return out,count
    size_thr=lambda a,b:max(5,int(np.ceil(2.2*np.log1p(min(sizes[a],sizes[b])))))
    sorted_contacts=sorted(contact.items(),key=lambda z:(-len(z[1]),z[0]))
    edges_geom=[k for k,s in sorted_contacts if len(s)>=(2 if anis[k[0]]>=3 or anis[k[1]]>=3 else size_thr(*k))]
    edges_size=[k for k,s in sorted_contacts if len(s)>=size_thr(*k)]
    ret={'unmodified':(y.copy(),0),'geometric_contact':combine(edges_geom),'size_only_contact':combine(edges_size),'min_distance':combine(sorted(prox))}
    return ret

def run(repeats,out,start=0):
    result=[];tstart=time.time()
    for rep in range(start,repeats):
        h.SEED=6100+10000*rep
        for i,name in enumerate(h.NAMES):
            X,truth=h.generated(i)
            cands,single=h.dbscan_candidates(X)
            tuned=h.select(X,cands)['y']
            hcands=h.hdbscan_candidates(X)
            htuned=h.select(X,hcands)['y']
            for alg,labels in [('DBSCAN-Single',single),('DBSCAN-15-selected',tuned),('HDBSCAN-15-selected',htuned)]:
                before=adjusted_rand_score(truth,labels)
                for method,(yy,merges) in variants(X,labels).items():
                    result.append({'dataset':name,'family':FAM[i//4],'seed':6100+10000*rep+i,'repeat':rep,'base':alg,'control':method,'ARI':adjusted_rand_score(truth,yy),'delta_from_base':adjusted_rand_score(truth,yy)-before,'K':len(set(yy)-{-1}),'coverage':float(np.mean(yy!=-1)),'merges':merges})
        print('rep',rep+1,'/',repeats,'seconds',round(time.time()-tstart,2),flush=True)
        pd.DataFrame(result).to_csv(out,index=False)
    d=pd.DataFrame(result)
    print(d.groupby(['base','control']).agg(mean_ari=('ARI','mean'),mean_delta=('delta_from_base','mean'),mean_merges=('merges','mean')).round(5).to_string(),flush=True)
    print('BY FAMILY',d.groupby(['family','base','control']).delta_from_base.mean().round(4).to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=1);p.add_argument('--start',type=int,default=0);p.add_argument('--out',default=str(ROOT/'results'/'recomputed_controls.csv'));a=p.parse_args();run(a.repeats,a.out,a.start)
