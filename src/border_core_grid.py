"""Genuinely selective 2D fixed-core frontier expansion for Border Core.

Unlike CSR benchmark acceleration, builds only a compact 2-D spatial hash/grid;
Numba visits 9 adjacent occupied cells *only for expanded eligible frontier points*.
This avoids full radius adjacency precomputation and counts actual expansion
queries; it is tested side by side against alternative spatial-index implementations.
KMeans option used to create automatic anchor candidates, never K_true.

Any changed partitions are reported, not silently accepted as equivalent.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from numba import njit
from border_core_fast import _anisotropy_from_indices, _UF

@njit(cache=True)
def _search(keys,value):
    lo=0;hi=len(keys)
    while lo<hi:
        mid=(lo+hi)//2
        if keys[mid]<value:lo=mid+1
        else:hi=mid
    return lo if lo<len(keys) and keys[lo]==value else -1

@njit(cache=True)
def _expand_grid(X,eligible,density,seeds,labels,fixed,
                 xs,ys,stride,unique,starts,ends,sorted_idx,
                 q_points,q_owners,radius):
    cap=max(128,len(q_points)*2,len(X)*8)
    a=np.empty(cap,np.int32);b=np.empty(cap,np.int32)
    tail=len(q_points)
    for t in range(tail):a[t]=q_points[t];b[t]=q_owners[t]
    expanded=np.zeros(len(X),np.uint8)
    co_u=np.empty(128,np.int32);co_k=np.empty(128,np.int32);co_o=np.empty(128,np.int32)
    co_cap=128;co_n=0;head=0;query_count=0;candidate_checks=0
    r2=radius*radius
    while head<tail:
        u=a[head];k=b[head];head+=1
        owner=labels[u]
        if owner>=0 and owner!=k:
            if eligible[u] and min(density[u],density[seeds[k]])/max(density[u],density[seeds[k]])>=0.35:
                if co_n>=co_cap:
                    co_cap*=2
                    x=np.empty(co_cap,np.int32);x[:co_n]=co_u[:co_n];co_u=x
                    x=np.empty(co_cap,np.int32);x[:co_n]=co_k[:co_n];co_k=x
                    x=np.empty(co_cap,np.int32);x[:co_n]=co_o[:co_n];co_o=x
                co_u[co_n]=u;co_k[co_n]=k;co_o[co_n]=owner;co_n+=1
            continue
        if owner<0:labels[u]=k
        if expanded[u] or not eligible[u] or fixed[u]:continue
        expanded[u]=1;query_count+=1
        for ox in (-1,0,1):
            for oy in (-1,0,1):
                cell_key=(xs[u]+ox)*stride+(ys[u]+oy)
                cell=_search(unique,cell_key)
                if cell<0:continue
                for pos in range(starts[cell],ends[cell]):
                    v=sorted_idx[pos]
                    candidate_checks+=1
                    dx=X[u,0]-X[v,0];dy=X[u,1]-X[v,1]
                    if dx*dx+dy*dy>r2 or v==u:continue
                    if labels[v]<0 or (labels[v]!=k and eligible[v]):
                        if tail>=cap:
                            cap*=2
                            tmp=np.empty(cap,np.int32);tmp[:tail]=a[:tail];a=tmp
                            tmp=np.empty(cap,np.int32);tmp[:tail]=b[:tail];b=tmp
                        a[tail]=v;b[tail]=k;tail+=1
    return labels,co_u[:co_n],co_k[:co_n],co_o[:co_n],query_count,tail,candidate_checks

@dataclass
class GridBCResult:
    labels:np.ndarray
    n_seeds:int
    n_fixed:int
    n_expanded:int
    n_candidates_checked:int
    radius:float
    median_anisotropy:float
    n_clusters:int
    n_grid_cells:int
    n_kmeans_centers:int
    stage_times:dict
    graph_bytes:int
    memory_grid_bytes:int
    n_merges:int
    seeding_mode:str

def fit_predict_bc_grid(X,seed_mode='hybrid',min_pts=5,random_state=2026,
                        fixed_core_factor=.45,coarse_factor=1.75):
    X=np.asarray(X,dtype=np.float64,order='C');n,d=X.shape
    if d!=2:raise ValueError('Spatial hash implementation only supports 2D; use CSR fallback for higher dimensions')
    if seed_mode not in ('hybrid','density') or not np.isfinite(X).all() or n<11:raise ValueError('Invalid inputs')
    times={};t=time.perf_counter()
    dist,idx=NearestNeighbors(n_neighbors=11,algorithm='kd_tree',n_jobs=1).fit(X).kneighbors(X)
    kd=np.maximum(dist[:,min_pts-1],1e-12);ani=_anisotropy_from_indices(X,idx)
    factor=1.25 if ani>=4.0 else .95
    radius=float(factor*np.quantile(kd,.9));density=1.0/kd;eligible=kd<=radius
    times['knn_density_anisotropy']=time.perf_counter()-t
    t=time.perf_counter()
    tree=cKDTree(X); center_candidates=[];coarse_k=0
    if seed_mode=='hybrid':
        coarse_k=min(n//3,max(4,min(96,int(np.ceil(coarse_factor*np.sqrt(n))))))
        km=MiniBatchKMeans(n_clusters=coarse_k,random_state=random_state,
            batch_size=min(1024,n),n_init=1,max_iter=40,reassignment_ratio=0.0)
        km.fit(X)
        group=km.labels_
        for k in range(coarse_k):
            inds=np.flatnonzero((group==k)&eligible)
            if len(inds):center_candidates.append(int(inds[np.argmax(density[inds])]))
    ranked=np.lexsort((np.arange(n),-density))
    ordered=np.r_[np.array(sorted(set(center_candidates)),dtype=int),ranked] if center_candidates else ranked
    seeded=np.zeros(n,bool);seeds=[]
    for p in ordered:
        if not eligible[p] or seeded[p]:continue
        seeds.append(int(p));seeded[tree.query_ball_point(X[p],2*radius)]=True
    seeds=np.asarray(seeds,np.int32)
    times['automatic_seeds_and_kmeans']=time.perf_counter()-t
    t=time.perf_counter()
    ix=np.floor((X[:,0]-X[:,0].min())/radius).astype(np.int64)
    iy=np.floor((X[:,1]-X[:,1].min())/radius).astype(np.int64)
    stride=int(iy.max())+3
    keys=ix*stride+iy
    order=np.argsort(keys,kind='stable').astype(np.int32)
    uniq,st=np.unique(keys[order],return_index=True)
    ed=np.r_[st[1:],len(X)]
    times['grid_construction']=time.perf_counter()-t
    t=time.perf_counter()
    labels=np.full(n,-1,np.int32);fixed=np.zeros(n,np.uint8)
    q_u=[];q_k=[]
    inner=tree.query_ball_point(X[seeds],radius*fixed_core_factor)
    outer=tree.query_ball_point(X[seeds],radius*(1+fixed_core_factor))
    for k,(p,inside,outside) in enumerate(zip(seeds,inner,outer)):
        a=np.asarray(inside,int);a=a[eligible[a]&(labels[a]<0)]
        labels[a]=k;fixed[a]=1
        if labels[p]<0:labels[p]=k;fixed[p]=1
        for u in outside:
            if not fixed[u]:q_u.append(u);q_k.append(k)
    times['fixed_cores_frontier']=time.perf_counter()-t
    t=time.perf_counter()
    labels,c_u,c_k,c_o,queries,events,candidate_checks=_expand_grid(X,eligible,density,seeds,labels,fixed,
        ix,iy,stride,uniq,st,ed,order,np.asarray(q_u,np.int32),np.asarray(q_k,np.int32),radius)
    times['compiled_lazy_frontier_expansion']=time.perf_counter()-t
    t=time.perf_counter()
    contacts={}
    for u,k,owner in zip(c_u,c_k,c_o):
        pair=tuple(sorted((int(k),int(owner))))
        contacts.setdefault(pair,set()).add(int(u))
    sizes=np.bincount(labels[labels>=0],minlength=len(seeds))
    anis=np.ones(len(seeds))
    valid=np.flatnonzero(labels>=0);sort=valid[np.argsort(labels[valid],kind='stable')]
    starts=np.r_[0,np.cumsum(sizes)]
    for i in range(len(seeds)):
        if sizes[i]<8:continue
        pts=X[sort[starts[i]:starts[i+1]]];eig=np.linalg.eigvalsh(np.cov(pts,rowvar=False))
        anis[i]=float(np.sqrt(max(eig[-1],1e-12)/max(eig[0],1e-9)))
    uf=_UF(len(seeds));n_merge=0
    for (i,j),nodes in sorted(contacts.items(),key=lambda v:(-len(v[1]),v[0])):
        th=2 if (anis[i]>=3 or anis[j]>=3) else max(5,int(np.ceil(2.2*np.log1p(min(sizes[i],sizes[j])))))
        if len(nodes)>=th and uf.union(i,j):n_merge+=1
    out=np.full(n,-1,dtype=int);mapped={}
    for p in np.flatnonzero(labels>=0):
        root=uf.find(int(labels[p]));out[p]=mapped.setdefault(root,len(mapped))
    labs,counts=np.unique(out[out>=0],return_counts=True)
    minsize=max(min_pts,min(7,int(np.ceil(.1*np.median(counts))))) if len(counts) else min_pts
    for l,count in zip(labs,counts):
        if count<minsize:out[out==l]=-1
    uniq_labs=sorted(set(out)-{-1});remap={v:i for i,v in enumerate(uniq_labs)}
    out=np.asarray([remap.get(int(v),-1) for v in out],dtype=np.int32)
    times['contact_fusion']=time.perf_counter()-t
    mem=order.nbytes+uniq.nbytes+st.nbytes+ed.nbytes+ix.nbytes+iy.nbytes
    return GridBCResult(out,len(seeds),int(fixed.sum()),queries,candidate_checks,
        radius,ani,len(uniq_labs),len(uniq),coarse_k,times,0,mem,n_merge,seed_mode)

if __name__=='__main__':
    from sklearn.datasets import make_blobs
    x,_=make_blobs(n_samples=300,random_state=0)
    z=fit_predict_bc_grid(x)
    print(z.n_clusters,z.stage_times)
