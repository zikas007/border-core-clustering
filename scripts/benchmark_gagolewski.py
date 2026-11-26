#!/usr/bin/env python3
"""Version-frozen, label-blind external clustering validation.

Benchmark Suite for Clustering Algorithms v1.2.0, Gagolewski et al.
Official tagged files are fetched from:
https://github.com/gagolews/clustering-data-v1/tree/v1.2.0
Reference labels0 are used ONLY to evaluate fitted partitions.

No threshold or model grid is changed after examining reference labels.
Methods: BC density (fixed), DBSCAN-Single (fixed), DBSCAN-15-selected,
HDBSCAN-15-selected. No oracle-K fitting. The benchmark should be run only
once with the predeclared selections below and archived with logs/hashes.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[k]='1'
import sys, argparse, gzip, hashlib, json, time, urllib.request, urllib.error
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, scipy, numba
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from border_core_fast import fit_predict_bc_fast,_anisotropy_from_indices
from border_core_grid import fit_predict_bc_grid
from synthetic_data import select

TAG='v1.2.0'
SUITE_DOI='10.5281/zenodo.23563578'
RAW='https://raw.githubusercontent.com/gagolews/clustering-data-v1/'+TAG+'/'
# Predeclared batteries/cases selected before reading performance metrics.
CASE_IDS=(
 'wut/mk1','wut/mk2','wut/mk3','wut/mk4','wut/smile',
 'wut/twosplashes','wut/x1','wut/x2','wut/x3','wut/z1',
 'fcps/atom','fcps/chainlink','fcps/hepta','fcps/lsun',
 'fcps/target','fcps/tetra','fcps/twodiamonds','fcps/wingnut',
 'graves/dense','graves/line','graves/parabolic','graves/ring','graves/zigzag',
 'cdc/DS1','cdc/DS10','cdc/DS11','cdc/DS16','cdc/DS17',
)

def get_file(rel,cache):
    path=cache/rel;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        req=urllib.request.Request(RAW+rel,headers={'User-Agent':'BorderCoreExternalValidation/1.0 research'})
        with urllib.request.urlopen(req,timeout=45) as r: body=r.read()
        if len(body)<12: raise RuntimeError('Unexpectedly small file: '+rel)
        path.write_bytes(body)
    return path,hashlib.sha256(path.read_bytes()).hexdigest()

def read_data(case,cache):
    p,xhash=get_file(case+'.data.gz',cache)
    q,yhash=get_file(case+'.labels0.gz',cache)
    with gzip.open(p,'rt') as f: X=np.loadtxt(f,ndmin=2)
    with gzip.open(q,'rt') as f: y=np.loadtxt(f,dtype=int,ndmin=1)
    if X.ndim!=2 or y.ndim!=1 or len(X)!=len(y) or len(X)<11: raise ValueError('Invalid dimensions: '+case)
    if not np.isfinite(X).all(): raise ValueError('Nonfinite dataset: '+case)
    # Gagolewski: reference labels 0 denote noise; ARI/AMI treat noise as a separate class.
    assert np.all(y>=0)
    Kref=len(set(y)-{0}) if 0 in set(y) else len(set(y))
    meta={'dataset':case,'battery':case.split('/')[0],'n':len(X),'dimension':X.shape[1],
          'reference_K':Kref,'reference_noise':int(np.sum(y==0)),
          'data_sha256':xhash,'labels_sha256':yhash,'annotation':'labels0'}
    return StandardScaler().fit_transform(X),y,meta

def candidates(X):
    # Label-free J0 selector shared across DBSCAN and HDBSCAN.
    dist,idx=NearestNeighbors(n_neighbors=11,n_jobs=1).fit(X).kneighbors(X)
    d=X.shape[1]
    alpha=(1.25 if _anisotropy_from_indices(X,idx)>=4 else .95) if d==2 else .75
    cand=[]
    for m in (3,5,8):
        base=alpha*np.quantile(np.maximum(dist[:,m-1],1e-12),.9)
        for f in (.7,.9,1.,1.15,1.4):
            eps=max(1e-12,float(base*f))
            lab=DBSCAN(eps=eps,min_samples=m,n_jobs=1).fit_predict(X)
            cand.append((f'{m}:{f:g}',lab))
    return cand

def hdb_candidates(X):
    return [(f'{mcs}:{ms}',HDBSCAN(min_cluster_size=mcs,min_samples=ms,n_jobs=1,copy=False).fit_predict(X))
            for mcs in (5,10,20,40,80) for ms in (3,5,8)]

def predict(X):
    # JIT warmup excluded for BC, as in the source experiments. Times are not cross-hardware claims.
    t=time.perf_counter()
    if X.shape[1]==2: bc=fit_predict_bc_grid(X,seed_mode='density').labels
    else: bc=fit_predict_bc_fast(X,seed_mode='density').labels
    t_bc=time.perf_counter()-t
    t=time.perf_counter()
    dist,idx=NearestNeighbors(n_neighbors=11,n_jobs=1).fit(X).kneighbors(X)
    alpha=(1.25 if _anisotropy_from_indices(X,idx)>=4 else .95) if X.shape[1]==2 else .75
    eps=float(max(1e-12,alpha*np.quantile(np.maximum(dist[:,4],1e-12),.9)))
    single=DBSCAN(eps=eps,min_samples=5,n_jobs=1).fit_predict(X)
    t_single=time.perf_counter()-t
    t=time.perf_counter(); cand=candidates(X);t_db=time.perf_counter()-t
    t=time.perf_counter(); selected=select(X,cand,penalty=False,cap=280);t_selection=time.perf_counter()-t
    t=time.perf_counter(); hcand=hdb_candidates(X);t_hdb=time.perf_counter()-t
    t=time.perf_counter(); hselected=select(X,hcand,penalty=False,cap=280); t_hselect=time.perf_counter()-t
    # Every method is timed as a standalone complete fitting policy; single and tuned
    # DBSCAN are measured independently, each including its own scale estimation.
    return [
       ('BC-density',bc,'fixed',t_bc),
       ('DBSCAN-Single',single,'fixed',t_single),
       ('DBSCAN-15-selected',selected['y'],str(selected['param']),t_db+t_selection),
       ('HDBSCAN-15-selected',hselected['y'],str(hselected['param']),t_hdb+t_hselect)]

def offline_cases():
    # Pure code smoke test; NEVER to be reported as official Gagolewski validation.
    from sklearn.datasets import make_blobs,make_moons
    X1,y1=make_blobs(n_samples=180,centers=3,random_state=411)
    X2,y2=make_moons(n_samples=240,noise=.09,random_state=412)
    for i,(X,y) in enumerate(((X1,y1),(X2,y2))):
        yield f'smoke/only_{i}',StandardScaler().fit_transform(X),y,{'dataset':f'smoke/only_{i}','battery':'smoke','n':len(X),'dimension':X.shape[1], 'reference_K':len(np.unique(y)),'reference_noise':0,'data_sha256':'NOT_OFFICIAL','labels_sha256':'NOT_OFFICIAL','annotation':'synthetic-smoke'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',default=str(ROOT/'benchmark_gagolewski'))
    ap.add_argument('--max-cases',type=int,default=0,help='0 means all 28 locked cases')
    ap.add_argument('--offline-smoke',action='store_true',help='unit test only, not Gagolewski results')
    args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if args.offline_smoke: cases=list(offline_cases())
    else:
        names=CASE_IDS[:args.max_cases] if args.max_cases>0 else CASE_IDS
        cases=[]
        for c in names:
            X,y,meta=read_data(c,out/'input_cache')
            cases.append((c,X,y,meta))
    # JIT warmup ahead of measurements, no reference labels used.
    W=np.random.default_rng(2026).normal(size=(40,2))
    fit_predict_bc_grid(W,seed_mode='density')
    rows=[];provenance=[]
    for case,X,truth,meta in cases:
        provenance.append(meta)
        for method,pred,params,seconds in predict(X):
            row=dict(meta,method=method,chosen_parameters=params,K_hat=len(set(pred)-{-1}),
                     coverage=float(np.mean(pred!=-1)),ARI=float(adjusted_rand_score(truth,pred)),
                     AMI=float(adjusted_mutual_info_score(truth,pred)),elapsed_seconds=float(seconds))
            rows.append(row)
        print('completed',case,'n',len(X),'BC_ARI',f"{rows[-4]['ARI']:.4f}",flush=True)
        pd.DataFrame(rows).to_csv(out/'dataset_results.csv',index=False)
        (out/'dataset_manifest.json').write_text(json.dumps(provenance,indent=2))
    d=pd.DataFrame(rows)
    summary=d.groupby('method').agg(n_datasets=('ARI','size'),mean_ARI=('ARI','mean'),median_ARI=('ARI','median'),
        mean_AMI=('AMI','mean'),mean_coverage=('coverage','mean')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    env=dict(source_repository='https://github.com/gagolews/clustering-data-v1',source_tag=TAG,
         source_suite_DOI=SUITE_DOI,selected_cases=list(CASE_IDS),actual_cases=[c[0] for c in cases],
         offline_smoke=args.offline_smoke,reference_labels_used_in_fit=False,
         selector='J0 = silhouette_nonnoise * coverage - 0.20*(1-coverage); 280 point cap, no K penalty',
         python=sys.version,scikit_learn=sklearn.__version__,scipy=scipy.__version__,numpy=np.__version__,numba=numba.__version__)
    (out/'environment_and_design.json').write_text(json.dumps(env,indent=2))
    print(summary.to_string(index=False),flush=True)
    print('OUTPUT:',out,flush=True)

if __name__=='__main__':main()
