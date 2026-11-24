import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import synthetic_data as h
from merge_controls import variants
from sklearn.metrics import adjusted_rand_score
import pandas as pd
h.SEED=6100
X,y=h.generated(0)
cands,single=h.dbscan_candidates(X)
base=h.select(X,cands)['y']
results={k:adjusted_rand_score(y,pred) for k,(pred,_) in variants(X,base).items()}
ref=pd.read_csv(Path(__file__).resolve().parents[1]/'results'/'merge_controls.csv')
ref=ref[(ref['dataset']=='blobs_K3') & (ref['base']=='DBSCAN-15-selected') & (ref['repeat']==0)]
for control,value in results.items():
    expected=float(ref[ref['control']==control]['ARI'].iloc[0])
    assert abs(value-expected)<1e-12,(control,value,expected)
print('SMOKE TEST PASSED: four ARI values for seed 6100 match the archived CSV exactly.')
