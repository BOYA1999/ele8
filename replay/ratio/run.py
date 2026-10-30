from pathlib import Path
import hashlib, numpy as np, pandas as pd
p=Path(__file__).resolve().parent/'inputs'
out=Path(__file__).resolve().parent
f=p/'prediction_shift_cells.csv'; old=p/'prediction_shift_bootstrap_replicates.csv'
w=pd.read_csv(f);w['labels_differ']=w.labels_differ.astype(str).str.lower().eq('true')
seeds=np.sort(w.seed.unique()); assert len(seeds)==23
rng=np.random.default_rng(130905); ii,jj=np.triu_indices(23,k=1); assert len(ii)==253
reps=5000; rows=[]; points=[]; duplicate_rows=[]
for model in ['ridge','xgboost','cpmlp','cptransformer']:
 cur=w[w.model.eq(model)]
 a0=cur.pivot(index='seed',columns='file_name',values='release').loc[seeds]
 b0=cur.pivot(index='seed',columns='file_name',values='current_generator').loc[seeds,a0.columns]
 meta=cur.drop_duplicates('file_name').set_index('file_name').loc[a0.columns]
 for status,mask in [('all',np.ones(len(meta),bool)),('changed',meta.labels_differ.to_numpy(bool)),('unchanged',~meta.labels_differ.to_numpy(bool))]:
  a=a0.to_numpy(float)[:,mask];b=b0.to_numpy(float)[:,mask];src=meta.dataset_group.to_numpy()[mask]; ns=sorted(set(src)); parts=[np.flatnonzero(src==s) for s in ns]
  draws=np.concatenate([rng.choice(g,size=(reps,len(g)),replace=True) for g in parts],axis=1); sd=rng.integers(0,23,size=(reps,23))
  weights={'cell':np.full(len(src),1/len(src)), 'source_equal':np.concatenate([np.full(len(g),1/(len(ns)*len(g))) for g in parts])}
  source_order=np.concatenate(parts); orig_weights={'cell':np.full(len(src),1/len(src)), 'source_equal':np.array([1/(len(ns)*len(np.flatnonzero(src==s))) for s in src])}
  for weighting,wt in orig_weights.items():
   cross=np.abs(a-b)@wt; within=np.concatenate([np.abs(a[ii]-a[jj])@wt,np.abs(b[ii]-b[jj])@wt]);
   for reducer in ['median','mean']:
    fun=np.median if reducer=='median' else np.mean; den=fun(within);points.append(dict(model=model,status=status,weighting=weighting,reducer=reducer,n=len(src),n_sources=len(ns),numerator=fun(cross),denominator=den,ratio=fun(cross)/den if den else np.nan))
  if model=='ridge': continue
  vals={f'{wt}_{red}':np.empty(reps) for wt in weights for red in ['median','mean']}
  for start in range(0,reps,25):
   stop=min(start+25,reps);cells=draws[start:stop];d=sd[start:stop]
   c=np.abs(b[d[:,:,None],cells[:,None,:]]-a[d[:,:,None],cells[:,None,:]])
   wa=np.abs(a[d[:,ii,None],cells[:,None,:]]-a[d[:,jj,None],cells[:,None,:]])
   wb=np.abs(b[d[:,ii,None],cells[:,None,:]]-b[d[:,jj,None],cells[:,None,:]])
   for weighting,wt in weights.items():
    cm=np.sum(c*wt,axis=2);wm=np.concatenate([np.sum(wa*wt,axis=2),np.sum(wb*wt,axis=2)],axis=1)
    for reducer in ['median','mean']:
     fun=np.median if reducer=='median' else np.mean;den=fun(wm,axis=1);vals[f'{weighting}_{reducer}'][start:stop]=np.divide(fun(cm,axis=1),den,out=np.full(stop-start,np.nan),where=den>0)
  for key,series in vals.items():
   weighting,reducer=key.split('_',1) if key.startswith('cell_') else ('source_equal',key[len('source_equal_'):])
   for i,val in enumerate(series):rows.append((model,status,weighting,reducer,i,val))
  dup=(sd[:,ii]==sd[:,jj]).sum(axis=1)
  duplicate_rows.append(dict(model=model,status=status,mean_duplicate_pairs=float(dup.mean()),median_duplicate_pairs=float(np.median(dup)),mean_fraction=float(np.mean(dup/253)),min=int(dup.min()),max=int(dup.max())))
rep=pd.DataFrame(rows,columns=['model','status','weighting','reducer','replicate','ratio']); pts=pd.DataFrame(points)
rep.to_csv(out/'ratio_bootstrap_replicates.csv',index=False);pts.to_csv(out/'ratio_point_estimates.csv',index=False);pd.DataFrame(duplicate_rows).to_csv(out/'duplicate_seed_draws.csv',index=False)
s=rep.groupby(['model','status','weighting','reducer']).ratio.agg(bootstrap_median='median',lo=lambda x:x.quantile(.025),hi=lambda x:x.quantile(.975)).reset_index();s.to_csv(out/'ratio_bootstrap_summary.csv',index=False)
original=pd.read_csv(old); orig=original[original.model.ne('ridge')][['model','label_status','replicate','cross_to_within_ratio']].rename(columns={'label_status':'status','cross_to_within_ratio':'orig'})
ours=rep[(rep.weighting=='cell')&(rep.reducer=='median')].merge(orig,on=['model','status','replicate'],validate='one_to_one'); delta=np.nanmax(np.abs(ours.ratio-ours.orig));print('original bootstrap max difference',delta,'rows',len(ours)); assert delta<1e-10
(out/'input_hashes.txt').write_text(''.join(hashlib.sha256(x.read_bytes()).hexdigest()+'  '+str(x)+'\n' for x in [f,old]),encoding='utf-8')
print(s.to_string(index=False));print(pd.DataFrame(duplicate_rows).to_string(index=False))
