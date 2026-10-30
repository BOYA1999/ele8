from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd

here = Path(__file__).parent
project = here/'inputs'
old = pd.read_csv(project/'04_analysis/reviewer_v13_20260904/core/blind_predictions.csv')
keys = ['fit_contract','split','file_name','model','seed']
original = pd.concat([pd.read_csv(here/'original_budget/blind_predictions.csv'),pd.read_csv(here/'original_budget_extra/blind_predictions.csv')],ignore_index=True)
doubled = pd.concat([pd.read_csv(here/'double_budget/blind_predictions.csv'),pd.read_csv(here/'double_budget_extra/blind_predictions.csv')],ignore_index=True)
assert len(original)==len(doubled)==12800 and not original.duplicated(keys).any() and not doubled.duplicated(keys).any()
check=original.merge(old,on=keys,suffixes=('_new','_old'),validate='1:1')
assert len(check)==len(original)
replay=(check.predicted_eol_new-check.predicted_eol_old).abs()
(here/'replay_check.json').write_text(json.dumps({'n':len(replay),'max_abs_difference':float(replay.max()),'exact_matches':int((replay==0).sum())},indent=2))
labels={c:pd.read_csv(project/'03_data_interaction/model_input/gate1_v2_20260821_0656/labels'/c/'test_labels_sealed.csv').set_index('file_name').eol_cycle for c in ['release','current_generator']}
assert len(labels['release'])==len(labels['current_generator'])==158
same=(labels['release']==labels['current_generator'])
ratio=[]; centres=[]
for budget,pred in [('original',original),('double',doubled)]:
    for model in ['cpmlp','cptransformer']:
        q=pred[(pred.model==model)&(pred.split=='test')]
        wide=q.pivot(index='file_name',columns=['fit_contract','seed'],values='predicted_eol').loc[labels['release'].index]
        for contract in ['release','current_generator']:
            for seed in range(10):
                centres.append({'budget':budget,'model':model,'contract':contract,'seed':seed,'mean_prediction':wide[contract,seed].mean(),'median_prediction':wide[contract,seed].median()})
        for subset,mask in [('all',same.notna()),('changed',~same),('unchanged',same)]:
            w=wide.loc[mask]
            cross=[(w['release',s]-w['current_generator',s]).abs().mean() for s in range(10)]
            within=[(w[c,a]-w[c,b]).abs().mean() for c in ['release','current_generator'] for a,b in combinations(range(10),2)]
            numerator=float(np.median(cross)); denominator=float(np.median(within))
            ratio.append({'budget':budget,'model':model,'subset':subset,'n_cells':len(w),'n_seeds':10,'n_pairs':len(within),'numerator':numerator,'denominator':denominator,'ratio':numerator/denominator})
pd.DataFrame(ratio).to_csv(here/'budget_ratios.csv',index=False)
pd.DataFrame(centres).to_csv(here/'budget_prediction_centres.csv',index=False)
def score(y,p,n):
    e=np.abs(y-p)
    return {'mape':lambda:np.mean(e/y),'smape':lambda:np.mean(2*e/(np.abs(y)+np.abs(p))),'mae':lambda:np.mean(e),'rmse':lambda:np.sqrt(np.mean((y-p)**2)),'log_mae':lambda:np.mean(np.abs(np.log(y)-np.log(p)))}[n]()
rows=[]
for budget,pred in [('original',original),('double',doubled)]:
    baseline=old[(old.model.isin(['ridge','xgboost'])) & old.seed.isin(range(10))]
    pred=pd.concat([baseline,pred],ignore_index=True)
    for (model,seed,contract),g in pred[pred.split.eq('test')].groupby(['model','seed','fit_contract']):
        y=labels[contract].loc[g.file_name].to_numpy(); p=g.predicted_eol.to_numpy()
        for metric in ['mape','smape','mae','rmse','log_mae']:
            rows.append({'budget':budget,'model':model,'seed':int(seed),'contract':contract,'metric':metric,'score':score(y,p,metric)})
metrics=pd.DataFrame(rows)
assert len(metrics)==800
metrics['rank']=metrics.groupby(['budget','seed','contract','metric']).score.rank(method='min')
metrics.to_csv(here/'budget_matched_contract_metrics_ranks.csv',index=False)
ranks=metrics.pivot(index=['seed','contract','metric','model'],columns='budget',values='rank').reset_index()
by=ranks.groupby(['seed','contract','metric']).apply(lambda x: pd.Series({'same_full_order':bool((x['original']==x['double']).all()),'same_winner':set(x.loc[x['original']==1,'model'])==set(x.loc[x['double']==1,'model'])}),include_groups=False).reset_index()
by.to_csv(here/'budget_rank_comparisons.csv',index=False)
print('replay',len(replay),float(replay.max()),int((replay==0).sum()))
print(pd.DataFrame(ratio).to_string(index=False))
print('rank comparisons',len(by),'same_order',by.same_full_order.sum(),'same_winner',by.same_winner.sum())
base_test=original[original.split.eq('test')].merge(doubled[doubled.split.eq('test')],on=keys,suffixes=('_original','_double'),validate='1:1')
base_test['budget_shift']=base_test.predicted_eol_double-base_test.predicted_eol_original
center_shift=base_test.groupby(['model','seed','fit_contract']).budget_shift.agg(mean_signed='mean',mean_absolute=lambda x:x.abs().mean(),median_absolute=lambda x:x.abs().median()).reset_index()
center_shift.to_csv(here/'budget_center_shift_by_seed.csv',index=False)
print('budget shift across test cells: median of seed means')
print(center_shift.groupby(['model','fit_contract'])[['mean_signed','mean_absolute','median_absolute']].median().to_string())
comparisons=[]
for budget in ['original','double']:
    q=metrics[metrics.budget.eq(budget)]
    piv=q.pivot(index=['seed','metric','model'],columns='contract',values='rank').reset_index()
    for (seed,metric),g in piv.groupby(['seed','metric']):
        comparisons.append({'budget':budget,'seed':seed,'metric':metric,'same_full_order':bool((g.release==g.current_generator).all()),'same_winner':set(g.loc[g.release==1,'model'])==set(g.loc[g.current_generator==1,'model'])})
comparisons=pd.DataFrame(comparisons)
comparisons.to_csv(here/'budget_AB_rank_comparisons.csv',index=False)
print('A/B comparisons')
print(comparisons.groupby('budget')[['same_full_order','same_winner']].sum().to_string())
rng=np.random.default_rng(130930)
draws=rng.integers(0,10,size=(5000,10)); ii,jj=np.triu_indices(10,1)
boot_rows=[]; boot_cache={}
for budget,pred in [('original',original),('double',doubled)]:
    for model in ['cpmlp','cptransformer']:
        q=pred[(pred.model==model)&(pred.split=='test')]
        wide=q.pivot(index='file_name',columns=['fit_contract','seed'],values='predicted_eol').loc[labels['release'].index]
        for subset,mask in [('all',same.notna()),('changed',~same),('unchanged',same)]:
            w=wide.loc[mask]
            a=np.stack([w['release',s].to_numpy() for s in range(10)])
            b=np.stack([w['current_generator',s].to_numpy() for s in range(10)])
            cross=np.mean(np.abs(a-b),axis=1)
            distances=np.stack([np.mean(np.abs(x[:,None,:]-x[None,:,:]),axis=2) for x in [a,b]])
            numerator=np.median(cross[draws],axis=1)
            paired=np.concatenate([distances[c,draws[:,ii],draws[:,jj]] for c in range(2)],axis=1)
            identical=np.tile(draws[:,ii]==draws[:,jj],(1,2))
            for method in ['include_repeat','exclude_repeat']:
                value=paired if method=='include_repeat' else np.where(identical,np.nan,paired)
                denominator=np.nanmedian(value,axis=1)
                r=numerator/denominator
                assert np.isfinite(r).all()
                boot_cache[(budget,model,subset,method)]=r
                boot_rows.append({'budget':budget,'model':model,'subset':subset,'method':method,'resamples':5000,'seed':130930,'bootstrap_median':np.median(r),'lo':np.quantile(r,.025),'hi':np.quantile(r,.975),'cross_seed_min':cross.min(),'cross_seed_max':cross.max(),'within_pair_min':np.min(distances[:,ii,jj]),'within_pair_max':np.max(distances[:,ii,jj])})
pd.DataFrame(boot_rows).to_csv(here/'budget_seed_bootstrap.csv',index=False)
print('paired seed bootstrap; fixed test cells, conditional exploration')
print(pd.DataFrame(boot_rows).to_string(index=False))
deltas=[]
for model in ['cpmlp','cptransformer']:
    for subset in ['all','changed','unchanged']:
        for method in ['include_repeat','exclude_repeat']:
            diff=boot_cache['double',model,subset,method]-boot_cache['original',model,subset,method]
            deltas.append({'model':model,'subset':subset,'method':method,'median_delta':np.median(diff),'lo':np.quantile(diff,.025),'hi':np.quantile(diff,.975)})
pd.DataFrame(deltas).to_csv(here/'budget_seed_bootstrap_ratio_delta.csv',index=False)
print('paired bootstrap ratio change double minus original')
print(pd.DataFrame(deltas).to_string(index=False))
