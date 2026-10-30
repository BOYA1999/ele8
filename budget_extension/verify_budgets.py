from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd

base = Path(__file__).resolve().parent
replay = base.parent / 'replay' / 'budget'
original = pd.read_csv(replay / 'inputs/04_analysis/reviewer_v13_20260904/core/blind_predictions.csv')
label_root = replay / 'inputs/03_data_interaction/model_input/gate1_v2_20260821_0656/labels'
labels = {c: pd.read_csv(label_root / c / 'test_labels_sealed.csv').set_index('file_name').eol_cycle for c in ['release', 'current_generator']}
changed = labels['release'].ne(labels['current_generator'])
baseline = original[original.seed.lt(10) & original.model.isin(['ridge', 'xgboost'])]
budgets = {
    'original23': original,
    'original10': original[original.seed.lt(10)],
    'double': pd.concat([baseline, pd.read_csv(replay/'double_budget/blind_predictions.csv'), pd.read_csv(replay/'double_budget_extra/blind_predictions.csv')]),
    'internal40': pd.concat([baseline, pd.read_csv(base/'internal40.csv')]),
    'internal80': pd.concat([baseline, pd.read_csv(base/'internal80.csv')]),
}
ratios, scores = [], []
for budget, predictions in budgets.items():
    test = predictions[predictions.split.eq('test')]
    for model, q in test.groupby('model'):
        w = q.pivot(index='file_name', columns=['fit_contract', 'seed'], values='predicted_eol').loc[changed.index]
        seeds = sorted(q.seed.unique())
        for subset, mask in [('all', changed.notna()), ('changed', changed), ('unchanged', ~changed)]:
            a, b = (w.loc[mask, c].to_numpy().T for c in ['release', 'current_generator'])
            numerator = np.median(np.abs(a-b).mean(axis=1))
            denominator = np.median([np.abs(v[i]-v[j]).mean() for v in [a,b] for i,j in combinations(range(len(seeds)),2)])
            ratios.append(dict(budget=budget,model=model,subset=subset,numerator=numerator,denominator=denominator,ratio=numerator/denominator if denominator else None))
    for (model, seed, c), g in test.groupby(['model','seed','fit_contract']):
        y, p = labels[c].loc[g.file_name].to_numpy(), g.predicted_eol.to_numpy()
        e = np.abs(y-p)
        values = dict(mape=np.mean(e/y),smape=np.mean(2*e/(np.abs(y)+np.abs(p))),mae=e.mean(),rmse=np.sqrt(np.mean((y-p)**2)),log_mae=np.abs(np.log(y)-np.log(p)).mean())
        scores.extend(dict(budget=budget,model=model,seed=int(seed),contract=c,metric=k,score=v) for k,v in values.items())
scores = pd.DataFrame(scores)
scores['rank'] = scores.groupby(['budget','seed','contract','metric']).score.rank(method='min')
ab, cross = [], []
for (budget, seed, metric), g in scores.groupby(['budget','seed','metric']):
    a, b = (g[g.contract.eq(c)].set_index('model')['rank'] for c in ['release','current_generator'])
    ab.append(dict(budget=budget,seed=int(seed),metric=metric,winner=set(a[a.eq(1)].index)==set(b[b.eq(1)].index),top2=set(a[a.le(2)].index)==set(b[b.le(2)].index),order=bool(a.sort_index().eq(b.sort_index()).all())))
for target, reference in [('double','original10'),('internal40','original10'),('internal80','original10'),('internal80','internal40')]:
    pair = scores[scores.budget.isin([reference,target])]
    for (seed, contract, metric), g in pair.groupby(['seed','contract','metric']):
        a, b = (g[g.budget.eq(c)].set_index('model')['rank'] for c in [reference,target])
        cross.append(dict(base=reference,target=target,seed=int(seed),contract=contract,metric=metric,winner=set(a[a.eq(1)].index)==set(b[b.eq(1)].index),order=bool(a.sort_index().eq(b.sort_index()).all())))
ab, cross = pd.DataFrame(ab), pd.DataFrame(cross)
summary = ab.groupby('budget')[['winner','top2','order']].sum()
for budget, winner, order in [('original23',115,87),('original10',50,39),('double',47,40),('internal40',43,27),('internal80',38,31)]:
    assert (int(summary.loc[budget,'winner']), int(summary.loc[budget,'order'])) == (winner,order)
first, extended = pd.read_csv(base/'validation40.csv'), pd.read_csv(base/'validation80.csv')
keys=['model','seed','contract','epoch']
check=first.merge(extended[extended.epoch.le(40)],on=keys,suffixes=('_40','_80'),validate='1:1')
assert len(check)==800 and len(extended)==1600
assert all(np.array_equal(check[v+'_40'],check[v+'_80']) for v in ['train_loss','validation_mape'])
selection={}
for cap, df in [(40,first),(80,extended)]:
    mean=df.groupby(['model','epoch']).validation_mape.mean().reset_index()
    selection[cap]={m:int(g.loc[g.validation_mape.idxmin(),'epoch']) for m,g in mean.groupby('model')}
assert selection == {40: {'cpmlp':38,'cptransformer':40},80:{'cpmlp':74,'cptransformer':78}}, selection
for name, frame in [('ratios.csv',pd.DataFrame(ratios)),('scores.csv',scores),('ab.csv',ab),('across.csv',cross)]:
    frame.to_csv(base/name,index=False)
print(json.dumps({'status':'PASS','selection':selection,'AB_counts':summary.to_dict('index'),'across_counts':cross.groupby(['base','target'])[['winner','order']].sum().reset_index().to_dict('records')},indent=2))
