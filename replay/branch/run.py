from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
out = Path(__file__).resolve().parent
records_path = out / 'inputs/records.csv'
paired_path = out / 'inputs/paired.csv'
r = pd.read_csv(records_path)
p = pd.read_csv(paired_path)
keys = ['split', 'dataset_group', 'file_name']
assert not r.duplicated(keys).any() and not p.duplicated(keys).any()
r['included_in_paired_cohort'] = r['included_in_paired_cohort'].astype(bool)
eligible = r[r.included_in_paired_cohort].copy()
d = eligible.merge(p, on=keys, validate='one_to_one')
assert len(r) == 843 and len(d) == len(p) == 823
assert np.allclose(d.released_eol, d.release_eol) and np.allclose(d.recomputed_eol, d.current_generator_eol)
assert (d.release_after_100 & d.generator_after_100).all()
assert len(r[(r.release_after_100 != r.generator_after_100)]) == 0
rows = []
for label, g in d.groupby('endpoint_kind'):
    a, b = g.release_eol.to_numpy(), g.current_generator_eol.to_numpy()
    i, j = np.triu_indices(len(g), 1)
    sa, sb = np.sign(a[i]-a[j]), np.sign(b[i]-b[j])
    comparable = (sa != 0) & (sb != 0)
    discordant = ((sa != sb) & comparable).sum()
    delta = np.abs(a-b)
    rows.append({'endpoint_kind':label, 'n_cells':len(g), 'changed':int((delta != 0).sum()), 'changed_fraction':float((delta != 0).mean()), 'median_absolute_delta':float(np.median(delta)), 'p90_absolute_delta':float(np.quantile(delta,.9)), 'discordant_pairs':int(discordant), 'comparable_pairs':int(comparable.sum()), 'discordant_fraction':float(discordant/comparable.sum()) if comparable.sum() else np.nan})
pd.DataFrame(rows).to_csv(out/'branch_summary.csv',index=False)
pd.crosstab(d.dataset_group,d.endpoint_kind).to_csv(out/'source_by_branch.csv')
pd.crosstab(r.dataset_group,r.v4_cohort_status).to_csv(out/'source_by_eligibility.csv')
r[~r.included_in_paired_cohort].to_csv(out/'excluded_records.csv',index=False)
d.groupby(['endpoint_kind','dataset_group']).agg(n=('file_name','size'),changed=('labels_differ','sum')).to_csv(out/'branch_source_discrepancy.csv')
with (out/'input_sha256.txt').open('w',encoding='utf-8') as f:
    for path in (records_path,paired_path):
        f.write(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n')
