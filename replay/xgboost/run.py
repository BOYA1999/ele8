from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
source = Path(__file__).resolve().parent / 'inputs' / 'xgboost_factor_contrasts.csv'
out = Path(__file__).resolve().parent
df = pd.read_csv(source)
rng = np.random.default_rng(20261030)
draws = rng.integers(0, 23, size=(10000, 23))
rows = []
for metric in ['mape', 'mae']:
    p = df[df.metric.eq(metric)].pivot(index='seed', columns='setup', values='fitting_contrast').sort_index()
    assert len(p) == 23 and p.notna().all().all()
    for kind, x in [('auto', p.auto), ('fixed', p.fixed), ('fixed_minus_auto', p.fixed-p.auto)]:
        x = x.to_numpy()
        q = np.quantile(x[draws].mean(axis=1), [.025, .975])
        rows.append(dict(metric=metric, contrast=kind, mean=x.mean(), seed_min=x.min(), seed_max=x.max(), lo=q[0], hi=q[1]))
pd.DataFrame(rows).to_csv(out / 'xgboost_fitting_seed_bootstrap.csv', index=False)
(out / 'input_sha256.txt').write_text(hashlib.sha256(source.read_bytes()).hexdigest() + '  ' + str(source) + '\n', encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
