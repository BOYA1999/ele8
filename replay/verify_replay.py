from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import pandas as pd

here = Path(__file__).resolve().parent
checksums = here / 'SHA256_INPUTS_AND_EXPECTED.txt'
for line in checksums.read_text(encoding='utf-8').splitlines():
    digest, rel = line.split('  ', 1)
    assert hashlib.sha256((here / rel).read_bytes()).hexdigest() == digest, rel

results = []
for name in ['branch', 'xgboost', 'budget', 'ratio']:
    root = here / name
    start = time.monotonic()
    result = subprocess.run([sys.executable, str(root/'run.py')], cwd=root, capture_output=True, text=True, encoding='utf-8', errors='replace')
    (root/'replay.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{name} failed: see {root}/replay.log')
    checked = []
    for reference in sorted((root/'expected').iterdir()):
        actual = root / reference.name
        if reference.suffix == '.json':
            assert json.loads(reference.read_text()) == json.loads(actual.read_text()), reference.name
        else:
            expected = pd.read_csv(reference)
            observed = pd.read_csv(actual)
            pd.testing.assert_frame_equal(expected, observed, check_dtype=False, rtol=1e-9, atol=1e-9)
        checked.append(reference.name)
    results.append({'analysis': name, 'expected_outputs_matched': checked, 'seconds': round(time.monotonic()-start, 2)})
print(json.dumps({'result':'PASS','python':sys.version,'pandas':pd.__version__,'numpy':__import__('numpy').__version__,'checks':results}, ensure_ascii=False, indent=2))
