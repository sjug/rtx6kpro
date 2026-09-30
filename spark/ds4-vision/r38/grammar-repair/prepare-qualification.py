#!/usr/bin/env python3
"""Reuse the executed DS4 battery without changing the separate benchmark repo."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SPARK = ROOT.parents[2]
SOURCE = SPARK / 'jj-main-sm121/ds4-vision'
image = json.loads((ROOT / 'receipts/build/image-inspect.json').read_text())[0]['Id']
common = {
    '6633e678fee74f5e1060290a01df812d34d10edcf30e34dd1c9bf7db88260ef3': image,
    'out=$base/../qualification/ds4-vision': 'out=$base/receipts/qualification',
    'ds4-vision-jj-main-tp2': 'ds4-vision-jj-r38p-tp2',
    'jj-main-spark-sm121': 'jj-r38p-spark-sm121',
    '$base/../../ds4-vision/qualify.py': '$base/../../qualify.py',
    '2026-09-jj-main-sm121-qualification': '2026-09-r38p-grammar-qualification',
    'jj-main-sm121-tp2-dspark-k3-dglin-524k': 'jj-r38p-tp2-dspark-k3-dglin-524k',
    '$base/../../glm53/r38-spark/qualification/validate-grid.py': '$base/../../../glm53/r38-spark/qualification/validate-grid.py',
    'DS4-JJ-MAIN-QUALIFICATION-COMPLETE': 'DS4-R38P-QUALIFICATION-COMPLETE',
}
for name in ('execute.sh', 'benchmark.sh'):
    text = (SOURCE / name).read_text()
    for old, new in common.items():
        text = text.replace(old, new)
    text = text.replace('DS4_JJ_MAIN_APPROVED', 'DS4_R38P_APPROVED')
    text = text.replace('remote=/home/jugs/git/ds4-vision-r38/jj-main-sm121',
                        'remote=/home/jugs/git/ds4-vision-r38/grammar-repair/runtime')
    # Preserve startup-to-finish journals separately from the benchmark window.
    text = text.replace('$out/$node-kernel.log', '$out/$node-' +
                        ('full-window' if name == 'execute.sh' else 'benchmark') + '-kernel.log')
    (ROOT / name).write_text(text)
shutil.copyfile(SOURCE / 'probe-structured.py', ROOT / 'probe-structured.py')
comparison = (SOURCE / 'compare.py').read_text().replace(
    '6633e678fee74f5e1060290a01df812d34d10edcf30e34dd1c9bf7db88260ef3', image
).replace("parents[2] / 'glm53", "parents[3] / 'glm53").replace("'jj_main': b", "'r38p': b")
(ROOT / 'compare.py').write_text(comparison)
campaign = ROOT.parents[3] / 'runs/deepseek-v4-flash/vision-exp/2026-09-r38p-grammar-qualification/campaign.yaml'
if not campaign.is_file():
    raise RuntimeError('Missing campaign metadata')
print('Driver scripts prepared; fresh campaign metadata exists.')
