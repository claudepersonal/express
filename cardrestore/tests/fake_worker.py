#!/usr/bin/env python3
"""Stand-in for restore_worker.py in HTTP tests: same event protocol, no model.
Copies each input to the output dir as the "restored" PNG and zips them."""
import json
import os
import shutil
import sys
import zipfile

if len(sys.argv) > 1 and sys.argv[1] == '--check':
    print(json.dumps({'event': 'ready', 'device': 'cpu'}))
    sys.exit(0)

job = json.load(sys.stdin)
os.makedirs(job['outdir'], exist_ok=True)
print(json.dumps({'event': 'ready', 'device': 'cpu'}), flush=True)
done = []
for i, item in enumerate(job['inputs']):
    if item['name'].startswith('bad'):
        print(json.dumps({'event': 'error', 'index': i, 'file': item['name'], 'message': 'cannot identify image file'}), flush=True)
        continue
    out = f'card{i}_restored_{job["scale"]}x.png'
    shutil.copy(item['path'], os.path.join(job['outdir'], out))
    done.append(out)
    row = {'file': item['name'], 'output': out, 'input_pixels': '1x1', 'output_pixels': '2x2', 'scale': job['scale'],
           'seconds': 0.01, 'input_edge_metric': 0, 'output_edge_metric': 0, 'output_bytes': 1, 'sha256_prefix': 'x',
           'ocr': 'TEXT' if job['ocr'] else None, 'profile_seen': job['profile'], 'tile_seen': job['tile'], 'fit_seen': job['fit'], 'fitted_from': None}
    print(json.dumps({'event': 'file', 'index': i, 'row': row}), flush=True)
with zipfile.ZipFile(os.path.join(job['outdir'], 'restored_cards.zip'), 'w') as z:
    for out in done:
        z.write(os.path.join(job['outdir'], out), out)
print(json.dumps({'event': 'done', 'zip': 'restored_cards.zip'}), flush=True)
