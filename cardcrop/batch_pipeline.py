#!/usr/bin/env python3
"""Batch CLI using the same worker/settings as the web GUI."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from enhance_worker import enhance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--cleanup', choices=['manual', 'classical', 'cnn'], default='classical')
    parser.add_argument('--detection', choices=['yolo', 'contour', 'none'], default='contour')
    parser.add_argument('--strength', type=float, default=0.45)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--no-auto-rotate', action='store_true')
    parser.add_argument('--zip', action='store_true', help='Package successful PNGs and the manifest')
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error('input_dir must exist')
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error('use a separate output directory')
    options = {'cleanup': args.cleanup, 'detection': args.detection,
               'cleanupStrength': args.strength, 'scale': args.scale,
               'autoRotate': not args.no_auto_rotate, 'conservative': False}
    from enhance_worker import validate_options
    validate_options(options)
    images = sorted(p for p in args.input_dir.iterdir() if p.is_file() and
                    p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'})
    if not images:
        parser.error('no supported images found')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, outputs = [], []
    for source in images:
        # Including the source extension avoids collisions such as card.jpg/card.png.
        destination = args.output_dir / (source.name + '.enhanced.png')
        record = {'source': source.name}
        try:
            record['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            result = enhance(str(source), str(destination), options)
            record.update(status='enhanced', output=destination.name, enhancement=result)
            outputs.append(destination)
        except Exception as exc:
            record.update(status='failed', error=str(exc))
        records.append(record)
    manifest = args.output_dir / 'manifest.json'
    temporary = args.output_dir / 'manifest.json.tmp'
    temporary.write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    temporary.replace(manifest)
    if args.zip:
        temporary_zip = args.output_dir / 'enhanced.zip.tmp'
        with zipfile.ZipFile(temporary_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in [*outputs, manifest]:
                archive.write(file, file.name)
        temporary_zip.replace(args.output_dir / 'enhanced.zip')
    print(json.dumps({'total': len(records), 'enhanced': len(outputs),
                      'failed': len(records) - len(outputs), 'manifest': str(manifest)}))
    return int(len(outputs) != len(records))


if __name__ == '__main__':
    sys.exit(main())
