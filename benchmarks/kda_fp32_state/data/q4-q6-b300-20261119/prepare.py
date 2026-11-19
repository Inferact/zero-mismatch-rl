"""Build a frozen, self-contained benchmark snapshot from this checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    package=Path(__file__).resolve().parent
    repo=package.parents[3]
    source=args.destination/'source'
    source.mkdir(parents=True,exist_ok=False)
    for name in ['benchmark.py','adapters.py','upstream_adapter.py','fi_q1.py','fi_q8.py','workloads.json','verify.py']:
        shutil.copy2(package/name,source/name)
    shutil.copytree(repo/'vllm/models/kimi_k3/nvidia/kda_fp32_state',source/'kda_consistent',ignore=shutil.ignore_patterns('__pycache__'))
    (source/'tests').mkdir()
    for name in ['__init__.py','helpers.py','fp64_reference.py']:
        data=(repo/'tests/kernels/kda_fp32_state'/name).read_text()
        data=data.replace('vllm.models.kimi_k3.nvidia.kda_fp32_state','kda_consistent')
        (source/'tests'/name).write_text(data)
    provenance=json.loads((package/'upstream-provenance.json').read_text())
    for name,want in provenance['sha256'].items():
        url=f"https://raw.githubusercontent.com/{provenance['repository']}/{provenance['commit']}/{name}"
        with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
        if hashlib.sha256(data).hexdigest()!=want:raise RuntimeError(f'Upstream hash mismatch: {name}')
        path=source/'upstream'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    shutil.copy2(package/'upstream-provenance.json',source/'upstream/provenance.json')
    manifest={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()}
    (source/'hashes.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
    digest=hashlib.sha256((source/'hashes.json').read_bytes()).hexdigest()
    (args.destination/'source-digest.txt').write_text(digest+'\n')
    for folder in ['results','cache/torch','cache/triton','cache/ccache']:(args.destination/folder).mkdir(parents=True,exist_ok=True)
    print(f'Snapshot: {source.resolve()}\nSHA256: {digest}')


if __name__=='__main__':main()
