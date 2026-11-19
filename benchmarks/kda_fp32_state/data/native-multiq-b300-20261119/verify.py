import hashlib,json,os
from pathlib import Path
root=Path(__file__).parent
p=root/'hashes.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()==os.environ['SOURCE_DIGEST']
for name,want in json.loads(p.read_text()).items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==want,name
print('SOURCE_VERIFIED',flush=True)
