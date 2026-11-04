#!/bin/bash
# Final stage of the K31 preview image (the Dockerfile and the enroot dev-image job both run this file): installs the
# mLite wheel without dependencies, then lays the staged K31 files over the base image's vLLM install.
#   install_k31.sh <out dir of build_k31.sh> <sources file> [<source tree>...]
# Each source tree (the Dockerfile passes /opt/k31/verl and /opt/k31/megatron, VERL_ROOT and MEGATRON_ROOT) is listed
# file by file in SOURCES, as are megatron.lite and megatron.core, so verify_k31.sh covers all the code a run imports.
# Gates, after everything is installed: (1) the only files of the vLLM install that change are the staged ones (eight
# K31 modules + the files the delivery branch changed + _version.py), so every base binary (_C, _moe_C, flash-attn,
# FlashMLA, DeepGEMM, vllm-rs, ...) stays byte-identical; (2) every file of the delivery tree's vllm/ equals the
# installed one. Bytecode of replaced or deleted modules is removed, and the vllm dist-info carries the overlay's
# version and file hashes, so pip sees what is installed. Writes /opt/k31/SOURCES (source commits + sha256 of
# everything installed), /opt/k31/vllm.manifest (sha256 of every vllm file and its dist-info after the gates) and
# /opt/k31/verify_k31.sh, which later install steps run to prove the install is unchanged.
set -euo pipefail
trap 'echo "${BASH_SOURCE[0]##*/} failed at line ${LINENO}" >&2' ERR
OUT=$1 SOURCES=$2 HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREES=("${@:3}")
for tree in "${TREES[@]}"; do [[ "${tree}" == /* && -d "${tree}" ]] || { echo "source tree ${tree} is not an absolute directory"; exit 1; }; done
cd /
VP=$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')
manifest() { (cd "${VP}" && find . -type f -not -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum); }
manifest >/tmp/k31-before
python3 -m pip install -q --no-cache-dir --no-deps --force-reinstall "${OUT}"/dist/nemotron_megatron_lite-*.whl
suffix=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
KP=$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("megatron.lite.primitive.kernels").origin))')
# megatron.core of the same Megatron-LM tree as mLite, next to megatron.lite (megatron is a namespace package); the base
# image has none, so nothing is replaced.
MEG=$(dirname "$(dirname "$(dirname "${KP}")")")
: >/tmp/k31-mcore
[[ -d "${OUT}/mcore/megatron/core" ]] || { echo "build_k31.sh staged no megatron.core"; exit 1; }
[[ ! -e "${MEG}/core" && ! -e "${MEG}/__init__.py" ]] || { echo "${MEG} already has core/ or __init__.py"; exit 1; }
cp -r "${OUT}/mcore/megatron/core" "${MEG}/core"
(cd "$(dirname "${MEG}")" && find megatron/core -type f -not -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum) \
  >/tmp/k31-mcore
(cd "${OUT}/mcore" && find megatron/core -type f -print0 | sort -z | xargs -0 sha256sum) | diff -q - /tmp/k31-mcore
python3 - "${MEG}" <<'PY'
import os
import sys

import megatron.core
import megatron.lite

here = os.path.realpath(sys.argv[1])
for m in (megatron.core, megatron.lite):
    if os.path.dirname(os.path.dirname(os.path.realpath(m.__file__))) != here:
        sys.exit(f"{m.__name__} imports from {m.__file__}, not {here}")
print("megatron.core and megatron.lite from", here)
PY
# The import surface of megatron.lite on megatron.core (frozen where Megatron-LM dropped it, while lite moved on):
# every `from megatron.core... import name` in lite, module-level or deferred (pipeline-parallel schedules, p2p,
# optimizer, checkpointing run only with PP > 1 or at save time), resolves. The one exception is the CSA attention
# modules that Megatron-LM added after the drop; only DeepSeek V4 imports them, never the K31 model.
python3 - <<'PY'
import ast
import importlib
import os

import megatron.lite

top = os.path.dirname(os.path.realpath(megatron.lite.__file__))
CSA = "megatron.core.transformer.experimental_attention_variant"
ok, bad = 0, []
for root, dirs, files in os.walk(top):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for name in files:
        if not name.endswith(".py"):
            continue
        path = os.path.join(root, name)
        rel = os.path.relpath(path, top)
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").startswith("megatron.core")):
                continue
            for alias in node.names:
                target = f"{node.module}.{alias.name}"
                if target.startswith(CSA + ".csa") or node.module.startswith(CSA + ".csa"):
                    if rel.startswith("model/k31/"):
                        bad.append(f"{rel}: the K31 model imports {target}, absent from the frozen megatron.core")
                    continue
                try:
                    module = importlib.import_module(node.module)
                    if not hasattr(module, alias.name):
                        importlib.import_module(target)
                    ok += 1
                except Exception as e:  # noqa: BLE001
                    bad.append(f"{rel}: from {node.module} import {alias.name}: {type(e).__name__}: {e}")
if bad:
    raise SystemExit("megatron.lite imports that megatron.core does not provide:\n  " + "\n  ".join(bad))
print(f"megatron.lite -> megatron.core: {ok} imported names resolve")
PY
ls "${KP}"/_k31_mxfp4_dequant"${suffix}" >/dev/null
(cd "${OUT}/stage/vllm" && find . -type f -print0 | sort -z | xargs -0 sha256sum) >/tmp/k31-staged
cp -r "${OUT}/stage/vllm/." "${VP}/"
while read -r path; do rm "${VP}/${path#vllm/}"; done <"${OUT}/stage/DELETED"
# bytecode of every replaced or deleted module, then directories the deletions emptied
{ awk '{print $2}' /tmp/k31-staged; sed 's#^vllm/#./#' "${OUT}/stage/DELETED"; } | grep '\.py$' | while read -r f; do
  rm -f "${VP}/$(dirname "${f}")/__pycache__/$(basename "${f}" .py)".*.pyc
done
while read -r path; do
  d=$(dirname "${VP}/${path#vllm/}")
  while [[ "${d}" != "${VP}" ]]; do
    rmdir "${d}/__pycache__" 2>/dev/null || true
    rmdir "${d}" 2>/dev/null || break
    d=$(dirname "${d}")
  done
done <"${OUT}/stage/DELETED"
manifest >/tmp/k31-after
changed=$( (diff <(sort -k2 /tmp/k31-before) <(sort -k2 /tmp/k31-after) || true) | sed -n 's/^[<>] [0-9a-f]* *//p' | sort -u)
expected=$( (awk '{print $2}' /tmp/k31-staged; sed 's#^vllm/#./#' "${OUT}/stage/DELETED") | sort -u)
unexpected=$(comm -23 <(echo "${changed}") <(echo "${expected}"))
[[ -z "${unexpected}" ]] || { echo "files outside the K31 overlay changed: ${unexpected}"; exit 1; }
(cd "$(dirname "${VP}")" && sha256sum --quiet -c "${OUT}/stage/SOURCE.sha256") \
  || { echo "installed vllm/ differs from the delivery tree"; exit 1; }
for m in tree tp_ordered lin_lora moe_lora_fuse moe_lora_mma bi_gemm kda_pf4 kda_dsync8 kda_v6; do
  [[ -f "${VP}/_k31_${m}_C${suffix}" ]] || { echo "missing _k31_${m}_C${suffix}"; exit 1; }
done
# the vllm dist-info: the overlay's version, RECORD rows rehashed / added / dropped
python3 - "${VP}" <<'PY'
import base64, csv, glob, hashlib, io, os, sys

vp = sys.argv[1]
site = os.path.dirname(vp)
ns = {}
exec(open(os.path.join(vp, "_version.py")).read(), ns)
version = ns["__version__"]
(old,) = glob.glob(os.path.join(site, "vllm-*.dist-info"))
new = os.path.join(site, f"vllm-{version}.dist-info")
meta = os.path.join(old, "METADATA")
head, sep, body = open(meta, encoding="utf-8").read().partition("\n\n")  # header fields end at the first blank line
head = "\n".join(f"Version: {version}" if line.startswith("Version: ") else line for line in head.split("\n"))
open(meta, "w", encoding="utf-8").write(head + sep + body)
rec_name = os.path.basename(new) + "/RECORD"


def row(rel):
    data = open(os.path.join(site, rel), "rb").read()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return [rel, f"sha256={digest}", str(len(data))]


rows, seen = [], set()
for r in csv.reader(open(os.path.join(old, "RECORD"), encoding="utf-8")):
    rel = r[0].replace(os.path.basename(old) + "/", os.path.basename(new) + "/", 1)
    seen.add(rel)
    if rel == rec_name:
        continue
    src = os.path.join(site, r[0])
    if not os.path.exists(src):
        continue  # deleted by the overlay, or bytecode removed with its module
    rows.append(row(r[0]) if r[0].startswith("vllm/") else [rel] + r[1:])
for dirpath, _, files in os.walk(vp):
    for f in files:
        rel = os.path.relpath(os.path.join(dirpath, f), site)
        if rel not in seen and "__pycache__" not in rel:
            rows.append(row(rel))
os.rename(old, new)
rows = [[r[0].replace(os.path.basename(old) + "/", os.path.basename(new) + "/", 1)] + r[1:] for r in rows]
rows = [row(r[0]) if r[0].startswith(os.path.basename(new) + "/") else r for r in rows]  # METADATA was rewritten
rows.append([rec_name, "", ""])
buf = io.StringIO()
csv.writer(buf, lineterminator="\n").writerows(rows)
open(os.path.join(new, "RECORD"), "w", encoding="utf-8").write(buf.getvalue())
print("dist-info", os.path.basename(new), len(rows), "RECORD rows")
PY
mkdir -p /opt/k31
# symbolic links under <dir> of <base> as `L <path> -> <target>` lines: the file lists below hold regular files only,
# and a link (to a directory especially) would otherwise add code no list covers
links() { (cd "$1" && find "$2" \( -name .git -o -name __pycache__ \) -prune -o -type l -printf 'L %p -> %l\n' \
  | LC_ALL=C sort); }
{
  cat "${SOURCES}"
  links "$(dirname "${VP}")" vllm
  links "$(dirname "${MEG}")" megatron/core
  links "$(dirname "${MEG}")" megatron/lite
  for tree in "${TREES[@]}"; do links / "${tree#/}"; done
  echo "# vLLM overlay ($(wc -l </tmp/k31-staged) files, deleted $(wc -l <"${OUT}/stage/DELETED")) over ${VP}"
  sed 's#  \./#  vllm/#' /tmp/k31-staged
  (cd "${OUT}/dist" && sha256sum ./*.whl)
  echo "# megatron.core ($(wc -l </tmp/k31-mcore) files) next to megatron.lite"
  cat /tmp/k31-mcore
  echo "# megatron.lite (the installed wheel, bytecode excluded)"
  # the same walk as verify_k31.sh: .git and __pycache__ pruned
  (cd "$(dirname "${MEG}")" && find megatron/lite \( -name .git -o -name __pycache__ \) -prune -o -type f -print0 \
    | sort -z | xargs -0 -r sha256sum)
  for tree in "${TREES[@]}"; do
    echo "# tree ${tree}"
    (cd / && find "${tree#/}" \( -name .git -o -name __pycache__ \) -prune -o -type f -print0 | sort -z \
      | xargs -0 -r sha256sum)
  done
} >/opt/k31/SOURCES
(cd "$(dirname "${VP}")" && find vllm vllm-*.dist-info -name __pycache__ -prune -o -type f -print0 | sort -z \
  | xargs -0 sha256sum) >/opt/k31/vllm.manifest
install -m 755 "${HERE}/verify_k31.sh" /opt/k31/verify_k31.sh
n=$(wc -l </tmp/k31-staged)
rm -f /tmp/k31-before /tmp/k31-after /tmp/k31-staged /tmp/k31-mcore
/opt/k31/verify_k31.sh
echo "INSTALL_K31_OK ${n} staged, $(wc -l <"${OUT}/stage/DELETED") deleted, $(wc -l <"${OUT}/stage/SOURCE.sha256") source files equal, ${VP}"
