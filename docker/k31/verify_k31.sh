#!/bin/bash
# The image still carries the vllm install exactly as install_k31.sh left it: every file of the vllm package and its
# dist-info (base binaries included, bytecode excluded) matches /opt/k31/vllm.manifest, written after the install
# gates passed, with no file added or missing; the overlay entries of /opt/k31/SOURCES match too; pip's version of
# vllm is the overlay's. megatron.lite, megatron.core and the source trees (VERL_ROOT / MEGATRON_ROOT of the
# Dockerfile) match their SOURCES entries file by file, nothing added or missing. Run after any install step that follows install_k31.sh (a later `pip install` that pulls
# vllm would otherwise replace the overlay silently). K31_PYTHON names the interpreter whose packages are checked
# (default python3 on PATH); the success line names the vllm directory it found.
set -euo pipefail
trap 'echo "${BASH_SOURCE[0]##*/} failed at line ${LINENO}" >&2' ERR
cd /
PY=${K31_PYTHON:-python3}
VP=$("${PY}" -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')
SITE=$(dirname "${VP}")
current() { (cd "${SITE}" && find vllm vllm-*.dist-info -name __pycache__ -prune -o -type f -print0 | sort -z \
  | xargs -0 sha256sum); }
current | diff -q /opt/k31/vllm.manifest - >/dev/null || {
  echo "the vllm install differs from /opt/k31/vllm.manifest:"; current | diff /opt/k31/vllm.manifest - | head -20 || true
  exit 1; }
n=$(grep -cE '^[0-9a-f]{64}  vllm/' /opt/k31/SOURCES)
[[ "${n}" -gt 0 ]] || { echo "no overlay entries in /opt/k31/SOURCES"; exit 1; }
(cd "${SITE}" && grep -E '^[0-9a-f]{64}  vllm/' /opt/k31/SOURCES | sha256sum --quiet -c) \
  || { echo "the K31 overlay of ${VP} differs from /opt/k31/SOURCES"; exit 1; }
# The symbolic links under <prefix> of <base> (same listing as install_k31.sh: `L <path> -> <target>`) are the ones
# SOURCES records: the file lists cover regular files only, and a link to a directory would add code none covers.
# Every directory scan checks its own exit status (functions run inside $(...) without errexit): a directory find
# cannot read fails the check instead of shrinking the listing.
# Every link must also resolve inside <base>/<prefix>, outside its pruned .git / __pycache__ directories: there the
# lists cover the files (a link elsewhere would run code no list covers).
# SOURCES lines under <prefix>/, matched as fixed strings (a path may hold regex characters): `sum` gives the
# `<sha256>  <path>` lines, `link` the `L <path> -> <target>` lines.
entries() {
  awk -v kind="$1" -v p="$2/" '
    kind == "sum" && length($1) == 64 && $1 ~ /^[0-9a-f]+$/ && substr($0, 65, 2 + length(p)) == "  " p { print }
    kind == "link" && substr($0, 1, 2 + length(p)) == "L " p { print }' /opt/k31/SOURCES
}
same_links() {
  local base=$1 prefix=$2 what=$3 found root outside
  found=$(cd "${base}" && find "${prefix}" \( -name .git -o -name __pycache__ \) -prune -o -type l -printf 'L %p -> %l\n' \
    | LC_ALL=C sort) || { echo "${what} in ${base}: cannot list ${prefix}" >&2; return 1; }
  [[ "${found}" == "$(entries link "${prefix}")" ]] \
    || { echo "${what} in ${base}: symbolic links differ from /opt/k31/SOURCES" >&2; return 1; }
  root=$(cd "${base}" && realpath -e -- "${prefix}") || { echo "${what}: no ${base}/${prefix}" >&2; return 1; }
  outside=$(cd "${base}" && find "${prefix}" \( -name .git -o -name __pycache__ \) -prune -o -type l -print0 \
    | xargs -0 -r realpath -m -- | awk -v r="${root}/" '
      substr($0 "/", 1, length(r)) != r || index(substr($0 "/", length(r)), "/.git/") || index(substr($0 "/", length(r)), "/__pycache__/")') \
    || { echo "${what} in ${base}: cannot resolve the links under ${prefix}" >&2; return 1; }
  [[ -z "${outside}" ]] || { echo "${what} in ${base}: links resolve outside ${root} or into its .git / __pycache__: ${outside}" >&2; return 1; }
}
same_links "${SITE}" vllm vllm || exit 1
# The files install_k31.sh listed in SOURCES under <prefix> (paths relative to <base>): same content, nothing added
# or missing (bytecode and .git excluded). Prints the count.
listed() {
  local base=$1 prefix=$2 what=$3 count have
  same_links "$@" || return 1
  count=$(entries sum "${prefix}" | wc -l)
  [[ "${count}" -gt 0 ]] || { echo "no ${what} entries in /opt/k31/SOURCES" >&2; return 1; }
  (cd "${base}" && entries sum "${prefix}" | sha256sum --quiet -c >&2) \
    || { echo "${what} in ${base} differs from /opt/k31/SOURCES" >&2; return 1; }
  # pruned, not filtered: an unreadable __pycache__ or .git is not entered at all
  have=$(cd "${base}" && find "${prefix}" \( -name .git -o -name __pycache__ \) -prune -o -type f -print | wc -l) \
    || { echo "${what} in ${base}: cannot list ${prefix}" >&2; return 1; }
  [[ "${have}" == "${count}" ]] || { echo "${what} has ${have} files, /opt/k31/SOURCES lists ${count}" >&2; return 1; }
  echo "${count}"
}
# megatron.lite and megatron.core (next to it) of this interpreter, and every source tree install_k31.sh was given
site_of() { "${PY}" -c 'import importlib.util, os, sys; print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(importlib.util.find_spec(sys.argv[1]).origin)))))' "$1"; }
MSITE=$(site_of megatron.lite)
# megatron is a namespace package: core must resolve next to lite, not from another portion (a venv, PYTHONPATH)
CSITE=$(site_of megatron.core)
[[ "${CSITE}" == "${MSITE}" ]] || { echo "megatron.core resolves in ${CSITE}, megatron.lite in ${MSITE}"; exit 1; }
nc=$(listed "${MSITE}" megatron/core megatron.core)
nl=$(listed "${MSITE}" megatron/lite megatron.lite)
nt=0
while read -r tree; do
  count=$(listed / "${tree#/}" "source tree ${tree}")
  nt=$((nt + count))
done < <(sed -n 's/^# tree //p' /opt/k31/SOURCES)
"${PY}" - "${VP}" <<'PY'
import glob
import importlib.metadata as md
import os
import sys

# The dist-info next to the installed package, not the first one on sys.path (a venv may shim vllm's version).
infos = glob.glob(os.path.join(os.path.dirname(sys.argv[1]), "vllm-*.dist-info"))
if len(infos) != 1:
    sys.exit(f"expected one vllm dist-info beside {sys.argv[1]}, found {infos}")
version = md.Distribution.at(infos[0]).version
ns = {}
exec(open(os.path.join(sys.argv[1], "_version.py")).read(), ns)
if version != ns["__version__"]:
    sys.exit(f"vllm dist-info version {version} is not the overlay's {ns['__version__']}")
PY
echo "VERIFY_K31_OK $(wc -l </opt/k31/vllm.manifest) files, ${n} overlay, ${nc} megatron.core, ${nl} megatron.lite, ${nt} source-tree files, vllm=$(realpath "${VP}")"
