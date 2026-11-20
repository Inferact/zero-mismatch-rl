"""Runtime fingerprint of the numerical contract the FP32-state kernels were built for.

The kernels use fast-math and approximate FP32 functions (ex2.approx, rcp.approx),
so identical source can produce different bits under another CuTe DSL, another
assembler or another GPU model. The fingerprint records the inputs that decide the
bits so that a rollout process and a trainer process can log them at start-up and
compare them. Equal fingerprints rule out the toolchain differences recorded here;
they do not prove bitwise equality, which still needs a bytewise comparison of
dumped outputs and states. The source digest covers raw file bytes, so any edit,
even to a docstring, changes it: read a mismatch as "re-validate", not as "the bits
differ".

Bump CONTRACT_ID whenever a change alters the output bits for any valid input.
Nothing in this package calls this module.
"""

import functools
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess

import torch

CONTRACT_ID = "kda-fp32-state/1"

PINNED_PACKAGES = {
    "nvidia-cutlass-dsl": "4.8.1",
    "nvidia-cutlass-dsl-libs-base": "4.8.1",
    "nvidia-cutlass-dsl-libs-core": "4.8.1",
    "nvidia-cutlass-dsl-libs-cu13": "4.8.1",
    "apache-tvm-ffi": "0.1.10",
}
OTHER_PACKAGES = ("cuda-python", "cuda-bindings", "torch", "triton")
STRICT_PREFIXES = (
    "contract_id",
    "kernel_sources.sha256",
    "packages.nvidia-cutlass-dsl",
    "cute_dsl_env",
    "device.name",
    "device.capability",
    "device.sm_count",
)
SOURCE_ROOT = pathlib.Path(__file__).resolve().parent


class ContractMismatch(ValueError):
    pass


def _distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hash_sources(root):
    paths = sorted(root.glob("kernels/**/*.py"))
    paths += [root / "api.py", root / "_primitives.py", root / "_checkpoint.py"]
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }
    if not files:
        return {"sha256": None, "files": {}}
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(f"{name}\0{files[name]}\n".encode())
    return {"sha256": digest.hexdigest(), "files": files}


@functools.cache
def _ptxas():
    candidates = [
        os.path.join(os.environ[name], "bin", "ptxas")
        for name in ("CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH")
        if os.environ.get(name)
    ]
    on_path = shutil.which("ptxas")
    if on_path:
        candidates.append(on_path)
    for path in candidates:
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        match = re.search(r"release (\d+\.\d+), V(\d+\.\d+\.\d+)", result.stdout)
        if match:
            return {"release": match.group(1), "version": match.group(2)}
    return {"release": None, "version": None}


def _driver_version():
    try:
        from cuda.bindings import driver

        status, version = driver.cuDriverGetVersion()
        return int(version) if int(status) == 0 else None
    except Exception:
        return None


def _device(device):
    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(device)
    return {
        "name": properties.name,
        "capability": [properties.major, properties.minor],
        "sm_count": properties.multi_processor_count,
        "driver_version": _driver_version(),
    }


def contract_fingerprint(device=None, *, probe_ptxas=True):
    """Return a JSON-serializable record of what decides the kernels' bits.

    device selects the GPU (default: the current one). probe_ptxas runs
    `ptxas --version` from CUDA_TOOLKIT_PATH, CUDA_HOME, CUDA_PATH or PATH; the
    CuTe DSL may use a different assembler, so the result is informational.
    """
    return {
        "contract_id": CONTRACT_ID,
        "kernel_sources": _hash_sources(SOURCE_ROOT),
        "packages": {
            name: _distribution_version(name)
            for name in (*PINNED_PACKAGES, *OTHER_PACKAGES)
        },
        "cute_dsl_env": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("CUTE_DSL_")
        },
        "ptxas": _ptxas() if probe_ptxas else {"release": None, "version": None},
        "device": _device(device),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
        "python": platform.python_version(),
    }


def _flatten(value, prefix=""):
    if isinstance(value, dict) and value:
        flat = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else key))
        return flat
    return {prefix: value}


def compare_contracts(a, b):
    """Return the differing fields, split into strict and informational ones.

    Strict fields (contract id, kernel source digest, CuTe DSL packages, CUTE_DSL_*
    environment, GPU name, capability and SM count) can change the bits. The rest
    (per-file digests, torch, tvm-ffi, cuda-python, ptxas, driver, Python) are
    recorded for diagnosis.
    """
    left, right = _flatten(a), _flatten(b)
    strict, informational = {}, {}
    for key in sorted(left.keys() | right.keys()):
        if left.get(key) != right.get(key):
            target = strict if key.startswith(STRICT_PREFIXES) else informational
            target[key] = (left.get(key), right.get(key))
    return {"strict": strict, "informational": informational}


def fingerprint_digest(fingerprint):
    """One short digest of the strict fields, for a start-up log line."""
    strict = {
        key: value
        for key, value in _flatten(fingerprint).items()
        if key.startswith(STRICT_PREFIXES)
    }
    encoded = json.dumps(strict, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def assert_same_contract(expected, actual=None, *, informational=False):
    """Raise ContractMismatch if the strict fields differ (all fields if informational)."""
    actual = contract_fingerprint() if actual is None else actual
    differences = compare_contracts(expected, actual)
    shown = dict(differences["strict"])
    if informational:
        shown.update(differences["informational"])
    if shown:
        lines = [f"  {key}: {a!r} != {b!r}" for key, (a, b) in shown.items()]
        raise ContractMismatch("FP32-state KDA contract differs:\n" + "\n".join(lines))


def pinned_toolchain_problems():
    """List installed packages that differ from the versions pinned in requirements.txt."""
    problems = []
    for name, pinned in PINNED_PACKAGES.items():
        installed = _distribution_version(name)
        if installed != pinned:
            problems.append(f"{name}: installed {installed}, pinned {pinned}")
    return problems


def require_pinned_toolchain():
    """Opt-in check that the installed CuTe DSL and tvm-ffi match the pinned versions."""
    problems = pinned_toolchain_problems()
    if problems:
        raise ContractMismatch(
            "toolchain differs from requirements.txt: " + "; ".join(problems)
        )
