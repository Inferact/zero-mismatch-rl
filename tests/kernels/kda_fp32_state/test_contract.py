"""The contract fingerprint is stable, tracks what decides the bits and matches the pins."""

import json
import pathlib

import pytest
import torch
from vllm.models.kimi_k3.nvidia.kda_fp32_state import contract

pytestmark = pytest.mark.gpu_free


def fingerprint():
    return contract.contract_fingerprint(probe_ptxas=False)


def test_fingerprint_is_json_and_stable():
    first, second = fingerprint(), fingerprint()
    assert json.loads(json.dumps(first)) == first
    assert first == second
    assert contract.fingerprint_digest(first) == contract.fingerprint_digest(second)
    contract.assert_same_contract(first, second, informational=True)


def test_fingerprint_records_contract_packages_and_kernel_sources():
    record = fingerprint()
    assert record["contract_id"] == contract.CONTRACT_ID
    assert set(contract.PINNED_PACKAGES) <= set(record["packages"])
    files = record["kernel_sources"]["files"]
    assert record["kernel_sources"]["sha256"]
    assert {"kernels/prefill/scan.py", "_primitives.py", "api.py"} <= set(files)


def test_strict_fields_fail_and_informational_fields_do_not(monkeypatch):
    before = fingerprint()
    real = contract._distribution_version
    monkeypatch.setattr(
        contract,
        "_distribution_version",
        lambda name: "9.9.9" if name == "nvidia-cutlass-dsl" else real(name),
    )
    after = fingerprint()
    assert contract.fingerprint_digest(after) != contract.fingerprint_digest(before)
    with pytest.raises(contract.ContractMismatch, match="nvidia-cutlass-dsl"):
        contract.assert_same_contract(before, after)

    informational = json.loads(json.dumps(before))
    informational["torch"]["version"] = "0.0"
    informational["python"] = "0.0"
    contract.assert_same_contract(before, informational)
    with pytest.raises(contract.ContractMismatch, match="torch.version"):
        contract.assert_same_contract(before, informational, informational=True)


def test_cute_dsl_environment_is_part_of_the_fingerprint(monkeypatch):
    before = fingerprint()
    monkeypatch.setenv("CUTE_DSL_TEST_KNOB", "1")
    after = fingerprint()
    assert after["cute_dsl_env"]["CUTE_DSL_TEST_KNOB"] == "1"
    with pytest.raises(contract.ContractMismatch, match="CUTE_DSL_TEST_KNOB"):
        contract.assert_same_contract(before, after)


def test_source_digest_tracks_kernel_files(tmp_path):
    root = tmp_path / "package"
    (root / "kernels" / "prefill").mkdir(parents=True)
    (root / "kernels" / "a.py").write_text("a = 1\n")
    (root / "kernels" / "prefill" / "b.py").write_text("b = 2\n")
    (root / "kernels" / "notes.txt").write_text("ignored")
    (root / "_primitives.py").write_text("p = 3\n")
    (root / "api.py").write_text("d = 4\n")
    first = contract._hash_sources(root)
    assert set(first["files"]) == {
        "kernels/a.py",
        "kernels/prefill/b.py",
        "_primitives.py",
        "api.py",
    }
    (root / "kernels" / "notes.txt").write_text("also ignored")
    assert contract._hash_sources(root) == first
    (root / "kernels" / "prefill" / "b.py").write_text("b = 3\n")
    changed = contract._hash_sources(root)
    assert changed["sha256"] != first["sha256"]
    assert [
        name
        for name, digest in changed["files"].items()
        if digest != first["files"][name]
    ] == ["kernels/prefill/b.py"]
    assert contract._hash_sources(tmp_path / "missing") == {"sha256": None, "files": {}}


def test_pins_match_requirements_txt():
    path = pathlib.Path(contract.__file__).with_name("requirements.txt")
    if not path.is_file():
        pytest.skip("requirements.txt is not shipped with this install")
    text = path.read_text()
    pins = dict(
        line.split("==")
        for line in text.splitlines()
        if "==" in line and not line.startswith("#")
    )
    assert pins == contract.PINNED_PACKAGES


def test_require_pinned_toolchain(monkeypatch):
    pinned = contract.PINNED_PACKAGES
    monkeypatch.setattr(contract, "_distribution_version", pinned.get)
    assert contract.pinned_toolchain_problems() == []
    contract.require_pinned_toolchain()
    monkeypatch.setattr(
        contract,
        "_distribution_version",
        lambda name: "4.6.2" if name == "nvidia-cutlass-dsl" else pinned.get(name),
    )
    assert contract.pinned_toolchain_problems() == [
        "nvidia-cutlass-dsl: installed 4.6.2, pinned 4.8.1"
    ]
    with pytest.raises(contract.ContractMismatch, match="4.6.2"):
        contract.require_pinned_toolchain()


def test_device_section_describes_the_current_gpu():
    device = fingerprint()["device"]
    if not torch.cuda.is_available():
        assert device is None
        return
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    assert device["name"] == properties.name
    assert device["capability"] == [properties.major, properties.minor]
    assert device["sm_count"] == properties.multi_processor_count
