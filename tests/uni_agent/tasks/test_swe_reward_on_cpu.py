import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from uni_agent.sandbox.base import ExecResult
from uni_agent.tasks.swe_bench import reward

UNRESOLVED_OUTPUT = "\n".join(
    (
        reward.START_TEST_OUTPUT,
        "FAILED test_requests.py::test_fix - AssertionError",
        "PASSED test_requests.py::test_existing",
        reward.END_TEST_OUTPUT,
    )
)
RESOLVED_OUTPUT = UNRESOLVED_OUTPUT.replace(
    "FAILED test_requests.py::test_fix - AssertionError", "PASSED test_requests.py::test_fix"
)
EMPTY_TEST_OUTPUT = f"{reward.START_TEST_OUTPUT}\n{reward.END_TEST_OUTPUT}"
UNRESOLVED_STATUS_MAP = {"test_requests.py::test_fix": "FAILED", "test_requests.py::test_existing": "PASSED"}
RESOLVED_STATUS_MAP = {**UNRESOLVED_STATUS_MAP, "test_requests.py::test_fix": "PASSED"}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exit_code,output,found,status_map,resolved",
    [
        pytest.param(1, UNRESOLVED_OUTPUT, True, UNRESOLVED_STATUS_MAP, False, id="nonzero-unresolved"),
        pytest.param(1, RESOLVED_OUTPUT, True, RESOLVED_STATUS_MAP, True, id="nonzero-resolved"),
        pytest.param(1, "collection failed", False, {}, False, id="missing-markers"),
    ],
)
async def test_swe_reward_preserves_execution_evidence_without_changing_resolution(
    exit_code, output, found, status_map, resolved
):
    metadata = {
        "instance_id": "psf__requests-test",
        "repo": "psf/requests",
        "version": "2.2",
        "base_commit": "0" * 40,
        "test_patch": "",
        "FAIL_TO_PASS": json.dumps(["test_requests.py::test_fix"]),
        "PASS_TO_PASS": json.dumps(["test_requests.py::test_existing"]),
    }
    sandbox = SimpleNamespace(
        write_file=AsyncMock(),
        exec_shell=AsyncMock(
            return_value=ExecResult(
                exit_code=exit_code,
                stdout=output,
                stderr="exec timed out after 600.0s" if exit_code == -1 else "",
            )
        ),
    )

    result = await reward.compute_reward(metadata, sandbox, eval_timeout=600.0)

    assert result["eval_exit_code"] == exit_code
    assert result["eval_completed"] is (exit_code == 0)
    assert result["eval_report"]["found_eval_status"] is found
    assert result["eval_report"]["status_map"] == status_map
    assert result["eval_report"]["resolved"] is resolved
    assert result["resolved"] is resolved
    if not found:
        assert result["eval_report"]["test_status"] is None
    assert sandbox.exec_shell.await_args.kwargs == {"workdir": "/testbed", "timeout": 600.0}


# --------------------------- SWE-bench Pro eval infra failures ---------------------------

PRO_METADATA = {
    "instance_id": "pro-test",
    "selected_test_files_to_run": ["tests/test_a.py"],
    "test_patch": "",
    "run_script": "",
    "parser": "",
    "fail_to_pass": ["test_fix"],
    "pass_to_pass": [],
}
_MISSING = object()


def _pro_sandbox(eval_result, *, output=_MISSING, probe_exit=1, probe_infra_error=None, eval_seconds=0.0):
    async def exec_shell(*args, **kwargs):
        await asyncio.sleep(eval_seconds)
        return eval_result

    async def read_file(path):
        if path.endswith("output.json") and output is not _MISSING:
            return json.dumps(output).encode()
        raise FileNotFoundError(path)

    return SimpleNamespace(
        write_file=AsyncMock(),
        exec_shell=exec_shell,
        exec=AsyncMock(return_value=ExecResult(probe_exit, "", "", infra_error=probe_infra_error)),
        read_file=read_file,
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("eval_result", "sandbox_kwargs"),
    [
        pytest.param(
            ExecResult(127, "", "envd gone", infra_error="SandboxUnreachableException: x"), {}, id="unreachable"
        ),
        pytest.param(ExecResult(127, "", "hung", infra_error="SandboxChannelError: hung"), {}, id="hung-stream"),
        pytest.param(ExecResult(-1, "", "exec timed out"), {}, id="early-transport-timeout"),
        pytest.param(ExecResult(0, "", ""), {"probe_exit": 0}, id="output-unreadable"),
        pytest.param(ExecResult(0, "", ""), {"probe_infra_error": "RuntimeError: down"}, id="probe-unreachable"),
    ],
)
async def test_swe_pro_eval_infra_failure_raises_only_when_requested(eval_result, sandbox_kwargs):
    """An eval that could not run must not become reward 0 when the task drops infra failures."""
    from uni_agent.tasks.swe_bench_pro import reward as pro_reward

    with pytest.raises(pro_reward.EvalInfraError):
        await pro_reward.compute_reward(
            PRO_METADATA, _pro_sandbox(eval_result, **sandbox_kwargs), eval_timeout=900.0, raise_on_infra_error=True
        )

    result = await pro_reward.compute_reward(
        PRO_METADATA, _pro_sandbox(eval_result, **sandbox_kwargs), eval_timeout=900.0
    )
    assert result["resolved"] is False and result["eval_completed"] is False


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("eval_result", "eval_timeout", "output"),
    [
        pytest.param(ExecResult(1, "", "patch failed"), 900.0, _MISSING, id="test-patch-did-not-apply"),
        pytest.param(ExecResult(-1, "", "exec timed out"), 0.01, _MISSING, id="hit-eval-timeout"),
        pytest.param(
            ExecResult(0, "", ""), 900.0, {"tests": [{"name": "test_fix", "status": "FAILED"}]}, id="tests-fail"
        ),
    ],
)
async def test_swe_pro_policy_failures_still_score_zero(eval_result, eval_timeout, output):
    """Failing tests and the eval's own timeout are policy outcomes and keep reward 0."""
    from uni_agent.tasks.swe_bench_pro import reward as pro_reward

    result = await pro_reward.compute_reward(
        PRO_METADATA,
        _pro_sandbox(eval_result, output=output, eval_seconds=0.02 if eval_result.exit_code == -1 else 0.0),
        eval_timeout=eval_timeout,
        raise_on_infra_error=True,
    )
    assert result["resolved"] is False


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("instance_timeout", "expected"),
    [
        pytest.param(None, 900.0, id="no-calibration"),
        pytest.param(180.0, 180.0, id="calibrated-tighter"),
        pytest.param(5000.0, 900.0, id="never-above-global"),
    ],
)
async def test_swe_pro_instance_eval_timeout_caps_global(instance_timeout, expected):
    """A calibrated per-instance eval_timeout_s cuts patch-induced hangs short but never extends the global cap."""
    from uni_agent.tasks.swe_bench_pro import reward as pro_reward

    metadata = dict(PRO_METADATA, eval_timeout_s=instance_timeout)
    sandbox = _pro_sandbox(ExecResult(0, "", ""), output={"tests": [{"name": "test_fix", "status": "PASSED"}]})
    seen = {}

    async def exec_shell(*args, **kwargs):
        seen.update(kwargs)
        return ExecResult(0, "", "")

    sandbox.exec_shell = exec_shell
    result = await pro_reward.compute_reward(metadata, sandbox, eval_timeout=900.0)
    assert seen["timeout"] == expected
    assert result["eval_timeout"] == expected
    assert result["resolved"] is True
