"""SWE-bench Pro patch evaluation."""

from __future__ import annotations

import json
import logging
import shlex
import time
import uuid
from typing import Any

from ...sandbox import ExecResult, SandboxBackend

logger = logging.getLogger(__name__)

REPO_WORKDIR = "/app"


def _grade(metadata: dict[str, Any], output: Any) -> dict[str, Any]:
    if not isinstance(output, dict) or not isinstance(output.get("tests"), list):
        raise ValueError("SWE-bench Pro parser output must contain a 'tests' list")

    fail_to_pass = set(metadata["fail_to_pass"])
    pass_to_pass = set(metadata["pass_to_pass"])
    tests = output["tests"]
    passed = {
        test.get("name")
        for test in tests
        if isinstance(test, dict) and test.get("status") == "PASSED" and isinstance(test.get("name"), str)
    }
    missing_fail_to_pass = sorted(fail_to_pass - passed)
    missing_pass_to_pass = sorted(pass_to_pass - passed)
    return {
        "resolved": not missing_fail_to_pass and not missing_pass_to_pass,
        "found_eval_status": True,
        "test_status": tests,
        "required_count": len(fail_to_pass | pass_to_pass),
        "passed_count": len((fail_to_pass | pass_to_pass) & passed),
        "missing_fail_to_pass": missing_fail_to_pass,
        "missing_pass_to_pass": missing_pass_to_pass,
    }


def _effective_eval_timeout(metadata: dict[str, Any], eval_timeout: float) -> float:
    """Tighten ``eval_timeout`` to the instance's calibrated ``eval_timeout_s``, if any."""
    instance_timeout = metadata.get("eval_timeout_s")
    if instance_timeout is None:
        return eval_timeout
    instance_timeout = float(instance_timeout)
    if not instance_timeout > 0:
        raise ValueError(f"eval_timeout_s must be positive, got {instance_timeout!r}")
    return min(eval_timeout, instance_timeout)


def _selected_test_args(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("selected_test_files_to_run must be a string or a list of strings")


async def _read_text(sandbox: SandboxBackend, path: str) -> str:
    try:
        return (await sandbox.read_file(path)).decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


class EvalInfraError(RuntimeError):
    """The evaluation could not run, so its result says nothing about the patch."""


_PROBE_TIMEOUT_SECONDS = 60.0


def _eval_infra_failure(response: ExecResult, execution_time: float, eval_timeout: float) -> str | None:
    """Why the eval command's result is not a real outcome, or ``None`` if it is."""
    if response.infra_error:
        return f"eval command did not complete: {response.infra_error}"
    # exit -1 is a timeout; one that fires before eval_timeout elapsed came from
    # the transport, not from the eval's own deadline.
    if response.exit_code == -1 and execution_time < eval_timeout:
        return f"eval command timed out after {execution_time:.0f}s, before eval_timeout={eval_timeout:g}s"
    return None


async def _output_missing_reason(sandbox: SandboxBackend, path: str) -> str | None:
    """``None`` if ``path`` is genuinely absent; otherwise why it could not be read."""
    probe = await sandbox.exec(["test", "-e", path], timeout=_PROBE_TIMEOUT_SECONDS)
    if probe.infra_error:
        return f"sandbox unreachable after eval: {probe.infra_error}"
    if probe.exit_code == 0:
        return f"{path} exists but could not be read"
    if probe.exit_code != 1:
        return f"probing {path} failed with exit code {probe.exit_code}: {probe.stderr.strip()}"
    return None


async def compute_reward(
    metadata: dict[str, Any],
    sandbox: SandboxBackend,
    eval_timeout: float = 3600.0,
    raise_on_infra_error: bool = False,
) -> dict[str, Any]:
    """Evaluate the worktree with the official per-instance test scripts.

    With ``raise_on_infra_error``, an eval that could not run (sandbox
    unreachable, hung command stream, output unreadable) raises
    :class:`EvalInfraError` instead of scoring 0. Test failures and the eval
    hitting ``eval_timeout`` still score 0. An instance's optional
    ``eval_timeout_s`` metadata (calibrated from its gold-patch eval time)
    lowers ``eval_timeout`` so a patch that hangs the tests fails fast.
    """
    instance_id = metadata["instance_id"]
    eval_timeout = _effective_eval_timeout(metadata, eval_timeout)
    selected_tests = _selected_test_args(metadata["selected_test_files_to_run"])

    workspace = f"/tmp/swe_bench_pro_eval_{uuid.uuid4().hex}"
    paths = {
        "test_patch": f"{workspace}/test.patch",
        "run_script": f"{workspace}/run_script.sh",
        "parser": f"{workspace}/parser.py",
        "stdout": f"{workspace}/stdout.log",
        "stderr": f"{workspace}/stderr.log",
        "output": f"{workspace}/output.json",
    }
    await sandbox.write_file(paths["test_patch"], metadata["test_patch"])
    await sandbox.write_file(paths["run_script"], metadata["run_script"])
    await sandbox.write_file(paths["parser"], metadata["parser"])

    run_tests = shlex.join(["bash", paths["run_script"], *selected_tests])
    eval_script = "\n".join(
        [
            f"cd {shlex.quote(REPO_WORKDIR)}",
            f"git apply -v {shlex.quote(paths['test_patch'])} || exit $?",
            f"{run_tests} > {shlex.quote(paths['stdout'])} 2> {shlex.quote(paths['stderr'])}",
            shlex.join(["python", paths["parser"], paths["stdout"], paths["stderr"], paths["output"]]),
        ]
    )

    started = time.perf_counter()
    response = await sandbox.exec_shell(
        eval_script,
        workdir=REPO_WORKDIR,
        timeout=eval_timeout,
    )
    execution_time = time.perf_counter() - started
    if raise_on_infra_error:
        reason = _eval_infra_failure(response, execution_time, eval_timeout)
        if reason:
            raise EvalInfraError(f"SWE-bench Pro eval for {instance_id}: {reason}")

    result: dict[str, Any] = {
        "eval_completed": False,
        "eval_execution_time": execution_time,
        "eval_timeout": eval_timeout,
        "eval_report": {
            "resolved": False,
            "found_eval_status": False,
            "eval_exit_code": response.exit_code,
            "error": None,
        },
        "resolved": False,
    }
    output_read = False
    try:
        raw_output = await sandbox.read_file(paths["output"])
        output_read = True
        report = _grade(metadata, json.loads(raw_output.decode("utf-8")))
    except Exception as exc:
        # An eval that hit eval_timeout never wrote output; nothing to probe.
        if raise_on_infra_error and not output_read and response.exit_code != -1:
            reason = await _output_missing_reason(sandbox, paths["output"])
            if reason:
                raise EvalInfraError(f"SWE-bench Pro eval for {instance_id}: {reason}") from exc
        details = [
            response.stderr.strip(),
            response.stdout.strip(),
            await _read_text(sandbox, paths["stderr"]),
            await _read_text(sandbox, paths["stdout"]),
        ]
        detail = next((item for item in details if item), "")
        result["eval_report"]["error"] = detail or f"{type(exc).__name__}: {exc}"
        logger.warning("SWE-bench Pro evaluation failed for %s: %s", instance_id, result["eval_report"]["error"])
        return result

    result["eval_report"].update(report)
    result["eval_completed"] = True
    result["resolved"] = report["resolved"]
    logger.info(
        "SWE-bench Pro reward for %s: resolved=%s required=%s passed=%s",
        instance_id,
        result["resolved"],
        report["required_count"],
        report["passed_count"],
    )
    return result
