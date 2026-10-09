from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import os
import re
import shlex
import threading
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import ExecResult, Sandbox, SandboxChannelError, _env_number, _to_str
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import SandboxConfig

logger = logging.getLogger(__name__)

_TEMPLATE_BUILD_LOCKS: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)

# E2B starts each command's shell with the image's environment, but the login
# shell's /etc/profile then resets PATH to the distribution default, hiding
# toolchains the image put on PATH (e.g. /usr/local/go/bin). Restore the PATH
# the shell was started with.
_RESTORE_IMAGE_PATH = (
    "_p=$(tr '\\0' '\\n' < /proc/$$/environ 2>/dev/null | sed -n 's/^PATH=//p')\n"
    '[ -n "$_p" ] && export PATH="$_p"\n'
    "unset _p\n"
)
_RESTORE_IMAGE_PATH_FILE = "/etc/profile.d/zz-uni-agent-image-path.sh"

# A provider at capacity (503 "no capacity available"), rate limiting (429), or
# briefly failing (500/502/504, or an error body the SDK cannot parse) is not a failed start:
# wait, up to E2B_CAPACITY_WAIT seconds, instead of dropping the episode.
# Raise SANDBOX_STARTUP_TIMEOUT above this when it is raised.
_DEFAULT_CAPACITY_WAIT = 1800.0
# Bound on commands run without a timeout (tmux bookkeeping, file probes), which
# are expected to return in seconds; override with E2B_EXEC_HARD_TIMEOUT.
_DEFAULT_EXEC_HARD_TIMEOUT = 600.0
_EXEC_HARD_TIMEOUT_GRACE = 60.0
# Bound on the best-effort kill of a command whose stream already failed.
_COMMAND_KILL_TIMEOUT = 30.0
_TRANSIENT_STATUS = re.compile(r"^(429|500|502|503|504):")
_TRANSIENT_EXCEPTIONS = frozenset({"ServiceBusyException", "RateLimitException"})


def _is_transient_provider_error(exc: BaseException) -> bool:
    if type(exc).__name__ in _TRANSIENT_EXCEPTIONS or "no capacity" in str(exc).lower():
        return True
    if isinstance(exc, KeyError) and exc.args == ("code",):
        return True
    return type(exc).__module__.startswith("e2b") and bool(_TRANSIENT_STATUS.match(str(exc)))


def _ensure_template(image: str, cpu_count: int, memory_mb: int) -> str:
    """Build the image-backed template once and return its deterministic name."""
    import e2b

    image = image.strip()
    if not image:
        raise ValueError("image must be a non-empty string")

    slug = re.sub(r"[^a-z0-9_-]+", "-", image.lower()).strip("-_")[:64] or "image"
    digest = hashlib.sha256(image.encode()).hexdigest()[:12]
    name = f"uni-agent-{slug}-{cpu_count}c-{memory_mb}mb-{digest}"
    with _TEMPLATE_BUILD_LOCKS[name]:
        if not e2b.Template.exists(name):
            e2b.Template.build(
                e2b.Template().from_image(image),
                name,
                cpu_count=cpu_count,
                memory_mb=memory_mb,
                on_build_logs=None,
            )
    return name


@register_sandbox("e2b")
class E2BSandbox(Sandbox):
    """Create an E2B sandbox from an image-backed template."""

    def __init__(
        self,
        *,
        image: str | None = None,
        template: str | None = None,
        cpu_count: int = 2,
        memory_mb: int = 4096,
        runtime_timeout: float = 3600.0,
        user: str | None = None,
        metadata: dict[str, str] | None = None,
        envs: dict[str, str] | None = None,
        secure: bool = True,
        allow_internet_access: bool = True,
        network: dict[str, Any] | None = None,
        lifecycle: dict[str, Any] | None = None,
        file_request_timeout: float = 600.0,
        lease_seconds: float | None = None,
    ) -> None:
        if runtime_timeout <= 0:
            raise ValueError("runtime_timeout must be positive")
        if lease_seconds is not None and lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive when provided")
        if file_request_timeout <= 0:
            raise ValueError("file_request_timeout must be positive")
        if cpu_count < 1:
            raise ValueError("cpu_count must be at least 1")
        if memory_mb < 512:
            raise ValueError("memory_mb must be at least 512")
        if image is not None and not image.strip():
            raise ValueError("image must be non-empty when provided")
        if template is not None and not template.strip():
            raise ValueError("template must be non-empty when provided")
        if image is None and template is None:
            raise ValueError("either image or template must be provided")

        self.image = image.strip() if image is not None else None
        self.template = template.strip() if template is not None else None
        self.cpu_count = cpu_count
        self.memory_mb = memory_mb
        self.runtime_timeout = float(runtime_timeout)
        self.user = user
        self.metadata = dict(metadata or {})
        # Lets a job-level reaper find this job's sandboxes after a hard cancel.
        if os.environ.get("SLURM_JOB_ID"):
            self.metadata.setdefault("slurm_job_id", os.environ["SLURM_JOB_ID"])
        self.envs = dict(envs or {})
        self.secure = secure
        self.allow_internet_access = allow_internet_access
        self.network = network
        self.lifecycle = lifecycle
        self.file_request_timeout = float(file_request_timeout)
        # With a lease the sandbox is created with a short E2B timeout that a
        # keepalive task renews (up to runtime_timeout). If this process dies
        # without stop() (SIGKILL, job cancel, forced ray.cancel) the sandbox
        # expires after one lease instead of living out runtime_timeout.
        self.lease_seconds = None if lease_seconds is None else float(lease_seconds)

        self._sandbox: Any | None = None
        self.sandbox_id: str | None = None
        self._keepalive_task: asyncio.Task | None = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> E2BSandbox:
        return cls(
            image=config.image,
            runtime_timeout=config.runtime_timeout,
            **config.sandbox_kwargs,
        )

    async def start(self) -> None:
        if self._sandbox is not None:
            return

        from e2b import AsyncSandbox

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _env_number("E2B_CAPACITY_WAIT", _DEFAULT_CAPACITY_WAIT)
        delay = 5.0

        async def backoff(exc: Exception) -> None:
            nonlocal delay
            if not _is_transient_provider_error(exc) or loop.time() + delay > deadline:
                raise exc
            logger.info("E2B unavailable, retrying in %.0fs: %r", delay, exc)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)

        template = self.template
        while template is None:
            assert self.image is not None
            try:
                template = await asyncio.to_thread(
                    _ensure_template,
                    self.image,
                    self.cpu_count,
                    self.memory_mb,
                )
            except Exception as exc:
                await backoff(exc)

        use_lease = self.lease_seconds is not None and self.lease_seconds < self.runtime_timeout
        create_kwargs: dict[str, Any] = {
            "template": template,
            "timeout": int(self.lease_seconds if use_lease else self.runtime_timeout),
            "metadata": self.metadata,
            "envs": self.envs,
            "secure": self.secure,
            "allow_internet_access": self.allow_internet_access,
        }
        if self.network is not None:
            create_kwargs["network"] = self.network
        if self.lifecycle is not None:
            create_kwargs["lifecycle"] = self.lifecycle

        while True:
            try:
                sandbox = await AsyncSandbox.create(**create_kwargs)
                break
            except Exception as exc:
                await backoff(exc)
        self._sandbox = sandbox
        self.sandbox_id = str(sandbox.sandbox_id)
        logger.info("E2B sandbox %s started (template=%s metadata=%s)", self.sandbox_id, template, self.metadata)
        if use_lease:
            self._keepalive_task = asyncio.create_task(self._keep_alive(sandbox, loop.time() + self.runtime_timeout))
        await sandbox.files.write(
            _RESTORE_IMAGE_PATH_FILE,
            _RESTORE_IMAGE_PATH,
            user="root",
            request_timeout=self.file_request_timeout,
        )

    async def _keep_alive(self, sandbox: Any, deadline: float) -> None:
        """Renew the sandbox lease until ``deadline`` (start + runtime_timeout)."""
        assert self.lease_seconds is not None
        loop = asyncio.get_running_loop()
        interval = max(1.0, self.lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            try:
                await asyncio.wait_for(
                    sandbox.set_timeout(max(1, int(min(self.lease_seconds, remaining)))),
                    timeout=interval,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("E2B sandbox %s: lease renewal failed: %r", self.sandbox_id, exc)

    async def stop(self) -> None:
        keepalive, self._keepalive_task = self._keepalive_task, None
        if keepalive is not None:
            keepalive.cancel()
        sandbox, self._sandbox = self._sandbox, None
        sandbox_id, self.sandbox_id = self.sandbox_id, None
        if sandbox is None:
            return
        try:
            await sandbox.kill()
        except Exception as exc:
            logger.warning("Failed to kill E2B sandbox %s: %s", sandbox_id or "?", exc)

    def _require_sandbox(self) -> Any:
        if self._sandbox is None:
            raise RuntimeError("E2BSandbox not started; call start() first")
        return self._sandbox

    async def is_alive(self) -> bool:
        if self._sandbox is None:
            return False
        try:
            return bool(await self._sandbox.is_running())
        except Exception:
            return False

    def _is_timeout_error(self, exc: BaseException) -> bool:
        return type(exc).__name__ == "TimeoutException" or super()._is_timeout_error(exc)

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        from e2b import CommandExitException

        # E2B executes command strings through a shell. ``exec`` replaces that
        # wrapper with the requested process so handle.kill() targets the real
        # command rather than leaving an orphaned child behind.
        command = f"exec {shlex.join(argv)}"
        command_timeout = timeout if timeout is not None and timeout > 0 else 0
        # The SDK opens the stream under ``asyncio.timeout(command_timeout)``
        # inside an async generator, so that deadline is bound to whichever task
        # opens the stream, while the handle reads it from its own task. Opened
        # here, an expiring command timeout would cancel this caller mid-await
        # with a bare CancelledError; a throwaway task absorbs it instead.
        handle = await asyncio.ensure_future(
            self._require_sandbox().commands.run(
                command,
                background=True,
                timeout=command_timeout,
                user=self.user,
                envs=env,
                cwd=workdir,
            )
        )
        # A sandbox that becomes unreachable can leave wait() hanging forever;
        # bound it past the command's own timeout. Hitting this bound is a broken
        # channel, not a command timeout, so it raises SandboxChannelError.
        # asyncio.wait (not wait_for) keeps a TimeoutError raised by wait()
        # itself distinguishable from this bound.
        if command_timeout:
            hard_timeout = command_timeout + _EXEC_HARD_TIMEOUT_GRACE
        else:
            hard_timeout = _env_number("E2B_EXEC_HARD_TIMEOUT", _DEFAULT_EXEC_HARD_TIMEOUT)
        wait_task = asyncio.ensure_future(handle.wait())
        # On the hung-stream path the task is abandoned and may still fail later.
        wait_task.add_done_callback(lambda t: t.cancelled() or t.exception())
        try:
            done, _ = await asyncio.wait({wait_task}, timeout=hard_timeout if hard_timeout > 0 else None)
            if not done:
                raise SandboxChannelError(
                    f"E2B command stream in sandbox {self.sandbox_id} returned nothing for {hard_timeout:g}s "
                    f"(command timeout {command_timeout:g}s)"
                )
            result = wait_task.result()
        except CommandExitException as exc:
            return ExecResult(
                exit_code=int(exc.exit_code),
                stdout=_to_str(exc.stdout),
                stderr=_to_str(exc.stderr),
            )
        except BaseException:
            wait_task.cancel()
            # The channel that just failed may also hang the kill RPC.
            try:
                await asyncio.wait_for(asyncio.shield(handle.kill()), timeout=_COMMAND_KILL_TIMEOUT)
            except Exception:
                logger.warning("Failed to kill E2B command in sandbox %s", self.sandbox_id)
            raise
        return ExecResult(
            exit_code=int(result.exit_code or 0),
            stdout=_to_str(result.stdout),
            stderr=_to_str(result.stderr),
        )

    async def read_file(self, path: str) -> bytes:
        data = await self._require_sandbox().files.read(
            path,
            format="bytes",
            user=self.user,
            request_timeout=self.file_request_timeout,
        )
        return bytes(data)

    async def write_file(self, path: str, content: bytes | str) -> None:
        await self._require_sandbox().files.write(
            path,
            content,
            user=self.user,
            request_timeout=self.file_request_timeout,
            use_octet_stream=isinstance(content, bytes),
        )

    async def upload_file(self, local_file: Path | str, remote_file: str) -> None:
        with Path(local_file).open("rb") as stream:
            await self._require_sandbox().files.write(
                remote_file,
                stream,
                user=self.user,
                request_timeout=self.file_request_timeout,
                use_octet_stream=True,
            )

    async def download_file(self, remote_file: str, local_file: Path | str) -> None:
        data = await self.read_file(remote_file)
        destination = Path(local_file)

        def _write() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)

        await asyncio.to_thread(_write)

    async def expose_port(self, port: int) -> str:
        host = self._require_sandbox().get_host(port)
        if inspect.isawaitable(host):
            host = await host
        host = str(host)
        return host if "://" in host else f"https://{host}"
