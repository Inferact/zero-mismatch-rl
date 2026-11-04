# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# cumem-based pytorch pluggable allocator to implement sleep mode.
# other approaches tried but failed:
# - cuda-python package binding
# - custom libcuda driver ctypes wrapper
# both of them failed because of cuda context mismatch.
# not sure why, they are created from a different context.
# the only successful approach is to call cuda driver API in C.
import atexit
import ctypes
import gc
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from vllm.device_allocator import DEFERRABLE_TAGS, AllocationData, HandleType
from vllm.device_allocator.alloc_conf import (
    EXPANDABLE_SEGMENTS,
    conf_flag_enabled,
    current_alloc_conf,
    set_alloc_conf,
    with_conf_flag,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.system_utils import find_loaded_library

logger = init_logger(__name__)

_K3_SLEEP_BACKUP = os.environ.get("K3_SLEEP_BACKUP", "cudahost")
assert _K3_SLEEP_BACKUP in ("cudahost", "disk"), _K3_SLEEP_BACKUP
_k3_cudart = None


class _K3HostBuffer:
    def __init__(self, nbytes: int) -> None:
        global _k3_cudart
        if _k3_cudart is None:
            _k3_cudart = ctypes.CDLL(find_loaded_library("libcudart"))
            _k3_cudart.cudaMallocHost.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
            ]
            _k3_cudart.cudaMallocHost.restype = ctypes.c_int
            _k3_cudart.cudaFreeHost.argtypes = [ctypes.c_void_p]
            _k3_cudart.cudaFreeHost.restype = ctypes.c_int
        ptr = ctypes.c_void_p()
        rc = _k3_cudart.cudaMallocHost(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        if rc != 0 or not ptr.value:
            raise RuntimeError(f"cudaMallocHost({nbytes}) failed with code {rc}")
        self._ptr = ptr.value
        self._nbytes = nbytes

    def data_ptr(self) -> int:
        return self._ptr

    def numel(self) -> int:
        return self._nbytes

    def element_size(self) -> int:
        return 1

    def __del__(self) -> None:
        ptr, self._ptr = getattr(self, "_ptr", None), None
        if ptr and _k3_cudart is not None:
            _k3_cudart.cudaFreeHost(ctypes.c_void_p(ptr))


_K31_DISK_DIR = os.environ.get("K31_SLEEP_DISK_DIR", "")
_K31_DISK_MIN = 4 << 20
_K31_DISK_THREADS = 8
_K31_DISK_CHUNK = 64 << 20


class _K31CudaArr:
    def __init__(self, ptr: int, n: int):
        self.__cuda_array_interface__ = {"shape": (n,), "typestr": "|u1", "data": (ptr, False), "version": 3}


def _k31_gpu_bytes(ptr: int, n: int) -> torch.Tensor:
    return torch.as_tensor(_K31CudaArr(ptr, n), device=torch.device("cuda", torch.cuda.current_device()))


def _k31_checksum(ptr: int, n: int) -> int:
    t = _k31_gpu_bytes(ptr, n)
    w = n // 4
    acc = 0
    for off in range(0, w, 1 << 28):
        m = min(1 << 28, w - off)
        v = int(t[4 * off : 4 * (off + m)].view(torch.int32).sum(dtype=torch.int64).item())
        acc = (acc * 1000003 + v) % (1 << 61)
    if n % 4:
        acc = (acc * 1000003 + int(t[4 * w :].sum(dtype=torch.int64).item())) % (1 << 61)
    return acc


def _k31_param_ranges() -> list:
    rs = set()

    def add(t):
        try:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                st = t.untyped_storage()
                p, n = st.data_ptr(), st.nbytes()
                if n >= _K31_DISK_MIN:
                    rs.add((p, p + n))
        except Exception:  # noqa: BLE001
            pass

    frozen = gc.get_freeze_count()
    if frozen:
        gc.unfreeze()
    try:
        objs = gc.get_objects()
    finally:
        if frozen:
            gc.freeze()
    for o in objs:
        if isinstance(o, torch.nn.Module):
            for t in o._parameters.values():
                add(t)
        elif isinstance(o, torch.nn.Parameter):
            add(o)
    del objs
    logger.info("K31 sleep-disk: %d parameter storages >= %d MiB (%.2f GiB), gc frozen objects %d",
                len(rs), _K31_DISK_MIN >> 20, sum(b - a for a, b in rs) / 2**30, frozen)
    merged = []
    for a, b in sorted(rs):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    cap = os.environ.get("K31_SLEEP_DISK_MAX_GIB")
    if cap:
        cap_b, kept, acc = int(float(cap) * 2**30), [], 0
        for a, b in merged:
            if acc + (b - a) > cap_b:
                continue
            kept.append((a, b))
            acc += b - a
        logger.info("K31 sleep-disk cap: %.2f GiB of %.2f GiB on disk (cap %.2f GiB)", acc / 2**30,
                    sum(b - a for a, b in merged) / 2**30, cap_b / 2**30)
        merged = kept
    return [tuple(x) for x in merged]


class _K31Backup:
    def __init__(self, ptr: int, size: int, imm: list):
        comp, cur = [], 0
        for off, n in imm:
            if off > cur:
                comp.append((cur, off - cur))
            cur = off + n
        if cur < size:
            comp.append((cur, size - cur))
        self.comp = comp
        tot = sum(n for _, n in comp)
        self.buf = _K3HostBuffer(tot) if tot else None
        self.nbytes = tot
        hp = self.buf.data_ptr() if self.buf else 0
        for off, n in comp:
            libcudart.cudaMemcpy(hp, ptr + off, n)
            hp += n

    def restore(self, ptr: int) -> None:
        hp = self.buf.data_ptr() if self.buf else 0
        for off, n in self.comp:
            libcudart.cudaMemcpy(ptr + off, hp, n)
            hp += n
        self.buf = None

    def numel(self) -> int:
        return self.nbytes

    def element_size(self) -> int:
        return 1


class _K31DiskStore:
    def __init__(self):
        import threading

        assert _K31_DISK_DIR, "K3_SLEEP_BACKUP=disk needs K31_SLEEP_DISK_DIR"
        os.makedirs(_K31_DISK_DIR, exist_ok=True)
        self.entries: dict = {}
        self.ranges = None
        self._starts = []
        self.stats: dict = {}
        self._fds: dict = {}
        self._lock = threading.Lock()
        self._tmp: set = set()
        atexit.register(self.close)

    def close(self) -> None:
        for fd in list(self._fds.values()):
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()
        for p in list(self._tmp):
            try:
                os.unlink(p)
            except OSError:
                pass

    def _fd(self, path: str) -> int:
        with self._lock:
            fd = self._fds.get(path)
            if fd is None:
                fd = self._fds[path] = os.open(path, os.O_RDONLY)
            return fd

    def imm_for(self, ptr: int, size: int) -> list:
        import bisect

        if self.ranges is None:
            self.ranges = _k31_param_ranges()
            self._starts = [a for a, _ in self.ranges]
        out = []
        i = max(bisect.bisect_right(self._starts, ptr) - 1, 0)
        while i < len(self.ranges) and self.ranges[i][0] < ptr + size:
            a, b = self.ranges[i]
            lo, hi = max(a, ptr), min(b, ptr + size)
            if hi > lo:
                out.append((lo - ptr, hi - lo))
            i += 1
        return out

    def _pool(self, nthr: int):
        import queue

        import numpy as np

        dev = torch.cuda.current_device()
        stages, keep = queue.Queue(), []
        for _ in range(nthr):
            hb = _K3HostBuffer(_K31_DISK_CHUNK)
            arr = np.ctypeslib.as_array((ctypes.c_uint8 * _K31_DISK_CHUNK).from_address(hb.data_ptr()))
            keep.append(hb)
            stages.put((arr, torch.from_numpy(arr), torch.cuda.Stream(device=dev)))
        return dev, stages, keep

    def _write_ranges(self, ranges: list) -> list:
        import concurrent.futures as cf
        import hashlib
        import itertools

        nthr = max(1, min(_K31_DISK_THREADS, len(ranges)))
        dev, stages, keep = self._pool(nthr)
        seq = itertools.count()

        def run(r):
            g, n = r
            torch.cuda.set_device(dev)
            arr, ht, strm = stages.get()
            tmp = os.path.join(_K31_DISK_DIR, f".tmp-{os.getpid()}-{next(seq)}")
            self._tmp.add(tmp)
            try:
                h = hashlib.blake2b(digest_size=16)
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    for k in range(0, n, _K31_DISK_CHUNK):
                        m = min(_K31_DISK_CHUNK, n - k)
                        with torch.cuda.stream(strm):
                            ht[:m].copy_(_k31_gpu_bytes(g + k, m), non_blocking=True)
                            strm.synchronize()
                        mv = memoryview(arr)[:m]
                        h.update(mv)
                        done = 0
                        while done < m:
                            done += os.write(fd, mv[done:])
                    os.fsync(fd)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                final = os.path.join(_K31_DISK_DIR, f"{h.hexdigest()}-{n}.bin")
                try:
                    os.link(tmp, final)
                    new = True
                except FileExistsError:
                    new = False
                return final, new
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                self._tmp.discard(tmp)
                stages.put((arr, ht, strm))

        with cf.ThreadPoolExecutor(nthr) as ex:
            res = list(ex.map(run, ranges))
        torch.cuda.synchronize()
        del keep
        return res

    def _read_ranges(self, jobs: list) -> None:
        import concurrent.futures as cf

        tasks = []
        for g, n, path in jobs:
            for k in range(0, n, _K31_DISK_CHUNK):
                tasks.append((g + k, min(_K31_DISK_CHUNK, n - k), path, k))
        nthr = max(1, min(_K31_DISK_THREADS, len(tasks)))
        dev, stages, keep = self._pool(nthr)

        def run(t):
            g, m, path, fo = t
            torch.cuda.set_device(dev)
            arr, ht, strm = stages.get()
            try:
                fd = self._fd(path)
                mv = memoryview(arr)[:m]
                done = 0
                while done < m:
                    r = os.preadv(fd, [mv[done:]], fo + done)
                    if r <= 0:
                        raise RuntimeError(f"K31 sleep-disk: short read {path}@{fo + done}")
                    done += r
                with torch.cuda.stream(strm):
                    _k31_gpu_bytes(g, m).copy_(ht[:m], non_blocking=True)
                    strm.synchronize()
            finally:
                stages.put((arr, ht, strm))

        with cf.ThreadPoolExecutor(nthr) as ex:
            list(ex.map(run, tasks))
        torch.cuda.synchronize()
        del keep

    def save_or_verify(self, items: list) -> None:
        import time

        t0 = time.time()
        new, staged, nb_ver = [], {}, 0
        for ptr, _size, imm in items:
            if ptr in self.entries:
                for off, n, _p, cs in self.entries[ptr]:
                    if _k31_checksum(ptr + off, n) != cs:
                        raise RuntimeError(
                            f"K31 sleep-disk: parameter bytes at {ptr + off:#x}+{n} changed since the "
                            "snapshot (base weights are not frozen); use K3_SLEEP_BACKUP=cudahost")
                    nb_ver += n
                continue
            for off, n in imm:
                new.append((ptr, off, n, _k31_checksum(ptr + off, n)))
            staged[ptr] = []
        t1 = time.time()
        nb_new = nb_written = 0
        if new:
            res = self._write_ranges([(p + off, n) for p, off, n, _ in new])
            for (p, off, n, cs), (path, wrote) in zip(new, res):
                staged[p].append((off, n, path, cs))
                nb_new += n
                nb_written += n if wrote else 0
        self.entries.update(staged)
        self.stats = {"snapshot_gib": nb_new / 2**30, "written_gib": nb_written / 2**30,
                      "verified_gib": nb_ver / 2**30, "checksum_s": t1 - t0, "write_s": time.time() - t1}

    def restore(self, ptr: int) -> int:
        ents = self.entries.get(ptr, [])
        if ents:
            self._read_ranges([(ptr + off, n, path) for off, n, path, _ in ents])
            for off, n, path, cs in ents:
                if _k31_checksum(ptr + off, n) != cs:
                    raise RuntimeError(f"K31 sleep-disk: bytes read back from {path} differ")
        return sum(e[1] for e in ents)

    def drop_cache(self) -> None:
        for fd in list(self._fds.values()):
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass


_k31_store = None


def _k31_disk_store() -> "_K31DiskStore":
    global _k31_store
    if _k31_store is None:
        _k31_store = _K31DiskStore()
        logger.info("K31 sleep-disk active: dir=%s min=%d MiB threads=%d", _K31_DISK_DIR,
                    _K31_DISK_MIN >> 20, _K31_DISK_THREADS)
    return _k31_store


cumem_available = False
libcudart: Any = None
try:
    from vllm.cumem_allocator import (
        init_module,
        python_create_and_map,
        python_unmap_and_release,
    )
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib_name = find_loaded_library("cumem_allocator")
    libcudart = CudaRTLibrary()
    cumem_available = True
except ModuleNotFoundError:
    # only cuda and rocm platforms support cumem allocator
    init_module = None
    python_create_and_map = None
    python_unmap_and_release = None
    lib_name = None


def create_and_map(allocation_handle: HandleType) -> None:
    python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    python_unmap_and_release(*allocation_handle)


def get_pluggable_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
) -> torch.cuda.memory.CUDAPluggableAllocator:
    init_module(python_malloc_fn, python_free_func)
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        lib_name, "my_malloc", "my_free"
    )
    return new_alloc


@contextmanager
def use_memory_pool_with_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
) -> Iterator[
    tuple[torch.cuda.memory.MemPool, torch.cuda.memory.CUDAPluggableAllocator]
]:
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)
    mem_pool = torch.cuda.memory.MemPool(new_alloc._allocator)
    with torch.cuda.memory.use_mem_pool(mem_pool):
        yield mem_pool, new_alloc


class CuMemAllocator:
    """A singleton class that manages a memory pool for CUDA tensors.
    The memory in this pool can be offloaded or discarded when the
    allocator sleeps.

    Inside the `use_memory_pool(tag)` context, all tensors created will
    be allocated in the memory pool, and has the same tag as the
    tag passed to the context.

    When we call `sleep`, all tensors with the specified tag will be
    offloaded to CPU memory, and the rest of the tensors will be discarded.
    When we call `wake_up`, all tensors that are previously offloaded
    will be loaded back to GPU memory, and the rest of the tensors will
    have empty memory (zeroed on ROCm).

    Why it needs to be a singleton?
    When allocated tensors are garbage collected, PyTorch will call
    the free callback, which will call the `python_free_callback` method.
    The C-extension uses a global variable to store the function of an
    instance of this class. If we create multiple instances of this class,
    the global variable will be overwritten and the free callback will
    not work as expected.
    """

    instance: "CuMemAllocator | None" = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CuMemAllocator":
        """CuMemAllocator is a singleton class.
        We cannot call the constructor directly.
        Call this method to get the instance.
        """
        assert cumem_available, "cumem allocator is not available"
        if CuMemAllocator.instance is None:
            CuMemAllocator.instance = CuMemAllocator()
            # Ensure MemPool/allocator wrappers are released before interpreter
            # finalization tears down PyTorch allocator internals.
            atexit.register(CuMemAllocator._shutdown_singleton)
        return CuMemAllocator.instance

    @staticmethod
    def _shutdown_singleton() -> None:
        instance = CuMemAllocator.instance
        if instance is None:
            return
        try:
            instance.release_pools()
        except Exception:
            logger.exception("CuMemAllocator singleton shutdown failed")

    def __init__(self):
        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = CuMemAllocator.default_tag
        self.allocator_and_pools: dict[str, list[Any]] = {}
        # Creating strong references to the two callbacks here to prevent
        # these ephemeral bound-method objects being garbage collected.
        # See discussions in https://github.com/vllm-project/vllm/pull/22724
        self.python_malloc_callback = self._python_malloc_callback
        self.python_free_callback = self._python_free_callback

    def release_pools(self) -> None:
        """Drop Python references to MemPool/pluggable allocators eagerly.

        A cumem ``MemPool`` outlives the ``use_memory_pool`` context (a strong
        reference is kept in ``allocator_and_pools`` to work around
        pytorch/pytorch#146431), and a captured CUDA graph can keep it alive
        longer still. ``MemPool`` only holds a non-owning pointer to the
        allocator, whose owning reference lives in the Python
        ``CUDAPluggableAllocator``. If both are instead dropped during
        interpreter shutdown, GC may finalize the allocator first; the eventual
        ``~MemPool`` -> ``emptyCache`` -> ``release_block`` then makes a virtual
        call into the freed allocator -- aborting the process with "pure virtual
        method called" (pytorch/pytorch#145168).

        Release the kept-alive pools before interpreter finalization, and keep
        the pluggable allocator wrappers alive while MemPool destructors run.
        This is safe to call more than once.
        """
        if not self.allocator_and_pools:
            return

        pool_entries = [e for es in self.allocator_and_pools.values() for e in es]
        self.allocator_and_pools.clear()

        mem_pools = [entry[0] for entry in pool_entries]
        allocators = [entry[1] for entry in pool_entries]
        pool_entries.clear()

        # Phase 1: drop MemPool refs while allocators are still strongly held.
        mem_pools.clear()
        gc.collect()

        # Phase 2: now it is safe to release allocator wrappers.
        allocators.clear()

    def close(self) -> None:
        """Compatibility alias for deterministic pool release."""
        self.release_pools()

    def _python_malloc_callback(self, allocation_handle: HandleType) -> None:
        """Internal method to store the allocation data
        when memory is allocated in the memory pool.
        """
        py_d_mem = allocation_handle[2]
        self.pointer_to_data[py_d_mem] = AllocationData(
            allocation_handle, self.current_tag
        )
        logger.debug(
            "Allocated %s bytes for %s with address %s from cumem allocator",
            allocation_handle[1],
            self.current_tag,
            py_d_mem,
        )
        return

    def _python_free_callback(self, ptr: int) -> HandleType:
        """Internal method to look up the allocation data
        when memory is freed in the memory pool.
        """
        data = self.pointer_to_data.pop(ptr)
        if data.cpu_backup_tensor is not None:
            data.cpu_backup_tensor = None
        if data.is_asleep and current_platform.is_rocm():
            # On ROCm, sleep() already unmapped and released this allocation's
            # physical chunks and holds its virtual address as a placeholder
            # reservation. Return a handle with an empty chunk list so the C
            # extension skips unmap/release (avoiding a double-free) while
            # still freeing the placeholder address.
            device, size, d_mem, _ = data.handle
            return (device, size, d_mem, [])
        # Drain pending kernels before the C extension's cuMemUnmap.
        # The pluggable allocator path doesn't defer reclaim like the
        # regular caching allocator, so without this, in-flight work
        # (e.g. quant helpers' transient tensors during weight loading)
        # races the unmap and surfaces as CUDA_ERROR_ILLEGAL_ADDRESS.
        torch.accelerator.synchronize(data.handle[0])
        logger.debug(
            "Freed %s bytes for %s with address %s from cumem allocator",
            data.handle[1],
            data.tag,
            ptr,
        )
        return data.handle

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        """Put the allocator in sleep mode.
        All data in the memory allocation with the specified tag will be
        offloaded to CPU memory, and others will be discarded.

        Args:
            offload_tags: The tags of the memory allocation that will be
                offloaded. The rest of the memory allocation will be discarded.

        """
        if offload_tags is None:
            # by default, allocated tensors are offloaded
            # when the allocator sleeps
            offload_tags = (CuMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        assert isinstance(offload_tags, tuple)

        total_bytes = 0
        backup_bytes = 0
        has_policy_conflict = False
        k31_disk = _K3_SLEEP_BACKUP == "disk" and "weights" in offload_tags
        k31_items = {}
        if k31_disk:
            import time as _t

            _t0 = _t.time()
            torch.cuda.synchronize()
            store = _k31_disk_store()
            for ptr, data in self.pointer_to_data.items():
                if not data.is_asleep and data.tag == "weights":
                    k31_items[ptr] = store.imm_for(ptr, data.handle[1])
            store.save_or_verify([(p, self.pointer_to_data[p].handle[1], imm) for p, imm in k31_items.items()])
            k31_disk_gib = sum(n for imm in k31_items.values() for _, n in imm) / 2**30

        for ptr, data in self.pointer_to_data.items():
            if data.is_asleep:
                requests_offload = data.tag in offload_tags
                was_offloaded = data.cpu_backup_tensor is not None
                if requests_offload != was_offloaded:
                    has_policy_conflict = True
                continue
            handle = data.handle
            total_bytes += handle[1]
            if k31_disk and ptr in k31_items:
                data.cpu_backup_tensor = _K31Backup(ptr, handle[1], k31_items[ptr])
                backup_bytes += data.cpu_backup_tensor.nbytes
            elif data.tag in offload_tags:
                backup_bytes += handle[1]
                size_in_bytes = handle[1]
                cpu_backup_tensor = _K3HostBuffer(size_in_bytes)
                cpu_ptr = cpu_backup_tensor.data_ptr()
                libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)
                data.cpu_backup_tensor = cpu_backup_tensor
            try:
                unmap_and_release(handle)
            finally:
                data.is_asleep = True

        logger.info(
            "CuMemAllocator: sleep freed %.2f GiB memory in total, of which "
            "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded "
            "directly.",
            total_bytes / 1024**3,
            backup_bytes / 1024**3,
            (total_bytes - backup_bytes) / 1024**3,
        )

        if has_policy_conflict:
            logger.warning(
                "CuMemAllocator: sleep cannot change the policy of "
                "already-asleep allocations; the existing policy was kept."
            )
        if k31_disk:
            logger.info("K31 sleep-disk: %.2f GiB on disk (%s), %.2f GiB in host buffers, %.2f s",
                        k31_disk_gib, {k: round(v, 2) for k, v in _k31_store.stats.items()},
                        backup_bytes / 1024**3, _t.time() - _t0)

        gc.collect()
        torch.accelerator.empty_cache()

    def discard(self, tags: tuple[str, ...] | str) -> None:
        """Discard mapped allocations with the given tags without CPU backup."""
        if isinstance(tags, str):
            tags = (tags,)

        discarded_bytes = 0
        has_policy_conflict = False
        for data in self.pointer_to_data.values():
            if data.tag not in tags:
                continue
            if data.is_asleep:
                if data.cpu_backup_tensor is not None:
                    has_policy_conflict = True
                continue
            torch.accelerator.synchronize(data.handle[0])
            unmap_and_release(data.handle)
            data.is_asleep = True
            discarded_bytes += data.handle[1]

        logger.info(
            "CuMemAllocator: discarded %.2f GiB for tags %s.",
            discarded_bytes / 1024**3,
            tags,
        )

        if has_policy_conflict:
            logger.warning(
                "CuMemAllocator: discard cannot change the policy of "
                "already-asleep allocations; the existing policy was kept."
            )

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest will have empty memory (zeroed on ROCm).

        Args:
            tags: The deferrable tags (weights, kv_cache) to load back to GPU
                memory; every other tag is always loaded back. If None, all
                memory allocation will be loaded back to GPU memory.

        """
        gc.collect()
        torch.accelerator.empty_cache()
        k31_disk_restore = []

        for ptr, data in self.pointer_to_data.items():
            if not data.is_asleep:
                continue
            if tags is None or data.tag in tags or data.tag not in DEFERRABLE_TAGS:
                handle = data.handle
                create_and_map(handle)
                data.is_asleep = False
                if isinstance(data.cpu_backup_tensor, _K31Backup):
                    k31_disk_restore.append((ptr, data))
                elif data.cpu_backup_tensor is not None:
                    cpu_backup_tensor = data.cpu_backup_tensor
                    if cpu_backup_tensor is not None:
                        size_in_bytes = (
                            cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                        )
                        cpu_ptr = cpu_backup_tensor.data_ptr()
                        libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                        data.cpu_backup_tensor = None
                elif current_platform.is_rocm():
                    # amdgpu <6.14 may return stale VRAM (#44972); drop with its support
                    libcudart.cudaMemset(ptr, 0, handle[1])
                    torch.accelerator.synchronize()
        if k31_disk_restore:
            import time as _t

            _t0 = _t.time()
            store = _k31_disk_store()
            nd = nh = 0
            for ptr, data in k31_disk_restore:
                nd += store.restore(ptr)
                nh += data.cpu_backup_tensor.nbytes
                data.cpu_backup_tensor.restore(ptr)
                data.cpu_backup_tensor = None
            torch.cuda.synchronize()
            store.drop_cache()
            logger.info("K31 sleep-disk wake: %.2f GiB from disk, %.2f GiB from host, %.2f s",
                        nd / 2**30, nh / 2**30, _t.time() - _t0)

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        """A context manager to use the memory pool.
        All memory allocation created inside the context will be allocated
        in the memory pool, and has the specified tag.

        Args:
            tag: The tag of the memory allocation. If None, the default tag
                will be used.

        """
        if tag is None:
            tag = CuMemAllocator.default_tag

        assert isinstance(tag, str)

        # Expandable segments are incompatible with the memory pool used for
        # sleep mode (see https://github.com/pytorch/pytorch/issues/147851).
        # If the user has enabled expandable segments, temporarily disable
        # them for the duration of the memory pool context and restore on
        # exit. The whole config string is rewritten, not just this one
        # field: torch resets every option that is absent from the string it
        # is handed, so a bare "expandable_segments:False" would also drop
        # max_split_size_mb, garbage_collection_threshold and
        # roundup_power2_divisions for the rest of the process.
        prev_conf = current_alloc_conf()
        expandable_was_enabled = conf_flag_enabled(prev_conf, EXPANDABLE_SEGMENTS)
        if expandable_was_enabled:
            set_alloc_conf(with_conf_flag(prev_conf, EXPANDABLE_SEGMENTS, False))

        old_tag = self.current_tag
        self.current_tag = tag
        try:
            if tag != old_tag:
                # Older pools of this tag are never reused, so trim them.
                for pool, _ in self.allocator_and_pools.get(tag, []):
                    self._trim(pool)
            with use_memory_pool_with_allocator(
                self.python_malloc_callback, self.python_free_callback
            ) as data:
                # start to hit another PyTorch bug in PyTorch 2.6,
                # possibly because of gc-related issue w.r.t. the allocator
                # and the memory pool.
                # to avoid the issue, we keep a reference of the data.
                # see https://github.com/pytorch/pytorch/issues/146431 .
                self.allocator_and_pools.setdefault(tag, []).append(data)
                yield
                # PyTorch's bug, torch.accelerator.empty_cache() does not
                # release memory held by a pluggable allocator, see
                # https://github.com/pytorch/pytorch/issues/145168 .
                # if we have some memory allocated and then freed,
                # the memory will not be released, e.g. in online
                # quantization, where the model is created in higher
                # precision, and then quantized in lower precision.
                # Find all unused allocations and manually release them.
                # TODO: we should expose `empty_cache` method in the memory
                # pool.
                # TODO: ask for help from PyTorch team to expose this method.
                self._trim(data[0])
        finally:
            self.current_tag = old_tag
            if expandable_was_enabled:
                set_alloc_conf(prev_conf)

    def _trim(self, pool: Any) -> None:
        """Release free segments of a pool that is never allocated from again."""
        for allocation in pool.snapshot():
            data = self.pointer_to_data.get(allocation["address"])
            if allocation["allocated_size"] == 0 and data and not data.is_asleep:
                unmap_and_release(self._python_free_callback(allocation["address"]))

    @contextmanager
    def cudagraph_pool(self) -> Iterator[tuple[int, int]]:
        """Tag CUDA graph capture allocations and yield the graph pool id."""
        if "cudagraph" not in self.allocator_and_pools:
            allocator = get_pluggable_allocator(
                self.python_malloc_callback, self.python_free_callback
            )
            mem_pool = torch.cuda.memory.MemPool(allocator._allocator)
            self.allocator_and_pools["cudagraph"] = [(mem_pool, allocator)]
        old_tag = self.current_tag
        self.current_tag = "cudagraph"
        try:
            # capture_begin routes allocations to this pool; no use_mem_pool.
            yield self.allocator_and_pools["cudagraph"][0][0].id
        finally:
            self.current_tag = old_tag

    def get_current_usage(self) -> int:
        """Get the total number of bytes allocated in the memory pool."""
        sum_bytes: int = 0
        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            sum_bytes += handle[1]
        return sum_bytes
