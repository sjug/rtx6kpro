"""Bounded disk row staging shared by PLE and Engram.

The native reader owns bounded request planning and I/O. This module owns
storage, source registration and the host/GPU transaction boundary; hashing
and numerical decoding belong to each embedding operation.
"""

from __future__ import annotations

import math
import operator
import os
import sys
import threading
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Iterator

import torch
if TYPE_CHECKING:
    from cuda.bindings import runtime as cudart


def _check_cuda(error: cudart.cudaError_t, operation: str) -> None:
    from cuda.bindings import runtime as cudart

    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {error}")


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= int(extent)
    return tuple(reversed(result))


def _tensor_from_pointer(
    pointer: int,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    nbytes: int,
) -> torch.Tensor:
    constructor = getattr(torch._C, "_construct_storage_from_data_pointer", None)
    if constructor is None:
        raise RuntimeError(
            "mapped-host storage requires torch._C._construct_storage_from_data_pointer"
        )
    storage = constructor(int(pointer), device, int(nbytes))
    return torch.empty(0, dtype=dtype, device=device).set_(
        storage, 0, shape, _contiguous_strides(shape)
    )


class MappedHostAllocation:
    """Own a mapped page-locked allocation and its CPU/CUDA tensor aliases."""

    def __init__(
        self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> None:
        from cuda.bindings import runtime as cudart

        self._host_pointer = 0
        self._closed = True
        if device.type != "cuda" or device.index is None:
            raise ValueError(
                f"mapped-host storage requires an indexed CUDA device, got {device}"
            )
        nbytes = math.prod(shape) * dtype.itemsize
        if nbytes <= 0:
            raise ValueError(
                f"mapped-host allocation size must be positive, got {nbytes}"
            )
        self.device = device
        self.nbytes = nbytes
        with torch.cuda.device(device):
            error, pointer = cudart.cudaHostAlloc(
                nbytes, cudart.cudaHostAllocMapped | cudart.cudaHostAllocWriteCombined
            )
            _check_cuda(error, "cudaHostAlloc")
            self._host_pointer = int(pointer)
            self._closed = False
            try:
                error, device_pointer = cudart.cudaHostGetDevicePointer(pointer, 0)
                _check_cuda(error, "cudaHostGetDevicePointer")
                self.host_view = _tensor_from_pointer(
                    self._host_pointer,
                    shape=shape,
                    dtype=dtype,
                    device=torch.device("cpu"),
                    nbytes=nbytes,
                )
                self.device_view = _tensor_from_pointer(
                    int(device_pointer),
                    shape=shape,
                    dtype=dtype,
                    device=device,
                    nbytes=nbytes,
                )
            except Exception:
                cudart.cudaFreeHost(pointer)
                self._host_pointer = 0
                self._closed = True
                raise

    def close(self) -> None:
        from cuda.bindings import runtime as cudart

        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        _check_cuda(cudart.cudaFreeHost(self._host_pointer)[0], "cudaFreeHost")
        self._host_pointer = 0
        self._closed = True

    def __del__(self) -> None:
        if self._closed or self._host_pointer == 0 or sys is None or sys.is_finalizing():
            return
        with suppress(Exception):
            self.close()


class DiskBatchJob:
    """Outcome of one stream-ordered native read over several row caches.

    ``launch_batch_read`` creates it. The read itself is a CUDA host function
    queued on the caches' transaction stream; this handle only reports it.
    Dropping it early is safe: the native job keeps the objects the host
    function touches (reader, ID and row buffers, status) until that function
    has run, and releases them when the handle is dropped afterwards.

    It does not own anything the GPU uses later. Lookups and any publication
    queued after the read consume the caches' mapped rows and the status word
    when the stream reaches them, possibly after ``result()`` returns; keep
    the caches open (``close()`` waits for their cache_done) and the status
    allocation alive until that stream work has completed.
    """

    __slots__ = ("_native", "_handle")

    def __init__(self, native, handle) -> None:
        self._native = native
        self._handle = handle

    def done(self) -> bool:
        """True once the host function has run or its launch failed."""
        return self._native.ple_batch_result(self._handle, 0.0)[0] != "pending"

    def result(self, timeout: float | None = None) -> None:
        """Wait for the host read (GIL released); raise its failure, if any.

        Reports the host function only, not the GPU work queued after it.
        ``timeout`` is in seconds: None or +inf waits indefinitely, zero or
        negative polls, NaN is refused. A host function that CUDA skips
        (context error) never completes, so bounded waits are recommended.
        """
        if timeout is not None:
            timeout = float(timeout)
            if math.isnan(timeout):
                raise ValueError("timeout must not be NaN")
        state, message = self._native.ple_batch_result(
            self._handle, -1.0 if timeout is None else max(timeout, 0.0)
        )
        if state == "pending":
            raise TimeoutError("disk batch read has not completed")
        if message is not None:
            raise RuntimeError(message)


def launch_batch_read(caches, counts, status_host: torch.Tensor) -> DiskBatchJob:
    """Queue one native read of every cache on their shared transaction stream.

    Call inside every cache's ``transaction`` on one thread and one stream,
    after each cache's ``_stage_ids`` for the same count. The read runs as a
    CUDA host function after the queued ID copies and before work queued
    later on that stream; tables are read concurrently on native threads.
    It stores 1 (all tables read) or -1 (any failure) into ``status_host``,
    a caller-owned int64 host word that is never reset by this API. The
    io_uring backend is required; GDS is refused.
    """
    caches = tuple(caches)
    counts = tuple(operator.index(count) for count in counts)
    if not caches or len(caches) != len(counts):
        raise ValueError("one staged count per disk row cache")
    thread = threading.get_ident()
    stream = caches[0]._transaction_stream
    if stream is None:
        raise RuntimeError("batch reads require an active disk row transaction")
    for cache, count in zip(caches, counts):
        cache._require_open()
        if cache._gds is not None or cache._native is None:
            raise NotImplementedError(
                "stream-ordered disk reads require the io_uring backend; GDS is refused"
            )
        if (
            cache._transaction_thread != thread
            or cache._transaction_stream is None
            or cache._transaction_stream.cuda_stream != stream.cuda_stream
        ):
            raise RuntimeError("batch reads require every cache in a transaction on one stream")
        if getattr(cache, "_staged_count", None) != count:
            raise RuntimeError("batch read count differs from the staged ID count")
    if len({id(cache) for cache in caches}) != len(caches):
        raise ValueError("batch reads require distinct disk row caches")
    if (
        not isinstance(status_host, torch.Tensor)
        or status_host.device.type != "cpu"
        or status_host.dtype != torch.int64
        or status_host.numel() != 1
        or not status_host.is_contiguous()
    ):
        raise ValueError("status_host must be one contiguous CPU int64 element")
    native = caches[0]._native
    handle = native.ple_batch(
        tuple(cache._reader for cache in caches),
        tuple(cache._ids_buffer for cache in caches),
        tuple(cache._weight_buffer for cache in caches),
        tuple(cache._scale_buffer for cache in caches),
        counts,
        memoryview(status_host.numpy()).cast("B"),
    )
    native.ple_batch_launch(handle, stream.cuda_stream)
    return DiskBatchJob(native, handle)


class DiskRowCache:
    """Read immutable row planes into a reusable, batch-sized cache.

    Source rows have one weight byte plane and an optional independent scale
    byte plane. Neither file data nor full table allocations are retained.
    Hold ``transaction`` across ID production, ``read_rows`` and GPU decoding.
    Downstream graphs consume the decoder's fixed device output, not disk I/O.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        max_lookups: int,
        table_rows: int,
        shard_start: int,
        shard_end: int,
        shard_rows: int,
        weight_row_bytes: int,
        scale_row_bytes: int = 0,
        queue_depth: int = 64,
    ) -> None:
        from b12x.loader._native import load

        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("disk row staging requires CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.max_lookups = operator.index(max_lookups)
        self.table_rows = operator.index(table_rows)
        self.shard_start = operator.index(shard_start)
        self.shard_end = operator.index(shard_end)
        self.shard_rows = operator.index(shard_rows)
        self.weight_row_bytes = operator.index(weight_row_bytes)
        self.scale_row_bytes = operator.index(scale_row_bytes)
        queue_depth = operator.index(queue_depth)
        if not 0 < self.shard_rows <= (1 << 63) - 1:
            raise ValueError("shard_rows must be a positive signed int64")
        self._backend = os.environ.get("B12X_DISK_BACKEND", "io_uring")
        if self._backend not in ("io_uring", "gds"):
            raise ValueError("B12X_DISK_BACKEND must be io_uring or gds")
        self._gds = None
        self._native = None
        self._reader = None
        if self._backend == "io_uring":
            self._native = load()
            self._reader = self._native.ple_reader(
                self.shard_rows,
                self.table_rows,
                self.shard_start,
                self.shard_end,
                self.weight_row_bytes,
                self.scale_row_bytes,
                self.max_lookups,
                queue_depth,
            )
        self.shard_count = (self.table_rows + self.shard_rows - 1) // self.shard_rows
        self.ids_host = torch.empty(
            (self.max_lookups,), dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._ids_buffer = memoryview(self.ids_host.numpy())
        self._weight_allocation = self._scale_allocation = None
        self.weight_host = self.scale_host = None
        self._weight_buffer = self._scale_buffer = None
        if self._backend == "gds":
            from ._gds import GdsRows
            self._gds = GdsRows(self, queue_depth)
            self.weight, self.scale = self._gds.weight, self._gds.scale
        else:
            self._weight_allocation = MappedHostAllocation(
                (self.max_lookups, self.weight_row_bytes), torch.uint8, device
            )
            self.weight = self._weight_allocation.device_view
            self.weight_host = self._weight_allocation.host_view
            self._weight_buffer = memoryview(self.weight_host.numpy())
            self._scale_allocation = None
            self.scale = None
            self.scale_host = None
            self._scale_buffer = None
            if self.scale_row_bytes:
                self._scale_allocation = MappedHostAllocation(
                    (self.max_lookups, self.scale_row_bytes), torch.uint8, device
                )
                self.scale = self._scale_allocation.device_view
                self.scale_host = self._scale_allocation.host_view
                self._scale_buffer = memoryview(self.scale_host.numpy())
        self._sources: set[tuple[bool, int]] = set()
        self._frozen = False
        self._lock = threading.RLock()
        self._transaction_thread: int | None = None
        self._transaction_stream: torch.cuda.Stream | None = None
        self._ids_ready = torch.cuda.Event()
        self._cache_done = torch.cuda.Event()
        self._cache_used = False
        self._closed = False

    def _require_open(self):
        if self._closed:
            raise RuntimeError("disk row cache is closed")

    def __del__(self):
        if not getattr(self, "_closed", True):
            with suppress(Exception):
                self.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._transaction_thread is not None:
                raise RuntimeError("cannot close a disk row cache during a transaction")
            if self._cache_used:
                with torch.cuda.device(self.device):
                    self._cache_done.synchronize()
            if self._gds is not None:
                self._gds.close()
            for allocation in (self._scale_allocation, self._weight_allocation):
                if allocation is not None:
                    allocation.close()
            self._reader = self._native = None
            self._closed = True

    def add_shard(
        self, shard_index: int, path: str, offset: int, *, scale: bool = False
    ) -> None:
        self._require_open()
        with self._lock:
            if self._frozen:
                raise RuntimeError("cannot change disk shards after binding")
            shard_index = operator.index(shard_index)
            offset = operator.index(offset)
            if not 0 <= shard_index < self.shard_count:
                raise ValueError("checkpoint shard index is out of range")
            if offset < 0:
                raise ValueError("checkpoint file offset must be nonnegative")
            if scale and not self.scale_row_bytes:
                raise ValueError("disk table has no row scale plane")
            key = (scale, shard_index)
            if key in self._sources:
                raise ValueError("checkpoint shard is already registered")
            start = shard_index * self.shard_rows
            end = min(start + self.shard_rows, self.table_rows)
            if end <= self.shard_start or start >= self.shard_end:
                return
            if self._gds is not None:
                self._gds.native.add(self._gds.reader, shard_index, os.fspath(path), offset, scale)
            else:
                self._native.ple_reader_add(self._reader, shard_index, os.fspath(path), offset, scale)
            self._sources.add(key)

    def require_complete(self) -> None:
        self._require_open()
        with self._lock:
            first = self.shard_start // self.shard_rows
            last = (self.shard_end + self.shard_rows - 1) // self.shard_rows
            for shard in range(first, last):
                if (False, shard) not in self._sources:
                    raise ValueError(f"missing disk weight shard {shard}")
                if self.scale_row_bytes and (True, shard) not in self._sources:
                    raise ValueError(f"missing disk scale shard {shard}")

    def freeze(self) -> None:
        with self._lock:
            self.require_complete()
            self._frozen = True

    @contextmanager
    def transaction(self) -> Iterator[DiskRowCache]:
        self._require_open()
        if torch.compiler.is_compiling():
            raise RuntimeError("disk table preparation cannot run under torch.compile")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "disk table preparation must run outside CUDA graph capture"
                )
            with self._lock:
                if self._transaction_thread is not None:
                    raise RuntimeError("disk row transactions cannot be nested")
                stream = torch.cuda.current_stream(self.device)
                if self._cache_used:
                    stream.wait_event(self._cache_done)
                self._transaction_thread = threading.get_ident()
                self._transaction_stream = stream
                try:
                    yield self
                finally:
                    self._cache_done.record(stream)
                    self._cache_used = True
                    self._transaction_stream = None
                    self._transaction_thread = None

    def read_rows(self, ids: torch.Tensor, count: int) -> None:
        self._stage_ids(ids, count)
        self._read_staged(count)

    def _stage_ids(self, ids: torch.Tensor, count: int) -> None:
        if self._transaction_thread != threading.get_ident():
            raise RuntimeError("read_rows requires an active disk row transaction")
        count = operator.index(count)
        if not 0 <= count <= self.max_lookups:
            raise ValueError("disk lookup count exceeds batch capacity")
        if (
            ids.device != self.device
            or ids.dtype != torch.int64
            or not ids.is_contiguous()
        ):
            raise ValueError(
                "disk row IDs must be contiguous CUDA int64 on the cache device"
            )
        if ids.numel() < count:
            raise ValueError("disk row IDs do not cover the requested count")
        self.ids_host[:count].copy_(ids.view(-1)[:count], non_blocking=True)
        self._ids_ready.record(self._transaction_stream)
        self._staged_count = count

    def _read_staged(self, count: int) -> None:
        # Completes ID production and all prior cache readers before host writes.
        with torch.cuda.device(self.device):
            self._ids_ready.synchronize()
            if self._gds is not None:
                self._gds.read(self._ids_buffer, count, self._transaction_stream)
                return
            self._native.ple_reader_run(
                self._reader,
                self._ids_buffer,
                self._weight_buffer,
                self._scale_buffer,
                count,
            )

    def stats(self) -> dict[str, int | float]:
        self._require_open()
        with self._lock:
            result = dict(self._gds.native.stats(self._gds.reader) if self._gds is not None
                          else self._native.ple_reader_stats(self._reader))
            result["ids_host_bytes"] = (
                self.ids_host.numel() * self.ids_host.element_size()
            )
            result["weight_cache_bytes"] = self.weight.numel() * self.weight.element_size()
            result["scale_cache_bytes"] = (
                self.scale.numel() * self.scale.element_size() if self.scale is not None else 0
            )
            result["cache_bytes"] = (
                result["weight_cache_bytes"] + result["scale_cache_bytes"]
            )
            result["owned_staging_bytes"] = (
                result["staging_bytes"]
                + result["ids_host_bytes"]
                + result["cache_bytes"]
            )
            descriptors = result.get("descriptor_bytes", 0)
            result["gds_enabled"] = int(self._gds is not None)
            result["device_staging_bytes"] = (result["staging_bytes"] + result["cache_bytes"] + descriptors) if self._gds else 0
            result["owned_host_bytes"] = result["ids_host_bytes"] + descriptors + result["metadata_bytes"]
            if self._gds is None:
                result["owned_host_bytes"] += result["staging_bytes"] + result["cache_bytes"]
            result["owned_staging_bytes"] += 2 * descriptors
            return result
