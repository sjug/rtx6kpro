"""Compose the Candidate B B12X/native patch: stream-ordered native disk reads.

Review artifact only: reads pinned B12X blobs with `git show`, writes patched
copies, one unified diff and a lock into this task directory. Nothing is
installed, built into an image, or run on a GPU; no repository is modified.

Scope (B12X and native only; vLLM integration is the user's):
  * _ple_reader.c: the CPU-only read core of ple_reader_run is factored into
    ple_read_core() (statement for statement, verified by the tests), and a
    batch API runs that core for several tables from one CUDA host function.
  * _storage.c: registers the batch API.
  * disk_table.py: DiskBatchJob handle and launch_batch_read(); _stage_ids
    records the staged count so a batch cannot read a different count.
  * engram/_impl.py, api.py, __init__.py: enqueue_lookups().

Native API (module _b12x_loader_storage, ABI_VERSION unchanged at 1):
  ple_batch(readers, ids, weights, scales, counts, status) -> handle
  ple_batch_launch(handle, stream_handle) -> None
  ple_batch_result(handle, timeout_seconds) -> (state, message | None)
Test builds only (-DB12X_PLE_BATCH_TEST_LAUNCH): ple_batch_test_run_deferred,
ple_batch_test_orphans; stream values select a pthread test launcher.
"""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
B12X, COMMIT = Path.home() / 'git/b12x', 'a7d7d29b'

READER = 'b12x/loader/_ple_reader.c'
STORAGE = 'b12x/loader/_storage.c'
DISK = 'b12x/sequence/_shared/disk_table.py'
IMPL = 'b12x/sequence/engram/_impl.py'
API = 'b12x/sequence/engram/api.py'
INIT = 'b12x/sequence/engram/__init__.py'

# The original critical section of py_ple_reader_run (pinned lines 334-363).
RUN_BLOCK = '''    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&reader->api_mutex);
    struct timespec start, planned_at, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    reader->plan.failure.message[0] = 0;
    reader->plan.lookups = count;
    reader->plan.requested_bytes = reader->plan.read_bytes = reader->plan.read_calls = 0;
    reader->plan.unique_blocks = reader->plan.coalesced_reads = reader->plan.submit_calls = 0;
    reader->plan.planning_seconds = 0;
    if (reader->poisoned)
        snprintf(reader->plan.failure.message, sizeof(reader->plan.failure.message), "PLE io_uring reader is unusable after a submission/completion failure; create a new reader");
    else {
        bool planned = ple_plan(&reader->plan, ids.buf, weights.buf, scales.buf, count);
        clock_gettime(CLOCK_MONOTONIC, &planned_at);
        reader->plan.planning_seconds = (double)(planned_at.tv_sec - start.tv_sec) +
            (double)(planned_at.tv_nsec - start.tv_nsec) * 1e-9;
#ifdef B12X_HAVE_LIBURING
        reader->weights = weights.buf;
        reader->scales = scales.buf;
        if (planned && reader->plan.job_count) ple_uring_run(reader);
#else
        (void)planned;
#endif
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    reader->plan.execution_seconds = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) * 1e-9;
    failure = reader->plan.failure;
    pthread_mutex_unlock(&reader->api_mutex);
    Py_END_ALLOW_THREADS
'''
RUN_BLOCK_NEW = '''    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    ple_read_core(reader, ids.buf, weights.buf, scales.buf, count, &failure);
    Py_END_ALLOW_THREADS
'''
# (old, new, expected count) substitutions turning the block body into the core.
CORE_SUBSTITUTIONS = [
    ('ids.buf, weights.buf, scales.buf', 'ids, weights, scales', 1),
    ('reader->weights = weights.buf;', 'reader->weights = weights;', 1),
    ('reader->scales = scales.buf;', 'reader->scales = scales;', 1),
    ('    failure = reader->plan.failure;\n', '    *failure = reader->plan.failure;\n', 1),
]


def core_function():
    body = RUN_BLOCK.split('    Py_BEGIN_ALLOW_THREADS\n', 1)[1].rsplit('    Py_END_ALLOW_THREADS\n', 1)[0]
    for old, new, count in CORE_SUBSTITUTIONS:
        if body.count(old) != count:
            raise RuntimeError('core substitution anchor changed: ' + old)
        body = body.replace(old, new)
    return ('/* CPU-only read core shared by ple_reader_run and the batch host function.\n'
            ' * Takes only the reader\'s native api_mutex; makes no Python or CUDA call. */\n'
            'static void ple_read_core(ple_reader_t *reader, const char *ids, char *weights,\n'
            '                          char *scales, long long count, failure_t *failure) {\n'
            + body + '}\n\n')


BATCH_C = r'''
/* ---- Stream-ordered batch reads (CUDA host function) ----------------------
 * ple_batch() validates and retains one read per table; ple_batch_launch()
 * enqueues ple_batch_callback with cudaLaunchHostFunc, so the reads run after
 * the stream's earlier work (the IDs copies) and before its later work (the
 * lookups and any publication). The callback runs on a CUDA driver thread. It
 * makes no Python or CUDA call and frees nothing: it runs ple_read_core for
 * every table concurrently on native threads, fences, stores 1 (all tables
 * read) or -1 (any failure) into the caller's int64 status word, and marks
 * the job done.
 *
 * Lifetime: the Python handle owns the job and every Python reference it
 * retains (reader capsules and buffer exports). They are released only on a
 * thread holding the GIL: when the launch fails, or when the handle is
 * dropped after the callback finished. A handle dropped while its callback is
 * pending parks the job on an orphan list with its references intact; later
 * batch calls release finished orphans. A callback CUDA never runs (a context
 * error skips host functions) leaves its orphan parked: a bounded leak, never
 * a use after free.
 *
 * Scope of ownership: the job keeps alive only what the host function
 * touches, and only for the host function. GPU work queued after it on the
 * stream (lookups reading the mapped rows, a publication reading status)
 * runs later and is NOT covered: the caller must keep the row caches and the
 * status allocation alive until that work completes (DiskRowCache.close()
 * waits for cache_done). ple_batch_result reports the host read only. */
#define PLE_BATCH_CAPSULE "b12x.ple_batch"
#define PLE_BATCH_MAX_TABLES 8

enum { PLE_BATCH_PREPARED, PLE_BATCH_LAUNCHED, PLE_BATCH_DONE, PLE_BATCH_LAUNCH_FAILED };

typedef struct {
    ple_reader_t *reader;
    PyObject *capsule; /* strong reference: keeps the reader allocated */
    long long count;
    Py_buffer ids, weights, scales;
    failure_t failure;
} ple_batch_table_t;

typedef struct ple_batch ple_batch_t;
struct ple_batch {
    pthread_mutex_t mutex;
    pthread_cond_t done;
    int state;
    ple_batch_t *next_orphan;
    int64_t *status;
    Py_buffer status_buffer;
    failure_t failure;
    size_t tables;
    ple_batch_table_t table[];
};

static pthread_mutex_t ple_batch_orphan_mutex = PTHREAD_MUTEX_INITIALIZER;
static ple_batch_t *ple_batch_orphans;

static int ple_batch_state(ple_batch_t *job) {
    pthread_mutex_lock(&job->mutex);
    int state = job->state;
    pthread_mutex_unlock(&job->mutex);
    return state;
}

/* GIL held; the callback has finished or can no longer run. */
static void ple_batch_release_refs(ple_batch_t *job) {
    for (size_t i = 0; i < job->tables; i++) {
        ple_batch_table_t *table = &job->table[i];
        if (table->ids.obj) PyBuffer_Release(&table->ids);
        if (table->weights.obj) PyBuffer_Release(&table->weights);
        if (table->scales.obj) PyBuffer_Release(&table->scales);
        Py_CLEAR(table->capsule);
    }
    if (job->status_buffer.obj) PyBuffer_Release(&job->status_buffer);
}

static void ple_batch_free(ple_batch_t *job) {
    ple_batch_release_refs(job);
    pthread_cond_destroy(&job->done);
    pthread_mutex_destroy(&job->mutex);
    free(job);
}

/* GIL held. Release every orphan whose callback has finished. Finished jobs
 * are unlinked under the list mutex but freed after it is released: buffer
 * releases may run arbitrary deallocators that re-enter this API. */
static void ple_batch_sweep(void) {
    ple_batch_t *finished = NULL;
    pthread_mutex_lock(&ple_batch_orphan_mutex);
    ple_batch_t **link = &ple_batch_orphans;
    while (*link) {
        ple_batch_t *job = *link;
        if (ple_batch_state(job) == PLE_BATCH_LAUNCHED) {
            link = &job->next_orphan;
            continue;
        }
        *link = job->next_orphan;
        job->next_orphan = finished;
        finished = job;
    }
    pthread_mutex_unlock(&ple_batch_orphan_mutex);
    while (finished) {
        ple_batch_t *job = finished;
        finished = job->next_orphan;
        ple_batch_free(job);
    }
}

static void ple_batch_delete(PyObject *capsule) {
    ple_batch_t *job = PyCapsule_GetPointer(capsule, PLE_BATCH_CAPSULE);
    if (!job) {
        PyErr_Clear();
        return;
    }
    if (ple_batch_state(job) == PLE_BATCH_LAUNCHED) {
        pthread_mutex_lock(&ple_batch_orphan_mutex);
        job->next_orphan = ple_batch_orphans;
        ple_batch_orphans = job;
        pthread_mutex_unlock(&ple_batch_orphan_mutex);
        return;
    }
    ple_batch_free(job);
}

static void *ple_batch_read_table(void *argument) {
    ple_batch_table_t *table = argument;
    ple_read_core(table->reader, table->ids.buf, table->weights.buf, table->scales.buf,
                  table->count, &table->failure);
    return NULL;
}

/* Order every preceding row store, including write-combined mapped-host
 * stores, before the status store, and the status store before returning. */
static void ple_batch_fence(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("dsb sy" ::: "memory");
#elif defined(__x86_64__)
    __asm__ __volatile__("mfence" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

static void CUDART_CB ple_batch_callback(void *argument) {
    ple_batch_t *job = argument;
    pthread_t threads[PLE_BATCH_MAX_TABLES] = {0};
    bool started[PLE_BATCH_MAX_TABLES] = {false};
    for (size_t i = 1; i < job->tables; i++)
        started[i] = pthread_create(&threads[i], NULL, ple_batch_read_table, &job->table[i]) == 0;
    ple_batch_read_table(&job->table[0]);
    for (size_t i = 1; i < job->tables; i++) {
        if (started[i]) pthread_join(threads[i], NULL);
        else ple_batch_read_table(&job->table[i]); /* no thread: read in turn */
    }
    failure_t failure = {{0}};
    for (size_t i = 0; i < job->tables && !failure.message[0]; i++)
        if (job->table[i].failure.message[0]) failure = job->table[i].failure;
    ple_batch_fence();
    __atomic_store_n(job->status, failure.message[0] ? (int64_t)-1 : (int64_t)1, __ATOMIC_SEQ_CST);
    ple_batch_fence();
    pthread_mutex_lock(&job->mutex);
    job->failure = failure;
    job->state = PLE_BATCH_DONE;
    pthread_cond_broadcast(&job->done);
    pthread_mutex_unlock(&job->mutex);
}

#ifdef B12X_PLE_BATCH_TEST_LAUNCH
/* Test launcher: 1 runs the callback on a detached pthread, 2 fails the launch,
 * 3 defers it until ple_batch_test_run_deferred(), 4 never runs it, 5 fails
 * the launch after 200 ms (a window for concurrent result() waiters). */
static void *ple_batch_deferred;

static void *ple_batch_test_thread(void *argument) {
    ple_batch_callback(argument);
    return NULL;
}

static cudaError_t ple_batch_test_launch(cudaStream_t stream, cudaHostFn_t function, void *argument) {
    (void)function;
    uintptr_t mode = (uintptr_t)stream;
    pthread_t thread;
    if (mode == 1) {
        if (pthread_create(&thread, NULL, ple_batch_test_thread, argument)) return cudaErrorLaunchFailure;
        pthread_detach(thread);
        return cudaSuccess;
    }
    if (mode == 3) {
        ple_batch_deferred = argument;
        return cudaSuccess;
    }
    if (mode == 4) return cudaSuccess;
    if (mode == 5) {
        struct timespec pause = {0, 200000000L};
        nanosleep(&pause, NULL);
    }
    return cudaErrorInvalidResourceHandle;
}

/* Runs the deferred callback on a new pthread and joins it WITHOUT releasing
 * the GIL: it can finish only if the callback never needs the GIL. */
static PyObject *py_ple_batch_test_run_deferred(PyObject *self, PyObject *unused) {
    (void)self;
    (void)unused;
    if (!ple_batch_deferred) return PyErr_Format(PyExc_RuntimeError, "no deferred batch");
    pthread_t thread;
    void *argument = ple_batch_deferred;
    ple_batch_deferred = NULL;
    if (pthread_create(&thread, NULL, ple_batch_test_thread, argument))
        return PyErr_Format(PyExc_RuntimeError, "could not start the deferred callback");
    pthread_join(thread, NULL);
    Py_RETURN_NONE;
}

static PyObject *py_ple_batch_test_orphans(PyObject *self, PyObject *unused) {
    (void)self;
    (void)unused;
    long count = 0;
    pthread_mutex_lock(&ple_batch_orphan_mutex);
    for (ple_batch_t *job = ple_batch_orphans; job; job = job->next_orphan) count++;
    pthread_mutex_unlock(&ple_batch_orphan_mutex);
    return PyLong_FromLong(count);
}
#define PLE_BATCH_LAUNCH ple_batch_test_launch
#else
#define PLE_BATCH_LAUNCH cudaLaunchHostFunc
#endif

/* Same integer-address test as ple_overlap: no relational comparison of
 * pointers into unrelated objects. */
static bool ple_batch_overlaps(const Py_buffer *a, size_t a_bytes, const void *b, size_t b_bytes) {
    uintptr_t x = (uintptr_t)a->buf, y = (uintptr_t)b;
    return a_bytes && b_bytes && (x <= y ? y - x < a_bytes : x - y < b_bytes);
}

static PyObject *py_ple_batch(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *readers, *ids, *weights, *scales, *counts, *status;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O", &PyTuple_Type, &readers, &PyTuple_Type, &ids,
                          &PyTuple_Type, &weights, &PyTuple_Type, &scales, &PyTuple_Type, &counts,
                          &status)) return NULL;
    ple_batch_sweep();
    Py_ssize_t tables = PyTuple_GET_SIZE(readers);
    if (tables < 1 || tables > PLE_BATCH_MAX_TABLES || PyTuple_GET_SIZE(ids) != tables ||
        PyTuple_GET_SIZE(weights) != tables || PyTuple_GET_SIZE(scales) != tables ||
        PyTuple_GET_SIZE(counts) != tables)
        return PyErr_Format(PyExc_ValueError, "PLE batch needs 1 to %d tables with one of each input",
                            PLE_BATCH_MAX_TABLES);
    ple_batch_t *job = calloc(1, sizeof(*job) + (size_t)tables * sizeof(ple_batch_table_t));
    if (!job) return PyErr_NoMemory();
    pthread_condattr_t attributes;
    if (pthread_mutex_init(&job->mutex, NULL)) {
        free(job);
        return PyErr_Format(PyExc_RuntimeError, "PLE batch mutex initialization failed");
    }
    if (pthread_condattr_init(&attributes) || pthread_condattr_setclock(&attributes, CLOCK_MONOTONIC) ||
        pthread_cond_init(&job->done, &attributes)) {
        pthread_mutex_destroy(&job->mutex);
        free(job);
        return PyErr_Format(PyExc_RuntimeError, "PLE batch condition initialization failed");
    }
    pthread_condattr_destroy(&attributes);
    job->tables = (size_t)tables;
    job->state = PLE_BATCH_PREPARED;
    if (PyObject_GetBuffer(status, &job->status_buffer, PyBUF_CONTIG) < 0) goto failed;
    if (job->status_buffer.len != 8 || (uintptr_t)job->status_buffer.buf % 8) {
        PyErr_SetString(PyExc_ValueError, "PLE batch status must be one writable, 8-byte aligned int64");
        goto failed;
    }
    job->status = job->status_buffer.buf;
    for (Py_ssize_t i = 0; i < tables; i++) {
        ple_batch_table_t *table = &job->table[i];
        PyObject *capsule = PyTuple_GET_ITEM(readers, i);
        table->reader = PyCapsule_GetPointer(capsule, PLE_CAPSULE);
        if (!table->reader) goto failed;
        Py_INCREF(capsule);
        table->capsule = capsule;
        for (Py_ssize_t j = 0; j < i; j++)
            if (job->table[j].reader == table->reader) {
                PyErr_SetString(PyExc_ValueError, "PLE batch tables must use distinct readers");
                goto failed;
            }
        table->count = PyLong_AsLongLong(PyTuple_GET_ITEM(counts, i));
        if (table->count == -1 && PyErr_Occurred()) goto failed;
        if (table->count < 0 || (uint64_t)table->count > table->reader->plan.max_lookups) {
            PyErr_SetString(PyExc_ValueError, "PLE lookup count exceeds batch capacity");
            goto failed;
        }
        PyObject *scale_object = PyTuple_GET_ITEM(scales, i);
        if (PyObject_GetBuffer(PyTuple_GET_ITEM(ids, i), &table->ids, PyBUF_CONTIG_RO) < 0) goto failed;
        if (PyObject_GetBuffer(PyTuple_GET_ITEM(weights, i), &table->weights, PyBUF_CONTIG) < 0) goto failed;
        if (table->reader->plan.scale_bytes) {
            if (PyObject_GetBuffer(scale_object, &table->scales, PyBUF_CONTIG) < 0) goto failed;
        } else if (scale_object != Py_None) {
            PyErr_SetString(PyExc_ValueError, "PLE reader has no scale plane; pass None");
            goto failed;
        }
        size_t id_bytes = (size_t)table->count * 8;
        size_t weight_bytes = (size_t)table->count * table->reader->plan.weight_bytes;
        size_t scale_bytes = (size_t)table->count * table->reader->plan.scale_bytes;
        if ((size_t)table->ids.len < id_bytes || (size_t)table->weights.len < weight_bytes ||
            (size_t)table->scales.len < scale_bytes) {
            PyErr_SetString(PyExc_ValueError, "PLE buffers must be C-contiguous and cover count rows (IDs are native signed int64 bytes)");
            goto failed;
        }
        if (ple_overlap(&table->ids, id_bytes, &table->weights, weight_bytes) ||
            ple_overlap(&table->ids, id_bytes, &table->scales, scale_bytes) ||
            ple_overlap(&table->weights, weight_bytes, &table->scales, scale_bytes) ||
            ple_batch_overlaps(&table->ids, id_bytes, job->status, 8) ||
            ple_batch_overlaps(&table->weights, weight_bytes, job->status, 8) ||
            ple_batch_overlaps(&table->scales, scale_bytes, job->status, 8)) {
            PyErr_SetString(PyExc_ValueError, "PLE IDs, destinations and status must not overlap");
            goto failed;
        }
    }
    PyObject *handle = PyCapsule_New(job, PLE_BATCH_CAPSULE, ple_batch_delete);
    if (!handle) goto failed;
    return handle;
failed:
    ple_batch_free(job);
    return NULL;
}

static PyObject *py_ple_batch_launch(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    unsigned long long stream;
    if (!PyArg_ParseTuple(args, "OK", &capsule, &stream)) return NULL;
    ple_batch_t *job = PyCapsule_GetPointer(capsule, PLE_BATCH_CAPSULE);
    if (!job) return NULL;
    pthread_mutex_lock(&job->mutex);
    int state = job->state;
    if (state == PLE_BATCH_PREPARED) job->state = PLE_BATCH_LAUNCHED;
    pthread_mutex_unlock(&job->mutex);
    if (state != PLE_BATCH_PREPARED) return PyErr_Format(PyExc_ValueError, "PLE batch was already launched");
    cudaError_t error;
    Py_BEGIN_ALLOW_THREADS
    error = PLE_BATCH_LAUNCH((cudaStream_t)(uintptr_t)stream, ple_batch_callback, job);
    Py_END_ALLOW_THREADS
    if (error == cudaSuccess) Py_RETURN_NONE;
    pthread_mutex_lock(&job->mutex);
    job->state = PLE_BATCH_LAUNCH_FAILED;
    snprintf(job->failure.message, sizeof(job->failure.message), "cudaLaunchHostFunc: %s",
             cudaGetErrorString(error));
    /* Another thread may already wait in ple_batch_result: the state became
     * LAUNCHED before the GIL was released around the launch. */
    pthread_cond_broadcast(&job->done);
    pthread_mutex_unlock(&job->mutex);
    ple_batch_release_refs(job); /* the callback will never run */
    return PyErr_Format(PyExc_RuntimeError, "PLE batch launch failed: %s", cudaGetErrorString(error));
}

static PyObject *py_ple_batch_result(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    double timeout;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &timeout)) return NULL;
    ple_batch_t *job = PyCapsule_GetPointer(capsule, PLE_BATCH_CAPSULE);
    if (!job) return NULL;
    if (timeout != timeout) return PyErr_Format(PyExc_ValueError, "PLE batch timeout must not be NaN");
    /* Negative, +inf, or too large for a timespec deadline: wait indefinitely. */
    if (timeout > 1e9) timeout = -1.0;
    ple_batch_sweep();
    int state;
    failure_t failure;
    Py_BEGIN_ALLOW_THREADS
    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    if (timeout > 0) {
        double whole = (double)(long long)timeout;
        deadline.tv_sec += (time_t)whole;
        deadline.tv_nsec += (long)((timeout - whole) * 1e9);
        if (deadline.tv_nsec >= 1000000000L) {
            deadline.tv_sec += 1;
            deadline.tv_nsec -= 1000000000L;
        }
    }
    pthread_mutex_lock(&job->mutex);
    while (job->state == PLE_BATCH_LAUNCHED && timeout != 0) {
        if (timeout < 0) pthread_cond_wait(&job->done, &job->mutex);
        else if (pthread_cond_timedwait(&job->done, &job->mutex, &deadline) == ETIMEDOUT) break;
    }
    state = job->state;
    failure = job->failure;
    pthread_mutex_unlock(&job->mutex);
    Py_END_ALLOW_THREADS
    if (state == PLE_BATCH_PREPARED) return PyErr_Format(PyExc_ValueError, "PLE batch was not launched");
    static const char *names[] = {"prepared", "pending", "done", "launch_failed"};
    if (failure.message[0]) return Py_BuildValue("(ss)", names[state], failure.message);
    return Py_BuildValue("(sO)", names[state], Py_None);
}
'''

STORAGE_METHODS_OLD = '''    {"ple_reader_stats", py_ple_reader_stats, METH_O,
     "Last-call counters; staging_bytes and metadata_bytes describe persistent allocations."},
'''
STORAGE_METHODS_NEW = STORAGE_METHODS_OLD + '''    {"ple_batch", py_ple_batch, METH_VARARGS,
     "Retain one stream-ordered read per table: (readers, ids, weights, scales, counts, status)."},
    {"ple_batch_launch", py_ple_batch_launch, METH_VARARGS,
     "Enqueue the batch read as a CUDA host function on (handle, stream)."},
    {"ple_batch_result", py_ple_batch_result, METH_VARARGS,
     "(handle, timeout seconds; <0 waits, 0 polls) -> (state, failure message or None)."},
#ifdef B12X_PLE_BATCH_TEST_LAUNCH
    {"ple_batch_test_run_deferred", py_ple_batch_test_run_deferred, METH_NOARGS, NULL},
    {"ple_batch_test_orphans", py_ple_batch_test_orphans, METH_NOARGS, NULL},
#endif
'''

DISK_STAGE_OLD = '''        self.ids_host[:count].copy_(ids.view(-1)[:count], non_blocking=True)
        self._ids_ready.record(self._transaction_stream)
'''
DISK_STAGE_NEW = DISK_STAGE_OLD + '''        self._staged_count = count
'''
DISK_CLASS_ANCHOR = '''class DiskRowCache:
'''
DISK_BATCH = '''class DiskBatchJob:
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
'''

IMPL_ANCHOR = ''' return tuple(b.out for b in bindings)
'''
IMPL_NEW = IMPL_ANCHOR + '''def enqueue_lookups(bindings,token_counts,*,status_host,clear_tail=False):
 """Queue disk lookups on the current stream with no host wait.

 In stream order: each table's ID copy, ONE native host function that reads
 every table concurrently and stores 1 or -1 in ``status_host``, then the
 unchanged lookup kernels. Work queued later on the same stream sees the
 rows and the status. Every binding must be disk-backed (io_uring only).
 Returns the DiskBatchJob that reports the native read only; the queued
 lookups (and any work the caller queues after them) read the caches' mapped
 rows and ``status_host`` later, so the caller keeps both alive until that
 stream work completes.
 """
 from contextlib import ExitStack
 from b12x.sequence._shared.disk_table import launch_batch_read
 from ._kernels import lookup_op
 bindings=tuple(bindings); counts=tuple(operator.index(n) for n in token_counts)
 if not bindings or len(bindings)!=len(counts): raise ValueError("one token count per lookup binding")
 for b,n in zip(bindings,counts):
  if not isinstance(b,LookupBinding): raise TypeError("lookup run requires an Engram lookup binding")
  if b.disk_table is None: raise ValueError("enqueue_lookups requires disk-backed lookup bindings")
  if not 0<=n<=b.hash_ids.shape[0]: raise ValueError("token count exceeds capacity")
 with ExitStack() as stack:
  caches=[]
  for b,n in zip(bindings,counts):
   b.disk_table._require_open()
   cache=stack.enter_context(b.disk_table._cache.transaction())
   cache._stage_ids(b.hash_ids,n*24)
   caches.append(cache)
  job=launch_batch_read(caches,[n*24 for n in counts],status_host)
  for b,n in zip(bindings,counts):
   lookup_op(b.plan.handle,b.weight,b.scale_bytes,b.hash_ids,b.num_tokens,b.out,n,clear_tail)
 return job
'''

EDITS = {
    READER: [
        (RUN_BLOCK, RUN_BLOCK_NEW),
        ('static PyObject *py_ple_reader_run(PyObject *self, PyObject *args) {\n',
         None),  # core inserted before; filled in compose()
        ('static PyObject *py_ple_reader_stats(PyObject *self, PyObject *capsule) {\n',
         BATCH_C.lstrip('\n') + '\n' + 'static PyObject *py_ple_reader_stats(PyObject *self, PyObject *capsule) {\n'),
    ],
    STORAGE: [(STORAGE_METHODS_OLD, STORAGE_METHODS_NEW)],
    DISK: [(DISK_STAGE_OLD, DISK_STAGE_NEW), (DISK_CLASS_ANCHOR, DISK_BATCH)],
    IMPL: [(IMPL_ANCHOR, IMPL_NEW)],
    API: [('run, run_lookup, run_lookups\n', 'run, run_lookup, run_lookups, enqueue_lookups\n'),
          ('"run_lookups", "is_supported"]', '"run_lookups", "enqueue_lookups", "is_supported"]')],
    INIT: [('"run_lookups","is_supported")', '"run_lookups","enqueue_lookups","is_supported")')],
}
PACKAGED = ['claude_prepare_ple_batch.py', 'claude_test_ple_batch_native.py', 'claude_test_ple_batch_python.py']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def output_name(path):
    return 'claude-ple-batch-' + Path(path).name


def pinned(path):
    return subprocess.check_output(['git', '-C', str(B12X), 'show', f'{COMMIT}:{path}']).decode()


def compose():
    lock_files = json.loads((root / 'runtime.lock.json').read_text())['sources']['b12x']['files']
    composed = []
    for path, edits in EDITS.items():
        old = pinned(path)
        if lock_files[path]['sha256'] != sha(old.encode()):
            raise RuntimeError('Pinned blob differs from runtime.lock.json: ' + path)
        new = old
        for anchor, replacement in edits:
            if replacement is None:
                replacement = core_function() + anchor
            if new.count(anchor) != 1:
                raise RuntimeError(f'{path}: anchor count {new.count(anchor)}: {anchor[:60]!r}')
            new = new.replace(anchor, replacement)
        if path.endswith('.py'):
            compile(new, path, 'exec')
        composed.append((path, old, new))
    return composed


def main():
    composed = compose()
    patch = []
    for path, old, new in composed:
        (root / output_name(path)).write_text(new)
        patch.append(''.join(difflib.unified_diff(
            old.splitlines(True), new.splitlines(True), fromfile='a/' + path, tofile='b/' + path)))
    (root / 'claude-ple-batch.patch').write_text(''.join(patch))
    lock = {
        'kind': 'b12x-ple-batch-host-function-proposal',
        'status': 'proposed-for-review-not-built-not-applied',
        'b12x_commit': COMMIT,
        'targets': {path: {'input_sha256': sha(old.encode()), 'output_sha256': sha(new.encode()),
                           'source': output_name(path)} for path, old, new in composed},
        'inputs': {name: sha((root / name).read_bytes())
                   for name in PACKAGED + ['claude-ple-batch.patch'] + [output_name(p) for p, _, _ in composed]},
    }
    (root / 'claude-ple-batch.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    print('PLE-BATCH-PREPARED', sha((root / 'claude-ple-batch.lock.json').read_bytes()))


if __name__ == '__main__':
    main()
