/* Included by _storage.c. File payloads are never mapped or retained between runs. */
#include <fcntl.h>
#include <time.h>
#ifdef B12X_HAVE_LIBURING
#include <liburing.h>
#endif

#include "_row_plan.h"
#define PLE_CAPSULE "b12x.ple_reader"

typedef struct ple_reader ple_reader_t;
typedef struct {
    char *buffer;
    size_t job;
} ple_slot_t;

struct ple_reader {
    ple_plan_t plan;
    bool poisoned;
    pthread_mutex_t api_mutex;
    unsigned slots_count;
    ple_slot_t *slots;
    char *buffers;
    char *weights, *scales;
#ifdef B12X_HAVE_LIBURING
    struct io_uring ring;
    bool ring_ready;
    bool files_registered;
    int *registered_fds;
    struct iovec *iovecs;
    unsigned *free_slots;
#endif
};

#ifdef B12X_HAVE_LIBURING
static void ple_error(ple_reader_t *reader, const char *operation, int error) {
    if (!reader->plan.failure.message[0])
        snprintf(reader->plan.failure.message, sizeof(reader->plan.failure.message), "%s: %s",
                 operation, strerror(error));
}

static bool ple_scatter(ple_reader_t *reader, size_t index, char *buffer, int result) {
    ple_job_t *job = &reader->plan.jobs[index];
    reader->plan.read_calls++;
    if (result > 0) reader->plan.read_bytes += (unsigned)result;
    if (result < 0) {
        ple_error(reader, "io_uring READ_FIXED", -result);
        return false;
    }
    if ((unsigned)result != job->expected) {
        if (!reader->plan.failure.message[0])
            snprintf(reader->plan.failure.message, sizeof(reader->plan.failure.message),
                     "short PLE read at offset %lld: expected %u bytes, received %d; source changed or was truncated",
                     (long long)job->offset, job->expected, result);
        return false;
    }
    for (size_t i = job->begin; i < job->end; i++) {
        ple_fragment_t *fragment = &reader->plan.fragments[i];
        memcpy((fragment->scale ? reader->scales : reader->weights) + fragment->destination, buffer + (fragment->offset - job->offset),
               fragment->length);
    }
    return true;
}

static void ple_release(ple_reader_t *reader) {
    if (!reader) return;
    if (reader->ring_ready) io_uring_queue_exit(&reader->ring);
    free(reader->iovecs);
    free(reader->free_slots);
    free(reader->registered_fds);
    for (size_t i = 0; i < reader->plan.file_count; i++) close(reader->plan.files[i].fd);
    free(reader->plan.files);
    free(reader->plan.sources);
    free(reader->buffers);
    free(reader->slots);
    free(reader->plan.fragments);
    free(reader->plan.jobs);
    pthread_mutex_destroy(&reader->api_mutex);
    free(reader);
}

static void ple_delete(PyObject *capsule) {
    ple_reader_t *reader = PyCapsule_GetPointer(capsule, PLE_CAPSULE);
    if (reader) {
        Py_BEGIN_ALLOW_THREADS
        ple_release(reader);
        Py_END_ALLOW_THREADS
    }
}
#endif

static PyObject *py_ple_reader(PyObject *self, PyObject *args) {
    (void)self;
    long long shard_rows, padded_rows, tp_start, tp_end, weight_bytes, scale_bytes, max_lookups;
    int depth;
    if (!PyArg_ParseTuple(args, "LLLLLLLi", &shard_rows, &padded_rows,
                          &tp_start, &tp_end, &weight_bytes, &scale_bytes, &max_lookups,
                          &depth)) return NULL;
#ifndef B12X_HAVE_LIBURING
    return PyErr_Format(PyExc_RuntimeError,
                        "io_uring PLE support is unavailable: install liburing development headers and pkg-config, then rebuild the b12x native loader");
#else
    if (shard_rows <= 0 || padded_rows <= 0 || tp_start < 0 || tp_end < tp_start ||
        tp_end > padded_rows || weight_bytes <= 0 || scale_bytes < 0 || max_lookups <= 0 ||
        depth <= 0 || depth > 32768 ||
        weight_bytes > PY_SSIZE_T_MAX / max_lookups ||
        scale_bytes > PY_SSIZE_T_MAX / max_lookups || max_lookups > PY_SSIZE_T_MAX / 8)
        return PyErr_Format(PyExc_ValueError, "invalid PLE reader geometry or I/O capacity");
    /* At most ceil((row_bytes + block - 1) / block) fragments per plane. */
    size_t per_row = (size_t)weight_bytes / PLE_BLOCK + 2;
    if (scale_bytes) per_row += (size_t)scale_bytes / PLE_BLOCK + 2;
    if ((size_t)max_lookups > SIZE_MAX / per_row ||
        (size_t)max_lookups * per_row > SIZE_MAX / sizeof(ple_fragment_t) ||
        (size_t)max_lookups * per_row > SIZE_MAX / sizeof(ple_job_t))
        return PyErr_Format(PyExc_OverflowError, "PLE batch metadata is too large");
    ple_reader_t *reader = calloc(1, sizeof(*reader));
    if (!reader) return PyErr_NoMemory();
    /* Only an initialized API mutex reaches ple_release. */
    int error = pthread_mutex_init(&reader->api_mutex, NULL);
    if (error) { free(reader); return PyErr_Format(PyExc_RuntimeError, "PLE mutex: %s", strerror(error)); }
    reader->plan.shard_rows = shard_rows;
    reader->plan.padded_rows = padded_rows;
    reader->plan.tp_start = tp_start;
    reader->plan.tp_end = tp_end;
    reader->plan.weight_bytes = weight_bytes;
    reader->plan.scale_bytes = scale_bytes;
    reader->plan.max_lookups = max_lookups;
    reader->plan.capacity = (size_t)max_lookups * per_row;
    reader->slots_count = (unsigned)depth;
    reader->slots = calloc(reader->slots_count, sizeof(*reader->slots));
    reader->plan.fragments = calloc(reader->plan.capacity, sizeof(*reader->plan.fragments));
    reader->plan.jobs = calloc(reader->plan.capacity, sizeof(*reader->plan.jobs));
    error = posix_memalign((void **)&reader->buffers, PLE_BLOCK,
                          (size_t)reader->slots_count * PLE_READ_MAX);
    if (!reader->slots || !reader->plan.fragments || !reader->plan.jobs || error) {
        ple_release(reader);
        return PyErr_NoMemory();
    }
    for (unsigned i = 0; i < reader->slots_count; i++) {
        reader->slots[i].buffer = reader->buffers + (size_t)i * PLE_READ_MAX;
    }
    error = io_uring_queue_init(reader->slots_count, &reader->ring, 0);
    if (error < 0) {
        ple_release(reader);
        return PyErr_Format(PyExc_RuntimeError,
                            "io_uring initialization failed: %s; enable io_uring in the kernel/container policy",
                            strerror(-error));
    }
    reader->ring_ready = true;
    reader->iovecs = calloc(reader->slots_count, sizeof(*reader->iovecs));
    reader->free_slots = calloc(reader->slots_count, sizeof(*reader->free_slots));
    if (!reader->iovecs || !reader->free_slots) { ple_release(reader); return PyErr_NoMemory(); }
    for (unsigned i = 0; i < reader->slots_count; i++) {
        reader->iovecs[i].iov_base = reader->slots[i].buffer;
        reader->iovecs[i].iov_len = PLE_READ_MAX;
    }
    error = io_uring_register_buffers(&reader->ring, reader->iovecs, reader->slots_count);
    if (error < 0) {
        ple_release(reader);
        return PyErr_Format(PyExc_RuntimeError,
                            "io_uring buffer registration failed: %s; raise the locked-memory limit or reduce queue_depth",
                            strerror(-error));
    }
    PyObject *capsule = PyCapsule_New(reader, PLE_CAPSULE, ple_delete);
    if (!capsule) ple_release(reader);
    return capsule;
#endif
}

static int ple_register_files(ple_plan_t *plan, void *owner) {
    (void)plan;
    ple_reader_t *reader = owner;
#ifdef B12X_HAVE_LIBURING
    int *fds = realloc(reader->registered_fds, reader->plan.file_count * sizeof(*fds));
    if (!fds) return -ENOMEM;
    reader->registered_fds = fds;
    for (size_t i = 0; i < reader->plan.file_count; i++) fds[i] = reader->plan.files[i].fd;
    if (reader->files_registered) {
        int error = io_uring_unregister_files(&reader->ring);
        if (error < 0) { reader->poisoned = true; return error; }
        reader->files_registered = false;
    }
    int error = io_uring_register_files(&reader->ring, fds, (unsigned)reader->plan.file_count);
    if (!error) reader->files_registered = true;
    else reader->poisoned = true;
    return error;
#else
    (void)reader;
    return 0;
#endif
}

static PyObject *py_ple_reader_add(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    const char *path;
    long long shard, offset;
    int scale;
    if (!PyArg_ParseTuple(args, "OLsLp", &capsule, &shard, &path, &offset, &scale)) return NULL;
    ple_reader_t *reader = PyCapsule_GetPointer(capsule, PLE_CAPSULE);
    if (!reader) return NULL;
    failure_t failure = {{0}};
    int error;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&reader->api_mutex);
    reader->plan.failure.message[0] = 0;
    error = ple_add_source(&reader->plan, shard, path, offset, scale != 0,
                           ple_register_files, reader);
    failure = reader->plan.failure;
    pthread_mutex_unlock(&reader->api_mutex);
    Py_END_ALLOW_THREADS
    if (error == EINVAL) return PyErr_Format(PyExc_ValueError, "invalid PLE source shard, offset, or scale plane");
    if (error == EOVERFLOW) return PyErr_Format(PyExc_OverflowError, "PLE source byte range overflows int64");
    if (error) return PyErr_Format(PyExc_RuntimeError, "%s",
                                   failure.message[0] ? failure.message : "invalid PLE source geometry or byte range");
    Py_RETURN_NONE;
}

#ifdef B12X_HAVE_LIBURING
/* Prepare a whole free-slot wave before entering the kernel. A short positive
 * submit accounts for exactly that many SQEs; retry only the remaining entries.
 * A failed submit may leave pending SQEs, so stop issuing, drain every submitted
 * CQE, and retire the ring without submitting those pending entries. */
static void ple_submit_wave(ple_reader_t *reader, unsigned free_count,
                            size_t *next, unsigned *outstanding) {
    size_t remaining = reader->plan.job_count - *next;
    unsigned count = remaining < free_count ? (unsigned)remaining : free_count;
    for (unsigned i = 0; i < count; i++) {
        unsigned slot = reader->free_slots[i];
        struct io_uring_sqe *sqe = io_uring_get_sqe(&reader->ring);
        if (!sqe) {
            ple_error(reader, "io_uring SQ capacity", ENOSPC);
            reader->poisoned = true;
            return;
        }
        ple_job_t *job = &reader->plan.jobs[*next];
        reader->slots[slot].job = (*next)++;
        io_uring_prep_read_fixed(sqe, (int)job->file,
                                reader->slots[slot].buffer, job->length,
                                job->offset, (int)slot);
        sqe->flags |= IOSQE_FIXED_FILE;
        io_uring_sqe_set_data64(sqe, slot);
    }
    unsigned pending = count;
    while (pending) {
        reader->plan.submit_calls++;
        int result = io_uring_submit(&reader->ring);
        if (result == -EINTR) continue;
        if (result <= 0) {
            ple_error(reader, "io_uring submit", result < 0 ? -result : EIO);
            reader->poisoned = true;
            return;
        }
        *outstanding += (unsigned)result;
        pending -= (unsigned)result;
    }
}

static void ple_uring_run(ple_reader_t *reader) {
    unsigned outstanding = 0;
    size_t next = 0;
    for (unsigned i = 0; i < reader->slots_count; i++) reader->free_slots[i] = i;
    ple_submit_wave(reader, reader->slots_count, &next, &outstanding);
    while (outstanding) {
        struct io_uring_cqe *cqe = NULL;
        /* wait/peek do not submit pending SQEs when draining a failed submit. */
        int result = io_uring_wait_cqe(&reader->ring, &cqe);
        if (result < 0) {
            if (result != -EINTR && result != -EAGAIN) {
                ple_error(reader, "io_uring completion wait", -result);
                reader->poisoned = true;
            }
            continue;
        }
        unsigned free_count = 0, head;
        io_uring_for_each_cqe(&reader->ring, head, cqe) {
            unsigned slot = (unsigned)io_uring_cqe_get_data64(cqe);
            int status = cqe->res;
            outstanding--;
            ple_scatter(reader, reader->slots[slot].job, reader->slots[slot].buffer, status);
            reader->free_slots[free_count++] = slot;
            if (!outstanding) break;
        }
        io_uring_cq_advance(&reader->ring, free_count);
        /* Do not prepare refill SQEs until every collected completion has been
         * checked. An I/O error therefore leaves no speculative refill entries. */
        if (!reader->plan.failure.message[0] && next < reader->plan.job_count)
            ple_submit_wave(reader, free_count, &next, &outstanding);
    }
    if (reader->poisoned) {
        io_uring_queue_exit(&reader->ring);
        reader->ring_ready = false;
    }
}
#endif

static bool ple_overlap(const Py_buffer *a, size_t a_bytes, const Py_buffer *b, size_t b_bytes) {
    uintptr_t x = (uintptr_t)a->buf, y = (uintptr_t)b->buf;
    return a_bytes && b_bytes && (x <= y ? y - x < a_bytes : x - y < b_bytes);
}

/* CPU-only read core shared by ple_reader_run and the batch host function.
 * Takes only the reader's native api_mutex; makes no Python or CUDA call. */
static void ple_read_core(ple_reader_t *reader, const char *ids, char *weights,
                          char *scales, long long count, failure_t *failure) {
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
        bool planned = ple_plan(&reader->plan, ids, weights, scales, count);
        clock_gettime(CLOCK_MONOTONIC, &planned_at);
        reader->plan.planning_seconds = (double)(planned_at.tv_sec - start.tv_sec) +
            (double)(planned_at.tv_nsec - start.tv_nsec) * 1e-9;
#ifdef B12X_HAVE_LIBURING
        reader->weights = weights;
        reader->scales = scales;
        if (planned && reader->plan.job_count) ple_uring_run(reader);
#else
        (void)planned;
#endif
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    reader->plan.execution_seconds = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) * 1e-9;
    *failure = reader->plan.failure;
    pthread_mutex_unlock(&reader->api_mutex);
}

static PyObject *py_ple_reader_run(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule, *ids_object, *weights_object, *scales_object;
    long long count;
    if (!PyArg_ParseTuple(args, "OOOOL", &capsule, &ids_object, &weights_object, &scales_object, &count)) return NULL;
    ple_reader_t *reader = PyCapsule_GetPointer(capsule, PLE_CAPSULE);
    if (!reader) return NULL;
    if (count < 0 || (uint64_t)count > reader->plan.max_lookups)
        return PyErr_Format(PyExc_ValueError, "PLE lookup count exceeds batch capacity");
    Py_buffer ids = {0}, weights = {0}, scales = {0};
    if (PyObject_GetBuffer(ids_object, &ids, PyBUF_CONTIG_RO) < 0) return NULL;
    if (PyObject_GetBuffer(weights_object, &weights, PyBUF_CONTIG) < 0) goto failed;
    if (reader->plan.scale_bytes && PyObject_GetBuffer(scales_object, &scales, PyBUF_CONTIG) < 0) goto failed;
    if (!reader->plan.scale_bytes && scales_object != Py_None) {
        PyErr_SetString(PyExc_ValueError, "PLE reader has no scale plane; pass None");
        goto failed;
    }
    size_t id_bytes = (size_t)count * 8;
    size_t weight_bytes = (size_t)count * reader->plan.weight_bytes;
    size_t scale_bytes = (size_t)count * reader->plan.scale_bytes;
    if ((size_t)ids.len < id_bytes || (size_t)weights.len < weight_bytes || (size_t)scales.len < scale_bytes ||
        !PyBuffer_IsContiguous(&ids, 'C') || !PyBuffer_IsContiguous(&weights, 'C') ||
        (reader->plan.scale_bytes && !PyBuffer_IsContiguous(&scales, 'C'))) {
        PyErr_SetString(PyExc_ValueError, "PLE buffers must be C-contiguous and cover count rows (IDs are native signed int64 bytes)");
        goto failed;
    }
    if (ple_overlap(&ids, id_bytes, &weights, weight_bytes) ||
        ple_overlap(&ids, id_bytes, &scales, scale_bytes) ||
        ple_overlap(&weights, weight_bytes, &scales, scale_bytes)) {
        PyErr_SetString(PyExc_ValueError, "PLE IDs and destination byte ranges must not overlap");
        goto failed;
    }
    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    ple_read_core(reader, ids.buf, weights.buf, scales.buf, count, &failure);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&ids);
    PyBuffer_Release(&weights);
    if (scales.obj) PyBuffer_Release(&scales);
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    Py_RETURN_NONE;
failed:
    PyBuffer_Release(&ids);
    if (weights.obj) PyBuffer_Release(&weights);
    if (scales.obj) PyBuffer_Release(&scales);
    return NULL;
}

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

static PyObject *py_ple_reader_stats(PyObject *self, PyObject *capsule) {
    (void)self;
    ple_reader_t *reader = PyCapsule_GetPointer(capsule, PLE_CAPSULE);
    if (!reader) return NULL;
    uint64_t lookups, requested, bytes, calls, blocks, coalesced, submits, staging, metadata;
    double seconds, planning;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&reader->api_mutex);
    lookups = reader->plan.lookups;
    requested = reader->plan.requested_bytes;
    bytes = reader->plan.read_bytes;
    calls = reader->plan.read_calls;
    blocks = reader->plan.unique_blocks;
    coalesced = reader->plan.coalesced_reads;
    submits = reader->plan.submit_calls;
    seconds = reader->plan.execution_seconds;
    planning = reader->plan.planning_seconds;
    staging = (uint64_t)reader->slots_count * PLE_READ_MAX;
    metadata = sizeof(*reader) + reader->plan.capacity * (sizeof(*reader->plan.fragments) + sizeof(*reader->plan.jobs)) +
        reader->slots_count * sizeof(*reader->slots) + reader->plan.source_count * sizeof(*reader->plan.sources) +
        reader->plan.file_count * sizeof(*reader->plan.files);
#ifdef B12X_HAVE_LIBURING
    metadata += reader->slots_count * (sizeof(*reader->iovecs) + sizeof(*reader->free_slots));
    metadata += reader->plan.file_count * sizeof(*reader->registered_fds);
    if (reader->ring_ready) {
        metadata += reader->ring.sq.ring_sz;
        if (reader->ring.cq.ring_ptr != reader->ring.sq.ring_ptr) metadata += reader->ring.cq.ring_sz;
        metadata += reader->ring.sq.ring_entries * sizeof(struct io_uring_sqe);
    }
#endif
    pthread_mutex_unlock(&reader->api_mutex);
    Py_END_ALLOW_THREADS
    return Py_BuildValue("{s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:d,s:d}",
        "lookups", (unsigned long long)lookups, "requested_bytes", (unsigned long long)requested,
        "read_bytes", (unsigned long long)bytes, "read_calls", (unsigned long long)calls,
        "unique_blocks", (unsigned long long)blocks, "coalesced_reads", (unsigned long long)coalesced,
        "submit_calls", (unsigned long long)submits,
        "staging_bytes", (unsigned long long)staging, "metadata_bytes", (unsigned long long)metadata,
        "execution_seconds", seconds, "planning_seconds", planning);
}
