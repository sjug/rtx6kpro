/* Diagnostic CPU-only CUDA host callback. No CUDA or Python calls. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdint.h>
#include <time.h>

struct callback_state {
    uint32_t delay_ms;
    uint32_t completed;
    unsigned char *destination;
    uint64_t bytes;
};

void progress_host_callback(void *opaque) {
    struct callback_state *state = opaque;
    struct timespec delay = {state->delay_ms / 1000,
                            (state->delay_ms % 1000) * 1000000L};
    while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {}
    for (uint64_t i = 0; i < state->bytes; ++i)
        state->destination[i] = (unsigned char)(i & 255);
    __atomic_store_n(&state->completed, 1, __ATOMIC_RELEASE);
}
