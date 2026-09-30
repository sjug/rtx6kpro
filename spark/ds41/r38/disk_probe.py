"""Exercise R38's real io_uring reader on an immutable checkpoint shard."""

from array import array
from pathlib import Path

from contract import MODEL, require


def verify(native, snapshot=MODEL):
    # This is an I/O byte-parity probe, not an Engram numerical test. Small
    # artificial row planes reuse a real file through its read-only HF mount.
    path = Path(snapshot) / "model-00001-of-00048.safetensors"
    require(path.is_file() and path.stat().st_size >= 16384, "Disk probe shard is absent or too small")
    rows = array("q", [0, 7, 15])
    require(rows.itemsize == 8, "Reader IDs require int64")
    weights, scales = bytearray(3 * 256), bytearray(3 * 8)
    reader = native.ple_reader(16, 16, 0, 16, 256, 8, 128, 128)
    try:
        # Nonaligned offsets deliberately exercise the reader's aligned I/O
        # staging, as real safetensors plane offsets need not be 4K aligned.
        native.ple_reader_add(reader, 0, str(path), 37, False)
        native.ple_reader_add(reader, 0, str(path), 8205, True)
        native.ple_reader_run(reader, rows, weights, scales, len(rows))
        with path.open("rb", buffering=0) as reference:
            for width, offset, result in ((256, 37, weights), (8, 8205, scales)):
                expected = bytearray()
                for row in rows:
                    reference.seek(offset + row * width)
                    expected.extend(reference.read(width))
                require(result == expected, "Disk reader byte parity failed")
        stats = dict(native.ple_reader_stats(reader))
    finally:
        # The capsule destructor unregisters buffers/files and closes the ring.
        del reader
    print(f"DS41_DISK_DIRECT_REGISTERED_READ_PASS path={path} stats={stats}", flush=True)


if __name__ == "__main__":
    from b12x.loader._native import load

    verify(load())
