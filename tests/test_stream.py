"""The streaming runner's background reader changes when batches are read, not
which batches the model sees.

`scripts/stream_full.py` is a script rather than a module, so it is loaded by
path.
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "stream_full.py"

spec = importlib.util.spec_from_file_location("stream_full", SCRIPT)
assert spec is not None and spec.loader is not None
stream_full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream_full)


@pytest.fixture
def data(tmp_path: Path) -> tuple[str, str]:

    rng = np.random.default_rng(0)
    path_X, path_Y = tmp_path / "X.npy", tmp_path / "Y.npy"

    np.save(path_X, rng.standard_normal((300, 2, 8)).astype(np.float32))
    np.save(path_Y, rng.integers(0, 3, 300))

    return str(path_X), str(path_Y)


def raw_stream(data: tuple[str, str]):

    # small batches and frequent evictions, so the reader crosses epoch
    # boundaries and cache drops many times over
    indices = np.random.default_rng(1).permutation(300)[:250]

    return stream_full.RawStream(
        *data, indices, batch_size=16, seed=3, drop_cache_every=4
    )


@pytest.mark.parametrize("depth", [1, 2, 8])
def test_prefetch_yields_the_serial_batches(data, depth):

    serial = list(raw_stream(data).epochs(3))
    prefetched = list(stream_full.Prefetch(raw_stream(data).epochs(3), depth=depth))

    assert len(prefetched) == len(serial) == 3 * 16

    for (raw, y, rows), (raw_p, y_p, rows_p) in zip(serial, prefetched):
        np.testing.assert_array_equal(rows_p, rows)
        np.testing.assert_array_equal(raw_p, raw)
        np.testing.assert_array_equal(y_p, y)


def test_prefetch_raises_what_the_reader_raised():

    def batches():
        yield np.zeros(1), np.zeros(1), np.zeros(1)
        raise OSError("disk gone")

    pending = iter(stream_full.Prefetch(batches(), depth=1))

    next(pending)
    with pytest.raises(OSError, match="disk gone"):
        next(pending)


def test_prefetch_reads_at_most_depth_ahead_and_stops_with_the_loop():

    reads = []

    def batches():
        for i in range(100):
            reads.append(i)
            yield np.zeros(1), np.zeros(1), np.zeros(1)

    pending = iter(stream_full.Prefetch(batches(), depth=2))

    next(pending)
    next(pending)
    pending.close()

    # the two batches taken, and the two read ahead of them
    assert reads == [0, 1, 2, 3]


def test_prefetch_reads_while_the_consumer_works():

    def slow_batches():
        for _ in range(8):
            time.sleep(0.05)
            yield np.zeros(1), np.zeros(1), np.zeros(1)

    def consume(batches) -> float:
        began = time.perf_counter()
        for _ in batches:
            time.sleep(0.05)
        return time.perf_counter() - began

    inline = stream_full.Prefetch(slow_batches(), depth=0)
    ahead = stream_full.Prefetch(slow_batches(), depth=1)

    # about 0.8 s one after the other, 0.45 s overlapped
    assert consume(ahead) < 0.75 * consume(inline)

    # either way, busy_s is the time spent reading
    assert inline.busy_s >= 0.4 and ahead.busy_s >= 0.4
