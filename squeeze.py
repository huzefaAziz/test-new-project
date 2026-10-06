"""
squeeze.py -- the closest real thing to an "infinitely fast, infinitely
compressing" algorithm.

What it does
  * Never makes data more than 1 byte bigger (falls back to "stored").
  * Absurd ratios on highly regular data: a block repeated N times is saved
    as (block, N), so 100 MB of zeros becomes a handful of bytes.
  * zlib level 1 for speed by default; zlib 9 / lzma when strong=True.
  * Bails out early when a quick sample shows the data is incompressible.

What it can NOT do (and nothing can)
  * Shrink every input. There are 256**n inputs of n bytes but only
    (256**n - 1) / 255 possible shorter outputs in total, so some inputs
    must fail to shrink.
  * Run in O(1). A lossless compressor has to look at every byte at least once.

Run it:  python squeeze.py
"""
import lzma
import os
import random
import time
import zlib

STORED, ZLIB, LZMA, REPEAT = range(4)  # 1-byte format tag at the start of every blob


def _repeat_block(data: bytes, max_period: int = 4096):
    """Return the shortest block with data == block * k (k >= 2), else None."""
    n = len(data)
    for p in range(1, min(max_period, n // 2) + 1):
        if n % p:
            continue
        block = data[:p]
        # cheap rejections first, the full memcmp-speed check last
        if data[-p:] == block and data[p:2 * p] == block and data[p:] == data[:-p]:
            return block
    return None


def compress(data: bytes, strong: bool = False) -> bytes:
    # 1) Perfectly repetitive data: store the block and a repeat count.
    block = _repeat_block(data)
    if block is not None:
        k = len(data) // len(block)
        kb = k.to_bytes((k.bit_length() + 7) // 8, "big")
        packed = bytes([REPEAT, len(kb)]) + kb + compress(block, strong)
        if len(packed) <= len(data):
            return packed

    stored = bytes([STORED]) + data  # worst case: exactly 1 byte bigger

    # 2) Quick sample test: don't waste time on random-looking data.
    sample = data[:65536]
    if len(sample) >= 4096 and len(zlib.compress(sample, 1)) > 0.97 * len(sample):
        return stored

    # 3) General-purpose compression, keep whichever result is smallest.
    candidates = [stored, bytes([ZLIB]) + zlib.compress(data, 9 if strong else 1)]
    if strong:
        candidates.append(bytes([LZMA]) + lzma.compress(data))
    return min(candidates, key=len)


def decompress(blob: bytes) -> bytes:
    mode, body = blob[0], blob[1:]
    if mode == STORED:
        return body
    if mode == ZLIB:
        return zlib.decompress(body)
    if mode == LZMA:
        return lzma.decompress(body)
    if mode == REPEAT:
        width = body[0]
        count = int.from_bytes(body[1:1 + width], "big")
        return decompress(body[1 + width:]) * count
    raise ValueError("unknown format")


# ----------------------------------------------------------------------------
# Self-test and demo
# ----------------------------------------------------------------------------
def _selftest():
    rng = random.Random(1)
    for _ in range(300):
        n = rng.randrange(0, 300)
        alphabet = rng.choice([1, 2, 4, 256])
        raw = bytes(rng.randrange(alphabet) for _ in range(n))
        if n and rng.random() < 0.3:
            raw = raw[:rng.randrange(1, n + 1)] * rng.randrange(1, 6)
        for strong in (False, True):
            out = compress(raw, strong)
            assert decompress(out) == raw
            assert len(out) <= len(raw) + 1  # never worse than +1 byte


def _timed(fn, *args):
    t = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t


def _demo():
    rng = random.Random(0)
    words = "the quick brown fox jumps over a lazy dog while compression laughs".split()
    word_salad = " ".join(rng.choices(words, k=1_000_000)).encode()
    cases = {
        "100 MB of zeros": bytes(100_000_000),
        "'hello world ' x 1M": b"hello world " * 1_000_000,
        "word salad": word_salad,
        "random bytes (20 MB)": os.urandom(20_000_000),
    }
    print(f"{'input':<22}{'size':>13}{'packed':>13}{'ratio':>14}{'comp':>9}{'decomp':>9}")
    for name, data in cases.items():
        blob, tc = _timed(compress, data)
        back, td = _timed(decompress, blob)
        assert back == data, name
        ratio = len(data) / len(blob)
        shown = f"{ratio:,.0f}x" if ratio >= 100 else f"{ratio:.2f}x"
        print(f"{name:<22}{len(data):>13,}{len(blob):>13,}{shown:>14}{tc:>8.3f}s{td:>8.3f}s")

    print("\nFeeding the output back in, over and over ('infinite compression'?):")
    blob = b"hello world " * 1_000_000
    for i in range(1, 8):
        blob = compress(blob)
        print(f"  pass {i}: {len(blob):>10,} bytes")


if __name__ == "__main__":
    _selftest()
    print("self-test passed\n")
    _demo()
