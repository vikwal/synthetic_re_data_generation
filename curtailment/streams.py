"""Deterministic random streams keyed by name.

Every random quantity of the curtailment model is drawn from its own stream
rng(seed, *keys), e.g. rng(seed, "beta", node). A stream depends only on the
seed and its keys, never on the processing order, the park subset or the
number of workers.
"""

import hashlib

import numpy as np


def _words(key) -> list:
    """Stable 32-bit words of one key (str/int/float/timestamp via str())."""
    digest = hashlib.sha256(str(key).encode()).digest()
    return [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4)]


def seed_sequence(seed: int, *keys) -> np.random.SeedSequence:
    entropy = [int(seed) & 0xFFFFFFFF]
    for k in keys:
        entropy += _words(k)
    return np.random.SeedSequence(entropy)


def rng(seed: int, *keys) -> np.random.Generator:
    return np.random.default_rng(seed_sequence(seed, *keys))


def uniform_at(seed: int, keys: tuple, idx) -> np.ndarray:
    """Counter-based uniforms in [0, 1): one value per integer index, the same
    value for the same (seed, keys, index) whatever other indices are asked
    for (SplitMix64 finaliser over key ^ index)."""
    key = np.uint64(seed_sequence(seed, *keys).generate_state(1, dtype=np.uint64)[0])
    x = np.asarray(idx, dtype=np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        z = (x * np.uint64(0x9E3779B97F4A7C15)) ^ key
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))
