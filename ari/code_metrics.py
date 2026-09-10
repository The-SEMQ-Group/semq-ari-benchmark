"""Coordinate, bit and byte differences between two sets of SEMQ codes.

SEMQ QUANT returns bit-packed bytes, and it packs whether or not the
context asks it to. Several measurements in this repository compared
two code buffers with ``(cur != ref).mean()`` and reported the result
as a symbol or coordinate disagreement rate. It is a **byte**
disagreement rate. At 4 bits per coordinate a single changed
coordinate moves one byte that holds two coordinates, so the reported
rate is twice the coordinate rate; at 2 bits it is four times. The
factor falls back towards one as changes get dense, so it cannot be
divided out of a number that was already published.

Each quantity here carries its own denominator, and the byte rate is
kept under a name that says bytes so earlier results stay reproducible.

SEMQ 1.6 exposes this as ``Context.compare_codes``. This module repeats
the unpacking because the benchmark runs against whichever wheel
CodeArtifact serves, including builds that predate that method. The
layout is pinned against the SDK in
``tests/test_code_metrics.py::test_unpacking_matches_the_sdk``, so a
change to the C packing fails a test here rather than silently shifting
a published rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def bits_per_coordinate(n_bins: int) -> int:
    """Code bits one coordinate occupies at ``n_bins`` magnitude bins.

    The QUANT alphabet holds one bin per magnitude level for each sign,
    so the width is ``ceil(log2(2 * n_bins))``.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    return (2 * n_bins - 1).bit_length()


def unpack_symbols(codes: np.ndarray, n_bins: int, dim: int) -> np.ndarray:
    """Expand packed QUANT codes to one uint8 symbol per coordinate.

    Symbols run contiguously from the least significant bit of byte 0,
    and the final byte is zero-padded when ``dim * bits`` is not a
    multiple of eight.
    """
    arr = np.ascontiguousarray(codes, dtype=np.uint8)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"codes must be 1-D or 2-D, got {arr.ndim}-D")
    width = bits_per_coordinate(n_bins)
    needed = dim * width
    bits = np.unpackbits(arr, axis=1, bitorder="little")
    if bits.shape[1] < needed:
        raise ValueError(
            f"codes hold {bits.shape[1]} bits, need {needed} for dim={dim} "
            f"at {width} bits per coordinate")
    weights = 1 << np.arange(width, dtype=np.uint16)
    grouped = bits[:, :needed].reshape(len(arr), dim, width).astype(np.uint16)
    return (grouped * weights).sum(axis=2).astype(np.uint8)


@dataclass(frozen=True)
class CodeDiff:
    """Per-row differences, each against its own denominator.

    ``changed_coordinates`` holds one ascending index array per row, so a
    change can be located and not only counted.
    """

    n_coordinates: int
    bits_per_coordinate: int
    n_bytes: int
    codes_equal: np.ndarray
    n_coordinates_changed: np.ndarray
    bit_hamming_distance: np.ndarray
    n_bytes_changed: np.ndarray
    changed_coordinates: tuple[np.ndarray, ...]

    @property
    def n_bits(self) -> int:
        """Code bits per row, excluding padding."""
        return self.n_coordinates * self.bits_per_coordinate

    @property
    def coordinate_change_rate(self) -> np.ndarray:
        return self.n_coordinates_changed / self.n_coordinates

    @property
    def bit_hamming_rate(self) -> np.ndarray:
        return self.bit_hamming_distance / self.n_bits

    @property
    def byte_change_rate(self) -> np.ndarray:
        """Differing bytes over ``n_bytes``. Not a coordinate rate.

        Reported only for continuity with results published before the
        distinction was drawn.
        """
        return self.n_bytes_changed / self.n_bytes

    def summary(self) -> dict[str, float | int]:
        """Corpus means, with the unit of each field carried in its name."""
        return {
            "n_coordinates": self.n_coordinates,
            "bits_per_coordinate": self.bits_per_coordinate,
            "n_bits": self.n_bits,
            "n_bytes": self.n_bytes,
            "code_equality_rate": float(self.codes_equal.mean()),
            "coordinate_change_rate": float(self.coordinate_change_rate.mean()),
            "bit_hamming_bits_mean": float(self.bit_hamming_distance.mean()),
            "bit_hamming_rate": float(self.bit_hamming_rate.mean()),
            "byte_change_rate_legacy": float(self.byte_change_rate.mean()),
        }


def code_diff(ref: np.ndarray, cur: np.ndarray, *, n_bins: int, dim: int) -> CodeDiff:
    """Compare two QUANT code buffers of ``dim`` coordinates each."""
    a = np.ascontiguousarray(ref, dtype=np.uint8)
    b = np.ascontiguousarray(cur, dtype=np.uint8)
    if a.shape != b.shape:
        raise ValueError(f"code shapes differ: {a.shape} and {b.shape}")
    if a.ndim == 1:
        a, b = a[None, :], b[None, :]

    sym_xor = np.bitwise_xor(unpack_symbols(a, n_bins, dim),
                             unpack_symbols(b, n_bins, dim))
    differs = sym_xor != 0
    return CodeDiff(
        n_coordinates=dim,
        bits_per_coordinate=bits_per_coordinate(n_bins),
        n_bytes=int(a.shape[1]),
        codes_equal=~differs.any(axis=1),
        n_coordinates_changed=differs.sum(axis=1).astype(np.int64),
        bit_hamming_distance=_POPCOUNT[sym_xor].sum(axis=1).astype(np.int64),
        n_bytes_changed=(a != b).sum(axis=1).astype(np.int64),
        changed_coordinates=tuple(
            np.flatnonzero(row).astype(np.int64) for row in differs),
    )


def bytes_for(n_coordinates: int, n_bins: int) -> int:
    """Serialized bytes one vector of ``n_coordinates`` occupies."""
    bits = n_coordinates * bits_per_coordinate(n_bins)
    return -(-bits // 8)


def chunked_code_diff(ref: np.ndarray, cur: np.ndarray, *, n_bins: int,
                      widths: list[int]) -> CodeDiff:
    """Compare buffers that hold several separately encoded chunks.

    A vector wider than ``SEMQ_MAX_DIM`` is encoded one chunk at a time
    and the codes are concatenated. Every chunk pads its own final byte,
    so the joined buffer must be split back at the chunk boundaries
    before the symbols are read; otherwise one chunk's padding is
    decoded as coordinates of the next.
    """
    a = np.ascontiguousarray(ref, dtype=np.uint8)
    b = np.ascontiguousarray(cur, dtype=np.uint8)
    if a.shape != b.shape:
        raise ValueError(f"code shapes differ: {a.shape} and {b.shape}")
    if a.ndim == 1:
        a, b = a[None, :], b[None, :]

    sizes = [bytes_for(w, n_bins) for w in widths]
    if sum(sizes) != a.shape[1]:
        raise ValueError(
            f"widths imply {sum(sizes)} bytes per row, buffer holds {a.shape[1]}")

    parts, offsets, off, coord_off = [], [], 0, 0
    for width, size in zip(widths, sizes):
        parts.append(code_diff(a[:, off:off + size], b[:, off:off + size],
                               n_bins=n_bins, dim=width))
        offsets.append(coord_off)
        off += size
        coord_off += width
    return CodeDiff(
        n_coordinates=sum(widths),
        bits_per_coordinate=bits_per_coordinate(n_bins),
        n_bytes=int(a.shape[1]),
        codes_equal=np.logical_and.reduce([p.codes_equal for p in parts]),
        n_coordinates_changed=sum(p.n_coordinates_changed for p in parts),
        bit_hamming_distance=sum(p.bit_hamming_distance for p in parts),
        n_bytes_changed=sum(p.n_bytes_changed for p in parts),
        changed_coordinates=tuple(
            np.concatenate([p.changed_coordinates[row] + off
                            for p, off in zip(parts, offsets)]).astype(np.int64)
            for row in range(a.shape[0])),
    )


def chunk_widths(dim: int, max_dim: int) -> list[int]:
    """Coordinate counts of the chunks ``dim`` is encoded in."""
    bounds = list(range(0, dim, max_dim)) + [dim]
    return [hi - lo for lo, hi in zip(bounds, bounds[1:])]
