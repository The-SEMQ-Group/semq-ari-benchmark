"""Coordinate, bit and byte rates over SEMQ codes.

The unpacking here duplicates the SDK's bit layout, so the first test
pins it against the SDK itself. The rest fix the denominators that
`(cur != ref).mean()` got wrong.
"""
import numpy as np
import pytest

from ari.code_metrics import bits_per_coordinate, code_diff, unpack_symbols

semq = pytest.importorskip("semq")

from ari.semq_compat import quant_context  # noqa: E402


def _vectors(n, dim, seed=38):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.99, 0.99, size=(n, dim)).astype(np.float32)


def _encode(X, n_bins, scale=1.0):
    ctx = quant_context(X.shape[1], n_bins=n_bins, scale_max=scale)
    try:
        return np.asarray(ctx.batch_encode(np.ascontiguousarray(X, np.float32)))
    finally:
        ctx.close()


@pytest.mark.parametrize("n_bins,bits", [
    (2, 2), (3, 3), (4, 3), (8, 4), (16, 5), (32, 6), (64, 7),
])
def test_bits_per_coordinate_matches_the_encoded_width(n_bins, bits):
    assert bits_per_coordinate(n_bins) == bits
    codes = _encode(_vectors(1, 4096), n_bins)
    assert codes.shape[1] * 8 / 4096 == pytest.approx(bits)


@pytest.mark.parametrize("n_bins", [2, 8, 16])
@pytest.mark.parametrize("dim", [17, 384, 4096])
def test_unpacking_matches_the_sdk(n_bins, dim):
    """Pins the local bit layout against the C packing."""
    X = _vectors(4, dim)
    ctx = quant_context(dim, n_bins=n_bins, scale_max=1.0)
    try:
        codes = np.asarray(ctx.batch_encode(np.ascontiguousarray(X, np.float32)))
        if not hasattr(ctx, "unpack_codes"):
            pytest.skip("this SEMQ build has no unpack_codes")
        expected = np.asarray(ctx.unpack_codes(codes, dim))
    finally:
        ctx.close()
    np.testing.assert_array_equal(unpack_symbols(codes, n_bins, dim), expected)


@pytest.mark.parametrize("n_bins,factor", [(2, 4.0), (8, 2.0)])
def test_one_changed_coordinate_shows_the_byte_overstatement(n_bins, factor):
    dim = 4096
    a = _vectors(1, dim)
    b = a.copy()
    b[0, 1234] = -b[0, 1234]
    d = code_diff(_encode(a, n_bins), _encode(b, n_bins), n_bins=n_bins, dim=dim)

    assert d.n_coordinates_changed.tolist() == [1]
    assert d.coordinate_change_rate[0] == pytest.approx(1 / dim)
    assert d.byte_change_rate[0] == pytest.approx(factor / dim)
    assert d.codes_equal.tolist() == [False]


def test_identical_codes_agree_on_every_measure():
    dim, n_bins = 384, 8
    codes = _encode(_vectors(5, dim), n_bins)
    d = code_diff(codes, codes.copy(), n_bins=n_bins, dim=dim)
    assert d.codes_equal.all()
    assert d.n_coordinates_changed.sum() == 0
    assert d.bit_hamming_distance.sum() == 0
    assert d.n_bytes_changed.sum() == 0


def test_bit_hamming_excludes_padding():
    """dim=17 at 2 bits leaves 6 padding bits in the last byte."""
    dim, n_bins = 17, 2
    a = _vectors(1, dim)
    b = a.copy()
    b[0, 16] = -b[0, 16]
    d = code_diff(_encode(a, n_bins), _encode(b, n_bins), n_bins=n_bins, dim=dim)
    assert d.n_bytes == 5
    assert d.n_bits == 34
    assert d.bit_hamming_rate[0] == d.bit_hamming_distance[0] / 34


def test_summary_names_the_unit_of_every_rate():
    dim, n_bins = 384, 8
    a = _vectors(3, dim)
    b = a.copy()
    b[1, 5] = -b[1, 5]
    s = code_diff(_encode(a, n_bins), _encode(b, n_bins), n_bins=n_bins, dim=dim).summary()
    assert s["n_bits"] == dim * 4
    assert s["code_equality_rate"] == pytest.approx(2 / 3)
    assert s["byte_change_rate_legacy"] > s["coordinate_change_rate"]


def test_mismatched_shapes_are_rejected():
    codes = _encode(_vectors(2, 384), 8)
    with pytest.raises(ValueError, match="shapes differ"):
        code_diff(codes, codes[:, :-1], n_bins=8, dim=384)


def test_chunked_diff_locates_a_coordinate_in_the_last_chunk():
    """Each chunk pads its own final byte, so the split has to be exact."""
    from ari.code_metrics import chunk_widths, chunked_code_diff

    widths = chunk_widths(1000, 384)
    assert widths == [384, 384, 232]
    n_bins = 8
    X = _vectors(3, 1000)
    Y = X.copy()
    Y[1, 999] = -Y[1, 999]

    def enc(M):
        parts, off = [], 0
        for w in widths:
            parts.append(_encode(np.ascontiguousarray(M[:, off:off + w]), n_bins))
            off += w
        return np.concatenate(parts, axis=1)

    d = chunked_code_diff(enc(X), enc(Y), n_bins=n_bins, widths=widths)
    assert d.n_coordinates == 1000
    assert d.n_coordinates_changed.tolist() == [0, 1, 0]
    assert d.codes_equal.tolist() == [True, False, True]
    assert d.n_bits == 4000


def test_chunked_diff_rejects_widths_that_do_not_fill_the_buffer():
    from ari.code_metrics import chunked_code_diff

    codes = _encode(_vectors(2, 384), 8)
    with pytest.raises(ValueError, match="widths imply"):
        chunked_code_diff(codes, codes.copy(), n_bins=8, widths=[100])


def test_concatenating_chunks_before_comparing_misplaces_the_change():
    """Why chunked_code_diff exists: the naive join reads padding as coordinates.

    Chunk 0 is 17 coordinates at 2 bits, so its final byte carries 6
    padding bits. Reading the joined buffer as one 34-coordinate vector
    decodes those bits as three coordinates and shifts every later index
    by three.
    """
    from ari.code_metrics import chunk_widths, chunked_code_diff

    widths = chunk_widths(34, 17)
    n_bins = 2
    X = _vectors(1, 34)
    Y = X.copy()
    Y[0, 20] = -Y[0, 20]

    def enc(M):
        return np.concatenate(
            [_encode(np.ascontiguousarray(M[:, i * 17:(i + 1) * 17]), n_bins)
             for i in range(2)], axis=1)

    correct = chunked_code_diff(enc(X), enc(Y), n_bins=n_bins, widths=widths)
    naive = code_diff(enc(X), enc(Y), n_bins=n_bins, dim=34)

    assert correct.changed_coordinates[0].tolist() == [20]
    assert naive.changed_coordinates[0].tolist() == [23]
