"""
Scrambled Sobol points and the two quantile functions the quasi-random
posterior proposal needs, in plain NumPy.

WHY? SciPy provides all of this (scipy.stats.qmc.Sobol, norm.ppf,
chi2.ppf), but importing scipy.stats costs hundreds of milliseconds per
process. The proposal needs 5 dimensions, so the generator is small.

SciPy is BSD-3-Clause licensed; see SCIPY_LICENSE.txt

Reference:
S. Joe and F. Y. Kuo, "Constructing Sobol sequences with better two-dimensional
projections", SIAM J. Sci. Comput. 30, 2635-2654 (2008), file new-joe-kuo-6.21201.

Scrambling: a random linear matrix scramble (lower-triangular, unit
diagonal, over GF(2)) of each dimension's direction numbers, followed by a
random digital shift. That is the same family of scramble SciPy uses; the
bit streams are not identical to SciPy's, but the points are a valid
randomised Sobol net with the same equidistribution properties.

"""
import numpy as np

NBITS = 30

# (s, a, m_1..m_s) for dimensions 2..8 of new-joe-kuo-6.21201.
# Dimension 1 is the van der Corput sequence (all m_k = 1).
_JOE_KUO = (
    (1, 0, (1,)),
    (2, 1, (1, 3)),
    (3, 1, (1, 3, 1)),
    (3, 2, (1, 1, 1)),
    (4, 1, (1, 1, 3, 3)),
    (4, 4, (1, 3, 5, 13)),
    (5, 2, (1, 1, 5, 5, 17)),
)
MAX_DIM = 1 + len(_JOE_KUO)


def _direction_numbers(dim):
    """(dim, NBITS) uint64 direction numbers V[j, k] = m_k << (NBITS - 1 - k)."""
    V = np.zeros((dim, NBITS), dtype=np.uint64)
    for k in range(NBITS):
        V[0, k] = np.uint64(1) << np.uint64(NBITS - 1 - k)
    for j in range(1, dim):
        s, a, m_init = _JOE_KUO[j - 1]
        m = list(m_init)
        for k in range(s, NBITS):
            new = m[k - s] ^ (m[k - s] << s)
            for i in range(1, s):
                if (a >> (s - 1 - i)) & 1:
                    new ^= m[k - i] << i
            m.append(new)
        for k in range(NBITS):
            V[j, k] = np.uint64(m[k]) << np.uint64(NBITS - 1 - k)
    return V


def _scramble(V, rng):
    """Linear matrix scramble of each dimension's direction numbers."""
    dim = V.shape[0]
    out = np.zeros_like(V)
    bits = np.arange(NBITS - 1, -1, -1, dtype=np.uint64)       # MSB first
    for j in range(dim):
        # lower-triangular binary matrix with ones on the diagonal
        L = np.tril(rng.integers(0, 2, size=(NBITS, NBITS), dtype=np.uint8), -1)
        np.fill_diagonal(L, 1)
        # columns of V[j] as bit vectors (MSB first): (NBITS, NBITS)
        B = ((V[j][None, :] >> bits[:, None]) & np.uint64(1)).astype(np.uint8)
        C = (L.astype(np.int64) @ B.astype(np.int64)) % 2
        out[j] = (C.astype(np.uint64) << bits[:, None]).sum(axis=0).astype(np.uint64)
    return out


def sobol(n, dim, seed=None, scramble=True):
    """
    n points of a dim-dimensional Sobol sequence in [0, 1)^dim.

    Balance properties hold for n a power of two; callers should round up.
    Unscrambled, the first point is the origin, as in SciPy.
    """
    if not 1 <= dim <= MAX_DIM:
        raise ValueError(f"sobol: dim must be 1..{MAX_DIM}, got {dim}")
    if n > (1 << NBITS):
        raise ValueError("sobol: n exceeds 2**30")
    V = _direction_numbers(dim)
    shift = np.zeros(dim, dtype=np.uint64)
    if scramble:
        rng = np.random.default_rng(seed)
        V = _scramble(V, rng)
        shift = rng.integers(0, 1 << NBITS, size=dim, dtype=np.uint64)
    i = np.arange(n, dtype=np.uint64)
    gray = i ^ (i >> np.uint64(1))
    X = np.zeros((n, dim), dtype=np.uint64)
    for k in range(NBITS):
        bit = ((gray >> np.uint64(k)) & np.uint64(1)).astype(bool)
        if not bit.any():
            break
        X[bit] ^= V[:, k]
    X ^= shift
    return X.astype(np.float64) / float(1 << NBITS)


def norm_ppf(p):
    """Standard normal quantile: Acklam's rational approximation plus one
    Halley refinement step. Agrees with scipy.stats.norm.ppf to 1e-8 for
    p in [1e-15, 1 - 1e-15]."""
    p = np.asarray(p, dtype=np.float64)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    x = np.empty_like(p)
    plow = 0.02425
    lo, hi = p < plow, p > 1 - plow
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo]))
    x[lo] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    x[hi] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
             ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p[mid] - 0.5
    r = q * q
    x[mid] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
             (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    # one Halley step against the exact CDF
    from math import erfc
    e = 0.5 * np.vectorize(erfc)(-x / np.sqrt(2.0)) - p
    u = e * np.sqrt(2 * np.pi) * np.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def chi2_4_ppf(p):
    """Quantile of the chi-square distribution with 4 degrees of freedom.

    CDF(x) = 1 - exp(-u) (1 + u), u = x/2, inverted by bisection in log(u),
    which converges unconditionally. The CDF is evaluated as a series for
    small u, where the closed form cancels catastrophically. Agrees with
    scipy.stats.chi2.ppf(p, 4) to 1e-8 relative for p in [1e-15, 1 - 1e-9];
    beyond that 1 - p itself is not resolved in double precision."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-300, 1.0 - 1e-16)

    def cdf(u):
        small = u < 1e-3
        series = u * u * (0.5 - u / 3.0 + u * u / 8.0)
        closed = -np.expm1(-u) - u * np.exp(-u)
        return np.where(small, series, closed)

    lo = np.full_like(p, np.log(1e-160))
    hi = np.full_like(p, np.log(200.0))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = cdf(np.exp(mid)) < p
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 2.0 * np.exp(0.5 * (lo + hi))
