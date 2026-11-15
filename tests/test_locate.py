"""
EQLocator: the EDT likelihood, the search, residuals and quality metrics,
on the shared 50 x 50 x 20 km cartesian network (8 surface stations).
"""
import numpy as np
import pytest

from pykonal.locate import EQLocator

TRUE = np.array([22.0, 28.0, 9.0])
INITIAL = np.array([25.0, 25.0, 10.0, 99.0])
DELTA = np.array([15.0, 15.0, 9.0, 5.0])
OUTLIER = ("XX", "ST04", "P")


@pytest.fixture
def arrivals(cart_net):
    return cart_net.arrivals(TRUE, t0=100.0, sigma=0.05, seed=42)


def _open(cart_net, arrivals, seed=7):
    loc = EQLocator(cart_net.path, coord_sys="cartesian")
    loc.add_arrivals(arrivals)
    loc.add_pick_errors({k: 0.05 for k in arrivals})
    loc.add_stations(cart_net.stations)
    loc.locate_seed = seed
    return loc


def test_edt_locates_a_clean_event(cart_net, arrivals):
    with _open(cart_net, arrivals) as loc:
        soln = loc.locate(INITIAL.copy(), DELTA.copy(), 0.01, "edt")
    assert np.linalg.norm(soln[:3] - TRUE) < 1.0
    assert soln[3] == pytest.approx(100.0, abs=0.1)


def test_edt_resists_an_outlier_that_breaks_l1(cart_net, arrivals):
    arrivals[OUTLIER] += 2.5
    with _open(cart_net, arrivals) as loc:
        edt = loc.locate(INITIAL.copy(), DELTA.copy(), 0.01, "edt")
        l1 = loc.locate(INITIAL.copy(), DELTA.copy(), 0.01, "l1")
        res = loc.residuals(edt)
    err_edt = np.linalg.norm(edt[:3] - TRUE)
    assert err_edt < 1.5
    assert err_edt < np.linalg.norm(l1[:3] - TRUE)
    # the residuals at the EDT solution single out the bad pick
    worst = max(res, key=lambda k: abs(res[k]))
    assert worst == OUTLIER and res[worst] > 2.0
    assert max(abs(v) for k, v in res.items() if k != OUTLIER) < 0.3


def test_locate_respects_explicit_bounds(cart_net, arrivals):
    bounds = np.array([[20.0, 20.0, 12.0, 90.0], [30.0, 30.0, 18.0, 110.0]])
    with _open(cart_net, arrivals) as loc:
        soln = loc.locate(np.array([25., 25., 15., 100.]), DELTA.copy(), 0.01,
                          "edt", bounds=bounds)
    assert bounds[0, 2] - 1e-9 <= soln[2] <= bounds[1, 2] + 1e-9


def test_locate_is_reproducible_with_a_seed(cart_net, arrivals):
    with _open(cart_net, arrivals, seed=11) as loc:
        a = loc.locate(INITIAL.copy(), DELTA.copy(), 0.01, "edt")
    with _open(cart_net, arrivals, seed=11) as loc:
        b = loc.locate(INITIAL.copy(), DELTA.copy(), 0.01, "edt")
    assert np.array_equal(a, b)


def test_edt_many_matches_single_evaluations(cart_net, arrivals):
    rng = np.random.default_rng(1)
    pts = TRUE + rng.uniform(-5, 5, size=(16, 3))
    pts[:, 2] = np.clip(pts[:, 2], 0.5, 19.5)
    with _open(cart_net, arrivals) as loc:
        loc.read_traveltimes()
        ll, ot = loc.edt_many(np.ascontiguousarray(pts))
        single = -np.array([loc.edt(p) for p in pts])     # edt() is the misfit
        ot_single = np.array([loc.origin_time(p) for p in pts])
    assert np.allclose(ll, single, rtol=1e-12, atol=1e-12)
    assert np.allclose(ot, ot_single, rtol=1e-9, atol=1e-9)


def test_likelihood_peaks_at_the_truth(cart_net):
    clean = cart_net.arrivals(TRUE, sigma=0.0)
    with _open(cart_net, clean) as loc:
        loc.read_traveltimes()
        at_true = loc.edt(TRUE)                 # negative log-likelihood
        for step in np.eye(3):
            assert loc.edt(TRUE + 2.0 * step) > at_true
            assert loc.edt(TRUE - 2.0 * step) > at_true


def test_origin_time_stats_recover_t0(cart_net):
    clean = cart_net.arrivals(TRUE, t0=100.0, sigma=0.0)
    with _open(cart_net, clean) as loc:
        loc.read_traveltimes()
        mean, var, w, ot = loc.edt_ot_stats(TRUE)
    assert mean == pytest.approx(100.0, abs=0.05)
    assert len(ot) == len(clean) and np.all(np.asarray(w) > 0)


def test_quality_metrics(cart_net, arrivals):
    with _open(cart_net, arrivals) as loc:
        soln = loc.locate(INITIAL.copy(), DELTA.copy(), 0.01, "edt")
        q = loc.quality(soln)
    assert q["nobs"] == 8
    assert 0.0 < q["rms"] < 0.2
    assert 0.0 < q["azimuthal_gap"] < 120.0            # surrounded
    # nearest station to (22, 28) is ST07 at (10, 25): ~12.4 km
    assert q["min_station_dist"] == pytest.approx(12.4, abs=1.0)


def test_locate_detailed_returns_solution_quality_posterior(cart_net, arrivals):
    with _open(cart_net, arrivals) as loc:
        out = loc.locate_detailed(INITIAL.copy(), DELTA.copy(), alpha=0.01,
                                  method="edt", nsamples=1024, nscatter=256, seed=2)
    assert {"hypocenter", "quality", "posterior"} <= set(out)
    assert out["posterior"]["scatter"].shape == (256, 3)



@pytest.mark.parametrize("n_stations", [4, 8])
def test_search_finds_the_maximum_for_every_seed(cart_net, n_stations):
    keys = sorted(cart_net.fields)[:n_stations]
    clean = cart_net.arrivals(TRUE, sigma=0.0, keys=keys)
    for seed in range(20):
        with _open(cart_net, clean, seed=seed) as loc:
            soln = loc.locate(INITIAL.copy(), DELTA.copy(), 0.01, "edt")
            assert loc.edt(soln[:3]) <= loc.edt(TRUE) + 0.05, seed

