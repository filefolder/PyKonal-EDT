"""
The shared relocation driver (pykonal.relocate.relocate) in "auto" mode:
one search over auto_search_max_km, the uncertainty box sized at the
solution, the sample count set by an effective-sample-size target, and
depth bounds taken from the grids.
"""
import math

import numpy as np
import pytest

from pykonal import relocate as pkrel
from pykonal import sizing
from pykonal.locate import EQLocator


@pytest.fixture(scope="module")
def event(sph_net):
    true = sph_net.point(north_km=-6.0, east_km=-9.0, depth_km=12.0)
    return true, sph_net.arrivals(true, seed=0)


def test_locates_and_reports_uncertainty(sph_net, relocate_sph, event):
    true, arrivals = event
    r = relocate_sph(arrivals)
    off = sph_net.offset_km(r.hypocenter, true)
    assert np.all(np.abs(off) < 3.0), off
    assert r.time == pytest.approx(100.0, abs=0.5)
    assert set(r.solve_keys) == set(arrivals) and not r.excluded
    assert r.posterior is not None and r.posterior["center"] == "solution"
    assert not r.posterior["box_limited"], r.posterior["edge_density"]


def test_linearised_sigma_tracks_the_sampled_posterior(relocate_sph, event):
    _true, arrivals = event
    r = relocate_sph(arrivals)
    ratio = np.sqrt(np.diag(r.posterior["covariance_km"])) / np.asarray(r.box.sigma_km)
    # the EDT posterior is 1.4-1.9x the least-squares width; outside 0.8-3
    # the two no longer describe the same thing
    assert np.all(ratio > 0.8) and np.all(ratio < 3.0), ratio


def test_single_search_finds_the_event_from_a_distant_anchor(sph_net, relocate_sph):
    true = sph_net.point(north_km=-33.0, east_km=34.0, depth_km=18.0)
    arrivals = sph_net.arrivals(true, seed=4)
    # ~100 km from the anchor, so the search box has to be wider than that
    r = relocate_sph(arrivals, {"auto_search_max_km": [120.0, 120.0, 40.0]},
                     anchor_km=(39.0, -29.0, 5.0))
    off = sph_net.offset_km(r.hypocenter, true)
    assert math.hypot(off[1], off[2]) < 2.0 and abs(off[0]) < 5.0, off


def test_outlier_beyond_the_cutoff_is_excluded(relocate_sph, event):
    _true, arrivals = event
    key = ("XX", "S4", "P")
    bad = dict(arrivals)
    bad[key] += 5.0                       # default outlier_cutoff_s is 3 s
    r = relocate_sph(bad)
    assert r.excluded == [key] and key not in r.solve_keys
    assert key not in r.residuals         # dropped before the final relocation
    kept = dict(arrivals)
    kept[key] += 1.0
    assert not relocate_sph(kept, {"outlier_mad_factor": 1e6}).excluded


def test_depth_bounds_default_to_the_grids(sph_net, relocate_sph, event):
    T, keys = sph_net.extra["T"], list(sph_net.fields)
    with EQLocator(sph_net.path) as loc:
        lo, hi, from_grid = sizing.grid_depth_bounds(loc, T, keys, sph_net.centre)
        assert from_grid and lo == pytest.approx(0.0, abs=1e-6)
        assert hi == pytest.approx(sph_net.depth_max, abs=1e-6)
        # an operator's bound is never overridden
        lo, hi, _ = sizing.grid_depth_bounds(loc, T, keys, sph_net.centre, depth_min=5.0)
        assert lo == 5.0 and hi == pytest.approx(sph_net.depth_max, abs=1e-6)
        assert sizing.grid_depth_bounds(loc, T, keys, sph_net.centre,
                                        -1.0, 30.0) == (-1.0, 30.0, False)
    r = relocate_sph(event[1])
    c = sph_net.centre
    assert r.box.bounds[1][0] == pytest.approx(c[0], abs=1e-6)
    assert r.box.bounds[0][0] == pytest.approx(c[0] - sph_net.depth_max, abs=1e-6)


def test_too_few_gradients_falls_back_to_the_capped_box(sph_net):
    T = sph_net.extra["T"]
    at = sph_net.centre - np.array([10.0, 0, 0])
    with EQLocator(sph_net.path) as loc:
        loc.add_arrivals({("XX", "S0", "P"): 1.0, ("XX", "S1", "P"): 2.0})
        loc.read_traveltimes()
        sigma, info = sizing.linearised_sigma(loc, at, "spherical")
        assert sigma is None and info["n"] == 2
        cfg = {"auto_search_max_km": [60.0, 50.0, 40.0]}
        delta, pinfo = sizing.size_posterior_box(loc, T, at, cfg=cfg)
    assert pinfo["sigma_km"] is None
    assert np.allclose(delta, sizing.cap_delta_grid(T, at, cfg))


def test_fixed_depth_is_held_and_still_gets_an_uncertainty(relocate_sph, event):
    _true, arrivals = event
    r = relocate_sph(arrivals, {"fixed_depth_tolerance_km": 1.0}, fixed_depth=12.0)
    assert r.depth_pinned and abs(r.depth_km - 12.0) <= 1.0 + 1e-6
    assert r.box.sigma_km is not None            # gradients survive the thin slab
    assert np.sqrt(r.posterior["covariance_km"][0, 0]) > 0.5


def test_fixed_box_mode(relocate_sph, event):
    true, arrivals = event
    r = relocate_sph(arrivals, {"search_delta_km": [40.0, 40.0, 30.0]})
    assert r.posterior is not None
    assert np.all(np.isfinite(r.hypocenter))


def test_relocation_is_reproducible(relocate_sph, event):
    _true, arrivals = event
    a = relocate_sph(arrivals)
    b = relocate_sph(arrivals)
    assert np.array_equal(a.hypocenter, b.hypocenter)
    assert np.array_equal(a.posterior["covariance_km"], b.posterior["covariance_km"])


def test_removed_keys_are_reported_not_fatal(relocate_sph, event):
    r = relocate_sph(event[1], {"auto_search_cond_max": 1e6})
    notes = [t for lvl, t in r.messages if lvl == "detail"]
    assert any("auto_search_cond_max is no longer used" in t for t in notes)


def test_too_few_arrivals_is_an_error(relocate_sph, event):
    two = dict(list(event[1].items())[:2])
    with pytest.raises(pkrel.RelocationError):
        relocate_sph(two)


# ------------------------------------------------------- posterior sample count
def test_resolve_posterior_nsamples():
    assert pkrel.resolve_posterior_nsamples("auto") == (
        pkrel.NSAMPLES_AUTO_START, float(pkrel.ESS_TARGET))
    assert pkrel.resolve_posterior_nsamples(None)[1] == pkrel.ESS_TARGET
    assert pkrel.resolve_posterior_nsamples(2048) == (2048, None)
    assert pkrel.resolve_posterior_nsamples(0) == (0, None)
    with pytest.raises(ValueError):
        pkrel.resolve_posterior_nsamples("lots")


def test_auto_reaches_the_ess_target_or_the_cap(relocate_sph, event):
    _true, arrivals = event
    post = relocate_sph(arrivals).posterior
    assert (post["ess"] >= pkrel.ESS_TARGET
            or post["nsamples"] >= pkrel.NSAMPLES_SUGGEST_MAX)
    assert relocate_sph(arrivals, {"posterior_nsamples": 512}).posterior["nsamples"] == 512
    assert relocate_sph(arrivals, {"posterior_nsamples": 0}).posterior is None
    with pytest.raises(pkrel.RelocationError):
        relocate_sph(arrivals, {"posterior_nsamples": "lots"})


def test_phase_map():
    pm = pkrel.normalize_phase_map({"Pg": "P", "Sg": "S"})
    assert pkrel.map_phase("Pg", pm) == "P"
    assert pkrel.map_phase("P", pm) == "P"


def test_sparse_network_with_small_pick_errors(sph_net, relocate_sph):
    """Four stations, 0.02 s picks: likelihood peaks ~1 km wide in a 120 km
    box. Before the coarse-to-fine search 3 in 4 such events ended on a
    worse mode, tens of km from the best solution. The check is on the fit,
    not the distance to the truth: four P stations can leave a real
    depth/epicentre trade-off."""
    from pykonal import _neldermead as nm
    keys = [("XX", s, "P") for s in ("S0", "S3", "S5", "S6")]
    rng = np.random.default_rng(5)
    for ev in range(16):
        true = sph_net.point(north_km=rng.uniform(-40, 40), east_km=rng.uniform(-35, 35),
                             depth_km=rng.uniform(3, 30))
        arrivals = sph_net.arrivals(true, sigma=0.02, alpha=0.002, seed=ev, keys=keys)
        r = relocate_sph(arrivals, {"default_pick_error": 0.02, "alpha": 0.002,
                                    "auto_search_max_km": [60.0, 60.0, 40.0],
                                    "outlier_cutoff_s": 0},
                         anchor_km=(rng.uniform(-15, 15), rng.uniform(-15, 15), 10.0),
                         posterior=False)              # event always inside the box
        with EQLocator(sph_net.path) as loc:
            loc.add_arrivals(arrivals)
            loc.pick_errors = {k: 0.02 for k in arrivals}
            loc.alpha = 0.002
            loc.read_traveltimes()
            ref = nm.minimize(loc.edt, true, method="Nelder-Mead",
                              options={"xatol": 1e-9, "fatol": 1e-6, "maxiter": 400})
            found = loc.edt(np.ascontiguousarray(r.hypocenter[:3]))
        assert found <= min(ref.fun, loc.edt(true)) + 0.5, (ev, found, ref.fun)
