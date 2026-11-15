"""
EQLocator.sample_posterior: importance sampling of the EDT posterior on the
shared spherical network.
"""
import numpy as np
import pytest

from pykonal.locate import EQLocator

HALF_KM = 15.0


@pytest.fixture(scope="module")
def event(sph_net, relocate_sph):
    true = sph_net.point(north_km=-6.0, east_km=-9.0, depth_km=12.0)
    arrivals = sph_net.arrivals(true, seed=0)
    soln = relocate_sph(arrivals, posterior=False).hypocenter
    return true, arrivals, soln


@pytest.fixture
def loc(sph_net, event):
    _true, arrivals, soln = event
    with EQLocator(sph_net.path) as loc:
        loc.add_arrivals(arrivals)
        loc.alpha, loc.default_pick_error = 0.01, 0.05
        delta = _delta(soln)
        loc.read_traveltimes(min_coords=soln[:3] - delta, max_coords=soln[:3] + delta)
        yield loc


def _delta(soln):
    return np.array([HALF_KM, HALF_KM / soln[0], HALF_KM / (soln[0] * np.sin(soln[1]))])


def _sample(loc, soln, **kw):
    kw.setdefault("nsamples", 2048)
    kw.setdefault("seed", 11)
    return loc.sample_posterior(soln[:3], _delta(soln), **kw)


def test_result_shapes_and_basic_sanity(loc, event):
    _true, _arr, soln = event
    post = _sample(loc, soln, nscatter=300)
    assert post["scatter"].shape == (300, 3)
    cov = post["covariance_km"]
    assert cov.shape == (3, 3) and np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    assert post["ellipsoid_km"]["semi_axes"] == pytest.approx(
        np.sqrt(np.sort(np.linalg.eigvalsh(cov))[::-1]), rel=1e-9)
    assert 0 < post["ess"] <= post["nsamples"]
    assert np.isfinite(post["time"]) and post["time_sigma"] > 0


def test_posterior_contains_the_truth(loc, event, sph_net):
    true, _arr, soln = event
    post = _sample(loc, soln)
    off = sph_net.offset_km(soln, true)
    z = np.abs(off) / np.sqrt(np.diag(post["covariance_km"]))
    assert np.all(z < 3.0), z


def test_mode_centred_covariance_is_mean_centred_plus_offset(loc, event):
    """The default (center='solution') is never narrower than NLL's convention."""
    _true, _arr, soln = event
    post = _sample(loc, soln)
    d = post["mode_minus_mean_km"]
    assert post["center"] == "solution"
    assert np.allclose(post["covariance_km"],
                       post["covariance_km_about_mean"] + np.outer(d, d),
                       rtol=1e-9, atol=1e-9)
    assert np.all(np.diag(post["covariance_km"])
                  >= np.diag(post["covariance_km_about_mean"]) - 1e-12)


def test_center_mean_reports_the_mean_centred_covariance(loc, event):
    _true, _arr, soln = event
    a = _sample(loc, soln)
    b = _sample(loc, soln, center="mean")
    assert b["center"] == "mean"
    assert np.allclose(b["covariance_km"], a["covariance_km_about_mean"], rtol=1e-9)


def test_sampling_is_reproducible_with_a_seed(loc, event):
    _true, _arr, soln = event
    a, b = _sample(loc, soln, seed=5), _sample(loc, soln, seed=5)
    assert np.array_equal(a["covariance_km"], b["covariance_km"])


def test_ess_target_grows_the_sample_and_reuses_points(loc, event):
    _true, _arr, soln = event
    small = _sample(loc, soln, nsamples=64, nscatter=0)
    assert small["nsamples"] == 64
    target = 4.0 * small["ess"]
    grown = _sample(loc, soln, nsamples=64, nscatter=0, ess_target=target)
    assert grown["nsamples"] > 64 and grown["ess"] >= target
    # growing 64 -> n gives exactly what drawing n at once gives: the points
    # already evaluated were kept, not redrawn
    direct = _sample(loc, soln, nsamples=grown["nsamples"], nscatter=0)
    assert np.allclose(grown["covariance_km"], direct["covariance_km"],
                       rtol=1e-9, atol=1e-12)
    assert grown["ess"] == pytest.approx(direct["ess"], rel=1e-6)


def test_ess_growth_respects_the_cap(loc, event):
    _true, _arr, soln = event
    capped = _sample(loc, soln, nsamples=64, nscatter=0, ess_target=1e9,
                     max_nsamples=256)
    assert capped["nsamples"] == 256


def test_too_small_box_is_flagged(loc, event):
    _true, _arr, soln = event
    tiny = np.array([0.3, 0.3 / soln[0], 0.3 / (soln[0] * np.sin(soln[1]))])
    post = loc.sample_posterior(soln[:3], tiny, nsamples=1024, seed=1)
    assert post["box_limited"]
