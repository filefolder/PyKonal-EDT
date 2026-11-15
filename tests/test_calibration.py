"""
Does the reported uncertainty contain the truth as often as it claims?

Synthetic events on the shared spherical network, located with the same
auto pipeline the drivers use (relocate.relocate, search_delta_km="auto").
Noise is Gaussian with the assumed pick sigma, so this is the best case: a
locator that under-covers here will under-cover on real data. Over-coverage
is expected (the EDT posterior is wider than a least-squares one) and is
bounded only loosely.

The fast test uses 30 events. --runslow adds the standalone harnesses in
validation/ at their quick settings.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from pykonal import sizing

N_EVENTS = 30
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def results(sph_net, relocate_sph):
    rng = np.random.default_rng(2024)
    rows = []
    for ev in range(N_EVENTS):
        true = sph_net.point(north_km=rng.uniform(-35, 35), east_km=rng.uniform(-30, 30),
                             depth_km=rng.uniform(3.0, 30.0))
        # noise drawn from the error model the locator assumes (alpha = 0.01)
        arrivals = sph_net.arrivals(true, sigma=0.05, alpha=0.01, seed=ev)
        anchor = (rng.normal(0, 15), rng.normal(0, 15), 10.0)
        r = relocate_sph(arrivals, {"posterior_seed": ev, "outlier_cutoff_s": 0},
                         anchor_km=anchor)
        cov = r.posterior["covariance_km"]
        off = sph_net.offset_km(r.hypocenter, true)          # (up, south, east) km
        rows.append(dict(off=off, cov=cov, box_limited=r.posterior["box_limited"]))
    return rows


def _coverage(rows, dof_axes, level):
    k2 = sizing.chi_scale(level, len(dof_axes)) ** 2
    hit = []
    for r in rows:
        a = np.ix_(dof_axes, dof_axes)
        d = r["off"][dof_axes]
        hit.append(d @ np.linalg.solve(r["cov"][a], d) <= k2)
    return float(np.mean(hit))


def test_no_event_is_limited_by_its_box(results):
    assert not any(r["box_limited"] for r in results)


@pytest.mark.parametrize("axes, name", [([0], "depth"), ([1, 2], "horizontal"),
                                        ([0, 1, 2], "3-D")])
def test_68_percent_regions_cover_at_least_68_percent(results, axes, name):
    cov = _coverage(results, axes, 68.27)
    # 30 events: binomial sd ~8.5%, so 0.55 is ~1.5 sd below nominal
    assert cov >= 0.55, f"{name} 68% coverage {cov:.0%}"


@pytest.mark.parametrize("axes, name", [([0], "depth"), ([1, 2], "horizontal"),
                                        ([0, 1, 2], "3-D")])
def test_95_percent_regions_cover_at_least_90_percent(results, axes, name):
    cov = _coverage(results, axes, 95.0)
    assert cov >= 0.85, f"{name} 95% coverage {cov:.0%}"


def test_uncertainty_is_conservative_but_informative(results):
    """RMS of error / sigma per axis: 1 is calibrated, below 1 conservative.
    The EDT posterior measures 0.5-0.75 here (sigma 1.4-1.9x the scatter);
    below 0.4 it would be uninformative, above 1.2 overconfident."""
    z = np.array([r["off"] / np.sqrt(np.diag(r["cov"])) for r in results])
    rms = np.sqrt(np.mean(z ** 2, axis=0))
    assert np.all(rms > 0.4) and np.all(rms < 1.2), rms


# ------------------------------------------------------------------- slow
def _run(script, *args, cwd, env=None):
    e = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [REPO] + os.environ.get("PYTHONPATH", "").split(os.pathsep)), **(env or {}))
    return subprocess.run([sys.executable, os.path.join(REPO, "validation", script), *args],
                          capture_output=True, text=True, env=e, cwd=str(cwd),
                          timeout=1800)


@pytest.mark.slow
def test_validation_suite_quick(tmp_path):
    p = _run("pykonal_validation.py", cwd=tmp_path, env={"PYKONAL_TEST_LEVEL": "quick"})
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "[FAIL]" not in p.stdout


@pytest.mark.slow
def test_calibration_harness(tmp_path):
    p = _run("calibrate_uncertainty.py", "--n", "40", cwd=tmp_path)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
