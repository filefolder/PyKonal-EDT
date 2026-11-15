"""
Traveltime inventories: storage, distance cropping, on-demand building
(single file and sharded), locking, and fitting station requests to grids.

All inventories here are built in tmp_path on small grids.
"""
import os
import shutil
import subprocess
import sys
import textwrap
import time
import warnings

import h5py
import numpy as np
import pytest

import pykonal
from pykonal import inventory as inv_mod
from pykonal.fields import ScalarField3D
from pykonal.inventory import (TraveltimeInventory, ensure_traveltimes,
                               ensure_traveltimes_sharded, fit_requests_to_grid,
                               resolve_max_dist)
from pykonal.locate import EQLocator

STATIONS = {"A": (40., 40., 0.), "B": (160., 40., 0.), "C": (100., 160., 0.)}
TRUE, T0 = np.array([90., 90., 10.]), 50.0


def _model(path, speed=5.0, npts=(51, 51, 13), node=(4.0, 4.0, 2.0),
           coord_sys="cartesian", mins=(0.0, 0.0, 0.0)):
    f = ScalarField3D(coord_sys=coord_sys)
    f.min_coords, f.node_intervals, f.npts = mins, node, npts
    f.values = np.full(npts, speed)
    f.to_hdf(str(path), overwrite=True)
    return str(path)


def _solve_into(inv, vm_path, sta, phase="P", **add_kw):
    vm = pykonal.fields.read_hdf(vm_path)
    s = pykonal.solver.PointSourceSolver(coord_sys="cartesian")
    s.velocity.min_coords, s.velocity.node_intervals = vm.min_coords, vm.node_intervals
    s.velocity.npts, s.velocity.values = vm.npts, vm.values
    s.src_loc = np.array(STATIONS[sta])
    s.solve()
    inv.add(s.traveltime, f"XX/{sta}/{phase}", station_coords=s.src_loc, **add_kw)
    return s.traveltime


@pytest.fixture
def vp(tmp_path):
    return _model(tmp_path / "vp.h5")


@pytest.fixture
def ab_inventory(tmp_path, vp):
    """Inventory holding stations A and B only."""
    p = str(tmp_path / "tt.h5")
    with TraveltimeInventory(p, mode="w") as inv:
        for sta in ("A", "B"):
            _solve_into(inv, vp, sta, max_dist=300)
    return p


def _requests():
    return {("XX", s, "P"): np.array(c) for s, c in STATIONS.items()}


# ------------------------------------------------------------ storage
def test_stored_grid_reads_back(tmp_path, vp):
    p = str(tmp_path / "tt.h5")
    with TraveltimeInventory(p, mode="w") as inv:
        tt = _solve_into(inv, vp, "A", max_dist=None)
    with TraveltimeInventory(p) as inv:
        assert inv.has("XX/A/P") and not inv.has("XX/B/P")
        back = inv.read("XX/A/P")
    assert np.allclose(back.values, tt.values, rtol=1e-6, equal_nan=True)


def test_default_compression_is_gzip_with_shuffle(tmp_path, vp):
    p = str(tmp_path / "tt.h5")
    with TraveltimeInventory(p, mode="w") as inv:
        _solve_into(inv, vp, "A", max_dist=None)
    with h5py.File(p, "r") as f5:
        ds = f5["XX/A/P/values"]
        assert ds.compression == "gzip" and ds.shuffle


def test_read_subvolume(tmp_path, vp):
    p = str(tmp_path / "tt.h5")
    with TraveltimeInventory(p, mode="w") as inv:
        full = _solve_into(inv, vp, "A", max_dist=None)
    lo, hi = np.array([40., 40., 0.]), np.array([80., 80., 10.])
    with TraveltimeInventory(p) as inv:
        sub = inv.read("XX/A/P", min_coords=lo, max_coords=hi)
    assert np.all(sub.npts < full.npts)
    pt = np.array([60., 61., 5.])
    assert sub.value(pt) == pytest.approx(full.value(pt), rel=1e-5)


def test_max_dist_crops_and_masks(tmp_path, vp):
    p = str(tmp_path / "tt.h5")
    with TraveltimeInventory(p, mode="w") as inv:
        full = _solve_into(inv, vp, "A", phase="full", max_dist=None)
        _solve_into(inv, vp, "A", phase="P", max_dist=60.0)
    with TraveltimeInventory(p) as inv:
        lim = inv.read("XX/A/P")
        n_full = np.prod(inv.f5["XX/A/full/npts"][:])
        n_lim = np.prod(inv.f5["XX/A/P/npts"][:])
    assert n_lim < n_full
    near, far = np.array([80., 40., 5.]), np.array([140., 40., 5.])   # 40 / 100 km
    assert lim.value(near) == pytest.approx(full.value(near), rel=1e-5)
    assert not np.isfinite(lim.value(far, null=np.inf))


def test_max_dist_crops_spherical_grids(tmp_path, sph_net):
    key, src = ("XX", "S4", "P"), sph_net.station_grid[("XX", "S4")]
    full = sph_net.fields[key]
    p = str(tmp_path / "tt.h5")
    with TraveltimeInventory(p, mode="w") as inv:
        inv.add(full, "XX/S4/P", station_coords=src, max_dist=50.0)
    with TraveltimeInventory(p) as inv:
        lim = inv.read("XX/S4/P")
    assert np.prod(lim.npts) < np.prod(full.npts)
    km = src[0] * np.pi / 180.0                         # km per degree of arc
    near = src + np.array([-10.0, 30.0 / km * np.pi / 180.0, 0.0])
    far = src + np.array([-10.0, 0.0, 80.0 / (km * np.sin(src[1])) * np.pi / 180.0])
    assert lim.value(near) == pytest.approx(full.value(near), rel=1e-5)
    assert not np.isfinite(lim.value(far, null=np.inf))


@pytest.mark.parametrize("cfg, expected", [
    ({}, inv_mod.DEFAULT_MAX_DIST_KM), ({"max_dist_km": None}, None),
    ({"max_dist_km": float("inf")}, None), ({"max_dist_km": 250}, 250.0),
])
def test_resolve_max_dist(cfg, expected):
    assert resolve_max_dist(cfg) == expected


@pytest.mark.parametrize("bad", [0, -5, "far", float("nan")])
def test_resolve_max_dist_rejects(bad):
    with pytest.raises(ValueError):
        resolve_max_dist({"max_dist_km": bad})


# ------------------------------------------------------------ ensure
def test_arrival_without_grid_is_dropped_with_warning(ab_inventory):
    with TraveltimeInventory(ab_inventory) as inv:
        arr = {("XX", s, "P"): T0 + inv.read(f"XX/{s}/P").value(TRUE) for s in "AB"}
    arr[("XX", "C", "P")] = T0 + 25.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with EQLocator(ab_inventory, coord_sys="cartesian") as loc:
            loc.add_arrivals(arr)
            loc.locate_seed = 0
            soln = loc.locate(np.array([100., 100., 15., 49.]),
                              np.array([60., 60., 10., 10.]), 0.01, "edt")
            used = loc.residuals(soln)
    assert any("No traveltime grid" in str(x.message) for x in w)
    assert set(used) == {("XX", "A", "P"), ("XX", "B", "P")}


def test_ensure_computes_only_what_is_missing(ab_inventory, vp):
    rep = ensure_traveltimes(ab_inventory, _requests(), {"P": vp}, max_dist=300)
    assert sorted(rep["present"]) == [("XX", "A", "P"), ("XX", "B", "P")]
    assert rep["computed"] == [("XX", "C", "P")] and not rep["skipped"]

    rep = ensure_traveltimes(ab_inventory, _requests(), {"P": vp}, max_dist=300)
    assert len(rep["present"]) == 3 and not rep["computed"]

    with TraveltimeInventory(ab_inventory) as inv:
        arr = {k: T0 + inv.read("/".join(k)).value(TRUE) for k in _requests()}
    with EQLocator(ab_inventory, coord_sys="cartesian") as loc:
        loc.add_arrivals(arr)
        loc.locate_seed = 0
        soln = loc.locate(np.array([100., 100., 15., 49.]),
                          np.array([60., 60., 10., 10.]), 0.01, "edt")
        assert len(loc.residuals(soln)) == 3
    # three surface stations fix the epicentre but not the depth
    assert np.linalg.norm(soln[:2] - TRUE[:2]) < 2.0


def test_three_station_search_finds_the_maximum_for_every_seed(ab_inventory, vp):
    """Three stations, exact data, a 120 km box: the true peak is ~1 km wide.
    Without the coarse-to-fine variance floor 1 seed in 10 ended on a face."""
    ensure_traveltimes(ab_inventory, _requests(), {"P": vp}, max_dist=300)
    with TraveltimeInventory(ab_inventory) as inv:
        arr = {k: T0 + inv.read("/".join(k)).value(TRUE) for k in _requests()}
    for seed in range(40):
        with EQLocator(ab_inventory, coord_sys="cartesian") as loc:
            loc.add_arrivals(arr)
            loc.locate_seed = seed
            soln = loc.locate(np.array([100., 100., 15., 49.]),
                              np.array([60., 60., 10., 10.]), 0.01, "edt")
            assert loc.edt(soln[:3]) <= loc.edt(TRUE) + 0.05, seed


def test_ensure_skips_phase_without_model(ab_inventory, vp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = ensure_traveltimes(ab_inventory, {("XX", "A", "S"): np.array(STATIONS["A"])},
                                 {"P": vp}, max_dist=300)
    assert rep["skipped"] == [("XX", "A", "S")]


def test_concurrent_ensure_computes_once(ab_inventory, vp):
    worker = textwrap.dedent(f"""
        import numpy as np
        from pykonal.inventory import ensure_traveltimes
        rep = ensure_traveltimes({ab_inventory!r},
                                 {{("XX", "D", "P"): np.array([60., 120., 0.])}},
                                 {{"P": {vp!r}}}, max_dist=300)
        print("computed" if rep["computed"] else "present")
    """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.dirname(os.path.dirname(pykonal.__file__))]
        + os.environ.get("PYTHONPATH", "").split(os.pathsep)))
    procs = [subprocess.Popen([sys.executable, "-c", worker], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env=env)
             for _ in range(2)]
    outs = [p.communicate(timeout=120) for p in procs]
    assert all(p.returncode == 0 for p in procs), [o[1][-500:] for o in outs]
    assert sorted(o[0].strip() for o in outs) == ["computed", "present"]


# ------------------------------------------------------------ fitting to grid
@pytest.fixture
def sph_model(tmp_path):
    R = 6371.0
    return _model(tmp_path / "vsph.h5", coord_sys="spherical", npts=(40, 40, 40),
                  mins=(R - 50.0, 1.0, 2.0),
                  node=(50.0 / 39, 0.01 / 39, 0.02 / 39)), R


def test_station_above_datum_is_clamped_with_a_message(sph_model):
    vm, R = sph_model
    log = []
    out = fit_requests_to_grid({("AU", "KMBL", "P"): np.array([R + 0.3, 1.005, 2.01])},
                               {"P": vm}, vertical_axis=0, log=log.append)
    assert out[("AU", "KMBL", "P")][0] == pytest.approx(R, abs=1e-9)
    assert any("vertically" in m for m in log)


def test_station_inside_grid_is_untouched(sph_model):
    vm, R = sph_model
    log = []
    c = np.array([R - 10.0, 1.005, 2.01])
    out = fit_requests_to_grid({("AU", "FORT", "P"): c}, {"P": vm}, 0, log=log.append)
    assert np.array_equal(out[("AU", "FORT", "P")], c) and log == []


def test_station_laterally_outside_is_excluded(sph_model):
    vm, R = sph_model
    log = []
    req = {("XX", "FAR", "P"): np.array([R - 5.0, 9.9, 2.01]),
           ("AU", "FORT", "P"): np.array([R - 10.0, 1.005, 2.01])}
    out = fit_requests_to_grid(req, {"P": vm}, 0, log=log.append)
    assert set(out) == {("AU", "FORT", "P")}
    assert any("laterally" in m for m in log)


def test_fit_keeps_none_and_does_not_mutate(sph_model):
    vm, R = sph_model
    req = {("ZZ", "NONE", "P"): None,
           ("AU", "KMBL", "P"): np.array([R + 0.3, 1.005, 2.01])}
    before = req[("AU", "KMBL", "P")].copy()
    out = fit_requests_to_grid(req, {"P": vm}, 0)
    assert out[("ZZ", "NONE", "P")] is None
    assert np.array_equal(req[("AU", "KMBL", "P")], before)


# ------------------------------------------------------------ sharded
def _shard_requests():
    return {("XX", "A", "P"): np.array(STATIONS["A"]),
            ("XX", "A", "S"): np.array(STATIONS["A"]),
            ("XX", "B", "P"): np.array(STATIONS["B"]),
            ("XX", "NOCOORD", "P"): None}


@pytest.fixture
def models(tmp_path):
    return {"P": _model(tmp_path / "vp.h5", 6.0), "S": _model(tmp_path / "vs.h5", 3.5)}


def test_sharded_build_then_present(tmp_path, models):
    d = str(tmp_path / "shards")
    r = ensure_traveltimes_sharded(d, _shard_requests(), models, max_dist=300, nproc=1)
    assert len(r["computed"]) == 3 and r["skipped"] == [("XX", "NOCOORD", "P")]
    assert all(os.path.exists(inv_mod.shard_path(d, k)) for k in r["computed"])

    r = ensure_traveltimes_sharded(d, _shard_requests(), models, max_dist=300, nproc=1)
    assert len(r["present"]) == 3 and not r["computed"]


def test_sharded_master_links_are_relative(tmp_path, models):
    d = str(tmp_path / "shards")
    ensure_traveltimes_sharded(d, _shard_requests(), models, max_dist=300, nproc=1)
    with EQLocator(os.path.join(d, "inventory.h5"), coord_sys="cartesian") as loc:
        loc.add_arrivals({("XX", "A", "S"): 1.0})
        loc.read_traveltimes()
        expected = loc.traveltimes[("XX", "A", "S")].value(TRUE)
    moved = str(tmp_path / "moved")
    shutil.move(d, moved)
    with h5py.File(os.path.join(moved, "inventory.h5"), "r") as f5:
        assert "XX/A/S" in f5
    with TraveltimeInventory(os.path.join(moved, "inventory.h5")) as inv:
        assert inv.read("XX/A/S").value(TRUE) == pytest.approx(expected)


def test_sharded_rebuilds_after_model_change(tmp_path, models):
    d = str(tmp_path / "shards")
    ensure_traveltimes_sharded(d, _shard_requests(), models, max_dist=300, nproc=1)
    time.sleep(1.1)                                   # mtime resolution
    _model(models["P"], 6.5)
    log = []
    r = ensure_traveltimes_sharded(d, _shard_requests(), models, max_dist=300,
                                   nproc=1, log=log.append)
    assert sorted(r["computed"]) == [("XX", "A", "P"), ("XX", "B", "P")]
    assert r["present"] == [("XX", "A", "S")]
    assert any("superseded" in m for m in log)


def test_sharded_rebuilds_when_max_dist_grows(tmp_path, models):
    d = str(tmp_path / "shards")
    req = {("XX", "A", "P"): np.array(STATIONS["A"])}
    ensure_traveltimes_sharded(d, req, models, max_dist=100, nproc=1)
    log = []
    r = ensure_traveltimes_sharded(d, req, models, max_dist=300, nproc=1, log=log.append)
    assert r["computed"] == [("XX", "A", "P")]
    assert any("max_dist_km" in m for m in log)


def test_sharded_parallel_matches_serial(tmp_path, models):
    a = ensure_traveltimes_sharded(str(tmp_path / "s1"), _shard_requests(), models,
                                   max_dist=300, nproc=1)
    b = ensure_traveltimes_sharded(str(tmp_path / "s2"), _shard_requests(), models,
                                   max_dist=300, nproc=2)
    assert sorted(a["computed"]) == sorted(b["computed"])
    for k in a["computed"]:
        with h5py.File(inv_mod.shard_path(str(tmp_path / "s1"), k), "r") as f1, \
             h5py.File(inv_mod.shard_path(str(tmp_path / "s2"), k), "r") as f2:
            g = "/".join(k) + "/values"
            assert np.array_equal(f1[g][...], f2[g][...], equal_nan=True)


def test_sharded_masks_impossible_nodes(tmp_path, models, monkeypatch):
    def fake(path, requests, velocity_models, **kw):
        (key, _coords), = requests.items()
        vals = np.arange(27, dtype=float).reshape(3, 3, 3)
        vals[1, 1, 1], vals[0, 0, 1] = -np.inf, -5.0
        with h5py.File(path, "w") as f5:
            f5.create_group("/".join(key)).create_dataset("values", data=vals)
        return {"present": [], "computed": [key], "skipped": []}
    monkeypatch.setattr(inv_mod, "ensure_traveltimes", fake)
    d, log = str(tmp_path / "shards"), []
    key = ("WG", "BAD", "P")
    ensure_traveltimes_sharded(d, {key: np.array([1., 2., 3.])}, models,
                               nproc=1, log=log.append)
    with h5py.File(inv_mod.shard_path(d, key), "r") as f5:
        v = f5["WG/BAD/P/values"][...]
    assert np.isnan(v).sum() == 2 and np.nanmin(v) >= 0
    assert any("impossible traveltime node" in m for m in log)


def _coarse_spherical_model(tmp_path):
    R = 6371.0
    mins = (R - 44.0, np.radians(118.0), np.radians(119.0))
    node = (1.0, np.radians(0.25), np.radians(0.25))
    vp = _model(tmp_path / "vp.h5", 5.8, npts=(45, 25, 25), node=node,
                coord_sys="spherical", mins=mins)
    return vp, np.array(mins), np.array(node)


def test_station_between_coarse_nodes_gets_a_full_grid(tmp_path):
    vp, mins, node = _coarse_spherical_model(tmp_path)
    src = mins + node * np.array([43.0, 10.5, 10.5])    # between node columns
    d = str(tmp_path / "s")
    r = ensure_traveltimes_sharded(d, {("XX", "MID", "P"): src}, {"P": vp},
                                   max_dist=None, nproc=1)
    assert r["computed"] == [("XX", "MID", "P")]
    with TraveltimeInventory(os.path.join(d, "inventory.h5")) as inv:
        assert np.isfinite(inv.read("XX/MID/P").values).all()


@pytest.mark.parametrize("builder", ["single", "sharded"])
def test_station_outside_the_model_is_skipped_not_stored(tmp_path, builder):
    vp, mins, node = _coarse_spherical_model(tmp_path)
    above = mins + node * np.array([48.0, 10.0, 10.0])    # 4 km above the top
    key = ("XX", "UP", "P")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if builder == "single":
            p = str(tmp_path / "tt.h5")
            r = ensure_traveltimes(p, {key: above}, {"P": vp}, max_dist=None, nproc=1)
            with TraveltimeInventory(p) as inv:
                assert not inv.has("XX/UP/P")
            assert any("outside the velocity model" in str(x.message) for x in w)
        else:
            log = []
            d = str(tmp_path / "s")
            r = ensure_traveltimes_sharded(d, {key: above}, {"P": vp},
                                           max_dist=None, nproc=1, log=log.append)
            assert not os.path.exists(inv_mod.shard_path(d, key))
            assert any("outside the velocity model" in m for m in log), log
    assert r["computed"] == [] and r["skipped"] == [key]


def test_station_longitude_wraps_on_spherical_grids(tmp_path):
    vp, mins, node = _coarse_spherical_model(tmp_path)
    src = mins + node * np.array([43.0, 10.0, 10.0])
    src[2] -= 2 * np.pi                                   # same place, other branch
    r = ensure_traveltimes(str(tmp_path / "tt.h5"), {("XX", "W", "P"): src},
                           {"P": vp}, max_dist=None, nproc=1)
    assert r["computed"] == [("XX", "W", "P")]


def test_inventory_lock_shared_and_exclusive(tmp_path):
    p = str(tmp_path / "inventory.h5")
    for shared in (True, False, True):
        with inv_mod.inventory_lock(p, shared=shared):
            pass
