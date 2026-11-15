"""
Shared fixtures for the pykonal test suite.

Everything a test writes goes to a pytest temporary directory: the suite
never writes into the source tree and can be run from any directory.

The synthetic networks are built once per session (a few seconds of FMM
solves) and shared read-only. Tests that need to modify an inventory make
their own in ``tmp_path``.

Run
    pytest                 # fast suite
    pytest --runslow       # also the slow statistical tests
    pytest -m slow --runslow   # only the slow ones
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pytest


# --------------------------------------------------------------- options
def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run tests marked slow")


def pytest_configure(config):
    try:
        import pykonal.locate  # noqa: F401  (compiled extension)
    except ImportError as exc:
        raise pytest.UsageError(
            "pykonal is not built or not importable "
            f"({exc}). Install it with `pip install -e .` from the "
            "repository root, or build in place with "
            "`python setup.py build_ext --inplace` and run "
            "`python -m pytest` from the repository root.")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow: run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


# --------------------------------------------------------------- networks
Key = Tuple[str, str, str]
EARTH_R = 6371.0


@dataclass
class Network:
    """A traveltime inventory plus what is needed to make events for it."""
    path: str
    coord_sys: str
    fields: Dict[Key, object]                 # (net, sta, phase) -> traveltime field
    stations: Dict[Tuple[str, str], tuple]    # grid coords (cart.) or (lat, lon, elev_m)
    station_grid: Dict[Tuple[str, str], np.ndarray]
    centre: np.ndarray                        # grid coords of the surface centre
    depth_max: float
    extra: dict = field(default_factory=dict)

    # ---- geometry -------------------------------------------------------
    def km_scale(self, at):
        """Grid units -> km per axis at `at` (1 for cartesian)."""
        if self.coord_sys == "cartesian":
            return np.ones(3)
        return np.array([1.0, at[0], at[0] * math.sin(at[1])])

    def offset_km(self, a, b):
        """b - a in km, axis by axis, in grid axis order."""
        a = np.asarray(a[:3], float)
        return (np.asarray(b[:3], float) - a) * self.km_scale(a)

    def point(self, north_km=0.0, east_km=0.0, depth_km=10.0):
        """Grid coordinates of a point given in km from the centre."""
        c = self.centre
        if self.coord_sys == "cartesian":
            return np.array([c[0] + east_km, c[1] + north_km, depth_km])
        r = c[0] - depth_km
        return np.array([r, c[1] - north_km / c[0],
                         c[2] + east_km / (c[0] * math.sin(c[1]))])

    # ---- synthetic data -------------------------------------------------
    def arrivals(self, hypo, t0=100.0, sigma=0.05, seed=0, keys=None, alpha=0.0):
        """Arrival times with Gaussian noise. With alpha > 0 the noise sd is
        sqrt(sigma^2 + (alpha * traveltime)^2): the locator's error model."""
        rng = np.random.default_rng(seed)
        keys = list(self.fields) if keys is None else keys
        out = {}
        for k in keys:
            tt = float(self.fields[k].value(np.asarray(hypo, float)))
            out[k] = t0 + tt + rng.normal(0.0, math.hypot(sigma, alpha * tt))
        return out


def _solve(coord_sys, mins, node, npts, vel, src):
    import pykonal
    solver = pykonal.solver.PointSourceSolver(coord_sys=coord_sys)
    solver.velocity.min_coords = mins
    solver.velocity.node_intervals = node
    solver.velocity.npts = npts
    solver.velocity.values = vel
    solver.src_loc = np.asarray(src, float)
    solver.solve()
    return solver.traveltime


@pytest.fixture(scope="session")
def cart_net(tmp_path_factory):
    """50 x 50 x 20 km cartesian volume, 1 km nodes, v = 4 + 0.1 z, 8 stations."""
    from pykonal.inventory import TraveltimeInventory

    npts = (51, 51, 21)
    mins, node = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    vel = np.broadcast_to(4.0 + 0.1 * np.arange(npts[2]), npts).copy()
    sta_xyz = {
        "ST01": (5, 5), "ST02": (45, 5), "ST03": (5, 45), "ST04": (45, 45),
        "ST05": (25, 10), "ST06": (25, 40), "ST07": (10, 25), "ST08": (40, 25),
    }
    path = str(tmp_path_factory.mktemp("cart") / "tt.h5")
    fields, stations = {}, {}
    with TraveltimeInventory(path, mode="w") as inv:
        for sta, (x, y) in sta_xyz.items():
            src = np.array([x, y, 0.0], float)
            tt = _solve("cartesian", mins, node, npts, vel, src)
            inv.add(tt, f"XX/{sta}/P", station_coords=src, max_dist=None)
            fields[("XX", sta, "P")] = tt
            stations[("XX", sta)] = tuple(src)
    return Network(path=path, coord_sys="cartesian", fields=fields,
                   stations=stations, station_grid={k: np.array(v) for k, v in stations.items()},
                   centre=np.array([25.0, 25.0, 0.0]), depth_max=20.0,
                   extra=dict(mins=mins, node=node, npts=npts, vel=vel))


SPH_LAT0, SPH_LON0, SPH_DEPTH = -30.0, 138.0, 40.0
SPH_STATIONS = {                       # (dlat, dlon) degrees from the centre
    "S0": (0.45, 0.40), "S1": (0.50, -0.35), "S2": (-0.40, 0.50),
    "S3": (-0.45, -0.45), "S4": (0.10, 0.05), "S5": (-0.08, 0.15),
    "S6": (0.05, -0.20), "S7": (-0.25, -0.05),
}


@pytest.fixture(scope="session")
def sph_net(tmp_path_factory):
    """~200 x 170 km spherical cap to 40 km depth, gradient model, 8 P stations."""
    from pykonal.inventory import TraveltimeInventory
    from pykonal.transformations import GeoTransform

    T = GeoTransform()
    centre = np.asarray(T.geo_to_grid(SPH_LAT0, SPH_LON0, 0.0), dtype=float)
    half = math.radians(0.9)
    npts = (21, 41, 41)
    node = np.array([SPH_DEPTH / (npts[0] - 1), 2 * half / (npts[1] - 1),
                     2 * half / (npts[2] - 1)])
    mins = np.array([centre[0] - SPH_DEPTH, centre[1] - half, centre[2] - half])
    depth = np.linspace(SPH_DEPTH, 0.0, npts[0])
    vel = np.broadcast_to((5.0 + 0.05 * depth)[:, None, None], npts).copy()

    path = str(tmp_path_factory.mktemp("sph") / "tt.h5")
    fields, stations, grid = {}, {}, {}
    with TraveltimeInventory(path, mode="w") as inv:
        for name, (dla, dlo) in SPH_STATIONS.items():
            src = np.array([centre[0], centre[1] - math.radians(dla),
                            centre[2] + math.radians(dlo)])
            tt = _solve("spherical", mins, node, npts, vel, src)
            inv.add(tt, f"XX/{name}/P", station_coords=src)
            fields[("XX", name, "P")] = tt
            lat, lon, _dep = T.grid_to_geo(src)
            stations[("XX", name)] = (float(lat), float(lon), 0.0)
            grid[("XX", name)] = src
    return Network(path=path, coord_sys="spherical", fields=fields,
                   stations=stations, station_grid=grid, centre=centre,
                   depth_max=SPH_DEPTH,
                   extra=dict(T=T, mins=mins, node=node, npts=npts, vel=vel))


@pytest.fixture(scope="session")
def relocate_sph(sph_net):
    """Call relocate.relocate on the spherical network with test defaults."""
    from pykonal import relocate as pkrel
    from pykonal.locate import EQLocator

    T = sph_net.extra["T"]

    def run(arrivals, settings=None, anchor_km=(25.0, 20.0, 10.0), **kw):
        a = sph_net.point(north_km=anchor_km[0], east_km=anchor_km[1],
                          depth_km=anchor_km[2])
        lat, lon, _ = T.grid_to_geo(a)
        s = {"search_delta_km": "auto", "alpha": 0.01, "default_pick_error": 0.05,
             "auto_search_max_km": [60.0, 60.0, 40.0], "posterior_seed": 3}
        s.update(settings or {})
        with EQLocator(sph_net.path) as loc:
            loc.locate_seed = 5
            return pkrel.relocate(loc, T, arrivals, {k: 0.05 for k in arrivals},
                                  sph_net.stations,
                                  (float(lat), float(lon), anchor_km[2], 100.0),
                                  s, **kw)
    return run
