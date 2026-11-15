"""
Search-box sizing and depth handling (pykonal.sizing): pure functions, no
traveltime grids. The two "event" cases reproduce boxes diagnosed from real
SeisComP logs.
"""
import numpy as np
import pytest

from pykonal import sizing as S
from pykonal.transformations import GeoTransform

R = 6371.0


class CartTransform:
    coord_sys = "cartesian"

    def geo_to_grid(self, lat, lon, dep):
        return np.array([0.0, 0.0, dep])

    def grid_to_geo(self, xyz):
        return 0.0, 0.0, float(xyz[2])

    def delta_to_grid(self, a, b, c, at=None):
        return np.array([a, b, c], float)


def _depths(bounds):
    """(shallow, deep) km from spherical (2, 4) bounds."""
    return R - bounds[1][0], R - bounds[0][0]


# ----------------------------------------------------------------- basics
@pytest.mark.parametrize("level, expected", [(68.27, (1.000, 1.515, 1.878)),
                                             (95.0, (1.960, 2.448, 2.788))])
def test_chi_scale(level, expected):
    got = [S.chi_scale(level, d) for d in (1, 2, 3)]
    assert got == pytest.approx(expected, abs=2e-3)


def test_axis_order():
    assert list(S._config_to_grid_order([10., 11., 20.], "spherical")) == [20., 10., 11.]
    assert list(S._config_to_grid_order([10., 11., 20.], "cartesian")) == [10., 11., 20.]
    assert S.depth_axis("spherical") == 0 and S.depth_axis("cartesian") == 2


@pytest.mark.parametrize("fixed, dmin, dmax, expected", [
    (None, -3, 80, (-3.0, 80.0, False)),
    (5.0, -3, 80, (5.0, 5.0, True)),
    (0.0, None, None, (0.0, 0.0, True)),
    (None, None, None, (None, None, False)),
])
def test_resolve_depth_bounds(fixed, dmin, dmax, expected):
    assert S.resolve_depth_bounds(fixed, dmin, dmax) == expected


def test_only_the_current_auto_keys_remain():
    assert set(S.AUTO_DEFAULTS) == {"auto_search_max_km", "auto_search_k_sigma",
                                    "auto_search_min_km",
                                    "auto_search_depth_dist_factor"}
    cfg = {"auto_search_cond_max": 1e6, "auto_search_provisional_km": [9, 9, 9],
           "auto_search_k_sigma": 4.0}
    assert {k for k, _ in S.removed_auto_keys(cfg)} == {
        "auto_search_cond_max", "auto_search_provisional_km"}
    assert S.removed_auto_keys(None) == []


# --------------------------------------------------------- floors and cap
def test_floors_and_cap_reproduce_logged_event():
    """Event at (-27.12, 128.75): k*sigma = 50.4/9.3/7.4 km (dep, lat, lon)."""
    log = []
    sigma = np.array([12.60, 2.33, 1.85])
    hw = 4.0 * sigma
    cfg = {"auto_search_max_km": [50., 50., 40.], "auto_search_min_km": [10., 10., 20.],
           "auto_search_depth_dist_factor": 0.5}
    out = S.apply_floors_and_cap(hw, sigma, "spherical", cfg,
                                 provisional_offset_km=np.zeros(3),
                                 min_station_dist_km=236.6, log=log.append)
    assert out[0] == pytest.approx(40.0)            # depth capped
    assert np.all(out[1:] >= 10.0)                   # horizontal floors
    assert any("floored on depth" in m for m in log)
    assert any("capped by auto_search_max_km" in m for m in log)


def test_floor_never_exceeds_cap():
    cfg = {"auto_search_max_km": [5., 5., 5.], "auto_search_min_km": [10., 10., 20.]}
    out = S.apply_floors_and_cap(np.ones(3), np.ones(3), "spherical", cfg)
    assert np.all(out <= 5.0 + 1e-12)


# ------------------------------------------------------------- depth clamp
def test_depth_clamp_uses_capped_half_width():
    T = GeoTransform()
    dep0 = 23.9
    initial = np.append(T.geo_to_grid(-27.1264, 128.748, dep0), 0.0)
    delta = np.append(T.delta_to_grid(10.0, 10.0, 40.0, at=initial[:3]), 5.0)
    b = S.clamp_depth_to_bounds(T, initial, delta, dep0, -1.0, 80.0, -27.1264, 128.748)
    assert _depths(b) == pytest.approx((-1.0, 63.9), abs=1e-6)


@pytest.mark.parametrize("dep0, dmin, dmax, half, expected", [
    (1.5, -3.0, 600.0, 11.4, (-3.0, 12.9)),
    (1.5, -3.0, 80.0, 25.0, (-3.0, 26.5)),
])
def test_depth_clamp_truncates_without_moving_the_anchor(dep0, dmin, dmax, half, expected):
    T = GeoTransform()
    initial = np.append(T.geo_to_grid(-31.3882, 121.8168, dep0), 0.0)
    delta = np.append(T.delta_to_grid(6.0, 6.7, half, at=initial[:3]), 5.0)
    b = S.clamp_depth_to_bounds(T, initial, delta, dep0, dmin, dmax, -31.3882, 121.8168)
    assert _depths(b) == pytest.approx(expected, abs=1e-6)
    assert b[0][0] <= initial[0] <= b[1][0]


@pytest.mark.parametrize("pin", [0.0, 5.0, 23.9])
def test_pinned_depth_collapses_to_the_pin(pin):
    T = GeoTransform()
    init = np.append(T.geo_to_grid(-31.4, 121.8, pin), 0.0)
    delta = np.append(T.delta_to_grid(25., 25., 1e-3, at=init[:3]), 5.0)
    b = S.clamp_depth_to_bounds(T, init, delta, pin, pin, pin, -31.4, 121.8,
                                min_half_width_km=2.0)
    assert np.mean(_depths(b)) == pytest.approx(pin, abs=1e-3)


# ----------------------------------------------------------- fixed boxes
@pytest.mark.parametrize("pin", [0.0, 5.0, 23.9])
def test_fixed_depth_box_keeps_posterior_extent(pin):
    T = GeoTransform()
    box = S.size_fixed_box(T, np.append(T.geo_to_grid(-31.4, 121.8, pin), 0.0),
                           5.0, [25., 25., 25.], pin, fixed_depth=pin,
                           fixed_depth_tolerance=2.0, depth_min=-3.0, depth_max=80.0)
    lo, hi = _depths(box.bounds)
    assert box.depth_pinned
    assert box.delta_grid[0] == pytest.approx(2.0)
    assert box.posterior_delta[0] == pytest.approx(25.0)
    assert 0.5 * (lo + hi) == pytest.approx(pin, abs=1e-3)
    assert hi - lo == pytest.approx(4.0, abs=1e-6)


def test_fixed_depth_box_cartesian():
    box = S.size_fixed_box(CartTransform(), np.array([0., 0., 10., 0.]), 5.0,
                           [25., 25., 25.], 10.0, fixed_depth=10.0,
                           fixed_depth_tolerance=2.0, depth_min=0.0, depth_max=50.0)
    assert box.delta_grid[2] == pytest.approx(2.0)
    assert box.posterior_delta[2] == pytest.approx(25.0)


# --------------------------------------------------------------- widening
def test_depth_is_widened_only_when_asked():
    post = {"edge_density": np.array([0.4, 0.0, 0.3])}
    delta, cap = np.array([5.0, 0.01, 0.01]), np.array([60.0, 0.1, 0.1])
    new, widened, _ = S.widen_truncated_axes(post, delta, cap)
    assert widened == [2] and new[0] == 5.0
    new, widened, _ = S.widen_truncated_axes(post, delta, cap, widen_depth=True)
    assert widened == [0, 2] and new[0] > 5.0


def test_widening_respects_the_cap():
    post = {"edge_density": np.array([0.9, 0.9, 0.9])}
    delta, cap = np.array([50.0, 0.09, 0.09]), np.array([60.0, 0.1, 0.1])
    new, _, _ = S.widen_truncated_axes(post, delta, cap, widen_depth=True)
    assert np.all(new <= cap + 1e-12)


def test_suggest_search_delta_km_is_config_ordered():
    post = {"covariance_km": np.diag([9.0, 1.0, 4.0])}      # (up, south, east)
    north, east, depth = S.suggest_search_delta_km(post, k=6.0, min_km=(0, 0, 0))
    assert (north, east, depth) == (6.0, 12.0, 18.0)


def test_ellipsoid_orientation_vertical_major_axis():
    semi, plunge, _az, _rot = S.ellipsoid_orientation(np.diag([9.0, 1.0, 4.0]))
    assert list(semi) == pytest.approx([3.0, 2.0, 1.0])
    assert plunge == pytest.approx(90.0)


def test_ellipsoid_orientation_north_major_axis():
    semi, plunge, az, _rot = S.ellipsoid_orientation(np.diag([1.0, 9.0, 4.0]))
    assert list(semi) == pytest.approx([3.0, 2.0, 1.0])
    assert plunge == pytest.approx(0.0, abs=1e-9)
    assert min(az % 180.0, 180.0 - az % 180.0) == pytest.approx(0.0, abs=1e-6)


def test_ellipsoid_orientation_rejects_cartesian():
    with pytest.raises(NotImplementedError):
        S.ellipsoid_orientation(np.eye(3), "cartesian")
