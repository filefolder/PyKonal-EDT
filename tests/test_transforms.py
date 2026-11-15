"""Coordinate transforms and the local km frame used for uncertainties."""
import math

import numpy as np
import pytest

from pykonal import pykonal_obspy as P
from pykonal import transformations as tr
from pykonal.transformations import GeoTransform

R = 6371.0


@pytest.mark.parametrize("lat, lon, dep", [(-31.4, 121.8, 10.0), (52.5, -3.2, 0.0),
                                           (0.0, 179.9, 600.0), (-89.0, 10.0, 5.0)])
def test_geo_grid_round_trip(lat, lon, dep):
    T = GeoTransform()
    back = T.grid_to_geo(T.geo_to_grid(lat, lon, dep))
    assert back == pytest.approx((lat, lon, dep), abs=1e-6)


def test_grid_axes_are_radius_colatitude_longitude():
    g = GeoTransform().geo_to_grid(-30.0, 120.0, 10.0)
    assert g[0] == pytest.approx(R - 10.0)
    assert g[2] == pytest.approx(math.radians(120.0))
    # colatitude is geocentric: within the ~0.19 deg geodetic offset of 120
    assert abs(math.degrees(g[1]) - 120.0) < 0.2
    assert math.degrees(g[1]) < 120.0      # geocentric |lat| < geodetic |lat|


def test_delta_to_grid_is_km_on_every_axis():
    T = GeoTransform()
    g = T.geo_to_grid(-31.4, 121.8, 10.0)
    d = T.delta_to_grid(25.0, 20.0, 12.0, at=g)
    assert d[0] == pytest.approx(12.0)
    assert d[1] * g[0] == pytest.approx(25.0)                    # north km
    assert d[2] * g[0] * math.sin(g[1]) == pytest.approx(20.0)   # east km


def test_delta_to_grid_needs_a_position():
    with pytest.raises(ValueError):
        GeoTransform().delta_to_grid(1.0, 1.0, 1.0)


def test_distance_azimuth_and_destination_are_inverse():
    T = GeoTransform()
    lat2, lon2 = T.destination(-30.0, 138.0, 60.0, 250.0)
    dist_km, dist_deg, az = T.distance_azimuth(-30.0, 138.0, lat2, lon2)
    assert dist_km == pytest.approx(250.0, rel=1e-6)
    assert dist_deg == pytest.approx(250.0 / tr.constants.EARTH_RADIUS * 180 / math.pi,
                                     rel=1e-6)
    assert az == pytest.approx(60.0, abs=1e-6)


def test_due_north_and_east_azimuths():
    T = GeoTransform()
    assert T.distance_azimuth(0.0, 0.0, 1.0, 0.0)[2] == pytest.approx(0.0, abs=1e-9)
    assert T.distance_azimuth(0.0, 0.0, 0.0, 1.0)[2] == pytest.approx(90.0, abs=1e-9)


# ------------------------------------------- horizontal uncertainty frame
# The spherical km frame is (up, south, east): north is -theta.
def test_horizontal_uncertainty_east_elongated():
    sn, se, smaj, smin, az = P._horizontal_uncertainty(np.diag([4.0, 1.0, 9.0]))
    assert (sn, se, smaj, smin) == pytest.approx((1.0, 3.0, 3.0, 1.0))
    assert az == pytest.approx(90.0)


def test_horizontal_uncertainty_north_elongated():
    *_, smaj, smin, az = P._horizontal_uncertainty(np.diag([4.0, 9.0, 1.0]))
    assert (smaj, smin) == pytest.approx((3.0, 1.0))
    assert min(az, 180.0 - az) == pytest.approx(0.0, abs=1e-6)


def test_horizontal_uncertainty_northeast_sign():
    # +cov(south, east) means the ellipse runs south-east / north-west
    cov = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 1.5], [0.0, 1.5, 2.0]])
    *_, az = P._horizontal_uncertainty(cov)
    assert az == pytest.approx(135.0)
    cov[1, 2] = cov[2, 1] = -1.5
    *_, az = P._horizontal_uncertainty(cov)
    assert az == pytest.approx(45.0)
