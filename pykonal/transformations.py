"""
A module to facilitate coordinate-system transformations.
"""

import numpy as np
from . import constants


def geo2sph(nodes):
    """
    Transform geographical coordinates to new spherical coordinates system.

    Latitude is GEODETIC (the WGS84 convention that station files, SeisComP
    inventories and QuakeML all use); colatitude on the grid is GEOCENTRIC.
    Treating them as the same quantity is a latitude-dependent shift of up
    to 0.192 deg (21.4 km at 45 deg!) so the conversion is applied here asap.
    """
    geo = np.array(nodes, dtype=constants.DTYPE_REAL)
    sph = np.empty_like(geo)
    lat_gc = np.arctan((1.0 - constants.WGS84_E2) * np.tan(np.radians(geo[..., 0])))
    sph[..., 0] = constants.EARTH_RADIUS - geo[..., 2]
    sph[..., 1] = np.pi / 2 - lat_gc
    sph[..., 2] = np.radians(geo[..., 1])
    return sph


def sph2sph(nodes, origin=(0,0,0), force_phi_positive=False):
    """
    Transform spherical coordinates to new spherical coordinate system.

    :param nodes: Coordinates (spherical) to transform.
    :type nodes: numpy.ndarray(shape=(...,3), dtype=numpy.float)
    :param origin: Coordinates (spherical) of the origin of the new
                   coordinate system with respect to the old coordinate
                   system.
    :type origin: tuple(float, float, float)
    :return: Coordinates in new (spherical) coordinate system.
    :rtype: numpy.ndarray(shape=(...,3), dtype=numpy.float)
    """

    xx = nodes[...,0] * np.sin(nodes[...,1]) * np.cos(nodes[...,2])
    yy = nodes[...,0] * np.sin(nodes[...,1]) * np.sin(nodes[...,2])
    zz = nodes[...,0] * np.cos(nodes[...,1])
    x0 = origin[0] * np.sin(origin[1]) * np.cos(origin[2])
    y0 = origin[0] * np.sin(origin[1]) * np.sin(origin[2])
    z0 = origin[0] * np.cos(origin[1])
    xx -= x0
    yy -= y0
    zz -= z0
    xyz = np.moveaxis(np.stack([xx, yy, zz]), 0, -1)
    rr  = np.sqrt(np.sum(np.square(xyz), axis=-1))
    tt = np.arctan2(np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2])
    pp  = np.arctan2(xyz[...,1], xyz[...,0])
    pp = np.mod(pp, 2*np.pi)
    if force_phi_positive is False:
        np.mod(pp, -np.pi, pp, where=pp>np.pi)
    rtp = np.moveaxis(np.stack([rr, tt, pp]), 0, -1)
    return rtp


def xyz2sph(nodes, origin=(0,0,0), force_phi_positive=False):
    """
    Transform Cartesian coordinates to new spherical coordinate system.

    :param nodes: Coordinates (Cartesian) to transform.
    :type nodes: numpy.ndarray(shape=(...,3), dtype=numpy.float)
    :param origin: Coordinates (Cartesian) of the origin of the new
                   coordinate system with respect to the old coordinate
                   system.
    :type origin: tuple(float, float, float)
    :return: Coordinates in new (spherical) coordinate system.
    :rtype: numpy.ndarray(shape=(...,3), dtype=numpy.float)
    """
    xyz = nodes - origin
    rr  = np.sqrt(np.sum(np.square(xyz), axis=-1))
    #old = np.seterr(divide='ignore', invalid='ignore')
    #tt  = np.arccos(xyz[...,2] / rr)
    #np.seterr(**old)
    tt = np.arctan2(np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2])
    pp  = np.arctan2(xyz[...,1], xyz[...,0])
    if force_phi_positive:
        pp = np.mod(pp, 2*np.pi)
    rtp = np.moveaxis(np.stack([rr, tt, pp]), 0, -1)
    return rtp


def sph2geo(nodes):
    """
    Transform spherical coordinates to new geographical coordinate system.

    Inverse of geo2sph: grid colatitude is geocentric, returned latitude is
    geodetic.
    """
    sph = np.array(nodes, dtype=constants.DTYPE_REAL)
    geo = np.empty_like(sph)
    lat_gc = np.pi / 2 - sph[..., 1]
    geo[..., 0] = np.degrees(np.arctan(np.tan(lat_gc) / (1.0 - constants.WGS84_E2)))
    # canonical geographic convention: [-180, 180), regardless of whether
    # the grid's phi axis is [-pi, pi) or [0, 2*pi). Without this, a
    # [0, 2*pi) grid emits longitudes up to 360 into QuakeML/SeisComP.
    geo[..., 1] = np.mod(np.degrees(sph[..., 2]) + 180.0, 360.0) - 180.0
    geo[..., 2] = constants.EARTH_RADIUS - sph[..., 0]
    return geo


def sph2xyz(nodes, origin=(0,0,0)):
    """
    Transform spherical coordinates to new Cartesian coordinate system.

    :param nodes: Coordinates (spherical) to transform.
    :type nodes: numpy.ndarray(shape=(...,3), dtype=numpy.float)
    :param origin: Coordinates (spherical) of the origin of the new
                   coordinate system with respect to the old coordinate
                   system.
    :type origin: tuple(float, float, float)
    :return: Coordinates in new (Cartesian) coordinate system.
    :rtype: numpy.ndarray(shape=(...,3), dtype=numpy.float)
    """
    xx  = nodes[...,0] * np.sin(nodes[...,1]) * np.cos(nodes[...,2])
    yy  = nodes[...,0] * np.sin(nodes[...,1]) * np.sin(nodes[...,2])
    zz  = nodes[...,0] * np.cos(nodes[...,1])
    origin = [
        origin[0] * np.sin(origin[1]) * np.cos(origin[2]),
        origin[0] * np.sin(origin[1]) * np.sin(origin[2]),
        origin[0] * np.cos(origin[1])
    ]
    xx -= origin[0]
    yy -= origin[1]
    zz -= origin[2]
    xyz = np.moveaxis(np.stack([xx, yy, zz]), 0, -1)
    return xyz


def rotation_matrix(alpha, beta, gamma):
    """
    Rotation matrix used to rotate a Set of cartesian coordinates.

    The rotation matrix is defined such that coordinates are rotated by
    alpha radians about the z-axis, then beta radians about the y'-axis
    and then gamma radians about the z''-axis.

    :param alpha: Angle to rotate about the z-axis.
    :type alpha: float
    :param beta: Angle to rotate about the y'-axis.
    :type beta: float
    :param gamma: Angle to rotate about the z''-axis.
    :type gamma: float
    :return: Rotation matrix.
    :rtype: numpy.ndarray(shape=(3,3), dtype=numpy.float)
    """
    aa = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0           ],
            [np.sin(alpha),  np.cos(alpha), 0           ],
            [0,              0,             1           ]
        ]
    )
    bb = np.array(
        [
            [ np.cos(beta), 0,              np.sin(beta)],
            [ 0,            1,              0           ],
            [-np.sin(beta), 0,              np.cos(beta)]
        ]
    )
    cc = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0           ],
            [np.sin(gamma),  np.cos(gamma), 0           ],
            [0,              0,             1           ]
        ]
    )
    return (aa.dot(bb).dot(cc))


## Below is for the SeisComP/EQ Object locaters

def delta2sph(nodes, half_widths_km):
    """
    Convert search-box half-widths in km to spherical grid units at `nodes`.

    `half_widths_km` is (north_km, east_km, depth_km); the return is in
    GRID-AXIS order (dr, dtheta, dphi), matching geo2sph output.

    The angular steps need the local metric: north ground distance is
    r*dtheta, east is r*sin(theta)*dphi. Dropping the sin(theta) makes the
    east-west box narrower than requested by 1/sin(theta) -- 14.5% at
    latitude -31.4, 29% at -45 -- which silently rescales every half-width
    the caller specified in km.

    `nodes` is in SPHERICAL grid coordinates (as returned by geo2sph), so
    theta is geocentric and r is EARTH_RADIUS - depth. Both matter: using
    EARTH_RADIUS instead of r is a 0.3% error for a 20 km source, and using
    geodetic latitude instead of geocentric is a 0.2 deg error.
    """
    sph = np.asarray(nodes, dtype=constants.DTYPE_REAL)
    north_km, east_km, depth_km = (float(v) for v in half_widths_km)
    r = float(sph[..., 0])
    sin_theta = max(float(np.sin(sph[..., 1])), 1e-6)
    return np.array([depth_km, north_km / r, east_km / (r * sin_theta)],
                    dtype=constants.DTYPE_REAL)


def geo_distance_azimuth(lat1, lon1, lat2, lon2):
    """
    Epicentral distance and azimuth between two geographic points, measured
    on the traveltime GRID sphere: geodetic latitude converted to geocentric
    (as geo2sph does), radius EARTH_RADIUS.

    This is the metric the grids themselves use -- the same great-circle
    formula inventory.add applies when it crops a grid to max_dist -- so a
    distance compared with max_dist_km here agrees with what was stored.
    It is within ~0.05% of the WGS84 geodesic (0.5 km at 1000 km), whereas a
    great circle on GEODETIC latitudes is off by up to ~0.4% (4 km at 1000 km
    north-south at 29 deg S).

    Returns (distance_km, distance_deg, azimuth_deg), azimuth clockwise from
    north at point 1.
    """
    p1 = np.arctan((1.0 - constants.WGS84_E2) * np.tan(np.radians(float(lat1))))
    p2 = np.arctan((1.0 - constants.WGS84_E2) * np.tan(np.radians(float(lat2))))
    dl = np.radians(float(lon2) - float(lon1))
    h = (np.sin((p2 - p1) / 2.0) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2)
    arc = 2.0 * np.arcsin(np.sqrt(min(max(float(h), 0.0), 1.0)))
    az = np.degrees(np.arctan2(np.sin(dl) * np.cos(p2),
                               np.cos(p1) * np.sin(p2)
                               - np.sin(p1) * np.cos(p2) * np.cos(dl))) % 360.0
    return (float(arc * constants.EARTH_RADIUS), float(np.degrees(arc)),
            float(az))


def geo_destination(lat1, lon1, azimuth_deg, distance_km):
    """
    Forward counterpart to geo_distance_azimuth: the destination point
    reached from (lat1, lon1) by travelling distance_km along azimuth_deg
    (clockwise from north), on the SAME traveltime-grid sphere
    (geodetic->geocentric latitude, radius EARTH_RADIUS) that
    geo_distance_azimuth uses -- so a round trip through both functions is
    self-consistent to numerical precision.

    Returns (lat2, lon2), geodetic degrees.
    """
    p1 = np.arctan((1.0 - constants.WGS84_E2) * np.tan(np.radians(float(lat1))))
    lon1_r = np.radians(float(lon1))
    az = np.radians(float(azimuth_deg))
    d_r = float(distance_km) / constants.EARTH_RADIUS

    p2 = np.arcsin(np.sin(p1) * np.cos(d_r) + np.cos(p1) * np.sin(d_r) * np.cos(az))
    lon2 = lon1_r + np.arctan2(
        np.sin(az) * np.sin(d_r) * np.cos(p1),
        np.cos(d_r) - np.sin(p1) * np.sin(p2),
    )

    # p2 is geocentric -- convert back to geodetic, same inverse sph2geo uses
    lat2 = np.degrees(np.arctan(np.tan(p2) / (1.0 - constants.WGS84_E2)))
    lon2 = np.mod(np.degrees(lon2) + 180.0, 360.0) - 180.0
    return float(lat2), float(lon2)


class GeoTransform:
    """
    Geographic (lat, lon, depth_km) <-> pykonal spherical grid coordinates.

    The single implementation used by every driver. It previously lived in
    pykonal_locext.py, where a second, subtly different copy existed in the
    ObsPy relocator -- they disagreed on the east-west metric.
    """

    coord_sys = "spherical"

    def geo_to_grid(self, lat, lon, depth_km):
        return geo2sph(np.array([lat, lon, depth_km]))

    def grid_to_geo(self, coords):
        geo = sph2geo(np.atleast_2d(np.asarray(coords, dtype=float)))[0]
        return float(geo[0]), float(geo[1]), float(geo[2])

    def distance_azimuth(self, lat1, lon1, lat2, lon2):
        """(distance_km, distance_deg, azimuth_deg) on the grid sphere; see
        geo_distance_azimuth."""
        return geo_distance_azimuth(lat1, lon1, lat2, lon2)

    def delta_to_grid(self, dlat_km, dlon_km, ddep_km, at=None):
        """`at` is the spherical grid position the metric is evaluated at.
        Required -- there is no position-independent answer."""
        if at is None:
            raise ValueError(
                "delta_to_grid needs `at`: the km->radian conversion depends "
                "on where on the sphere the box sits")
        return delta2sph(at, (dlat_km, dlon_km, ddep_km))

    def destination(self, lat1, lon1, azimuth_deg, distance_km):
        """(lat2, lon2) reached from (lat1, lon1) along azimuth_deg for
        distance_km; see geo_destination."""
        return geo_destination(lat1, lon1, azimuth_deg, distance_km)

