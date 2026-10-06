import sys; sys.path.insert(0,"fake")
import numpy as np, math
from pykonal import pykonal_obspy as P
import pykonal.sizing as S
from pykonal.transformations import GeoTransform

print("=== depth-bound resolution ===")
for fd,dmin,dmax in [(None,-3,80),(5.0,-3,80),(0.0,None,None),(None,None,None)]:
    print("  ",(fd,dmin,dmax),"->",P._resolve_depth_bounds(fd,dmin,dmax))
assert P._resolve_depth_bounds(5.0,-3,80) == (5.0,5.0,True)

print("\n=== depth axis / grid ordering ===")
assert S.depth_axis("spherical")==0 and S.depth_axis("cartesian")==2
T = GeoTransform()
g = T.geo_to_grid(-31.4,121.8,10.0)
lat,lon,dep = T.grid_to_geo(g)
print(f"   round-trip: lat {lat:+.6f} lon {lon:+.6f} dep {dep:.6f}")
assert abs(lat+31.4)<1e-6 and abs(lon-121.8)<1e-6 and abs(dep-10.0)<1e-6

d = T.delta_to_grid(25.0, 25.0, 12.0, at=g)
print(f"   delta grid (dep_km, dtheta_rad, dphi_rad) = "
      f"{d[0]:.3f}, {d[1]:.6f}, {d[2]:.6f}")
r, theta = g[0], g[1]
assert abs(d[0]-12.0)<1e-9
assert abs(d[1]*r - 25.0)<1e-6                        # north km
assert abs(d[2]*r*math.sin(theta) - 25.0)<1e-6        # east km
print("   km -> grid conversion checks out on both horizontal axes")

print("\n=== depth clamp keeps the anchor, collapses when pinned ===")
for pin in (0.0, 5.0, 23.9):
    init = np.append(T.geo_to_grid(-31.4,121.8,pin), 0.0)
    delt = np.append(T.delta_to_grid(25.,25.,1e-3, at=init[:3]), 5.0)
    delt[0] = 1e-3
    b = S.clamp_depth_to_bounds(GeoTransform(), init, delt,
                                pin, pin, pin, -31.4, 121.8,
                                min_half_width_km=2.0)
    dlo = 6371.0-b[1][0]; dhi = 6371.0-b[0][0]
    print(f"   pin {pin:5.1f} -> depth interval [{dlo:.4f}, {dhi:.4f}] km")
    assert abs(0.5*(dlo+dhi)-pin) < 1e-3

print("\n   free depth, clamp truncates the shallow side only:")
init = np.append(T.geo_to_grid(-31.4,121.8,1.5), 0.0)
delt = np.append(T.delta_to_grid(25.,25.,25., at=init[:3]), 5.0)
b = S.clamp_depth_to_bounds(GeoTransform(), init, delt,
                            1.5, -3.0, 80.0, -31.4, 121.8)
dlo, dhi = 6371.0-b[1][0], 6371.0-b[0][0]
anchor_grid = init[0]
print(f"   interval [{dlo:.2f}, {dhi:.2f}] km; anchor 1.50 km still inside "
      f"and NOT recentred (centre would be {0.5*(dlo+dhi):.2f})")
assert abs(dlo+3.0)<1e-6 and abs(dhi-26.5)<1e-6
assert b[0][0] <= anchor_grid <= b[1][0]

print("\n=== horizontal uncertainty frame (north = -theta) ===")
cov = np.diag([4.0, 1.0, 9.0]).astype(float)   # (r, theta, phi) = up,south,east
sn,se,smaj,smin,az = P._horizontal_uncertainty(cov,"spherical")
print(f"   sigma_n={sn:.2f} sigma_e={se:.2f} major={smaj:.2f} minor={smin:.2f} az={az:.1f}")
assert abs(sn-1.0)<1e-9 and abs(se-3.0)<1e-9
assert abs(smaj-3.0)<1e-9 and abs(az-90.0)<1e-6    # major axis points east
cov2 = np.diag([4.0, 9.0, 1.0]).astype(float)
sn,se,smaj,smin,az = P._horizontal_uncertainty(cov2,"spherical")
assert abs(smaj-3.0)<1e-9 and (abs(az)<1e-6 or abs(az-180)<1e-6)
print(f"   north-elongated case: major={smaj:.2f} az={az:.1f} (north). OK")
print("\nall unit checks passed")
