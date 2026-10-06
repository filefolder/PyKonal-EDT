"""Regression tests: the shared module must reproduce the numbers from the
two events we diagnosed from real logs."""
import numpy as np, math
import sys as _s; _s.path.insert(0,"fake")
import pykonal.sizing as S

R = 6371.0

class SphTransform:
    coord_sys = "spherical"
    def geo_to_grid(self, lat, lon, dep):
        return np.array([R-dep, math.pi/2-math.radians(lat), math.radians(lon)])
    def grid_to_geo(self, xyz):
        return np.array([math.degrees(math.pi/2-xyz[1]),
                         math.degrees(xyz[2]), R-xyz[0]])
    def delta_to_grid(self, lat_km, lon_km, dep_km, at=None):
        if at is None:
            raise ValueError("delta_to_grid needs `at`")
        r = float(at[0]); sin_t = max(math.sin(float(at[1])), 1e-6)
        return np.array([dep_km, lat_km/r, lon_km/(r*sin_t)])

class CartTransform(SphTransform):
    coord_sys = "cartesian"
    def geo_to_grid(self, lat, lon, dep): return np.array([0.0, 0.0, dep])
    def grid_to_geo(self, xyz): return np.array([0.0, 0.0, xyz[2]])
    def delta_to_grid(self, a,b,c, at=None): return np.array([a,b,c])

LOG=[]
def log(m): LOG.append(m)

print("=== chi_scale matches the documented values ===")
for lvl,exp in [(68.27,(1.000,1.515,1.878)),(95.0,(1.960,2.448,2.788))]:
    got=tuple(round(S.chi_scale(lvl,d),3) for d in (1,2,3))
    print(f"  {lvl}%: {got}"); assert all(abs(a-b)<0.002 for a,b in zip(got,exp))

print("\n=== axis ordering: config [lat,lon,dep] -> grid order ===")
assert list(S._config_to_grid_order([10.,10.,20.],"spherical"))==[20.,10.,10.]
assert list(S._config_to_grid_order([10.,10.,20.],"cartesian"))==[10.,10.,20.]
assert S.depth_axis("spherical")==0 and S.depth_axis("cartesian")==2
print("  ok for both coordinate systems")

print("\n=== event 2 (-27.12, 128.75): floors and cap reproduce the log ===")
# from the log: k*sigma gave 50.4/9.3/7.4 km (grid order dep,lat,lon)
sigma = np.array([12.60, 2.33, 1.85]); hw = 4.0*sigma
cfg = {"auto_search_max_km":[50.,50.,40.], "auto_search_min_km":[10.,10.,20.],
       "auto_search_depth_dist_factor":0.5}
LOG.clear()
out = S.apply_floors_and_cap(hw, sigma, "spherical", cfg,
                             provisional_offset_km=np.array([0.,0.,0.]),
                             min_station_dist_km=236.6, log=log)
print(f"  k*sigma      = {np.round(hw,1)}")
print(f"  after floors+cap = {np.round(out,1)}")
for m in LOG: print("   ", m[:96])
assert abs(hw[0]-50.4)<0.05, hw
assert abs(out[0]-40.0)<1e-9, "depth not capped at auto_search_max_km"
assert any("floored on depth" in m for m in LOG)
assert any("capped by auto_search_max_km" in m for m in LOG)

print("\n=== the same event, depth clamp uses the CAPPED half-width ===")
t=SphTransform(); dep0=23.9
initial=np.append(t.geo_to_grid(-27.1264,128.748,dep0), 0.0)
delta=np.append(t.delta_to_grid(out[1],out[2],out[0], at=initial[:3]), 5.0)
LOG.clear()
b=S.clamp_depth_to_bounds(t,initial,delta,dep0,-1.0,80.0,-27.1264,128.748,log=log)
d_lo, d_hi = R-b[1][0], R-b[0][0]
print(f"  interval [{d_lo:.1f}, {d_hi:.1f}] km   (was [-1.0, 80.0] with the bug)")
for m in LOG: print("   ", m[:96])
assert abs(d_lo+1.0)<1e-6 and abs(d_hi-63.9)<1e-6

print("\n=== event 1 (-31.39, 121.82): anchor is never recentred ===")
dep0=1.5
initial=np.append(t.geo_to_grid(-31.3882,121.8168,dep0), 0.0)
delta=np.append(t.delta_to_grid(6.0,6.7,11.4, at=initial[:3]), 5.0)
b=S.clamp_depth_to_bounds(t,initial,delta,dep0,-3.0,600.0,-31.3882,121.8168)
d_lo,d_hi = R-b[1][0], R-b[0][0]
centre=0.5*(d_lo+d_hi)
print(f"  interval [{d_lo:.2f}, {d_hi:.2f}] km, centre would be {centre:.2f} km")
print(f"  anchor {dep0} km still inside the bounds and NOT moved")
assert b[0][0] <= initial[0] <= b[1][0]
assert abs(d_lo+3.0)<1e-6 and abs(d_hi-12.9)<1e-6

print("\n=== fixed depth: zero-width bound, posterior extent preserved ===")
for pin in (0.0, 5.0, 23.9):
    box=S.size_fixed_box(t, np.append(t.geo_to_grid(-31.4,121.8,pin),0.0),
                         5.0, [25.,25.,25.], pin, fixed_depth=pin,
                         fixed_depth_tolerance=2.0,
                         depth_min=-3.0, depth_max=80.0)
    d_lo,d_hi = R-box.bounds[1][0], R-box.bounds[0][0]
    print(f"  pin {pin:5.1f} -> location delta[dep]={box.delta_grid[0]:.4f} km, "
          f"bounds width={d_hi-d_lo:.4f} km, posterior delta[dep]="
          f"{box.posterior_delta[0]:.1f} km")
    assert box.depth_pinned and abs(box.delta_grid[0]-2.0)<1e-9
    assert abs(box.posterior_delta[0]-25.0)<1e-9, "posterior depth extent lost"
    assert abs(0.5*(d_lo+d_hi)-pin)<1e-3
    assert abs((d_hi-d_lo)-4.0)<1e-6      # +/-2 km tolerance, honoured

print("\n=== cartesian path ===")
ct=CartTransform()
box=S.size_fixed_box(ct, np.array([0.,0.,10.,0.]), 5.0, [25.,25.,25.], 10.0,
                     fixed_depth=10.0, fixed_depth_tolerance=2.0,
                     depth_min=0.0, depth_max=50.0)
assert abs(box.delta_grid[2]-2.0)<1e-9, "cartesian depth axis not pinned"
assert abs(box.posterior_delta[2]-25.0)<1e-9
print(f"  delta={np.round(box.delta_grid,4)} posterior={np.round(box.posterior_delta,1)}")

print("\nall sizing regression tests passed")
