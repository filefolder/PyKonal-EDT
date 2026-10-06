import sys, types, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# stand in for pykonal.fields.read_hdf
class Field:
    def __init__(self, minc, node, npts):
        self.min_coords, self.node_intervals, self.npts = minc, node, npts
fake_fields = types.ModuleType("fields")
MODELS = {}
fake_fields.read_hdf = lambda p: MODELS[p]

import pykonal.inventory as S
# the module does `from . import fields`; emulate for the standalone copy
S.__dict__["_test_fields"] = fake_fields
_orig = S.grid_extents
def grid_extents(velocity_models, log=None):
    out = {}
    for phase, vm in velocity_models.items():
        f = MODELS[vm]
        minc = np.asarray(f.min_coords, float); npts = np.asarray(f.npts)
        node = np.asarray(f.node_intervals, float)
        out[phase.upper()] = (minc, minc + (npts-1)*node)
    return out
S.grid_extents = grid_extents

R = 6371.0
# spherical grid: axis 0 = radius. Top of model is the datum (r = R).
# min_coords r = R-50 km (50 km deep), 40 nodes of 1.282 km -> top at R.
MODELS["vp.h5"] = Field(np.array([R-50.0, 1.0, 2.0]),
                        np.array([50.0/39, 0.01/39, 0.02/39]),
                        np.array([40, 40, 40]))
models = {"P": "vp.h5"}
LOG=[]; log=LOG.append

print("=== station above the datum (300 m elevation -> r = R+0.3) ===")
req = {("AU","KMBL","P"): np.array([R+0.3, 1.005, 2.01])}
LOG.clear()
out = S.fit_requests_to_grid(req, models, vertical_axis=0, log=log)
c = out[("AU","KMBL","P")]
print(f"  r before={R+0.3:.4f}  after={c[0]:.4f}  (grid top={R:.4f})")
for m in LOG: print("   LOG:", m[:110])
assert abs(c[0]-R) < 1e-9, "elevation not clamped to the grid top"
assert any("vertically" in m for m in LOG), "vertical clamp was silent"

print("\n=== station inside the grid: untouched, no warning ===")
req = {("AU","FORT","P"): np.array([R-10.0, 1.005, 2.01])}
LOG.clear()
out = S.fit_requests_to_grid(req, models, vertical_axis=0, log=log)
print(f"  r={out[('AU','FORT','P')][0]:.4f}  warnings={len(LOG)}")
assert LOG == []

print("\n=== station laterally outside: excluded, warned ===")
req = {("XX","FAR","P"): np.array([R-5.0, 9.9, 2.01]),
       ("AU","FORT","P"): np.array([R-10.0, 1.005, 2.01])}
LOG.clear()
out = S.fit_requests_to_grid(req, models, vertical_axis=0, log=log)
print(f"  kept={sorted('.'.join(k) for k in out)}")
for m in LOG: print("   LOG:", m[:110])
assert ("XX","FAR","P") not in out
assert any("laterally" in m for m in LOG)

print("\n=== None coords pass through (no station location) ===")
out = S.fit_requests_to_grid({("ZZ","NONE","P"): None}, models, 0, log=log)
assert out[("ZZ","NONE","P")] is None
print("  preserved as None for ensure_traveltimes to skip")

print("\n=== caller's dict is not mutated ===")
req = {("AU","KMBL","P"): np.array([R+0.3, 1.005, 2.01])}
before = req[("AU","KMBL","P")].copy()
S.fit_requests_to_grid(req, models, 0)
assert np.array_equal(req[("AU","KMBL","P")], before)
print("  input untouched")

print("\nall clamp tests passed")
