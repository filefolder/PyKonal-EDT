"""Exercise the sharded builder with a stubbed ensure_traveltimes."""
import os, sys, shutil, time
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pykonal.inventory as S
from pykonal import pykonal_obspy as P

CALLS = []
def fake_ensure_traveltimes(path, requests, velocity_models, max_dist=None,
                            nproc=None, **kw):
    """Write a minimal grid shaped like the real one."""
    (key, coords), = requests.items()
    CALLS.append(key)
    vals = np.arange(27, dtype=float).reshape(3,3,3)
    if key[1] == "BAD":                      # inject a corrupt patch
        vals[1,1,1] = -np.inf
        vals[0,0,1] = -5.0
    with h5py.File(path, "w") as f5:
        g = f5.create_group("/".join(key))
        g.create_dataset("values", data=vals)
    return {"present": [], "computed": [key], "skipped": []}
S.ensure_traveltimes = fake_ensure_traveltimes

LOG = []
def log(m): LOG.append(m)

D = "/tmp/shards"
shutil.rmtree(D, ignore_errors=True)
VM = "/tmp/vp.h5"
open(VM, "w").write("model v1")
models = {"P": VM, "S": VM}
req = {("AU","KMBL","P"): np.array([1.,2.,3.]),
       ("AU","KMBL","S"): np.array([1.,2.,3.]),
       ("WG","BAD","P"):  np.array([4.,5.,6.]),
       ("XX","NOCOORD","P"): None}

print("=== first build ===")
LOG.clear(); CALLS.clear()
r = S.ensure_traveltimes_sharded(D, req, models, nproc=1, log=log)
print(f"  computed={len(r['computed'])} present={len(r['present'])} "
      f"skipped={len(r['skipped'])}")
print(f"  master={os.path.basename(r['master'])}")
for m in LOG: print("   LOG:", m[:92])
assert len(r["computed"]) == 3 and r["skipped"] == [("XX","NOCOORD","P")]
assert any("impossible traveltime node" in m for m in LOG), "corrupt patch not reported"

print("\n=== corrupt nodes were masked to NaN ===")
with h5py.File(S.shard_path(D, ("WG","BAD","P"))) as f5:
    v = f5["WG/BAD/P"]["values"][...]
print(f"  NaN count={np.isnan(v).sum()}  any -inf={np.isneginf(v).any()}  "
      f"any negative={np.nanmin(v) < 0}")
assert np.isnan(v).sum() == 2 and not np.isneginf(v).any() and np.nanmin(v) >= 0

print("\n=== second run: everything present, nothing rebuilt ===")
LOG.clear(); CALLS.clear()
r = S.ensure_traveltimes_sharded(D, req, models, nproc=1, log=log)
print(f"  computed={len(r['computed'])} present={len(r['present'])} "
      f"solver calls={len(CALLS)}")
assert len(r["present"]) == 3 and CALLS == []

print("\n=== velocity model edited -> stale grids rebuilt ===")
time.sleep(1.1)
open(VM, "w").write("model v2 which is longer")
LOG.clear(); CALLS.clear()
r = S.ensure_traveltimes_sharded(D, req, models, nproc=1, log=log)
print(f"  computed={len(r['computed'])} solver calls={len(CALLS)}")
for m in LOG[:3]: print("   LOG:", m[:92])
assert len(r["computed"]) == 3, "stale grids were not rebuilt"
assert any("superseded" in m for m in LOG)

print("\n=== master index resolves through external links ===")
with h5py.File(r["master"], "r") as f5:
    keys = []
    f5.visit(lambda n: keys.append(n))
    print("  groups:", [k for k in keys if k.count('/')==2])
    val = f5["AU/KMBL/P"]["values"][0,0,0]
print(f"  AU/KMBL/P values[0,0,0] read through the link = {val}")
assert val == 0.0

print("\n=== links are relative: directory can be moved ===")
shutil.rmtree("/tmp/shards_moved", ignore_errors=True)
shutil.move(D, "/tmp/shards_moved")
with h5py.File("/tmp/shards_moved/inventory.h5", "r") as f5:
    v = f5["AU/KMBL/S"]["values"][2,2,2]
print(f"  after move, AU/KMBL/S values[2,2,2] = {v}")
assert v == 26.0

print("\n=== parallel path ===")
shutil.rmtree("/tmp/shards_par", ignore_errors=True)
LOG.clear()
r = S.ensure_traveltimes_sharded("/tmp/shards_par", req, models, nproc=2, log=log)
print(f"  computed={len(r['computed'])} (nproc=2)")
assert len(r["computed"]) == 3

print("\n=== lock is usable and re-entrant across sequential uses ===")
with S.inventory_lock("/tmp/shards_par/inventory.h5", shared=True):
    pass
with S.inventory_lock("/tmp/shards_par/inventory.h5", shared=False):
    pass
print("  shared and exclusive locks acquire and release")

print("\nall sharding tests passed")
