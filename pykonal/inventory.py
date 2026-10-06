"""
A module to handle the creation/population of velocity and traveltime hdf5 files.
"""

import h5py
import numpy as np
import os
import warnings
import contextlib

from . import constants
from . import fields

# Default storage settings for traveltime values, same pipeline Field3D.to_hdf uses
DEFAULT_COMPRESS_KWARGS = fields.DEFAULT_COMPRESS_KWARGS

class TraveltimeInventory(object):

    def __init__(self, path, mode="r"):
        self._mode = mode
        self._path = path
        self._f5 = h5py.File(path, mode=mode)

    def close(self):
        if getattr(self, "_f5", None) is not None:
            self._f5.close()
            self._f5 = None

    def __del__(self):
        self.close()

    def __exit__(self, *exc):
        self.close()

    def __enter__(self):
        return self

    @property
    def f5(self):
        return self._f5

    @property
    def path(self):
        return self._path


    # merge essentially only used for PyVoroTomo to place all in one file
    def merge(self, paths, station_coords=None, max_dist=None,
              mask=True, compress=True, overwrite=True, compress_kwargs=None):
        """
        Merge externally built traveltime fields (files named
        NET.STA.PHASE.h5, root-level pykonal fields) into this inventory.

        station_coords: optional dict mapping (network, station) or
        "NET.STA" to coordinates in the fields' coordinate system,
        passed through to add() for robust distance limiting.
        max_dist is in km.
        overwrite: replace a key that already exists instead of raising.
        Note: merged grids carry no velocity_model_fingerprint, so they are
        exempt from the staleness rebuild -- delete them by hand if the
        model they came from changes.
        """
        for path in paths:
            _, filename = os.path.split(path)
            filename, _ = os.path.splitext(filename)
            parts = filename.split(".")
            if len(parts) != 3:
                raise ValueError(
                    f"{filename!r}: expected NET.STA.PHASE.h5 "
                    f"(got {len(parts)} dot-separated parts)")
            network, station, phase = parts
            coords = None
            if station_coords is not None:
                coords = station_coords.get(
                    (network, station),
                    station_coords.get(f"{network}.{station}")
                )
            key = "/".join([network, station, phase])
            if overwrite and key in self.f5:
                del self.f5[key]
            field = fields.read_hdf(path)
            self.add(
                field, key,
                station_coords=coords,
                max_dist=max_dist, mask=mask, compress=compress,
                compress_kwargs=compress_kwargs
            )

        return True


    def add(self, field, key, station_coords=None, max_dist=None,
            mask=True, compress=True, compress_kwargs=None):
        """
        Store a traveltime field, by default distance-limited to nodes
        within max_dist (km) epicentral distance of the station.

        Parameters
        ----------
        field : ScalarField3D
            Traveltime field to store.
        key : str
            HDF5 group key, e.g. "NET/STA/PHASE".
        station_coords : array-like, recommended
            Station location in the field's coordinate system (e.g. the
            solver's src_loc). Strongly recommended whenever max_dist is
            used. If None, it is inferred as the grid node with the
            minimum traveltime (with a warning); inference can misbehave
            on unusual grids, so passing it explicitly is more robust.
            A sanity check warns if the given coordinates lie more than
            3 grid nodes from the field's traveltime minimum (i.e. they
            do not look like the source of this field), and raises if
            they fall outside the grid.
        max_dist : float or None
            Epicentral (horizontal / great-circle) distance cutoff in km.
            The stored grid is cropped to the bounding box of this radius,
            and (if *mask* is True) nodes beyond the radius are set to NaN
            so the corners of the box compress away and locators treat
            them as no-coverage. None disables distance limiting.
        mask : bool
            Mask nodes beyond max_dist with NaN (default True).
        compress : bool
            Store values chunked with gzip compression (default True).
            Traveltime fields are smooth and typically compress 2-4x on
            top of the distance-limiting savings.
        compress_kwargs : dict or None
            Partial overrides of inventory.DEFAULT_COMPRESS_KWARGS, merged
            over the defaults (unspecified keys keep their default). E.g.
            {"compression_opts": 1} = faster gzip, same codec;
            {"compression": "lzf"} = much faster, larger.
            NOTE reader portability: the chosen filter is stored per dataset
            in the file, so mixed archives are always readable where the
            codec itself is available -- but non-HDF5-core codecs (zstd via
            hdf5plugin) must be importable in EVERY reader process.

        Coordinate conventions: "cartesian" fields are (x, y, z[depth]) in
        km with distance measured in the x-y plane; "spherical" fields are
        (r, theta[colatitude], phi[longitude]) with distance measured as
        great-circle arc length at the Earth's surface. The depth/radial
        axis is never cropped.
        """

        values = np.asarray(field.values, dtype=np.float64)
        min_coords = np.asarray(field.min_coords, dtype=np.float64)
        node_intervals = np.asarray(field.node_intervals, dtype=np.float64)
        npts = np.asarray(field.npts, dtype=np.int64)

        idx_start = np.array([0, 0, 0], dtype=np.int64)
        idx_end = npts.copy()

        if max_dist is not None:
            src_idx = np.array(np.unravel_index(np.nanargmin(values), values.shape))
            if station_coords is None:
                station_coords = min_coords + src_idx * node_intervals
                warnings.warn(
                    f"{key}: station_coords not provided; inferred "
                    f"{np.round(station_coords, 4)} from the traveltime "
                    f"minimum. Pass station_coords explicitly for "
                    f"robustness.",
                    stacklevel=2
                )
            station_coords = np.asarray(station_coords, dtype=np.float64)

            # sanity check: traveltime at the station should be ~0. Compare
            # against the time to cross a few nodes at the local apparent
            # velocity implied by neighboring values.
            idx = np.rint(
                (station_coords - min_coords) / node_intervals
            ).astype(np.int64)
            if np.any(idx < 0) or np.any(idx >= npts):
                raise ValueError(
                    f"{key}: station_coords {station_coords} fall outside "
                    f"the grid."
                )
            # sanity: the claimed station location should coincide with
            # the field's traveltime minimum to within a few grid nodes.
            # Comparing node indices (rather than traveltime values) is
            # immune to the near-source interpolation error that leaves
            # the coarse source node with a small nonzero time.
            offset = np.max(np.abs(idx - src_idx))
            if offset > 3:
                warnings.warn(
                    f"{key}: station_coords are {offset} grid nodes from "
                    f"the field's traveltime minimum (node {tuple(src_idx)} "
                    f"vs claimed {tuple(idx)}). The provided coordinates "
                    f"may not match this field's source; the distance "
                    f"limit may crop the wrong region.",
                    stacklevel=2
                )

            axes = [
                min_coords[iax] + np.arange(npts[iax]) * node_intervals[iax]
                for iax in range(3)
            ]

            if field.coord_sys == "cartesian":
                # crop x (axis 0) and y (axis 1) to +/- max_dist; keep z
                for iax in (0, 1):
                    lo = station_coords[iax] - max_dist
                    hi = station_coords[iax] + max_dist
                    idx_start[iax] = np.searchsorted(axes[iax], lo, "left")
                    idx_end[iax] = np.searchsorted(axes[iax], hi, "right")
            else:
                # spherical: angular radius, crop theta (axis 1) and
                # phi (axis 2); keep r (axis 0)
                dd = max_dist / constants.EARTH_RADIUS
                lo = station_coords[1] - dd
                hi = station_coords[1] + dd
                idx_start[1] = np.searchsorted(axes[1], lo, "left")
                idx_end[1] = np.searchsorted(axes[1], hi, "right")
                # widen the phi window by 1/sin(theta) at the extreme
                # colatitudes of the (cropped) theta range
                theta_lo = axes[1][idx_start[1]]
                theta_hi = axes[1][min(idx_end[1], npts[1]) - 1]
                sin_min = min(np.sin(theta_lo), np.sin(theta_hi))
                if sin_min * constants.EARTH_RADIUS > 1e-3:
                    dphi = dd / sin_min
                    phi_lo = station_coords[2] - dphi
                    phi_hi = station_coords[2] + dphi
                    if phi_lo < axes[2][0] or phi_hi > axes[2][-1]:
                        # window crosses the +/-pi seam: a plain crop would drop the
                        # in-coverage nodes on the far side. Keep the full phi axis;
                        # the haversine mask below handles wraparound and NaNs the
                        # out-of-radius nodes.
                        idx_start[2], idx_end[2] = 0, npts[2]
                    else:
                        idx_start[2] = np.searchsorted(axes[2], phi_lo, "left")
                        idx_end[2]   = np.searchsorted(axes[2], phi_hi, "right")

            idx_start = np.clip(idx_start, 0, npts - 1)
            idx_end = np.clip(idx_end, idx_start + 1, npts)

            slices = tuple(
                slice(idx_start[iax], idx_end[iax]) for iax in range(3)
            )
            values = np.array(values[slices], dtype=np.float32, order="C")
            min_coords = min_coords + idx_start * node_intervals
            npts = idx_end - idx_start
            axes = [
                min_coords[iax] + np.arange(npts[iax]) * node_intervals[iax]
                for iax in range(3)
            ]

            if mask:
                if field.coord_sys == "cartesian":
                    dx = axes[0] - station_coords[0]
                    dy = axes[1] - station_coords[1]
                    dist = np.sqrt(
                        dx[:, None] ** 2 + dy[None, :] ** 2
                    )  # (nx, ny)
                    values[dist > max_dist, :] = np.nan
                else:
                    # great-circle distance on the sphere
                    lat0 = np.pi / 2 - station_coords[1]
                    lat = np.pi / 2 - axes[1]
                    dlon = axes[2] - station_coords[2]
                    a = (
                        np.sin((lat[:, None] - lat0) / 2) ** 2
                        + np.cos(lat0)
                        * np.cos(lat)[:, None]
                        * np.sin(dlon[None, :] / 2) ** 2
                    )
                    gc = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
                    dist = gc * constants.EARTH_RADIUS  # (ntheta, nphi)
                    values[:, dist > max_dist] = np.nan

        group = self.f5.create_group(key)
        group.attrs["coord_sys"] = field.coord_sys
        group.attrs["field_type"] = field.field_type
        if max_dist is not None:
            group.attrs["max_dist"] = max_dist
            group.attrs["station_coords"] = station_coords

        # write the actual times only as float32
        values = np.asarray(values, dtype=np.float32)

        if compress:
            _kw = dict(DEFAULT_COMPRESS_KWARGS)
            if compress_kwargs:
                _kw.update(compress_kwargs)
            group.attrs["compression"] = str(_kw.get("compression"))
            group.create_dataset("values", data=values, chunks=True, **_kw)
        else:
            group.create_dataset("values", data=values)

        group.create_dataset("min_coords", data=min_coords)
        group.create_dataset("node_intervals", data=node_intervals)
        group.create_dataset("npts", data=npts)

        return True


    def has(self, key):
        """True if a traveltime field is stored under *key* (str or tuple)"""
        if not isinstance(key, str):
            key = "/".join(key)
        return key in self.f5


    def keys(self):
        found = []
        self.f5.visit(
            lambda name: found.append(name)
            if name.count("/") == 2 and "values" not in name else None
        )
        out = [k for k in found if f"{k}/values" in self.f5]
        if not out:
            # Sharded master: external links are invisible to visit(), so
            # enumerate the sibling shard files instead.
            base = os.path.dirname(self.path)
            if os.path.isdir(base):
                out = ["/".join(f[:-3].split("."))
                       for f in sorted(os.listdir(base))
                       if f.endswith(".h5") and f != "inventory.h5"
                       and "/".join(f[:-3].split(".")) in self.f5]
        return out


    def read(self, key, min_coords=None, max_coords=None):
        """Field stored under *key* ("NET/STA/PHASE"), optionally cropped."""
        return fields.read_hdf(self.f5[key], min_coords, max_coords)



def symmetrize_pole_rows(values, min_coords, node_intervals, npts):
    """
    A spherical grid's theta = 0 and theta = pi rows are each ONE physical
    point repeated npts[2] times. The FMM updates those nodes independently
    along theta, leaving the row inconsistent by stencil noise, which the
    ~1e-16 phi metric then amplifies. Collapse each pole row to its mean so
    the stored field is single-valued and phi-differences are exactly zero.
    No-op for grids whose theta range excludes the poles.
    """
    th = min_coords[1] + np.arange(int(npts[1])) * node_intervals[1]
    pole_rows = np.nonzero(np.abs(np.sin(th)) < 1e-12)[0]
    if pole_rows.size == 0:
        return values                    # no copy, no write
    v = np.asarray(values, dtype=np.float64).copy()
    for i1 in pole_rows:
        v[:, i1, :] = np.nanmean(v[:, i1, :], axis=1, keepdims=True)
    return v


def _solve_one_traveltime(args):
    """
    Multiprocessing worker: one FMM solve. Takes/returns only picklable
    plain data (no open HDF5 handles cross the process boundary).
    """
    key, coords, vm_spec, solver_kwargs = args
    from . import solver as _solver

    try:
        if isinstance(vm_spec, (str, os.PathLike)):
            vmodel = fields.read_hdf(vm_spec)
            vm_spec = dict(
                coord_sys=vmodel.coord_sys, min_coords=vmodel.min_coords,
                node_intervals=vmodel.node_intervals, npts=vmodel.npts,
                values=vmodel.values
            )

        slv = _solver.PointSourceSolver(coord_sys=vm_spec["coord_sys"])
        slv.velocity.min_coords = vm_spec["min_coords"]
        slv.velocity.node_intervals = vm_spec["node_intervals"]
        slv.velocity.npts = vm_spec["npts"]
        slv.velocity.values = vm_spec["values"]
        if solver_kwargs:
            for attr, value in solver_kwargs.items():
                setattr(slv, attr, value)
        slv.src_loc = np.asarray(coords, dtype=np.float64)
        slv.solve()

        tt = slv.traveltime
        _mc = np.asarray(tt.min_coords, dtype=np.float64)
        _ni = np.asarray(tt.node_intervals, dtype=np.float64)
        _npts = np.asarray(tt.npts)
        return key, None, dict(
            coord_sys=tt.coord_sys,
            min_coords=_mc,
            node_intervals=_ni,
            npts=_npts,
            values=symmetrize_pole_rows(np.asarray(tt.values), _mc, _ni, _npts)
        )
    except Exception as err:
        return key, f"{type(err).__name__}: {err}", None


def ensure_traveltimes(path, requests, velocity_models, max_dist=None,
                       mask=True, compress=True, compress_kwargs=None,
                       solver_kwargs=None, nproc=None):
    """
    Cross-process-safe compute-and-store of missing traveltime grids,
    with parallel FMM solves when several stations are missing at once.

    Three phases:
      1. shared lock  : read-only check of which requested keys are
                        missing (fast path returns in ~1 ms if none).
      2. NO lock      : missing grids are solved in a multiprocessing
                        pool (nproc workers, default = min(n_missing,
                        cpu_count)). Readers are never blocked while
                        solves run.
      3. exclusive lock: keys are re-checked and results written. If a
                        concurrent process stored a grid meanwhile, its
                        copy wins and ours is discarded (duplicate solves
                        across racing processes are possible but
                        harmless; duplicate WRITES are not).

    requests : {("NET","STA","PHASE"): station_coords}
    velocity_models : {"P": ScalarField3D or hdf path, "S": ...}
    Returns {"present": [...], "computed": [...], "skipped": [...]}.

    Locking: <path>.lock, exclusive for writes, shared for reads. Hold a
    shared lock on the same file while reading the inventory elsewhere
    (the SeisComP wrapper does this).
    """
    import fcntl
    import multiprocessing

    requests = {tuple(k): v for k, v in requests.items()}
    result = {"present": [], "computed": [], "skipped": []}

    # ---- phase 1: what is missing? (shared lock, read-only) ----
    missing = dict(requests)
    if os.path.exists(path):
        fd = os.open(path + ".lock", os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            try:
                with TraveltimeInventory(path, mode="r") as inv:
                    for key in requests:
                        if inv.has(key):
                            result["present"].append(key)
                            missing.pop(key)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    if not missing:
        return result

    # resolve velocity model specs; pass file paths through untouched so
    # workers read them independently (cheaper than pickling big arrays)
    vm_specs = {}
    for phase, vm in velocity_models.items():
        if isinstance(vm, (str, os.PathLike)):
            vm_specs[phase.upper()] = str(vm)
        else:
            vm_specs[phase.upper()] = dict(
                coord_sys=vm.coord_sys, min_coords=np.asarray(vm.min_coords),
                node_intervals=np.asarray(vm.node_intervals),
                npts=np.asarray(vm.npts), values=np.asarray(vm.values)
            )

    tasks = []
    for key, coords in missing.items():
        phase = key[2].upper() if len(key) > 2 else "P"
        if phase not in vm_specs or coords is None:
            warnings.warn(
                f"{'/'.join(key)}: cannot compute traveltimes "
                f"({'no velocity model for phase ' + phase if phase not in vm_specs else 'no station coordinates'}); "
                f"skipping.",
                stacklevel=2
            )
            result["skipped"].append(key)
            continue
        tasks.append((key, np.asarray(coords, dtype=np.float64),
                      vm_specs[phase], solver_kwargs))

    if not tasks:
        return result

    # ---- phase 2: parallel solves, no lock held ----
    if nproc is None:
        nproc = min(len(tasks), os.cpu_count() or 1)
    nproc = max(1, int(nproc))
    if nproc == 1 or len(tasks) == 1:
        solved = [_solve_one_traveltime(t) for t in tasks]
    else:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=nproc) as pool:
            solved = pool.map(_solve_one_traveltime, tasks)

    # ---- phase 3: write under exclusive lock, re-checking ----
    coords_by_key = {key: coords for key, coords, _, _ in tasks}
    fd = os.open(path + ".lock", os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        mode = "a" if os.path.exists(path) else "w"
        try:
            with TraveltimeInventory(path, mode=mode) as inv:
                for key, err, data in solved:
                    if err is not None:
                        warnings.warn(f"{'/'.join(key)}: {err}; skipped")
                        result["skipped"].append(key)
                        continue
                    if inv.has(key):     # a racing process wrote it first
                        result["present"].append(key)
                        continue
                    field = fields.ScalarField3D(coord_sys=data["coord_sys"])
                    field.min_coords = data["min_coords"]
                    field.node_intervals = data["node_intervals"]
                    field.npts = data["npts"]
                    field.values = data["values"]
                    inv.add(
                        field, "/".join(key),
                        station_coords=coords_by_key[key],
                        max_dist=max_dist, mask=mask, compress=compress,
                        compress_kwargs=compress_kwargs
                    )
                    result["computed"].append(key)
                inv.f5.flush()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)

    return result


### the below is relevant to the station inventory creation / sharding


__all__ = ["inventory_lock", "ensure_traveltimes_sharded", "shard_path",
           "clamp_to_grid", "fit_requests_to_grid", "resolve_max_dist",
           "describe_max_dist", "DEFAULT_MAX_DIST_KM"]


def _noop(*_a, **_k):
    pass


@contextlib.contextmanager
def inventory_lock(inventory_path, shared):
    """
    Coordinate HDF5 access across concurrent processes: a shared lock while
    reading (locating), an exclusive lock while grids are being written.
    Uses <inventory>.lock, the same file ensure_traveltimes() locks on.
 
    fcntl is imported here rather than at module scope so that importing
    pykonal.inventory still works on platforms without it; only this
    function is Unix-only.
    """
    import fcntl
    with open(str(inventory_path) + ".lock", "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


DEFAULT_MAX_DIST_KM = 1000.0


def resolve_max_dist(cfg, key="max_dist_km"):
    """
    The station-event distance limit in km, from a driver config, with the
    one rule every driver (pykonal_locext, pykonal_obspy,
    pykonal_prepopulate) applies:

      key absent        -> DEFAULT_MAX_DIST_KM (1000)
      null / None       -> None: NO limit. Must be written explicitly;
                           grids are then stored uncropped (whole model),
                           which costs far more disk and memory.
      inf               -> None, same as null (lets a keyword argument, where
                           None already means "not given", ask for no limit)
      positive number   -> that many km

    Raises ValueError for anything else (zero, negative, non-numeric):
    those used to mean different things in different drivers.
    """
    if key not in cfg:
        return DEFAULT_MAX_DIST_KM
    val = cfg[key]
    if val is None:
        return None
    try:
        val = float(val)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a positive number or null, "
                         f"got {cfg[key]!r}") from None
    if val == float("inf"):
        return None
    if not val > 0:          # also rejects NaN
        raise ValueError(f"{key} must be a positive number or null "
                         f"(null = no limit), got {cfg[key]!r}")
    return val


def describe_max_dist(max_dist):
    """'1000 km' or 'unlimited', for log messages."""
    return "unlimited" if max_dist is None else f"{float(max_dist):g} km"


def shard_path(dirpath, key):
    """Per-station-phase file for key (net, sta, phase)."""
    return os.path.join(dirpath, "{}.{}.{}.h5".format(*key))


def clamp_to_grid(coords, lo, hi, vertical_axis):
    """
    Fit a station onto the velocity grid.

    The VERTICAL coordinate is clamped into range rather than rejected: a
    station a few hundred metres above sea level would otherwise fall
    outside a model whose top is the datum, and dropping it entirely is far
    worse than placing it at the grid's top node (~0.06 s of P traveltime
    for 340 m). Being outside LATERALLY is a real exclusion -- there is no
    sensible place to put such a source -- so that returns None.

    Returns (clamped_coords, was_clamped), or (None, False) if laterally
    outside. 'was_clamped' is reported so the caller can say so: moving a
    station is a silent change to the data otherwise, and on a sea-level
    datum it happens to nearly every station.
    """
    c = np.asarray(coords, dtype=float).copy()
    lateral = [i for i in range(3) if i != vertical_axis]
    if any(c[i] < lo[i] or c[i] > hi[i] for i in lateral):
        return None, False
    before = c[vertical_axis]
    c[vertical_axis] = min(max(before, lo[vertical_axis]),
                           hi[vertical_axis])
    return c, bool(c[vertical_axis] != before)


def grid_extents(velocity_models, log=None):
    """{PHASE: (min_coords, max_coords)} for each readable velocity model."""
    from . import fields as _fields
    log = log or _noop
    out = {}
    for phase, vm in velocity_models.items():
        try:
            with h5py.File(vm, "r") as f5:
                minc = f5["min_coords"][:]
                npts = f5["npts"][:]
                node = f5["node_intervals"][:]
            out[phase.upper()] = (minc, minc + (npts - 1) * node)
        except Exception as err:
            log(f"could not read velocity model extent for phase {phase} "
                f"({err}); off-grid pre-filter skipped for it")
    return out


def fit_requests_to_grid(requests, velocity_models, vertical_axis=0,
                         log=None):
    """
    Fit every requested station onto its phase's velocity grid, dropping
    those that cannot be fitted.

    Done BEFORE building so the solver is never asked for an out-of-grid
    source, which would raise and abort the whole relocation. Returns a new
    requests dict; the caller's is not modified.

    Reports both outcomes: stations excluded for being laterally outside,
    and stations whose elevation was clamped onto the grid top. The second
    is a real change to the station position and should not be silent.
    """
    log = log or _noop
    extents = grid_extents(velocity_models, log=log)
    fitted, offgrid, clamped = {}, [], []

    for key, coords in requests.items():
        key = tuple(key)
        if coords is None:
            fitted[key] = None
            continue
        ext = extents.get(key[2].upper())
        if ext is None:
            fitted[key] = coords            # no extent known; leave as-is
            continue
        c, was_clamped = clamp_to_grid(coords, ext[0], ext[1], vertical_axis)
        if c is None:
            offgrid.append(key)
        else:
            fitted[key] = c
            if was_clamped:
                clamped.append(key)

    if offgrid:
        log(f"{len(offgrid)} station/phase(s) fall outside the velocity "
            f"grid laterally and will be excluded: "
            + ", ".join(".".join(k) for k in offgrid))
    if clamped:
        log(f"{len(clamped)} station/phase(s) sit outside the velocity grid "
            f"vertically and were moved onto the nearest grid face before "
            f"solving (typically stations above a sea-level datum; roughly "
            f"0.06 s of P traveltime per 340 m): "
            + ", ".join(".".join(k) for k in clamped[:8])
            + ("..." if len(clamped) > 8 else ""))
    return fitted


def _model_fingerprint(vm_path):
    """
    Cheap identity for a velocity model file: size + mtime. Stored on each
    grid so grids built from a superseded model can be detected and
    rebuilt. Traveltime grids are otherwise cached forever, so editing the
    velocity model silently leaves stale grids that look perfectly valid
    but predict the wrong times.
    """
    try:
        st = os.stat(vm_path)
        return f"{st.st_size}:{int(st.st_mtime)}"
    except Exception:
        return "unknown"


def _stored_max_dist(f5, key):
    """
    The max_dist (km) a stored grid was cropped to, or None if it was stored
    uncropped (TraveltimeInventory.add writes the attribute only when it
    crops). Raises KeyError if the grid is not in the file.
    """
    val = f5["/".join(key)].attrs.get("max_dist")
    return None if val is None else float(val)


def _max_dist_covers(stored, requested):
    """
    True if a grid cropped to `stored` km serves a request for `requested`
    km. None means uncropped (unlimited) on either side. A LARGER stored crop
    is a superset and still serves: the extra nodes only matter if a driver
    locates with arrivals beyond its own cutoff, and the drivers filter those
    by distance themselves. Small tolerance so a float round trip through
    HDF5 or JSON cannot trigger a rebuild.
    """
    if stored is None:
        return True
    if requested is None:
        return False
    return stored >= float(requested) - 1e-6


def _shard_staleness(dirpath, key, velocity_models, max_dist):
    """
    Why the stored grid for `key` cannot be used as-is, or None if it can.

    Returns one of:
      None                          current: exists, built from the current
                                    velocity model, covers max_dist
      "missing"                     no grid file
      "velocity model superseded"   fingerprint mismatch (or unreadable)
      "max_dist <stored>"           cropped tighter than max_dist asks for;
                                    the stored value follows the space so
                                    the caller can report it

    The max_dist check always opens the file (~0.3 ms per grid): unlike the
    model fingerprint, max_dist is a request parameter with no file to
    compare modification times against, so there is no fast path for it.
    """
    path = shard_path(dirpath, key)
    try:
        shard_st = os.stat(path)
    except OSError:
        return "missing"

    try:
        with h5py.File(path, "r") as f5:
            stored_md = _stored_max_dist(f5, key)
            got = f5.attrs.get("velocity_model_fingerprint")
    except Exception:
        # unreadable or does not hold this key: rebuild it
        return "velocity model superseded"

    if not _max_dist_covers(stored_md, max_dist):
        return f"max_dist {stored_md:g}"

    vm = velocity_models.get(key[2]) or velocity_models.get(key[2].upper())
    if not isinstance(vm, str):
        return None          # cannot fingerprint an in-memory model

    # A shard written after the model file was last touched can only have
    # been built from the current model. Same failure mode as the
    # fingerprint itself: a model file restored with an older mtime looks
    # unchanged to both.
    try:
        if shard_st.st_mtime >= os.stat(vm).st_mtime:
            return None
    except OSError:
        pass                 # model file unreadable; fall through and check

    if isinstance(got, bytes):
        got = got.decode()
    if got != _model_fingerprint(vm):
        return "velocity model superseded"
    return None


def _build_one_shard(args):
    """
    Worker: build ONE station-phase grid into its own HDF5 file, using
    ensure_traveltimes (which handles its own per-file lock). Runs with
    nproc=1 because parallelism happens across shards here, not inside.

    Returns (key, error_or_None, messages). Messages are RETURNED rather
    than logged: this runs in a multiprocessing worker, where a driver's
    logger is not picklable and stderr ordering is not guaranteed.
    """
    key, coords, velocity_models, dirpath, max_dist, compress_kwargs = args
    messages = []
    shard = shard_path(dirpath, key)
    try:
        rep = ensure_traveltimes(
            shard, {key: coords}, velocity_models,
            max_dist=max_dist, nproc=1,
            compress_kwargs=compress_kwargs
        )
        if key in rep.get("skipped", []):
            # ensure_traveltimes' write phase creates the (empty) shard file
            # even when the solve failed; an empty shard would be linked into
            # the master and then accepted as current forever by
            # _shard_staleness's mtime fast path. Remove it.
            try:
                os.remove(shard)
            except OSError:
                pass
            return key, "skipped by ensure_traveltimes", messages

        # Sanity-guard the stored field. A traveltime must be finite and
        # non-negative; a point-source field's minimum must also be AT the
        # source. Rare eikonal-solver glitches can leave an isolated -inf
        # (or a small patch of impossibly low times), which then corrupts
        # the objective for any trial hypocenter near it. Replace such
        # nodes with NaN so they are treated as no-data rather than as an
        # attractive minimum, and report it.
        bad = 0
        with h5py.File(shard, "r+") as f5:
            dset = f5["/".join(key)]["values"]
            vals = dset[...]
            impossible = ~np.isfinite(vals) & ~np.isnan(vals)   # +/-inf
            impossible |= np.isfinite(vals) & (vals < 0)        # negative
            bad = int(impossible.sum())
            if bad:
                vals[impossible] = np.nan
                dset[...] = vals
        if bad:
            # repaired, not failed: the grid is still usable
            messages.append(
                f"{'.'.join(key)}: {bad} impossible traveltime node(s) "
                f"(-inf or negative) masked to NaN — the eikonal solve "
                f"produced a corrupt patch; check the velocity model there")

        # stamp with the model identity so a later model edit invalidates it
        vm = velocity_models.get(key[2]) or velocity_models.get(key[2].upper())
        if isinstance(vm, str):
            with h5py.File(shard, "r+") as f5:
                f5.attrs["velocity_model_fingerprint"] = _model_fingerprint(vm)
        return key, None, messages
    except Exception as err:
        try: os.remove(shard)
        except OSError: pass
        return key, f"{type(err).__name__}: {err}", messages


def _rebuild_master(dirpath):
    """
    Rebuild the master index: a small HDF5 file whose groups are EXTERNAL
    LINKS to the per-station files. TraveltimeInventory and EQLocator
    follow these transparently, so no change is needed elsewhere
    (single-file inventories keep working untouched).

    Rebuilt wholesale from a directory scan -- it is a few KB, so this is
    fast and avoids incremental corruption. Links are RELATIVE so the
    directory can be moved or copied as a unit.
    """
    master = os.path.join(dirpath, "inventory.h5")
    tmp = f"{master}.{os.getpid()}.tmp"
    with h5py.File(tmp, "w") as f5:
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith(".h5") or fname == "inventory.h5":
                continue
            parts = fname[:-3].split(".")
            if len(parts) != 3:
                continue
            net, sta, phase = parts
            f5[f"{net}/{sta}/{phase}"] = h5py.ExternalLink(
                fname, f"/{net}/{sta}/{phase}"
            )
    os.replace(tmp, master)   # atomic swap
    return master


def ensure_traveltimes_sharded(dirpath, requests, velocity_models,
                               max_dist=DEFAULT_MAX_DIST_KM, nproc=None, log=None,
                               compress_kwargs=None):
    """
    Alternative to ensure_traveltimes that stores one HDF5 file per
    station-phase instead of one large shared file.

    Why: an inventory that grows on demand from a live interactive session
    is badly served by a single file. Every new grid takes an exclusive
    lock on the whole inventory (blocking all other relocations), HDF5
    never reclaims space when grids are rebuilt after a velocity-model
    change, and a crash mid-write risks the entire archive. Per-station
    files make writes independent, make rebuilds a file delete, and limit
    any corruption to one station.

    requests : {("NET","STA","PHASE"): station_coords}, None to skip
    velocity_models : {"P": ScalarField3D or hdf path, "S": ...}
    max_dist : km, or None for no limit (grids stored uncropped). Drivers
        should obtain it from resolve_max_dist(cfg). A stored grid cropped
        tighter than this is rebuilt; a wider one is kept.
    log : callable taking one string, or None to discard diagnostics.

    Returns {"computed", "present", "skipped"} as ensure_traveltimes does,
    plus "master": the index path to hand to EQLocator.
    """
    log = log or _noop
    os.makedirs(dirpath, exist_ok=True)
    report = {"computed": [], "present": [], "skipped": []}

    todo = []
    rebuilt_model, rebuilt_dist, unremovable = [], {}, []
    for key, coords in requests.items():
        key = tuple(key)
        if coords is None:
            report["skipped"].append(key)
            continue
        why = _shard_staleness(dirpath, key, velocity_models, max_dist)
        if why is None:
            report["present"].append(key)
            continue
        if why != "missing":
            # A stale grid must be DELETED before rebuilding:
            # ensure_traveltimes sees the key already present in the file
            # and would skip it, silently keeping the old traveltimes.
            try:
                os.remove(shard_path(dirpath, key))
            except OSError as err:
                # the old grid is still linked in the master, so the
                # locator will keep using it; report it as present
                unremovable.append((key, why, err))
                report["present"].append(key)
                continue
            if why.startswith("max_dist "):
                rebuilt_dist.setdefault(why.split(" ", 1)[1], []).append(key)
            else:
                rebuilt_model.append(key)
        todo.append((key, coords, velocity_models, dirpath, max_dist,
                     compress_kwargs))

    # One line per cause rather than per grid: a config change typically
    # invalidates every station at once.
    def _names(keys):
        return (", ".join(".".join(k) for k in keys[:8])
                + ("..." if len(keys) > 8 else ""))
    want = describe_max_dist(max_dist)
    for stored, keys in sorted(rebuilt_dist.items()):
        log(f"{len(keys)} grid(s) were cropped to max_dist_km={stored} but "
            f"{want} is now requested, so they cannot cover the larger "
            f"distance range; deleted and rebuilding: {_names(keys)}")
    if rebuilt_model:
        log(f"{len(rebuilt_model)} grid(s) were built from a superseded or "
            f"unreadable velocity model; deleted and rebuilding: "
            f"{_names(rebuilt_model)}")
    for key, why, err in unremovable:
        log(f"{'.'.join(key)}: grid is stale ({why}) but could not be "
            f"removed ({err}); the old grid is still in use")

    if todo:
        n = nproc or min(len(todo), os.cpu_count() or 1)
        n = max(1, int(n))
        # Announced here rather than by the caller: the caller would have
        # to repeat the staleness scan to know what is missing, and that
        # scan may open every shard file. One scan, one announcement.
        log(f"computing {len(todo)} new traveltime grid(s) before locating "
            f"on {n} CPU(s) (one FMM solve each; this can take a while) — "
            f"{', '.join('.'.join(t[0]) for t in todo)}")
        if n > 1:
            import multiprocessing
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(processes=n) as pool:
                results = pool.map(_build_one_shard, todo)
        else:
            results = [_build_one_shard(t) for t in todo]
        for key, err, messages in results:
            for m in messages:          # emitted here: workers cannot log
                log(m)
            if err is None:
                report["computed"].append(key)
            else:
                log(f"{'.'.join(key)}: {err}")
                report["skipped"].append(key)

    report["master"] = _rebuild_master(dirpath)
    return report
 
