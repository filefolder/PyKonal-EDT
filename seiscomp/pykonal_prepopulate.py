#!/usr/bin/env python3
"""
Pre-populate pykonal traveltime grids for every station in the configured
station CSV, so that scolv relocations never have to wait for an FMM solve.

Usage:
    pykonal_prepopulate.py /path/to/pykonal_locext.json


Stations whose coordinates fall outside the velocity model are skipped and
listed - they cannot have a grid computed, and the plugin excludes their
arrivals at location time anyway. Grids that already exist and were built
from the current velocity model are left alone; grids built from a
superseded model are rebuilt.
"""
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pykonal.transformations import GeoTransform
from pykonal.inventory import (ensure_traveltimes_sharded, fit_requests_to_grid,
                               resolve_max_dist, describe_max_dist)


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    cfg_path = sys.argv[1]
    with open(cfg_path) as f:
        cfg = json.load(f)

    velocity_models = cfg.get("velocity_models") or {}
    if not velocity_models:
        print("config has no 'velocity_models': nothing can be built.",
              file=sys.stderr)
        return 1
    stations_csv = cfg.get("stations_csv")
    if not stations_csv:
        print("config has no 'stations_csv': no station list to build from.",
              file=sys.stderr)
        return 1

    transform = GeoTransform()

    # stations
    stations = {}
    with open(stations_csv) as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 4:
                continue
            net, sta = row[0].strip(), row[1].strip()
            if net.lower() == 'network': continue
            try:
                lat, lon = float(row[2]), float(row[3])
                elev = float(row[4]) if len(row) > 4 and row[4].strip() else 0.0
            except ValueError:
                continue           # header line or malformed row
            stations[(net, sta)] = (lat, lon, elev)
    print(f"stations in CSV                 : {len(stations)}")


    # build the request set
    requests = {}
    for (net, sta), (lat, lon, elev) in sorted(stations.items()):
        grid_xyz = transform.geo_to_grid(lat, lon, -elev / 1000.0)
        for phase in velocity_models:
            requests[(net, sta, phase)] = grid_xyz
    requests = fit_requests_to_grid(requests, velocity_models,
                                    vertical_axis=0, log=print)


    # build
    try:
        max_dist = resolve_max_dist(cfg)     # absent -> 1000, null -> none
    except ValueError as err:
        print(f"config: {err}", file=sys.stderr)
        return 1
    print(f"max_dist_km                     : {describe_max_dist(max_dist)}"
          + ("  (grids stored uncropped: whole model per station)"
             if max_dist is None else ""))
    nproc = cfg.get("ensure_nproc")
    shard_dir = cfg.get("traveltime_dir")
    t0 = time.time()
    if not shard_dir:
        print("config has no 'traveltime_dir': the sharded layout is the "
              "only supported storage.", file=sys.stderr)
        return 1
    report = ensure_traveltimes_sharded(
        shard_dir, requests, velocity_models,
        max_dist=max_dist, nproc=nproc, log=print)

    dt = time.time() - t0
    built = report.get("computed", [])
    present = report.get("present", [])
    failed = report.get("skipped", [])
    print()
    print(f"already current                 : {len(present)}")
    print(f"built now                       : {len(built)}")
    print(f"could not build                 : {len(failed)}")
    for k in failed:
        print(f"    {'.'.join(k)}")
    print(f"elapsed                         : {dt:.1f} s"
          + (f"  ({dt/len(built):.1f} s per grid)" if built else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
