# pykonal test suite

```bash
pip install -e ".[tests,obspy]"     # or: python setup.py build_ext --inplace
pytest                              # fast suite, ~15 s
pytest --runslow                    # also the validation harnesses, ~35 s
```

With an in-place build (no install), run `python -m pytest` from the
repository root so the source tree is importable.

The suite tests whatever `import pykonal` finds. It never writes into the
source tree: every grid, inventory and config is built in a pytest
temporary directory, so it can run from any directory and in parallel.

## Layout

| file | covers |
|---|---|
| `conftest.py` | shared synthetic networks (cartesian 50 x 50 x 20 km and a spherical regional cap, 8 stations each), built once per session; `--runslow` |
| `test_grid.py`, `test_interpolator.py`, `test_solver.py` | grid geometry, interpolation, eikonal solver (stored regression fixture in `data/`) |
| `test_transforms.py` | geographic <-> grid coordinates, km deltas, the local uncertainty frame |
| `test_sizing.py` | search-box sizing, depth clamping and pinning, widening (pure functions) |
| `test_inventory.py` | storage, compression, distance cropping, `ensure_traveltimes` (incl. two processes at once), sharded inventories, fitting stations to the grid |
| `test_locate.py` | EDT likelihood, search, outlier resistance, residuals, quality |
| `test_posterior.py` | posterior sampling, ESS growth, mode- vs mean-centred covariance |
| `test_relocate.py` | the shared driver in auto mode: search, sizing, outliers, fixed depth, sample count |
| `test_obspy.py` | the ObsPy driver end to end (skipped without ObsPy) |
| `test_calibration.py` | coverage of the reported uncertainty on synthetic events through the auto pipeline; slow: the harnesses in `validation/` |

## Regression tests for fixed defects

- `test_solver.py::PointSourceAspectRatioTestCase`: on spherical grids with
  coarse horizontal and fine radial spacing (e.g. 0.25 deg x 1 km) the
  point-source solver's near field did not reach any far-field node for a
  source between node columns, and the whole grid came back infinite.
  Sources off the nodes also had errors of tens of percent.
- `test_inventory.py::test_station_outside_the_model_is_skipped_not_stored`
  and `test_station_between_coarse_nodes_gets_a_full_grid`: failed solves
  used to be stored as all-NaN grids and reported as computed.
- `test_inventory.py::test_three_station_search_finds_the_maximum_for_every_seed`
  and `test_relocate.py::test_sparse_network_with_small_pick_errors`: with
  few stations and small pick errors the search often ended on a worse
  mode, tens of km away.

Mark a newly found defect `@pytest.mark.xfail(strict=True, reason=...)`
until it is fixed: a strict xfail that starts passing fails the run, so the
marker cannot outlive the fix.

## Writing tests

- Use the `cart_net` / `sph_net` fixtures; build anything you modify in
  `tmp_path`.
- Seed everything (`locate_seed`, `posterior_seed`, `seed=`): the search
  and the posterior are randomised.
- Assert; do not print. Anything taking more than a few seconds gets
  `@pytest.mark.slow`.
