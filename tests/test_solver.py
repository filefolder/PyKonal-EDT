"""
Regression and accuracy tests for EikonalSolver.

Ported from the original nose-based suite. Two things had rotted:

  * `import nose` / `nose.main()`. nose has been unmaintained since 2015
    and does not import on Python 3.10+, so the entire module was
    uncollectable. These are plain `unittest.TestCase` classes, so
    `unittest.main()` is a drop-in replacement and they then run under
    unittest, pytest and nose2 alike -- no test-runner dependency needed.

  * The 0.2-era API. `solver.vgrid` / `solver.pgrid` / `solver.vv` /
    `solver.uu` and `add_source()` no longer exist; a solver now exposes
    `velocity` and `traveltime` as ScalarField3D objects, and a source is
    seeded by zeroing its traveltime, marking it known, and pushing it
    onto the trial heap.

`pkg_resources` is also gone (deprecated, and its argument here was a path
already prefixed with "pykonal/", which only resolved by accident of the
source layout). Data files resolve relative to this module instead.
"""

import numpy as np
import pathlib
import pykonal
import unittest

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"


def solve_from_fixture(fname):
    """Rebuild the stored problem with the current API and solve it."""
    with np.load(fname) as inf:
        solver = pykonal.EikonalSolver(coord_sys="cartesian")
        solver.velocity.min_coords = inf["vgrid_min_coords"]
        solver.velocity.node_intervals = inf["vgrid_node_intervals"]
        solver.velocity.npts = inf["vgrid_npts"]
        solver.velocity.values = inf["vv"]

        # Seed the point source. The 0.2 API's add_source() did this
        # internally; there is no equivalent single call on a bare
        # EikonalSolver now (PointSourceSolver has its own src_loc).
        src = tuple(int(i) for i in inf["src"])
        solver.traveltime.values[src] = 0.0
        solver.unknown[src] = False
        solver.trial.push(*src)

        uu_ref = inf["uu"]

    solver.solve()
    return np.asarray(solver.traveltime.values), uu_ref


class EikonalSolverTestCase(unittest.TestCase):

    FIXTURE = "test_EikonalSolver_uniform_velocity_cartesian.npz"

    def test_uniform_velocity_cartesian_regression(self):
        """Current solver still reproduces the stored solution.

        Compared with a tolerance rather than assert_array_almost_equal:
        the stored array predates several solver changes and agrees to
        ~0.045 out of traveltimes reaching ~170. A six-decimal equality
        assertion here would only be testing that nobody ever touches the
        update stencil again.
        """
        uu, uu_ref = solve_from_fixture(DATA_DIR / self.FIXTURE)
        self.assertEqual(uu.shape, uu_ref.shape)
        self.assertLess(np.abs(uu - uu_ref).max(), 0.1)

    def test_uniform_velocity_cartesian_accuracy(self):
        """Unit velocity from a corner source: traveltime is the distance.

        This is the assertion the fixture comparison cannot make -- it
        checks the solver against the analytic answer rather than against
        its own past output, so it would catch an error that was baked
        into the fixture when it was generated. First-order FMM has known
        error along the diagonals near the source, so accuracy is asserted
        in the far field.
        """
        uu, _ = solve_from_fixture(DATA_DIR / self.FIXTURE)
        idx = np.meshgrid(*[np.arange(n) for n in uu.shape], indexing="ij")
        analytic = np.sqrt(sum(i.astype(float) ** 2 for i in idx))

        far = analytic > 10.0
        rel = np.abs(uu[far] - analytic[far]) / analytic[far]
        self.assertLess(rel.max(), 0.10)
        self.assertLess(np.median(rel), 0.01)

        self.assertGreaterEqual(uu.min(), 0.0)
        self.assertTrue(np.all(np.isfinite(uu)))


class PointSourceAspectRatioTestCase(unittest.TestCase):
    """
    Coarse-horizontal / fine-vertical spherical grids. Before the near field
    was sized to the far-field cells, a source between node columns on a
    0.25 deg x 1 km grid got no traveltimes at all, and sources off the
    nodes on 0.25 deg x 2 km grids had errors of tens of percent.
    """

    R = 6371.0
    V = 5.8

    def _solve(self, dtheta_deg, dr_km, frac, **near_field):
        n, nr = 25, int(round(44 / dr_km)) + 1
        mins = np.array([self.R - 44.0, np.radians(118.0), np.radians(119.0)])
        node = np.array([dr_km, np.radians(dtheta_deg), np.radians(dtheta_deg)])
        s = pykonal.solver.PointSourceSolver(coord_sys="spherical")
        s.velocity.min_coords, s.velocity.node_intervals = mins, node
        s.velocity.npts = (nr, n, n)
        s.velocity.values = np.full((nr, n, n), self.V)
        for k, v in near_field.items():
            setattr(s, k, v)
        src = mins + node * np.array([0.0, 10.0 + frac[0], 10.0 + frac[1]])
        src[0] = self.R - 1.0
        s.src_loc = src
        s.solve()
        return s, src

    @staticmethod
    def _xyz(p):
        r, t, f = p[..., 0], p[..., 1], p[..., 2]
        return np.stack([r * np.sin(t) * np.cos(f), r * np.sin(t) * np.sin(f),
                         r * np.cos(t)], -1)

    def _errors(self, s, src):
        d = np.linalg.norm(self._xyz(s.traveltime.nodes) - self._xyz(src), axis=-1)
        far = d > 20.0
        exact = d[far] / self.V
        return np.abs(s.traveltime.values[far] - exact) / exact

    def test_every_source_position_solves_accurately(self):
        for dth, dr in ((0.1, 2.0), (0.25, 2.0), (0.25, 1.0), (0.5, 1.0)):
            for frac in ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.3, 0.6)):
                with self.subTest(grid=(dth, dr), frac=frac):
                    s, src = self._solve(dth, dr, frac)
                    self.assertTrue(np.isfinite(s.traveltime.values).all())
                    rel = self._errors(s, src)
                    self.assertLess(np.median(rel), 0.04)
                    self.assertLess(np.percentile(rel, 99), 0.12)

    def test_near_field_reaches_the_far_field_cell(self):
        s, src = self._solve(0.25, 1.0, (0.5, 0.5))
        self.assertGreaterEqual(s.drho * (s.nrho - 1),
                                1.5 * s.far_field_cell_diagonal() - 1e-9)
        self.assertLessEqual(s.nrho, s.NRHO_MAX)

    def test_ordinary_grids_keep_the_default_near_field(self):
        s = pykonal.solver.PointSourceSolver(coord_sys="cartesian")
        s.velocity.min_coords, s.velocity.node_intervals = (0, 0, 0), (1, 1, 1)
        s.velocity.npts = (21, 21, 21)
        s.velocity.values = np.full((21, 21, 21), 5.0)
        s.src_loc = np.array([10.3, 10.6, 3.2])
        s.solve()
        self.assertEqual((s.nrho, s.drho), (64, 1.0 / 8))

    def test_explicit_near_field_settings_are_respected(self):
        s, _ = self._solve(0.1, 2.0, (0.0, 0.0), nrho=80, drho=0.3)
        self.assertEqual((s.nrho, s.drho), (80, 0.3))

    def test_unreachable_far_field_raises(self):
        with self.assertRaises(RuntimeError):
            self._solve(0.25, 1.0, (0.5, 0.5), nrho=16)


if __name__ == "__main__":
    unittest.main()
