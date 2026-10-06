cimport numpy as np

from libcpp cimport bool as bool_t

from . cimport constants
from . cimport fields
from . cimport heapq

cdef class EikonalSolver(object):
    cdef str                       cy_coord_sys
    cdef fields.ScalarField3D      cy_velocity
    cdef fields.ScalarField3D      cy_traveltime
    cdef heapq.Heap                cy_trial
    cdef constants.BOOL_t[:,:,:]   cy_known
    cdef constants.BOOL_t[:,:,:]   cy_unknown

    cpdef constants.BOOL_t _check_state(EikonalSolver self)
    cpdef constants.BOOL_t solve(EikonalSolver self, constants.REAL_t max_traveltime=*)
    cpdef np.ndarray[constants.REAL_t, ndim=2] trace_ray(
            EikonalSolver self,
            constants.REAL_t[:] end
    )
