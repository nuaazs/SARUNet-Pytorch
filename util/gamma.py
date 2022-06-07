# ██╗██╗███╗   ██╗████████╗
# ██║██║████╗  ██║╚══██╔══╝
# ██║██║██╔██╗ ██║   ██║
# ██║██║██║╚██╗██║   ██║
# ██║██║██║ ╚████║   ██║
# ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝

# @Time    : 2021-09-28 09:35:49
# @Author  : zhaosheng
# @email   : zhaosheng@nuaa.edu.cn
# @Blog    : iint.icu
# @File    : util/gamma.py
# @Describe: 


import numpy as np

def gamma_matrix(rm, tm, dta=1.0, dd=0.05):
    '''Compute matrix of gammma indices.
    :param rm: reference matrix (relative values assumed)
    :param tm: tested matrix (relative values assumed)
    :param dta: maximum distance-to-agreement (in voxels)
    :param dd: maximum dose difference
    It can be evaluated on matrices of any dimensionality.
    '''
    # Check validity of input
    if rm.shape != tm.shape:
        raise Exception("Cannot compute for matrices of different sizes.")

    # Result matrix
    output = np.ndarray(rm.shape, dtype=np.float64)

    # Help scaling variables
    dta_scale = dta ** 2
    dd_scale = dd ** 2

    # Index matrices
    indices = np.indices(rm.shape)

    it = np.nditer(rm, ("multi_index",))
    while not it.finished:
        index = it.multi_index

        # Dose difference to every point (squared)
        dd2 = (tm - it.value) ** 2

        # Distance to every point (squared)
        dist2 = np.sum((indices - np.array(index).reshape(len(rm.shape),1,1,1)) ** 2, axis=0)

        # Minimum of the sum
        output[index] = np.sqrt(np.nanmin(dd2 / dd_scale + dist2 / dta_scale))
        it.iternext()
    return output

def pass_gamma(rm, tm, dta=1.0, dd=0.05, ignore=lambda value: False):
    '''Effectively check which points in a matrix pair pass gamma index test.
    :param rm: reference matrix (relative values assumed)
    :param tm: tested matrix (relative values assumed)
    :param dta: maximum distance-to-agreement (in voxels)
    :param dd: maximum dose difference
    :param ignore: function called on dose in rm values
        if it returns True, point is ignored (gammma <- np.nan)

    It can be evaluated on matrices of any dimensionality.
    Optimized in that only surrounding region that can possibly
        pass dta criterion is checked for each point.
    '''
    # Check validity of input
    if rm.shape != tm.shape:
        raise Exception("Cannot compute for matrices of different sizes.")

    shape = rm.shape
    ndim = rm.ndim

    # Result matrix
    output = np.ndarray(rm.shape, dtype=np.float64)

    # Help scaling variables
    dta_scale = dta ** 2
    dd_scale = dd ** 2

    # Index matrices
    indices = np.indices(rm.shape)

    # How many points (*2 + 1)
    npoints = int(dta)

    it = np.nditer(rm, ("multi_index",))
    while not it.finished:
        index = tuple(it.multi_index)

        if ignore(it.value):
            output[index] = np.nan
            it.iternext()
            continue

        slices = [ slice(max(0, index[i] - npoints), min(shape[i], index[i] + npoints + 1)) for i in xrange(ndim) ]
        subtm = tm[slices]

        # Dose difference to every point (squared)
        dd2 = (subtm - it.value) ** 2

        # Distance to every point (squared)
        dist2 = np.sum((indices[[slice(None, None)] + slices] - np.array(index).reshape(ndim,1,1,1)) ** 2, axis=0)

        # Minimum of the sum
        output[index] = np.sqrt(np.nanmin(dd2 / dd_scale + dist2 / dta_scale))
        it.iternext()
    return output
