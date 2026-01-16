import math
from copy import deepcopy

import numpy as np

from modt.hypervolume import InnerHyperVolume


def compute_hypervolume(ep_objs_batch):
    n = len(ep_objs_batch[0])
    HV = InnerHyperVolume(np.zeros(n))
    return HV.compute(ep_objs_batch)


def compute_sparsity(ep_objs_batch):
    if len(ep_objs_batch) < 2:
        return 0.0

    sparsity = 0.0
    m = len(ep_objs_batch[0])
    ep_objs_batch_np = np.array(ep_objs_batch)
    for dim in range(m):
        objs_i = np.sort(deepcopy(ep_objs_batch_np.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity += np.square(objs_i[i] - objs_i[i - 1])
    sparsity /= len(ep_objs_batch) - 1
    return sparsity


def check_dominated(obj_batch, obj, tolerance=0):
    return (
        np.logical_and(
            (obj_batch * (1 - tolerance) >= obj).all(axis=1),
            (obj_batch * (1 - tolerance) > obj).any(axis=1),
        )
    ).any()


# return sorted indices of nondominated objs
def undominated_indices(obj_batch_input, tolerance=0):
    obj_batch = np.array(obj_batch_input)
    sorted_indices = np.argsort(obj_batch.T[0])
    indices = []
    for idx in sorted_indices:
        if (obj_batch[idx] >= 0).all() and not check_dominated(
            obj_batch, obj_batch[idx], tolerance
        ):
            indices.append(idx)
    return indices


def _is_close_to_one(x):
    return math.isclose(x, 1, rel_tol=1e-12)


def pref_grid(n_obj, max_prefs=None, min_prefs=None, granularity=5):
    max_prefs = np.ones(n_obj) if max_prefs is None else max_prefs
    min_prefs = np.zeros(n_obj) if min_prefs is None else min_prefs
    grid = np.array([x / granularity for x in range(granularity + 1)])
    prefs = [[]]
    grid = tuple(grid)
    for _ in range(n_obj):
        prefs = [
            x + [y]
            for x in prefs
            for y in grid
            if sum(x + [y]) < 1 or _is_close_to_one(sum(x + [y]))
        ]
    prefs = np.array([p for p in prefs if _is_close_to_one(sum(p))])
    for i in range(n_obj):
        prefs[:, i] = prefs[:, i] * (max_prefs[i] - min_prefs[i]) + min_prefs[i]
    prefs = prefs / np.sum(prefs, axis=1, keepdims=True)
    return prefs
