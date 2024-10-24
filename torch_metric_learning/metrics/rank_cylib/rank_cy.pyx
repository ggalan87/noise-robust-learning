# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

from __future__ import print_function
import numpy as np
from libc.stdint cimport int64_t, uint64_t
from libc.math cimport isnan

import cython

cimport numpy as np

import random
from collections import defaultdict

"""
Compiler directives:
https://github.com/cython/cython/wiki/enhancements-compilerdirectives

Cython tutorial:
https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html

Credit to https://github.com/luzai
"""

@cython.inline
cpdef clip_max_rank(int64_t num_g, int64_t max_rank):
    if num_g < max_rank:
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))
        return num_g

    return max_rank

cpdef cast_inputs(q_pids, g_pids, q_camids, g_camids):
    return (np.asarray(q_pids, dtype=np.int64),
            np.asarray(g_pids, dtype=np.int64),
            np.asarray(q_camids, dtype=np.int64),
            np.asarray(g_camids, dtype=np.int64))


cpdef eval_from_distmat(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, only_cmc=False):
    distmat = np.asarray(distmat, dtype=np.float32)

    cdef int64_t[:,:] sorted_indices = np.argsort(distmat, axis=1)

    return eval_market1501_base_cy(sorted_indices, *cast_inputs(q_pids, g_pids, q_camids, g_camids), max_rank, only_cmc)


cpdef eval_from_sorted_indices(sorted_indices, q_pids, g_pids, q_camids, g_camids, max_rank, only_cmc=False):
    sorted_indices = np.asarray(sorted_indices, dtype=np.int64)

    return eval_market1501_base_cy(sorted_indices, *cast_inputs(q_pids, g_pids, q_camids, g_camids), max_rank, only_cmc)


cpdef eval_market1501_base_cy(int64_t[:,:] indices, int64_t[:] q_pids, int64_t[:]g_pids,
                         int64_t[:]q_camids, int64_t[:]g_camids, int64_t max_rank, bint only_cmc):

    cdef int64_t num_q = len(q_pids)
    cdef int64_t num_g = len(g_pids)

    max_rank = clip_max_rank(num_g, max_rank)

    cdef:
        int64_t[:,:] matches = (np.asarray(g_pids)[np.asarray(indices)] == np.asarray(q_pids)[:, np.newaxis]).astype(np.int64)

        float[:,:] all_cmc = np.zeros((num_q, max_rank), dtype=np.float32)
        float[:] all_AP = np.zeros(num_q, dtype=np.float32)
        float num_valid_q = 0. # number of valid query

        int64_t q_idx, q_pid, q_camid, g_idx
        int64_t[:] order = np.zeros(num_g, dtype=np.int64)
        int64_t keep

        float[:] raw_cmc = np.zeros(num_g, dtype=np.float32) # binary vector, positions with value 1 are correct matches
        float[:] cmc = np.zeros(num_g, dtype=np.float32)
        int64_t num_g_real, rank_idx
        uint64_t meet_condition

        float num_rel
        float[:] tmp_cmc = np.zeros(num_g, dtype=np.float32)
        float tmp_cmc_sum

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        for g_idx in range(num_g):
            order[g_idx] = indices[q_idx, g_idx]
        num_g_real = 0
        meet_condition = 0

        for g_idx in range(num_g):
            if (g_pids[order[g_idx]] != q_pid) or (g_camids[order[g_idx]] != q_camid):
                raw_cmc[num_g_real] = matches[q_idx][g_idx]
                num_g_real += 1
                if matches[q_idx][g_idx] > 1e-31:
                    meet_condition = 1

        if not meet_condition:
            # this condition is true when query identity does not appear in gallery
            continue

        # compute cmc
        function_cumsum(raw_cmc, cmc, num_g_real)
        for g_idx in range(num_g_real):
            if cmc[g_idx] > 1:
                cmc[g_idx] = 1

        for rank_idx in range(max_rank):
            all_cmc[q_idx, rank_idx] = cmc[rank_idx]
        num_valid_q += 1.

        if not only_cmc:
            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            function_cumsum(raw_cmc, tmp_cmc, num_g_real)
            num_rel = 0
            tmp_cmc_sum = 0
            for g_idx in range(num_g_real):
                tmp_cmc_sum += (tmp_cmc[g_idx] / (g_idx + 1.)) * raw_cmc[g_idx]
                num_rel += raw_cmc[g_idx]
            all_AP[q_idx] = tmp_cmc_sum / num_rel

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    # compute averaged cmc
    cdef float[:] avg_cmc = np.zeros(max_rank, dtype=np.float32)
    for rank_idx in range(max_rank):
        for q_idx in range(num_q):
            avg_cmc[rank_idx] += all_cmc[q_idx, rank_idx]
        avg_cmc[rank_idx] /= num_valid_q

    cdef float mAP = 0
    for q_idx in range(num_q):
        mAP += all_AP[q_idx]
    mAP /= num_valid_q

    return np.asarray(avg_cmc).astype(np.float32), mAP


# Compute the cumulative sum
cdef void function_cumsum(cython.numeric[:] src, cython.numeric[:] dst, int64_t n):
    cdef int64_t i
    dst[0] = src[0]
    for i in range(1, n):
        dst[i] = src[i] + dst[i - 1]