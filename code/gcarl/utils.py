""" Utilities
"""


import numpy as np
import scipy as sp
import os
import shutil
import tarfile
import scipy.stats as ss
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from itertools import combinations
# from cdt.metrics import SHD

from subfunc.showdata import *
from subfunc.munkres import Munkres


# =============================================================
# =============================================================
def w_to_directed(w, zero_tolerance=1e-10):
    """ Convert w to directed graph
    Args:
        w: [node x node x dim]
    Returns:
        wdir: directed w, NaN if not determined
    """

    num_dim, _, _, num_comb = w.shape
    wdir = w.copy()
    for c in range(num_comb):
        for i in range(num_dim):
            for j in range(num_dim):
                if np.abs(wdir[i, j, 0, c]) > np.abs(wdir[j, i, 1, c]):
                    wdir[j, i, 1, c] = 0
                elif np.abs(wdir[i, j, 0, c]) < np.abs(wdir[j, i, 1, c]):
                    wdir[i, j, 0, c] = 0
                # elif (np.abs(wdir[i, j, 0, c]) == np.abs(wdir[j, i, 1, c])) and (wdir[i, j, 0, c] != 0):
                elif (np.abs(wdir[i, j, 0, c]) == np.abs(wdir[j, i, 1, c])) and (np.abs(wdir[i, j, 0, c]) > zero_tolerance):
                    # cannot determine the direction
                    wdir[i, j, 0, c] = np.nan
                    wdir[j, i, 1, c] = np.nan

    return wdir


# =============================================================
# =============================================================
def w_permute(w, sort_idx, num_group, win=None):
    """ Convert w to directed graph
    Args:
        w: [node x node x dim]
    Returns:
        wdir: directed w, NaN if not determined
    """

    num_dim, _, _, num_combs = w.shape
    group_combs = list(combinations(np.arange(num_group), 2))

    w_perm = np.zeros_like(w)
    for mc in range(len(group_combs)):
        a = group_combs[mc][0]
        b = group_combs[mc][1]
        # a -> b
        w_mc = w[:, :, 0, mc].copy()
        w_mc = w_mc[sort_idx[:, a], :]
        w_mc = w_mc[:, sort_idx[:, b]]
        w_perm[:, :, 0, mc] = w_mc
        # b -> a
        w_mc = w[:, :, 1, mc].copy()
        w_mc = w_mc[sort_idx[:, b], :]
        w_mc = w_mc[:, sort_idx[:, a]]
        w_perm[:, :, 1, mc] = w_mc

    if win is not None:
        win_perm = np.zeros_like(win)
        for m in range(num_group):
            w_m = win[:, :, m].copy()
            w_m = w_m[sort_idx[:, m], :]
            w_m = w_m[:, sort_idx[:, m]]
            win_perm[:, :, m] = w_m
    else:
        win_perm = None

    return w_perm, win_perm


# =============================================================
# =============================================================
def w_threshold(w, thresh_ratio=0, comb_wise=False):
    """ Apply threshold to w
    Args:
        w: [node x node x dim]
        thresh_ratio: Threshold ratio compared to the maximum absolute value
        comb_wise: Evaluate for each group-combination (True) or not (False)
    Returns:
        wthresh: thresholded w
    """

    num_node, _, _, num_comb = w.shape
    if comb_wise:
        wthresh = np.zeros_like(w)
        for c in range(num_comb):
            wc = w[:, :, :, c].copy()
            thval = np.max(np.abs(wc)) * thresh_ratio
            wc[np.abs(wc) <= thval] = 0
            wthresh[:, :, :, c] = wc
    else:
        wthresh = w.copy()
        thval = np.max(np.abs(wthresh)) * thresh_ratio
        wthresh[np.abs(wthresh) <= thval] = 0

    # if len(w.shape) > 3:
    #     w_shape_orig = w.shape
    #     w = np.reshape(w, [w.shape[0], w.shape[1], -1])
    #     w_reshape_flag = True
    # else:
    #     w_reshape_flag = False
    #
    # num_node, _, num_dim = w.shape
    # wthresh = np.zeros_like(w)
    # for d in range(num_dim):
    #     wd = w[:, :, d].copy()
    #     thval = np.max(np.abs(wd)) * thresh_ratio
    #     wd[np.abs(wd) <= thval] = 0
    #     wthresh[:, :, d] = wd
    #
    # if w_reshape_flag:
    #     wthresh = np.reshape(wthresh, w_shape_orig)

    return wthresh


# =============================================================
# =============================================================
def eval_dag(wtrue, west, num_group):
    """ Evaluate estimated causal sturcture
    Args:
        wtrue: [node x node x dim]
        west: [node x node x dim]
        conn_list: list of edges
    Returns:
        F1: [dim, dim]
        precision: [dim, dim]
        recall: [dim, dim]
        FPR: [dim, dim]
        sort_idx
    """

    num_dim, _, _, num_comb = wtrue.shape
    # group_combs = list(combinations(np.arange(num_group), 2))

    precision = np.zeros([4])
    recall = np.zeros([4])
    f1 = np.zeros([4])
    fpr = np.zeros([4])

    cause_true = wtrue != 0

    # evaluate across dimensions
    cause_est = west != 0
    precision[0] = precision_score(cause_true.reshape(-1), cause_est.reshape(-1))
    recall[0] = recall_score(cause_true.reshape(-1), cause_est.reshape(-1))
    f1[0] = f1_score(cause_true.reshape(-1), cause_est.reshape(-1))
    tn, fp, fn, tp = confusion_matrix(cause_true.reshape(-1), cause_est.reshape(-1)).flatten()
    fpr[0] = fp / (fp + tn)

    cause_est = np.transpose(west, [1, 0, 2, 3]) != 0  # transpose
    precision[1] = precision_score(cause_true.reshape(-1), cause_est.reshape(-1))
    recall[1] = recall_score(cause_true.reshape(-1), cause_est.reshape(-1))
    f1[1] = f1_score(cause_true.reshape(-1), cause_est.reshape(-1))
    tn, fp, fn, tp = confusion_matrix(cause_true.reshape(-1), cause_est.reshape(-1)).flatten()
    fpr[1] = fp / (fp + tn)

    cause_est = west[:, :, -1::-1, :] != 0  # flip direction
    precision[2] = precision_score(cause_true.reshape(-1), cause_est.reshape(-1))
    recall[2] = recall_score(cause_true.reshape(-1), cause_est.reshape(-1))
    f1[2] = f1_score(cause_true.reshape(-1), cause_est.reshape(-1))
    tn, fp, fn, tp = confusion_matrix(cause_true.reshape(-1), cause_est.reshape(-1)).flatten()
    fpr[2] = fp / (fp + tn)

    cause_est = np.transpose(west[:, :, -1::-1, :], [1, 0, 2, 3]) != 0  # flip direction and transpose
    precision[3] = precision_score(cause_true.reshape(-1), cause_est.reshape(-1))
    recall[3] = recall_score(cause_true.reshape(-1), cause_est.reshape(-1))
    f1[3] = f1_score(cause_true.reshape(-1), cause_est.reshape(-1))
    tn, fp, fn, tp = confusion_matrix(cause_true.reshape(-1), cause_est.reshape(-1)).flatten()
    fpr[3] = fp / (fp + tn)

    maxidx = np.argmax(f1)
    f1 = f1[maxidx]
    precision = precision[maxidx]
    recall = recall[maxidx]
    fpr = fpr[maxidx]

    # pre_mat = np.zeros([num_dim, num_dim])
    # rec_mat = np.zeros([num_dim, num_dim])
    # f1_mat = np.zeros([num_dim, num_dim])
    # fpr_mat = np.zeros([num_dim, num_dim])
    # for d1 in range(num_dim):
    #     for d2 in range(num_dim):
    #         cause_true = wtrue[:, :, d1].copy()
    #         cause_est = west[:, :, d2].copy()
    #         cause_est_t = cause_est.copy().T
    #
    #         # decide direction of nans favorably
    #         for i in range(num_node):
    #             for j in range(i + 1, num_node):
    #                 if np.isnan(cause_est[i, j]):
    #                     if cause_true[i, j] > 0:
    #                         cause_est[i, j] = 1
    #                         cause_est[j, i] = 0
    #                         cause_est_t[i, j] = 1
    #                         cause_est_t[j, i] = 0
    #                     else:
    #                         cause_est[i, j] = 0
    #                         cause_est[j, i] = 1
    #                         cause_est_t[i, j] = 0
    #                         cause_est_t[j, i] = 1
    #
    #         # vectorize & binarize
    #         cause_true = cause_true[conn_list[0], conn_list[1]] != 0
    #         cause_est = cause_est[conn_list[0], conn_list[1]] != 0
    #         cause_est_t = cause_est_t[conn_list[0], conn_list[1]] != 0
    #
    #         precision = precision_score(cause_true, cause_est)
    #         precision_t = precision_score(cause_true, cause_est_t)
    #
    #         recall = recall_score(cause_true, cause_est)
    #         recall_t = recall_score(cause_true, cause_est_t)
    #
    #         f1 = f1_score(cause_true, cause_est)
    #         f1_t = f1_score(cause_true, cause_est_t)
    #
    #         tn, fp, fn, tp = confusion_matrix(cause_true, cause_est).flatten()
    #         fpr = fp / (fp + tn)
    #         tn_t, fp_t, fn_t, tp_t = confusion_matrix(cause_true, cause_est_t).flatten()
    #         fpr_t = fp_t / (fp_t + tn_t)
    #
    #         # decide trasponse or not based on f1
    #         if f1 >= f1_t:
    #             pre_mat[d1, d2] = precision
    #             rec_mat[d1, d2] = recall
    #             f1_mat[d1, d2] = f1
    #             fpr_mat[d1, d2] = fpr
    #         else:
    #             pre_mat[d1, d2] = precision_t
    #             rec_mat[d1, d2] = recall_t
    #             f1_mat[d1, d2] = f1_t
    #             fpr_mat[d1, d2] = fpr_t

    # # sorting
    # munk = Munkres()
    # indexes = munk.compute(-f1_mat)
    # sort_idx = [idx[1] for idx in indexes]

    return f1, precision, recall, fpr


# =============================================================
# =============================================================
def eval_dag_mat(wtrue, west):
    """ Evaluate estimated causal sturcture
    Args:
        wtrue: [node x node x dim]
        west: [node x node x dim]
        conn_list: list of edges
    Returns:
        F1: [dim, dim]
        precision: [dim, dim]
        recall: [dim, dim]
        FPR: [dim, dim]
        sort_idx
    """

    # num_dim, _, _, num_comb = wtrue.shape
    # group_combs = list(combinations(np.arange(num_group), 2))

    # precision = np.zeros([4])
    # recall = np.zeros([4])
    # f1 = np.zeros([4])
    # fpr = np.zeros([4])

    wtrue_bin = wtrue.copy()
    wtrue_bin[(wtrue_bin != 0) & (~np.isnan(wtrue_bin))] = 1

    west_bin = west.copy()
    west_bin[np.isnan(west_bin)] = wtrue_bin[np.isnan(west_bin)]  # give true information for undetermined edges
    west_bin[(west_bin != 0) & (~np.isnan(west_bin))] = 1

    # remove nan true edges
    wtrue_bin_nonnan = wtrue_bin[~np.isnan(wtrue_bin)].copy()
    west_bin_nonnan = west_bin[~np.isnan(wtrue_bin)].copy()

    # # evaluate
    # shd = SHD(np.nan_to_num(wtrue_bin, nan=0), np.nan_to_num(west_bin, nan=0))
    precision = precision_score(wtrue_bin_nonnan, west_bin_nonnan)
    recall = recall_score(wtrue_bin_nonnan, west_bin_nonnan)
    f1 = f1_score(wtrue_bin_nonnan, west_bin_nonnan)
    tn, fp, fn, tp = confusion_matrix(wtrue_bin_nonnan, west_bin_nonnan).flatten()
    fpr = fp / (fp + tn)

    return f1, precision, recall, fpr


# =============================================================
# =============================================================
def remove_last_group(w, num_group):
    """ Evaluate estimated causal sturcture
    Args:
        w: [node x node x 2 x comb]
    Returns:
        F1: [dim, dim]
        precision: [dim, dim]
        recall: [dim, dim]
        FPR: [dim, dim]
        sort_idx
    """

    combs = list(combinations(np.arange(num_group), 2))
    num_comb = len(combs)

    keep_flag = np.ones(num_comb, dtype=bool)
    for c in range(num_comb):
        if num_group - 1 in combs[c]:
            keep_flag[c] = False
    w_removed = w[:, :, :, keep_flag]

    return w_removed


# =============================================================
# =============================================================
def w_to_wg(w, num_group=None, num_dim=None):
    """ Convert full adjacency matrix to gorup-wise representation
    Args:
        w: [variable x variable]
    Returns:
        w: [variable x variable x 2 x combination]
        win: [variable x variable x group]
    """

    if num_group is not None:
        num_dim = int(w.shape[0] / num_group)
    elif num_dim is not None:
        num_group = int(w.shape[0] / num_dim)
    else:
        raise ValueError

    combs = list(combinations(np.arange(num_group), 2))
    num_comb = len(combs)

    wg = np.zeros([num_dim, num_dim, 2, num_comb])
    for c in range(num_comb):
        a = combs[c][0]
        b = combs[c][1]
        wg[:, :, 0, c] = w[num_dim * a:num_dim * (a + 1), num_dim * b:num_dim * (b + 1)]
        wg[:, :, 1, c] = w[num_dim * b:num_dim * (b + 1), num_dim * a:num_dim * (a + 1)]

    wgin = np.zeros([num_dim, num_dim, num_group])
    for m in range(num_group):
        wgin[:, :, m] = w[num_dim * m:num_dim * (m + 1), num_dim * m:num_dim * (m + 1)]

    return wg, wgin


# =============================================================
# =============================================================
def wg_to_w(wg, num_group, wgin=None):
    """ Evaluate estimated causal sturcture
    Args:
        w: [node x node x 2 x comb]
    Returns:
        F1: [dim, dim]
        precision: [dim, dim]
        recall: [dim, dim]
        FPR: [dim, dim]
        sort_idx
    """

    num_dim, _, _, num_comb = wg.shape
    combs = list(combinations(np.arange(num_group), 2))

    w = np.zeros([num_dim * num_group, num_dim * num_group])
    w[:] = np.nan
    for c in range(num_comb):
        a = combs[c][0]
        b = combs[c][1]
        w[num_dim * a:num_dim * (a + 1), num_dim * b:num_dim * (b + 1)] = wg[:, :, 0, c]
        w[num_dim * b:num_dim * (b + 1), num_dim * a:num_dim * (a + 1)] = wg[:, :, 1, c]

    if wgin is not None:
        for m in range(num_group):
            w[num_dim * m:num_dim * (m + 1), num_dim * m:num_dim * (m + 1)] = wgin[:, :, m]

    return w




# =============================================================
# =============================================================
def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
         method: correlation method ('Pearson' or 'Spearman')
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
     """

    print('Calculating correlation...')

    x = x.copy().T
    y = y.copy().T
    dimx = x.shape[0]
    dimy = y.shape[0]

    # calculate correlation
    if method == 'Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dimy, dimy:]
    elif method == 'Spearman':
        corr, pvalue = sp.stats.spearmanr(y.T, x.T)
        corr = corr[0:dimy, dimy:]
    else:
        raise ValueError
    # corr[np.isnan(corr)] = 0
    if np.max(np.isnan(corr)):
        raise ValueError

    # sort
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dimy, dtype=int)
    for i in range(dimy):
        sort_idx[i] = indexes[i][1]
    sort_idx_other = np.setdiff1d(np.arange(0, dimx), sort_idx)
    sort_idx = np.concatenate([sort_idx, sort_idx_other])

    x_sort = x[sort_idx, :]

    # re-calculate correlation
    if method == 'Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dimy, dimy:]
    elif method == 'Spearman':
        corr_sort, pvalue = sp.stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dimy, dimy:]
    else:
        raise ValueError
    # corr_sort[np.isnan(corr_sort)] = 0

    return corr_sort, sort_idx, x_sort


# ===============================================================
# ===============================================================
def unzip(loadfile, unzipfolder, necessary_word='/storage'):
    """unzip trained model (loadfile) to unzipfolder
    """

    print('load: %s...' % loadfile)
    if loadfile.find(".tar.gz") > -1:
        if unzipfolder.find(necessary_word) > -1:
            if os.path.exists(unzipfolder):
                print('delete savefolder: %s...' % unzipfolder)
                shutil.rmtree(unzipfolder)  # remove folder
            archive = tarfile.open(loadfile)
            archive.extractall(unzipfolder)
            archive.close()
        else:
            assert False, "unzip folder doesn't include necessary word"
    else:
        if os.path.exists(unzipfolder):
            print('delete savefolder: %s...' % unzipfolder)
            shutil.rmtree(unzipfolder)  # remove folder
        os.makedirs(unzipfolder)
        src_files = os.listdir(loadfile)
        for fn in src_files:
            full_file_name = os.path.join(loadfile, fn)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, unzipfolder + '/')

    if not os.path.exists(unzipfolder):
        raise ValueError
