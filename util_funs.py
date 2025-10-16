import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
from scipy.io import loadmat

def EProjSimplex_new(v, k=1):
    n = v.numel()
    v0 = v - torch.mean(v) + k / n
    vmin = torch.min(v0)

    if vmin < 0:
        lambda_m = torch.tensor(0.0, device=v.device, dtype=v.dtype)
        eps = torch.finfo(v.dtype).eps

        for _ in range(100):
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = torch.sum(posidx)
            g = -npos.float()
            f = torch.sum(v1[posidx]) - k

            if abs(f) < 1e-10:
                break

            # 避免除零
            if g == 0:
                break
            lambda_m = lambda_m - f / g

        x = torch.clamp(v1, min=0)
    else:
        x = v0
    return x

def solve_hungarian(A):
    """
    使用匈牙利算法解决分配问题
    """
    from scipy.optimize import linear_sum_assignment
    device = A.device
    n = A.size(0)
    C = -A.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(C)
    P = torch.zeros(n, n, device=device)
    P[row_ind, col_ind] = 1
    return P

def L2_distance_1(a, b):
    """
    计算两个矩阵之间的平方欧几里得距离
    ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B

    Parameters:
        a, b: 两个矩阵，每列是一个数据点 (torch.Tensor)

    Returns:
        d: 距离矩阵 (torch.Tensor)
    """
    # 处理一维情况
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)

    if a.size(0) == 1:
        a = torch.cat([a, torch.zeros(1, a.size(1), device=a.device)], dim=0)
        b = torch.cat([b, torch.zeros(1, b.size(1), device=b.device)], dim=0)

    # 计算平方范数
    aa = torch.sum(a * a, dim=0)  # shape: (n_a,)
    bb = torch.sum(b * b, dim=0)  # shape: (n_b,)

    # 计算点积
    ab = torch.mm(a.t(), b)  # shape: (n_a, n_b)

    # 使用广播机制计算距离矩阵
    d = aa.unsqueeze(1) + bb.unsqueeze(0) - 2 * ab

    # 确保结果为实数和非负
    d = d.real
    d = torch.clamp(d, min=0)

    return d

def NormalizeFea(fea, inplace=False):
    """Normalize features using PyTorch only"""
    if not inplace:
        fea = fea.clone()

    # PyTorch 实现 L2 归一化
    norms = torch.norm(fea, p=2, dim=0, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)  # 避免除零
    fea = fea / norms

    return fea




def load_data_from_mat(file_path):

    # 加载MAT文件
    mat_data = loadmat(file_path)

    # 提取X和gt变量
    mat_X = mat_data['X']  # 假设X是cell数组，1行V列
    mat_gt = mat_data['gt']  # 真实标签列向量
    # 将MATLAB cell转换为Python列表
    X = []
    for i in range(mat_X.shape[1]):  # 遍历列
        # 从MATLAB cell中提取矩阵并转换为numpy数组
        view_data = mat_X[0, i]
        view_data = view_data.T  ##是否转置
        # 转换数据类型为PyTorch支持的格式
        if view_data.dtype == np.uint16:
            view_data = view_data.astype(np.float32)  # 转换为float32
        elif view_data.dtype == np.uint8:
            view_data = view_data.astype(np.float32)
        # 可以添加其他数据类型的转换

        X.append(view_data)

    # 确保gt是1维数组
    gt = mat_gt.flatten()

    return X, gt

def perform_clustering(Z_star, gt, Clus_num):
    U, _, _ = torch.svd(Z_star.t())
    UU = U[:, :Clus_num]

    result1 = []
    for i in range(10):
        kmeans = KMeans(n_clusters=Clus_num, n_init=20)
        pre_labels = kmeans.fit_predict(UU.cpu().real.numpy())
        result1.append(cluster(gt.cpu().numpy(), pre_labels))
    return np.mean(result1, axis=0)



import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as compute_nmi
from sklearn.metrics.cluster._supervised import check_clusterings
from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from sklearn.metrics import silhouette_score
from scipy.io import savemat



def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)
def get_cluster_result(features, n_clusters):
    km = KMeans(n_clusters=n_clusters, n_init=10)
    pred = km.fit_predict(features)
    return pred


def compute_acc(Y, Y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    from scipy.optimize import linear_sum_assignment

    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size


def compute_fscore_pre_recall(labels_true, labels_pred):
    # b3_precision_recall_fscore就是Fscore
    """Compute the B^3 variant of precision, recall and F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """
    # Check that labels_* are 1d arrays and have the same size

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError("input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return f_score,precision,recall


def cluster(labels, pred):
    labels = np.reshape(labels, np.shape(pred))
    if np.min(labels) == 1:
        labels -= 1

    nmi = compute_nmi(labels, pred)

    acc = compute_acc(labels, pred)

    fscore, pre, recal = compute_fscore_pre_recall(labels, pred)
    pur = purity(labels, pred)
    ari = adjusted_rand_score(labels, pred)

    results= [acc, nmi, pur, fscore, pre, recal, ari]
    return results