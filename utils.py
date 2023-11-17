import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, max_epochs=30, lambda_u=75):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u * linear_rampup(epoch, max_epochs)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def get_labels_from_nn(unlabeled_features, unlabeled_prediction, num_class, gt_label, graphk=5):
    all_features = F.normalize(unlabeled_features, dim=1, p=2)
    weight = torch.matmul(unlabeled_features, unlabeled_features.transpose(0, 1))
    weight[weight < 0] = 0
    weight.diagonal(0).fill_(0)
    values, indexes = torch.topk(weight, graphk)
    weight[weight < values[:, -1].view(-1, 1)] = 0
    weight[weight > 0.00001] = 1
    comb_pred = torch.torch.matmul(weight, unlabeled_prediction)
    comb_pred /= graphk
    scores, hard_label_nn = torch.max(comb_pred, dim=1)
    acc_nn = accuracy(comb_pred, gt_label)
    # print('acc of nn is: %3f' % (acc_nn))

    return comb_pred, hard_label_nn, acc_nn

def get_labels_from_lp(labeled_features, labeled_onehot_gt, unlabeled_features, gt_label, num_class, dis='cos', solver='closedform', graphk=20, alpha=0.75):

    num_labeled = labeled_features.size(0)
    if num_labeled > 100000:
        print('too many labeled data, randomly select a subset')
        indices = torch.randperm(num_labeled)[:10000]
        labeled_features = labeled_features[indices]
        labeled_onehot_gt  = labeled_onehot_gt[indices]
        num_labeled = 10000

    num_unlabeled = unlabeled_features.size(0)
    num_all = num_unlabeled + num_labeled
    all_features = torch.cat((labeled_features, unlabeled_features), dim=0)
    unlabeled_zero_gt = torch.zeros(num_unlabeled, num_class)
    all_gt = torch.cat((labeled_onehot_gt, unlabeled_zero_gt), dim=0)
    ### calculate the affinity matrix
    if dis == 'cos':
        all_features = F.normalize(all_features, dim=1, p=2)
        weight = torch.matmul(all_features, all_features.transpose(0, 1))
        weight[weight < 0] = 0
        values, indexes = torch.topk(weight, graphk)
        weight[weight < values[:, -1].view(-1, 1)] = 0
        weight = weight + weight.transpose(0, 1)
    weight.diagonal(0).fill_(0)  ## change the diagonal elements with inplace operation.
    if solver == 'closedform':
        D = weight.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + 1e-8))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, num_all)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(num_all, 1)
        S = D1 * weight * D2  ############ same with D3 = torch.diag(D_sqrt_inv)  S = torch.matmul(torch.matmul(D3, weight), D3)
        pred_all = torch.matmul(torch.inverse(torch.eye(num_all) - alpha * S + 1e-8), all_gt)
    del weight
    pred_unl = pred_all[num_labeled:, :]
    #### add a fix value
    min_value = torch.min(pred_unl, 1)[0]
    min_value[min_value > 0] = 0
    pred_unl = pred_unl - min_value.view(-1, 1)

    pred_unl = pred_unl / pred_unl.sum(1).view(-1, 1)

    soft_label_lp = pred_unl
    scores, hard_label_lp = torch.max(soft_label_lp, dim=1)
    acc_lp = accuracy(soft_label_lp, gt_label)
    # print('acc of lp is: %3f' % (acc_lp))

    return soft_label_lp, hard_label_lp, acc_lp

def get_labels_from_kmeans(initial_centers_array, target_u_feature, num_class, gt_label, T=1.0, max_iter=100, target_l_feature=None):
    ## initial_centers: num_cate * feature dim
    ## target_u_feature: num_u * feature dim
    if type(target_l_feature) == torch.Tensor:  ### if there are some labeled data of the same domain
        target_u_feature_array = torch.cat((target_u_feature, target_l_feature), dim=0).numpy()
    else:
        target_u_feature_array = target_u_feature.numpy()

    initial_centers_array = initial_centers_array.numpy()
    kmeans = KMeans(n_clusters=num_class, random_state=0, init=initial_centers_array,
                             max_iter=max_iter).fit(target_u_feature_array)
    Ind = kmeans.labels_
    Ind_tensor = torch.from_numpy(Ind)
    centers = kmeans.cluster_centers_  ### num_category * feature_dim
    centers_tensor = torch.from_numpy(centers)

    centers_tensor_unsq = torch.unsqueeze(centers_tensor, 0)
    target_u_feature_unsq = torch.unsqueeze(target_u_feature, 1)
    L2_dis = ((target_u_feature_unsq - centers_tensor_unsq)**2).mean(2)
    soft_label_kmeans = torch.softmax(1 + 1.0 / (L2_dis + 1e-8), dim=1)
    scores, hard_label_kmeans = torch.max(soft_label_kmeans, dim=1)

    acc_kmeans = accuracy(soft_label_kmeans, gt_label)
    # print('acc of kmeans is: %3f' % (acc_kmeans))

    return soft_label_kmeans, hard_label_kmeans, acc_kmeans

def accuracy(output, target):
    """Computes the precision"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct = correct[:1].view(-1).float().sum(0, keepdim=True)
    res = correct.mul_(100.0 / batch_size)
    return res

def to_onehot(label, num_classes):
    identity = torch.eye(num_classes).to(label.device)
    onehot = torch.index_select(identity, 0, label)
    return onehot