import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def TripletLoss(feat, label):
    assert feat.shape[0] == label.shape[0]

    loss = torch.Tensor([0.0]).float().cuda()
    #similarity_matrix = compute_similarity(feat)

    for i in range(feat.shape[0]):
        anchor = feat[i]

        positive_set = label[:] == label[i]
        positive_set[i] = False
        positive_set = feat[positive_set]

        negative_set = label[:] != label[i]
        negative_set = feat[negative_set]

        # print('anchor:{}, positive_set:{}, negative_set:{}'.format(anchor.shape, positive_set.shape, negative_set.shape))
        if positive_set.shape[0] > 0 and negative_set.shape[0] > 0:
            l_i = torch.stack([F.triplet_margin_loss(anchor.unsqueeze(0), positive_set[i].unsqueeze(0), negative_set, reduction='none')for i in range(positive_set.shape[0])])
            #l_i = torch.stack([F.triplet_margin_with_distances_loss(anchor.unsqueeze(0), positive_set[i].unsqueeze(0), negative_set, reduction='none')for i in range(positive_set.shape[0])])
            l_i = l_i.sum() / (l_i > 0).sum()
            if torch.isnan(l_i):
                pass
            else:
                loss += l_i
            #print('loss:{}'.format(l_i.sum() / (l_i > 0).sum()))


    loss /= feat.shape[0]
    #print('loss:{}'.format(type(loss)))
    
    return loss


def compute_similarity(feat):
    return torch.stack([F.pairwise_distance(feat[i], feat, p=2.0) for i in range(feat.shape[0])])
