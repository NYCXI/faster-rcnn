import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Memory(nn.Module):
    def __init__(self, classes, n_classes, bank_size=10):
        super(Memory, self).__init__()
        self.classed = classes
        self.n_classes = n_classes
        self.bank_size = bank_size
        self.ins_dim = 2048

        self.memory = torch.zeros(self.n_classes, self.bank_size, self.ins_dim).cuda().requires_grad_(False)

        self.memory_pos = torch.zeros(self.n_classes).long().cuda().requires_grad_(False)
        

    def forward(self, instances, instance_labels=None):
        if self.training:
            dis = torch.stack([F.pairwise_distance(ins, self.memory.mean(dim=1), p=2.0) for ins in instances])
            
            cls_prob = torch.argmin(dis, dim=1)
            cls_prob = cls_prob.unsqueeze(1)

            RCNN_cls_acc = (cls_prob == instance_labels).sum().type(torch.float32) / cls_prob.shape[0]

            # print('instances:{}, instance_label:{}'.format(instances.shape, instance_labels.shape))
            
            for i in range(instances.shape[0]):
                label = instance_labels[i]

                if self.memory_pos[label] < self.bank_size:
                    # fill memory bank if it is not full
                    self.memory[label, self.memory_pos[label]] = instances[i]
                    self.memory_pos[label] += 1
                    
                else:
                    # update memory bank
                    dis = F.pairwise_distance(instances[i], self.memory[label].squeeze(0), p=2.0)
                    self.memory[label, torch.argmax(dis)] = instances[i]
                    
            
            #print('memory state:{}'.format(self.memory_pos))
            
            return cls_prob, RCNN_cls_acc

        else:
            similarity = F.cosine_similarity(torch.unsqueeze(instances, 0), self.memory, dim=2)
            return torch.argmax(similarity) / self.n_classes
    
