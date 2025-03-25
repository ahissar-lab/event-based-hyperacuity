"""
This module defines loss functions used during training.
Add custom loss functions (e.g. NTXentLoss) as needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, device=None, p_drop_negative_samples=None):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.p_drop_negative_samples = p_drop_negative_samples
        self.large_negative_constant = -1e9

    def forward(self, z_i, z_j):
        '''
        :param z_i: first batch of embeddings
        :param z_j: second batch of embeddings
        :return: loss
        '''
        batch_size = z_i.size(0)
        z = torch.cat((z_i, z_j), dim=0)
        sim_matrix = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)
        positive_samples = torch.cat((sim_ij, sim_ji), dim=0).view(2 * batch_size, 1)
        negative_samples = sim_matrix[~torch.eye(2 * batch_size, dtype=bool, device=self.device)].view(2 * batch_size,
                                                                                                       -1)
        #ignore a fraction of negative samples by setting them to a large negative constant
        #this is done for both training and validation
        #the reason is that if we have to many negative samples, reducing batch size is more
        # computationally expensive than randomly reducing the number of negative samples
        if self.p_drop_negative_samples is not None:
            n_negative_samples = int(negative_samples.size(1))
            n_negative_samples_to_drop = int(n_negative_samples * self.p_drop_negative_samples)
            negative_samples_to_drop = torch.randint(n_negative_samples, (n_negative_samples_to_drop,))
            negative_samples[:, negative_samples_to_drop] = self.large_negative_constant


        labels = torch.zeros(2 * batch_size).to(self.device).long() #zero labels denote positive samples
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = F.cross_entropy(logits, labels)
        return loss


if __name__ == '__main__':
    loss = NTXentLoss()
    #random embeddings
    z_i = torch.randn(128, 64)
    z_j = torch.randn(128, 64)
    print('random embeddings',loss(z_i, z_j))
    #perfect embeddings
    z_i = torch.randn(128, 64)
    z_j = z_i
    print('perfect embeddings',loss(z_i, z_j))