from tkinter import Label
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random

from torch_geometric.nn import GCNConv
import gc

from torch_geometric.utils import num_nodes


def ADloss(z1, hard):
    device = z1.device
    hard_batch = hard.size()[0]
    num_nodes = z1.size()[0]  # 所有节点个数
    index = np.arange(num_nodes)
    random.shuffle(index)  # 打乱所有顺序
    mask = torch.tensor(index[:hard_batch], dtype=torch.long).to(device)
    loss = nn.MSELoss()
    losses = loss(z1[mask], hard)

    return losses


def Dis(Discriminator, z, hard):
    device = z.device
    hard_batch = hard.size()[0]
    num_nodes = z.size()[0]  # 所有节点个数
    index = np.arange(num_nodes)
    random.shuffle(index)  # 打乱所有顺序
    mask = torch.tensor(index[:hard_batch], dtype=torch.long).to(device)

    z_label = Discriminator(z[mask])
    hard_label = Discriminator(hard)
    loss = nn.MSELoss(reduction='mean')
    losses = loss(z_label, hard_label)
    return losses


class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Discriminator, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, out_channels),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        label = self.fcs(x) + self.linear_shortcut(x)
        label_activation = nn.Sigmoid()
        label = label_activation(label)
        return label


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model
        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class ADNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        # ! out_channels: 负样本batch；
        super(ADNet, self).__init__()
        self.base_model = base_model
        # self. = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]

    def Generate_hard(self, z1, hard):
        hard = torch.matmul(hard.T, z1)
        # hard = hard_T.T
        return hard


class HNGCL(torch.nn.Module):
    def __init__(self, encoder, num_hidden, num_proj_hidden, tau=0.5, alpha=0.6):
        super(HNGCL, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.num_hidden = num_hidden
        self.alpha = alpha

    def generate_hard_neg_samples(self, z: torch.Tensor, x: torch.Tensor, hard_sample_num: int):
        device = z.device
        num_nodes = x.size()[0]  # 所有节点个数
        index = np.arange(num_nodes)
        random.shuffle(index)  # 打乱所有顺序
        mask = torch.tensor(index[:hard_sample_num], dtype=torch.long).to(device)
        # print(x[mask].shape)# 256,128

        print(z.shape)
        z1 = torch.unsqueeze(z, 1).to(device)

        source = torch.ones(z1.size()[0], hard_sample_num, 1).to(device)
        # print(source.shape) #256,256,1
        source = torch.bmm(source, z1)  # 矩阵乘法 256, 256, 128 1*128的元素变成了256倍。
        # 256,hard_sample_num,128#print(source.shape)
        # hard_neg_samples = self.alpha * source + (1 - self.alpha) * x[mask] #先拓维
        alpha = torch.mm(F.normalize(z), F.normalize(x[mask]).t())

        hard_neg_samples = alpha * source + (1 - alpha) * x[mask]
        # hard_neg_samples = self.alpha * source + (1 - self.alpha) *
        # 256,128
        # print(hard_neg_samples,hard_neg_samples.shape)
        hard_neg_samples = hard_neg_samples.to(z1.device)
        return hard_neg_samples

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        # 256,128;  11701,128
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, semi = False):
        f = lambda x: torch.exp(x / self.tau)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        if semi == 'False':
            refl_sim = f(self.sim(z1, z1))
            between_sim = f(self.sim(z1, z2))
            return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        else:
            between_sim = f(self.sim(z1, z2))
            return -torch.log(between_sim.diag() /  between_sim.sum(1) )

    def semi_hard_loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        hard_sim = f(self.sim(z1, z3))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + hard_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss_hard_neg(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, batch_size: int, weight: float):
        # Space complexity: O(BN) (semi_loss: O(N^2))

        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        # ! Add mixup
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]
            hard_sim = f(self.sim(z1[mask], z3))
            # hard_sim_inter = f(self.sim(z2[mask], z3))

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1) + weight * hard_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))  # + hard_sim_inter.sum(1)
            # losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            #                          / (between_sim[:, i * batch_size:(i + 1) * batch_size].diag() + hard_sim.sum(1) + hard_sim_inter.sum(1)
            #                             )))
        return torch.cat(losses)

    def batched_simple_loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, batch_size: int, weight: float):
        # Space complexity: O(BN) (semi_loss: O(N^2))

        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        # ! Add mixup
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]
            hard_sim = f(self.sim(z1[mask], z3))
            # hard_sim_inter = f(self.sim(z2[mask], z3))

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (between_sim.sum(1) + weight * hard_sim.sum(1))))
        return torch.cat(losses)

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None, semi = False):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        if batch_size is None:
            l1 = self.semi_loss(h1, h2, semi)
            if hasattr(torch.cuda, 'empty_cache'):
                gc.collect()
                torch.cuda.empty_cache()
            l2 = self.semi_loss(h2, h1, semi)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            gc.collect()
            torch.cuda.empty_cache()
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        del h1, h2, l1, l2
        gc.collect()
        torch.cuda.empty_cache()

        return ret

    def loss_neg(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None, weight=1.0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        h3 = self.projection(z3)

        if batch_size is None:
            l1 = self.semi_hard_loss(h1, h2, h3)
            if hasattr(torch.cuda, 'empty_cache'):
                gc.collect()
                torch.cuda.empty_cache()
            l2 = self.semi_hard_loss(h2, h1, h3)
        else:
            l1 = self.batched_semi_loss_hard_neg(h1, h2, h3, batch_size, weight=weight)
            gc.collect()
            torch.cuda.empty_cache()
            l2 = self.batched_semi_loss_hard_neg(h2, h1, h3, batch_size, weight=weight)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        del h1, h2, h3, l1, l2
        gc.collect()
        torch.cuda.empty_cache()

        return ret


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
