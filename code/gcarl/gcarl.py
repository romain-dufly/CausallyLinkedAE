"""GCaRL"""

import numpy as np
import torch
import torch.nn as nn
from itertools import combinations

# =============================================================
# =============================================================
class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        m, _ = torch.max(x.reshape([*x.shape[:-1], x.shape[-1] // self._pool_size, self._pool_size]), dim=-1)
        return m


# =============================================================
# =============================================================
class Net(nn.Module):
    def __init__(self, h_sizes, num_group, directions, num_xdim, phi_type='gauss-maxout', hz_sizes=None, hp_sizes=None, phi_share=True, pool_size=2):
        """ Network model
        Args:
             h_sizes: number of nodes for each layer (excluding the input layer)
             num_group: number of groups
             num_xdim: dimension of input
             phi_type: model type of phi ('maxout')
             hz_sizes: number of nodes for each layer (hz)
             hp_sizes: number of nodes for each layer (hp)
             phi_share: share phi across group-pairs or not
             pool_size: pool size of max-out
        """
        super(Net, self).__init__()

        self.num_group = num_group
        self.num_xdim = num_xdim
        self.num_hdim = [hs[-1] for hs in h_sizes]
        self.group_combs = list(combinations(np.arange(num_group), 2))
        self.num_comb = len(self.group_combs)
        self.maxout = Maxout(pool_size)
        self.phi_type = phi_type
        self.phi_share = phi_share

        self.num_dim = max(self.num_hdim)

        if hz_sizes is None:
            hz_sizes = h_sizes.copy()
        # h
        if len(h_sizes) > 0:
            h = []
            bn = []
            for m in range(num_group):
                if len(h_sizes[m]) > 1:
                    hm = [nn.Linear(self.num_xdim[m], h_sizes[m][0] * pool_size)]
                    hm = hm + [nn.Linear(h_sizes[m][k - 1], h_sizes[m][k] * pool_size) for k in range(1, len(h_sizes[m]) - 1)]
                    hm.append(nn.Linear(h_sizes[m][-2], h_sizes[m][-1], bias=False))
                elif len(h_sizes[m]) > 0:
                    hm = []
                    hm = [nn.Linear(self.num_xdim[m], h_sizes[m][-1], bias=False)]      ###
                else:
                    hm = []
                h.append(nn.ModuleList(hm))
                bn.append(nn.BatchNorm1d(num_features=self.num_hdim[m]))
            #
            self.h = nn.ModuleList(h)
            self.bn = nn.ModuleList(bn)
        else:
            self.h = [[] for _ in np.arange(num_group)]
            self.bn = nn.ModuleList([nn.BatchNorm1d(num_features=self.num_hdim) for i in np.arange(num_group)])
        # hz
        if len(hz_sizes) > 0:
            hz = []
            bnz = []
            for m in range(num_group):
                num_xdim_hz = [x * 2 for x in self.num_hdim] if self.phi_type in {'gauss-mlp'} else [x for x in self.num_hdim]
                num_xdim_hz = [self.num_dim * 2 for x in self.num_hdim] if self.phi_type in {'gauss-mlp'} else [self.num_dim for x in self.num_hdim] ###
                if len(hz_sizes) > 1:
                    hm = [nn.Linear(num_xdim_hz[m], hz_sizes[0] * pool_size)]
                    hm = hm + [nn.Linear(hz_sizes[k - 1], hz_sizes[k] * pool_size) for k in range(1, len(hz_sizes) - 1)]
                    hm.append(nn.Linear(hz_sizes[-2], hz_sizes[-1], bias=False))
                else:
                    hm = [nn.Linear(num_xdim_hz[m], hz_sizes[0], bias=False)]
                hz.append(nn.ModuleList(hm))
                bnz.append(nn.BatchNorm1d(num_features=self.num_dim)) ###
            #
            self.hz = nn.ModuleList(hz)
            self.bnz = nn.ModuleList(bnz)
        else:
            self.hz = []
            self.bnz = nn.BatchNorm1d(num_features=self.num_hdim)
        # phi
        if self.phi_type == 'gauss-maxout':

            for _ in range(self.num_comb):
                self.w0 = nn.Parameter(torch.zeros([self.num_dim, self.num_dim]))
                self.w1 = nn.Parameter(torch.zeros([self.num_dim, self.num_dim]))

            self.zw = nn.Parameter(torch.zeros([num_group, self.num_dim, 2]))
            if phi_share:
                self.pw = nn.Parameter(torch.ones([1, 2]))
                self.pb = nn.Parameter(torch.zeros([1]))
            else:
                self.pw = nn.Parameter(torch.ones([self.num_comb, 2]))
                self.pb = nn.Parameter(torch.zeros([self.num_comb]))
        elif self.phi_type == 'gauss-relu':

            for _ in range(self.num_comb):
                self.w0 = nn.Parameter(torch.zeros([self.num_dim, self.num_dim]))
                self.w1 = nn.Parameter(torch.zeros([self.num_dim, self.num_dim]))

            print("num combs: ", self.num_comb)
            # freeze the weights that correspond to directions not in the directions matrix
            for l in range(self.num_comb):
                a = self.group_combs[l][0]
                b = self.group_combs[l][1]
                if directions[a, b] == 0:
                    print('Freezing weights for group pair %d-%d' % (a, b))
                    self.w0.requires_grad = False
                else:
                    print('Training weights for group pair %d-%d' % (a, b))
                    self.w0.requires_grad = True
                if directions[b, a] == 0:
                    print('Freezing weights for group pair %d-%d' % (b, a))
                    self.w1.requires_grad = False
                else:
                    print('Training weights for group pair %d-%d' % (b, a))
                    self.w1.requires_grad = True

            #self.w = torch.stack(self.w_directions_list).reshape([self.num_dim, self.num_dim, 2, self.num_comb])
            self.zw = nn.Parameter(torch.zeros([num_group, self.num_dim, 2]))
        elif self.phi_type == 'gauss-mlp':
            
            for _ in range(self.num_comb):
                self.w0 = nn.Parameter(torch.zeros([self.num_dim, self.num_dim]))
                self.w1 = nn.Parameter(torch.zeros([self.num_dim, self.num_dim]))

            self.zw = nn.Parameter(torch.zeros([num_group, self.num_dim, 2]))
            if phi_share:
                hp = [nn.Linear(1, hp_sizes[0])]
                hp = hp + [nn.Linear(hp_sizes[k - 1], hp_sizes[k]) for k in range(1, len(hp_sizes) - 1)]
                hp.append(nn.Linear(hp_sizes[-2], hp_sizes[-1]))
                bnp = nn.BatchNorm1d(num_features=1)
                self.hp = nn.ModuleList(hp)
                self.bnp = bnp
            else:
                hp = []
                bnp = []
                for c in range(self.num_comb):
                    if len(hz_sizes) > 1:
                        hc = [nn.Linear(1, hp_sizes[0])]
                        hc = hc + [nn.Linear(hp_sizes[k - 1], hp_sizes[k]) for k in range(1, len(hp_sizes) - 1)]
                        hc.append(nn.Linear(hp_sizes[-2], hp_sizes[-1]))
                    else:
                        hc = [nn.Linear(1, hp_sizes[0])]
                    hp.append(nn.ModuleList(hc))
                    bnp.append(nn.BatchNorm1d(num_features=1))
                self.hp = nn.ModuleList(hp)
                self.bnp = nn.ModuleList(bnp)
        elif self.phi_type == 'lap-mlp':
            self.w = nn.Parameter(torch.zeros([self.num_hdim, self.num_hdim, 2, self.num_comb]))
            self.w2 = nn.Parameter(torch.zeros([self.num_hdim, self.num_hdim, self.num_comb]))
            self.b2 = nn.Parameter(torch.zeros([self.num_hdim, self.num_hdim, 2, self.num_comb]))
            self.zw = nn.Parameter(torch.zeros([num_group, self.num_hdim, 2]))
            if phi_share:
                hp = [nn.Linear(1, hp_sizes[0])]
                hp = hp + [nn.Linear(hp_sizes[k - 1], hp_sizes[k]) for k in range(1, len(hp_sizes) - 1)]
                hp.append(nn.Linear(hp_sizes[-2], hp_sizes[-1]))
                bnp = nn.BatchNorm1d(num_features=1)
                self.hp = nn.ModuleList(hp)
                self.bnp = bnp
            else:
                hp = []
                bnp = []
                for c in range(self.num_comb):
                    if len(hz_sizes) > 1:
                        hc = [nn.Linear(1, hp_sizes[0])]
                        hc = hc + [nn.Linear(hp_sizes[k - 1], hp_sizes[k]) for k in range(1, len(hp_sizes) - 1)]
                        hc.append(nn.Linear(hp_sizes[-2], hp_sizes[-1]))
                    else:
                        hc = [nn.Linear(1, hp_sizes[0])]
                    hp.append(nn.ModuleList(hc))
                    bnp.append(nn.BatchNorm1d(num_features=1))
                self.hp = nn.ModuleList(hp)
                self.bnp = nn.ModuleList(bnp)
        else:
            raise ValueError

        self.b = nn.Parameter(torch.zeros([1]))

        # initialize
        for m in range(num_group):
            for k in range(len(self.h[m])):
                torch.nn.init.xavier_uniform_(self.h[m][k].weight)
        if self.phi_type == 'lap-mlp':
            if phi_share:
                for k in range(len(self.hp)):
                    torch.nn.init.constant_(self.hp[k].weight, 0)
                    torch.nn.init.uniform_(self.hp[k].bias, -0.01, 0.01)
                torch.nn.init.constant_(self.hp[-1].bias, 0)
                torch.nn.init.uniform_(self.w2, -0.1, 0.1)
            else:
                for g in range(num_group):
                    for k in range(len(self.hp[g])):
                        torch.nn.init.constant_(self.hp[g][k].weight, 0)
                        torch.nn.init.uniform_(self.hp[g][k].bias, -0.01, 0.01)
                    torch.nn.init.constant_(self.hp[g][-1].bias, 0)
                torch.nn.init.uniform_(self.w2, -0.1, 0.1)

    def forward(self, x, calc_logit=True):
        """ forward
         Args:
             x: input [batch, group, dim]
             calc_logit: obtain logits additionally to h, or not
         """
        batch_size = x[0].shape[0]
        num_group = len(x)
        # num_dim is the max of the latent dimensions
        num_dim = max(self.num_hdim)
        device = x[0].device
        # h
        h_bn = torch.zeros([batch_size, num_group, num_dim], device=device)         #WARNING: THIS IS WHERE IT FAILS WHEN USING DIFFERENT LATENT DIMENSIONS
        for m in range(num_group):
            hm = x[m]                           # x_m to h_m : this is the part that should be changed to use other NN than MLP
            for k in range(len(self.h[m])):
                hm = self.h[m][k](hm)           # Applying the MLP layers
                if k != len(self.h[m]) - 1:
                    hm = self.maxout(hm)        # Maxout inbetween   ### use RELU ?
            h_bn[:, m, :self.num_hdim[m]] = self.bn[m](hm)  # batch normalization
            # set the latest latent variables to zero if num_hdim is smaller than the maximum
            if self.num_hdim[m] < num_dim:
                h_bn[:, m, self.num_hdim[m]:] = 0

        if calc_logit:
            if self.phi_type == 'gauss-maxout':
                w = torch.stack([self.w0, self.w1], dim=2).reshape([self.num_dim, self.num_dim, 2, self.num_comb])
                logits = torch.zeros(batch_size, device=device)
                h_nonlin, _ = torch.max(self.pw[None, None, None, :, :] * (h_bn[:, :, :, None, None] - self.pb[None, None, None, :, None]), dim=-1)  # [batch, group, dim, comb]
                for c in range(len(self.group_combs)):
                    a = self.group_combs[c][0]
                    b = self.group_combs[c][1]
                    if self.phi_share:
                        phi_ab = h_bn[:, b, None, :] * h_nonlin[:, a, :, 0, None]
                        phi_ba = h_bn[:, a, None, :] * h_nonlin[:, b, :, 0, None]
                    else:
                        phi_ab = h_bn[:, b, None, :] * h_nonlin[:, a, :, c, None]
                        phi_ba = h_bn[:, a, None, :] * h_nonlin[:, b, :, c, None]
                    logits = logits + torch.sum(w[None, :, :, 0, c] * phi_ab + w[None, :, :, 1, c] * phi_ba, dim=[1, 2])
                # hz
                hz_bn = torch.zeros_like(h_bn)
                for m in range(num_group):
                    hzm = h_bn[:, m, :]
                    for k in range(len(self.hz[m])):
                        hzm = self.hz[m][k](hzm)
                        if k != len(self.hz[m]) - 1:
                            hzm = self.maxout(hzm)
                    hz_bn[:, m, :] = self.bnz[m](hzm)
                logits_z = torch.sum(self.zw[None, :, :, 0] * hz_bn ** 2 + self.zw[None, :, :, 1] * hz_bn, dim=[1, 2])

            elif self.phi_type == 'gauss-relu':
                w = torch.stack([self.w0, self.w1], dim=2).reshape([self.num_dim, self.num_dim, 2, self.num_comb])
                logits = torch.zeros(batch_size, device=device)
                h_nonlin = torch.relu(h_bn)[:, :, :, None]  # [batch, group, dim, comb]     # Apply ReLu to the latent variables
                for c in range(len(self.group_combs)):
                    a = self.group_combs[c][0]
                    b = self.group_combs[c][1]
                    if self.phi_share:
                        phi_ab = h_bn[:, b, None, :] * h_nonlin[:, a, :, 0, None]  # [batch, dim, dim] maxout       # phi_ab is the product of the latent variables (relu applied on h_a)
                        phi_ba = h_bn[:, a, None, :] * h_nonlin[:, b, :, 0, None]  # [batch, dim, dim] maxout
                    else:
                        raise NotImplementedError
                    logits = logits + torch.sum(w[None, :, :, 0, c] * phi_ab + w[None, :, :, 1, c] * phi_ba, dim=[1, 2])  # sum everything

                hz_bn = torch.zeros_like(h_bn)
                for m in range(num_group):
                    hzm = h_bn[:, m, :]
                    for k in range(len(self.hz[m])):
                        hzm = self.hz[m][k](hzm)        # mlp on the latent variables
                        if k != len(self.hz[m]) - 1:
                            hzm = self.maxout(hzm)      # maxout activation
                    hz_bn[:, m, :] = self.bnz[m](hzm)
                logits_z = torch.sum(self.zw[None, :, :, 0] * hz_bn ** 2 + self.zw[None, :, :, 1] * hz_bn, dim=[1, 2])      # sum for intra group

            elif self.phi_type == 'gauss-mlp':
                w = torch.stack([self.w0, self.w1], dim=2).reshape([self.num_dim, self.num_dim, 2, self.num_comb])
                logits = torch.zeros(batch_size, device=device)
                h_nonlin = torch.zeros([batch_size, 2, num_dim, self.num_comb], device=device)
                # pre-calculate phi if phi is shared
                if self.phi_share:
                    hp = h_bn.reshape([-1, 1])
                    for k in range(len(self.hp)):
                        hp = self.hp[k](hp)     # apply MLP
                        if k != len(self.hp) - 1:
                            hp = torch.tanh(hp) # tanh activation function
                    hp = self.bnp(hp).reshape(h_bn.shape)
                #
                for c in range(len(self.group_combs)):
                    a = self.group_combs[c][0]
                    b = self.group_combs[c][1]
                    if self.phi_share:
                        h_nonlin_c = hp[:, [a, b], :]
                    else:
                        hp = h_bn[:, [a, b], :].reshape([-1, 1])
                        for k in range(len(self.hp[c])):
                            hp = self.hp[c][k](hp)
                            if k != len(self.hp[c]) - 1:
                                hp = torch.tanh(hp)
                        hp = self.bnp[c](hp)
                        h_nonlin_c = hp.reshape([h_bn.shape[0], 2, h_bn.shape[-1]])
                    phi_ab = h_bn[:, b, None, :] * h_nonlin_c[:, 0, :, None]
                    phi_ba = h_bn[:, a, None, :] * h_nonlin_c[:, 1, :, None]
                    logits = logits + torch.mean(w[None, :, :, 0, c] * phi_ab + w[None, :, :, 1, c] * phi_ba, dim=[1, 2]) * 1e2
                    h_nonlin[:, :, :, c] = h_nonlin_c
                # hz
                hz_bn = torch.zeros_like(h_bn)
                for m in range(num_group):
                    if self.phi_share:
                        hzm = torch.cat([h_bn[:, m, :], hp[:, m, :]], dim=1)
                    else:
                        hzm = torch.cat([h_bn[:, m, :], torch.tanh(h_bn[:, m, :])], dim=1)
                    for k in range(len(self.hz[m])):
                        hzm = self.hz[m][k](hzm)
                        if k != len(self.hz[m]) - 1:
                            hzm = self.maxout(hzm)
                    hz_bn[:, m, :] = self.bnz[m](hzm)
                logits_z = torch.sum(self.zw[None, :, :, 0] * hz_bn ** 2 + self.zw[None, :, :, 1] * hz_bn, dim=[1, 2])

            elif self.phi_type == 'lap-mlp':
                logits = torch.zeros(batch_size, device=x.device)
                h_nonlin = torch.zeros([batch_size, 2, num_dim, self.num_comb], device=x[0].device)
                # pre-calculate phi if phi is shared
                if self.phi_share:
                    hp = h_bn.reshape([-1, 1])
                    for k in range(len(self.hp)):
                        hp = self.hp[k](hp)
                        if k != len(self.hp) - 1:
                            hp = torch.tanh(hp)
                    hp = self.bnp(hp).reshape(h_bn.shape)
                #
                for c in range(len(self.group_combs)):
                    a = self.group_combs[c][0]
                    b = self.group_combs[c][1]
                    if self.phi_share:
                        h_nonlin_c = hp[:, [a, b], :]
                    else:
                        hp = h_bn[:, [a, b], :].reshape([-1, 1])
                        for k in range(len(self.hp[c])):
                            hp = self.hp[c][k](hp)
                            if k != len(self.hp[c]) - 1:
                                hp = torch.tanh(hp)
                        hp = self.bnp[c](hp)
                        h_nonlin_c = hp.reshape([h_bn.shape[0], 2, h_bn.shape[-1]])
                    phi_ab = self.w2[None, :, :, c] * h_bn[:, b, None, :] - torch.abs(self.w2[None, :, :, c]) * h_nonlin_c[:, 0, :, None]
                    phi_ba = self.w2[:, :, c].T[None, :, :] * h_bn[:, a, None, :] - torch.abs(self.w2[:, :, c].T[None, :, :]) * h_nonlin_c[:, 1, :, None]
                    logits = logits + torch.mean(self.w[None, :, :, 0, c] * torch.abs(phi_ab) + self.w[None, :, :, 1, c] * torch.abs(phi_ba), dim=[1, 2]) * 1e2
                    h_nonlin[:, :, :, c] = h_nonlin_c
                # hz
                hz_bn = torch.zeros_like(x)
                for m in range(num_group):
                    if self.phi_share:
                        hzm = torch.cat([h_bn[:, m, :], hp[:, m, :]], dim=1)
                    else:
                        hzm = torch.cat([h_bn[:, m, :], torch.tanh(h_bn[:, m, :])], dim=1)
                    for k in range(len(self.hz[m])):
                        hzm = self.hz[m][k](hzm)
                        if k != len(self.hz[m]) - 1:
                            hzm = self.maxout(hzm)
                    hz_bn[:, m, :] = self.bnz[m](hzm)
                logits_z = torch.sum(self.zw[None, :, :, 0] * hz_bn ** 2 + self.zw[None, :, :, 1] * hz_bn, dim=[1, 2])

            logits = logits + logits_z + self.b     # sum all (inter and intra group) and add bias

        else:
            logits = None

        return logits, h_bn, h_nonlin