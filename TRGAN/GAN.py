import pandas as pd
import numpy as np
import copy
import random

import torch
from torch import optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance, entropy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from rdt.transformers.numerical import ClusterBasedNormalizer, GaussianNormalizer
from rdt.transformers.categorical import FrequencyEncoder
from scipy import signal
from functorch import vmap

from TRGAN.encoders import *









"""
GENERATOR
"""


def init_weight_seq(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.normal_(layer.weight, 0, 0.02)


def convolve_vec(tensor, filter):
    # return torch.nn.functional.conv1d(tensor.view(1, 1, -1), filter.to(device).view(1, 1, -1), padding='same').view(-1)
    return torch.nn.functional.conv1d(
        tensor.detach().cpu().view(1, 1, -1), filter.view(1, 1, -1), padding="same"
    ).view(-1)


conv_vec = vmap(convolve_vec)
# gauss_filter_dim = 25


class Generator(nn.Module):
    def __init__(self, z_dim, data_dim, h_dim, num_blocks, gauss_filter_dim, device):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.data_dim = data_dim
        self.h_dim = h_dim
        self.num_blocks = num_blocks
        self.gauss_filter_dim = gauss_filter_dim
        self.device = device

        self.fc1 = nn.Linear(self.z_dim, self.h_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.lrelu = nn.LeakyReLU(0.1)
        # self.relu = nn.ELU(0.9)
        # self.relu = nn.PReLU()

        self.fc2 = nn.Linear(self.h_dim, self.data_dim)
        self.tanh = nn.Tanh()
        self.layernorm0 = nn.LayerNorm(self.h_dim)
        # self.layernorm0 = nn.BatchNorm1d(self.h_dim)

        self.linear_layers = nn.ModuleList(
            [nn.Linear(3 * 2**6, self.h_dim) for _ in range(self.num_blocks)]
        )
        self.linear_layers_conv1 = nn.ModuleList(
            [nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)]
        )
        self.linear_layers_conv2 = nn.ModuleList(
            [nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)]
        )
        self.linear_layers_conv3 = nn.ModuleList(
            [nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)]
        )

        self.feed_forward_generator_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.h_dim, self.h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(self.h_dim, self.h_dim),
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.feed_forward_generator_layers2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.h_dim, self.h_dim),
                    nn.ReLU(),
                    # nn.Dropout(0.2),
                    nn.Linear(self.h_dim, self.h_dim),
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.layernorm_layers_1 = nn.ModuleList(
            [nn.LayerNorm(self.h_dim) for _ in range(self.num_blocks)]
        )
        self.layernorm_layers_2 = nn.ModuleList(
            [nn.LayerNorm(self.h_dim) for _ in range(self.num_blocks)]
        )
        # self.layernorm_layers_1 = nn.ModuleList([nn.BatchNorm1d(self.h_dim) for _ in range(self.num_blocks)])
        # self.layernorm_layers_2 = nn.ModuleList([nn.BatchNorm1d(self.h_dim) for _ in range(self.num_blocks)])

        self.filter1 = signal.windows.gaussian(gauss_filter_dim, 0.5)
        self.filter2 = signal.windows.gaussian(gauss_filter_dim, 1)
        self.filter3 = signal.windows.gaussian(gauss_filter_dim, 3)

        # self.init_weights()
        # self.linear_layers.apply(init_weight_seq)
        # self.linear_layers_conv1.apply(init_weight_seq)
        # self.linear_layers_conv2.apply(init_weight_seq)
        # self.linear_layers_conv3.apply(init_weight_seq)

    def init_weights(self):
        torch.nn.init.normal_(self.fc1.weight, 0, 0.02)
        torch.nn.init.normal_(self.fc2.weight, 0, 0.02)

    def forward(self, x):
        x_size = x.size()
        out = self.layernorm0(self.relu(self.fc1(x)))

        for i in range(self.num_blocks):
            res = out

            x1 = conv_vec(
                out,
                torch.FloatTensor(self.filter1).expand(
                    x_size[0], self.gauss_filter_dim
                ),
            ).to(self.device)
            x2 = conv_vec(
                out,
                torch.FloatTensor(self.filter2).expand(
                    x_size[0], self.gauss_filter_dim
                ),
            ).to(self.device)
            x3 = conv_vec(
                out,
                torch.FloatTensor(self.filter3).expand(
                    x_size[0], self.gauss_filter_dim
                ),
            ).to(self.device)

            x1 = self.lrelu(self.linear_layers_conv1[i](x1))
            x2 = self.lrelu(self.linear_layers_conv2[i](x2))
            x3 = self.lrelu(self.linear_layers_conv3[i](x3))

            out = torch.cat([x1, x2, x3], dim=1)

            out = self.linear_layers[i](out)

            # out = self.feed_forward_generator_layers2[i](out)

            # add & norm
            out += res
            out = self.layernorm_layers_1[i](out)

            # #feed forward
            # res = out
            # out = self.feed_forward_generator_layers[i](out)
            # # #add & norm
            # out += res
            # out = self.layernorm_layers_2[i](out)

        out = self.fc2(out)
        return self.tanh(out)


class Supervisor(nn.Module):
    def __init__(self, z_dim, data_dim, h_dim, num_blocks, gauss_filter_dim, device):
        super(Supervisor, self).__init__()

        self.z_dim = z_dim
        self.data_dim = data_dim
        self.h_dim = h_dim
        self.num_blocks = num_blocks
        self.gauss_filter_dim = gauss_filter_dim
        self.device = device

        self.fc1 = nn.Linear(self.z_dim, self.h_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.lrelu = nn.LeakyReLU(0.1)
        # self.relu = nn.ELU(0.9)
        # self.relu = nn.PReLU()

        self.fc2 = nn.Linear(self.h_dim, self.data_dim)
        self.tanh = nn.Tanh()

        self.layernorm0 = nn.LayerNorm(self.h_dim)
        # self.layernorm0 = nn.BatchNorm1d(self.h_dim)

        # self.linear_layers = nn.ModuleList([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.num_blocks)])
        self.linear_layers = nn.ModuleList(
            [nn.Linear(3 * 2**6, self.h_dim) for _ in range(self.num_blocks)]
        )
        self.linear_layers_conv1 = nn.ModuleList(
            [nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)]
        )
        self.linear_layers_conv2 = nn.ModuleList(
            [nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)]
        )
        self.linear_layers_conv3 = nn.ModuleList(
            [nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)]
        )

        self.feed_forward_generator_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.h_dim, self.h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(self.h_dim, self.h_dim),
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.feed_forward_generator_layers2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.h_dim, self.h_dim),
                    nn.ReLU(),
                    # nn.Dropout(0.2),
                    nn.Linear(self.h_dim, self.h_dim),
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.layernorm_layers_1 = nn.ModuleList(
            [nn.LayerNorm(self.h_dim) for _ in range(self.num_blocks)]
        )
        self.layernorm_layers_2 = nn.ModuleList(
            [nn.LayerNorm(self.h_dim) for _ in range(self.num_blocks)]
        )
        # self.layernorm_layers_1 = nn.ModuleList([nn.BatchNorm1d(self.h_dim) for _ in range(self.num_blocks)])
        # self.layernorm_layers_2 = nn.ModuleList([nn.BatchNorm1d(self.h_dim) for _ in range(self.num_blocks)])

        self.filter1 = signal.windows.gaussian(gauss_filter_dim, 0.5)
        self.filter2 = signal.windows.gaussian(gauss_filter_dim, 1)
        self.filter3 = signal.windows.gaussian(gauss_filter_dim, 3)

        # self.init_weights()
        # self.linear_layers.apply(init_weight_seq)
        # self.linear_layers_conv1.apply(init_weight_seq)
        # self.linear_layers_conv2.apply(init_weight_seq)
        # self.linear_layers_conv3.apply(init_weight_seq)

    def init_weights(self):
        torch.nn.init.normal_(self.fc1.weight, 0, 0.02)
        torch.nn.init.normal_(self.fc2.weight, 0, 0.02)

    def forward(self, x):
        x_size = x.size()
        out = self.layernorm0(self.relu(self.fc1(x)))

        for i in range(self.num_blocks):
            res = out

            x1 = conv_vec(
                out,
                torch.FloatTensor(self.filter1).expand(
                    x_size[0], self.gauss_filter_dim
                ),
            ).to(self.device)
            x2 = conv_vec(
                out,
                torch.FloatTensor(self.filter2).expand(
                    x_size[0], self.gauss_filter_dim
                ),
            ).to(self.device)
            x3 = conv_vec(
                out,
                torch.FloatTensor(self.filter3).expand(
                    x_size[0], self.gauss_filter_dim
                ),
            ).to(self.device)

            x1 = self.lrelu(self.linear_layers_conv1[i](x1))
            x2 = self.lrelu(self.linear_layers_conv2[i](x2))
            x3 = self.lrelu(self.linear_layers_conv3[i](x3))

            out = torch.cat([x1, x2, x3], dim=1)

            out = self.linear_layers[i](out)

            # out = self.feed_forward_generator_layers2[i](out)

            # add & norm
            out += res
            out = self.layernorm_layers_1[i](out)

            # #feed forward
            # res = out
            # out = self.feed_forward_generator_layers[i](out)
            # # #add & norm
            # out += res
            # out = self.layernorm_layers_2[i](out)

        out = self.fc2(out)
        return self.tanh(out)


class Discriminator(nn.Module):
    def __init__(self, data_dim, h_dim, num_blocks, gauss_filter_dim, device):
        super(Discriminator, self).__init__()

        self.data_dim = data_dim
        self.h_dim = h_dim
        self.num_blocks = num_blocks
        self.gauss_filter_dim = gauss_filter_dim
        self.device = device

        self.fc1 = nn.Linear(self.data_dim, self.h_dim)
        self.lrelu = nn.LeakyReLU(0.1)
        # self.relu = nn.ELU(0.9)
        self.relu = nn.PReLU()

        self.layernorm0 = nn.LayerNorm(self.h_dim)
        # self.layernorm0 = nn.BatchNorm1d(self.h_dim)

        # self.linear_layers = nn.ModuleList([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.num_blocks)])
        self.linear_layers = nn.ModuleList(
            [nn.Linear(3 * 2**6, self.h_dim) for _ in range(self.num_blocks)]
        )
        self.linear_layers_conv1 = nn.ModuleList(
            [nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)]
        )
        self.linear_layers_conv2 = nn.ModuleList(
            [nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)]
        )
        self.linear_layers_conv3 = nn.ModuleList(
            [nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)]
        )

        self.feed_forward_discriminator_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.h_dim, self.h_dim),
                    nn.ReLU(),
                    # nn.Dropout(0.15),
                    nn.Linear(self.h_dim, self.h_dim),
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.feed_forward_discriminator_layers2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.h_dim, self.h_dim),
                    nn.ReLU(),
                    # nn.Dropout(0.1),
                    nn.Linear(self.h_dim, self.h_dim),
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.layernorm_layers_1 = nn.ModuleList(
            [nn.LayerNorm(self.h_dim) for _ in range(self.num_blocks)]
        )
        self.layernorm_layers_2 = nn.ModuleList(
            [nn.LayerNorm(self.h_dim) for _ in range(self.num_blocks)]
        )
        # self.layernorm_layers_1 = nn.ModuleList([nn.BatchNorm1d(self.h_dim) for _ in range(self.num_blocks)])
        # self.layernorm_layers_2 = nn.ModuleList([nn.BatchNorm1d(self.h_dim) for _ in range(self.num_blocks)])

        self.filter1 = signal.windows.gaussian(gauss_filter_dim, 0.5)
        self.filter2 = signal.windows.gaussian(gauss_filter_dim, 1)
        self.filter3 = signal.windows.gaussian(gauss_filter_dim, 3)

        self.fc2 = nn.Linear(self.h_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # self.init_weights()
        # self.linear_layers.apply(init_weight_seq)
        # self.linear_layers_conv1.apply(init_weight_seq)
        # self.linear_layers_conv2.apply(init_weight_seq)
        # self.linear_layers_conv3.apply(init_weight_seq)

    def init_weights(self):
        torch.nn.init.normal_(self.fc1.weight, 0, 0.02)
        torch.nn.init.normal_(self.fc2.weight, 0, 0.02)

    def forward(self, x):
        x_size = x.size()
        out = self.layernorm0(self.relu(self.fc1(x)))

        for i in range(self.num_blocks):
            res = out

            x1 = conv_vec(
                out,
                torch.FloatTensor(self.filter1).expand(
                    x_size[0], self.gauss_filter_dim
                ),
            ).to(self.device)
            x2 = conv_vec(
                out,
                torch.FloatTensor(self.filter2).expand(
                    x_size[0], self.gauss_filter_dim
                ),
            ).to(self.device)
            x3 = conv_vec(
                out,
                torch.FloatTensor(self.filter3).expand(
                    x_size[0], self.gauss_filter_dim
                ),
            ).to(self.device)

            x1 = self.lrelu(self.linear_layers_conv1[i](x1))
            x2 = self.lrelu(self.linear_layers_conv2[i](x2))
            x3 = self.lrelu(self.linear_layers_conv3[i](x3))

            out = torch.cat([x1, x2, x3], dim=1)

            out = self.linear_layers[i](out)

            # out = self.feed_forward_discriminator_layers2[i](out)

            # add & norm
            out += res
            out = self.layernorm_layers_1[i](out)

            # # #feed forward
            # res = out
            # out = self.feed_forward_discriminator_layers[i](out)

            # # #add & norm
            # out += res
            # out = self.layernorm_layers_2[i](out)

        out = self.fc2(out)
        return out


def grad_penalty(discriminator, real_data, gen_data, device):
    batch_size = real_data.size()[0]
    t = torch.rand((batch_size, 1), requires_grad=True).to(device)
    t = t.expand_as(real_data)

    # mixed sample from real and fake; make approx of the 'true' gradient norm
    interpol = t * real_data + (1 - t) * gen_data

    prob_interpol = discriminator(interpol).to(device)
    torch.autograd.set_detect_anomaly(True)
    gradients = grad(
        outputs=prob_interpol,
        inputs=interpol.to(device),
        grad_outputs=torch.ones(prob_interpol.size()).to(device),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1).to(device)
    # grad_norm = torch.norm(gradients, dim=1).mean()
    # self.losses['gradient_norm'].append(grad_norm.item())

    # add epsilon for stability
    eps = 1e-10
    gradients_norm = torch.sqrt(
        torch.sum(gradients**2, dim=1, dtype=torch.double) + eps
    )

    return 10 * (
        torch.max(
            torch.zeros(1, dtype=torch.double).to(device), gradients_norm.mean() - 1
        )
        ** 2
    )





"""
SCENARIO MODELLING
"""


def change_scenario(X_oh, mcc, data, rate):
    X_oh = copy.deepcopy(X_oh)

    idx_scenario_1 = np.where(X_oh[mcc] == 1)[0]
    idx_scenario_1_compl = np.setdiff1d(X_oh.index, idx_scenario_1)

    new_idx_sc_1 = random.sample(list(idx_scenario_1_compl), rate)

    X_oh.loc[new_idx_sc_1, mcc] = np.ones(len(new_idx_sc_1)).astype(int)
    X_oh = X_oh.astype(int)

    X_oh.loc[
        new_idx_sc_1,
        np.setdiff1d(X_oh.iloc[:, : len(data["mcc"].unique())].columns, mcc),
    ] = 0

    return X_oh


# def create_cond_vector_scenario(X_oh_sc, encoder_onehot, date_transformations, behaviour_cl_enc, encoder, X_cont, X_cl, scaler):
#     X_oh_emb = encoder_onehot(torch.FloatTensor(X_oh_sc.values).to(device)).detach().numpy()

#     data_transformed = np.concatenate([X_cont, X_oh_emb], axis=1)
#     data_transformed = scaler.transform(data_transformed)
#     data_transformed = np.concatenate([X_cl, data_transformed], axis=1)

#     X_emb = encoder(torch.FloatTensor(data_transformed).to(device)).detach().cpu().numpy()

#     cond_vector = np.concatenate([X_emb, date_transformations, behaviour_cl_enc], axis=1)

#     return cond_vector


def change_scenario_rnf(X_oh, value, mcc_by_values, data, rate):
    X_oh = copy.deepcopy(X_oh)

    queue_idx = X_oh.index

    for mcc in mcc_by_values[value]:
        mcc = "mcc_" + str(mcc)

        idx_scenario_1 = np.where(X_oh[mcc] == 1)[0]
        idx_scenario_1_compl = np.setdiff1d(queue_idx, idx_scenario_1)
        queue_idx = idx_scenario_1_compl

        new_idx_sc_1 = random.sample(list(idx_scenario_1_compl), rate)

        X_oh.loc[new_idx_sc_1, mcc] = np.ones(len(new_idx_sc_1)).astype(int)
        X_oh = X_oh.astype(int)

        X_oh.loc[
            new_idx_sc_1,
            np.setdiff1d(X_oh.iloc[:, : len(data["mcc"].unique())].columns, mcc),
        ] = 0

    return X_oh


def sample_scenario(
    n_samples,
    generator,
    supervisor,
    noise_dim,
    cond_vector,
    X_oh_sc,
    scaler,
    X_cl,
    X_cont,
    encoder_onehot,
    encoder,
    data,
    behaviour_cl_enc,
    date_feature,
    name_client_id,
    time="synth",
    model_time="poisson",
    n_splits=2,
    opt_time=True,
    xi_array=[],
    q_array=[],
    device="cpu",
):
    X_oh_emb = (
        encoder_onehot(torch.FloatTensor(X_oh_sc.values).to(device)).detach().numpy()
    )

    data_transformed = np.concatenate([X_cont, X_oh_emb], axis=1)
    data_transformed = scaler.transform(data_transformed)
    data_transformed = np.concatenate([X_cl, data_transformed], axis=1)

    X_emb_cv = (
        encoder(torch.FloatTensor(data_transformed).to(device)).detach().cpu().numpy()
    )

    if n_samples <= len(cond_vector):
        synth_time, cond_vector_new = sample_cond_vector_with_time(
            n_samples,
            len(cond_vector),
            X_emb_cv,
            data,
            behaviour_cl_enc,
            date_feature,
            name_client_id,
            time,
            model_time,
            n_splits,
            opt_time,
            xi_array,
            q_array,
        )

        noise = torch.FloatTensor(dclProcess(n_samples - 1, noise_dim)).to(device)
        z = torch.cat(
            [
                noise.to(device),
                torch.FloatTensor(cond_vector_new[:n_samples]).to(device),
            ],
            axis=1,
        ).to(device)
        synth_data = (
            supervisor(
                torch.cat(
                    [
                        generator(z).detach(),
                        torch.FloatTensor(cond_vector_new[:n_samples]).to(device),
                    ],
                    dim=1,
                )
            )
            .detach()
            .cpu()
            .numpy()
        )

    else:
        synth_time, cond_vector_new = sample_cond_vector_with_time(
            n_samples,
            len(cond_vector),
            X_emb_cv,
            data,
            behaviour_cl_enc,
            date_feature,
            name_client_id,
            time,
            model_time,
            n_splits,
            opt_time,
            xi_array,
            q_array,
        )

        noise = torch.FloatTensor(dclProcess(n_samples - 1, noise_dim)).to(device)
        z = torch.cat(
            [noise.to(device), torch.FloatTensor(cond_vector_new).to(device)], axis=1
        ).to(device)
        synth_data = (
            supervisor(
                torch.cat(
                    [
                        generator(z).detach(),
                        torch.FloatTensor(cond_vector_new).to(device),
                    ],
                    dim=1,
                )
            )
            .detach()
            .cpu()
            .numpy()
        )

    return synth_data, synth_time


"""
PRIVACY PRESERVING
"""


def k_anonymity(data, quasi_id, sensitive_att):
    equiv_classes_k_anon = data.groupby(quasi_id).count()[sensitive_att].reset_index()

    return equiv_classes_k_anon.iloc[:, -1].min()


def l_diversity(data, quasi_id, sensitive_att):
    equiv_classes_l_div = data.groupby(quasi_id).nunique()[sensitive_att].reset_index()

    return equiv_classes_l_div.iloc[:, -1].min()


def t_closeness(data, quasi_id, sensitive_att):
    equiv_classes_t_clos = (
        data.groupby(quasi_id)
        .apply(
            lambda x: wasserstein_distance(x[sensitive_att], data[sensitive_att].values)
        )
        .reset_index()
    )

    return equiv_classes_t_clos.iloc[:, -1].max()


def l_diversity_cont(data, quasi_id, sensitive_att):
    equiv_classes_l_div_c = (
        data.groupby(quasi_id)
        .apply(lambda x: np.exp(entropy(x[sensitive_att])))
        .reset_index()
    )

    return equiv_classes_l_div_c.iloc[:, -1].min()
