from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pydantic import BaseModel
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from TRGAN.dataset.metadata import TRGANMetadata
from TRGAN.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorConfig,
    default_feature_extractor_config,
)
from TRGAN.GAN import Discriminator, Generator, Supervisor
from TRGAN.time_generation import optimize_xi_by_deltas_split, synthetic_time
from TRGAN.utils import deltas_by_client


class TRGANDataset:
    def __init__(
        self, df: pd.DataFrame, metadata: TRGANMetadata, n_splits: int
    ):
        self.data = df
        client_id = metadata.get_big_n_unique_id_feature()
        time_id = metadata.get_time_feature()

        self.deltas = deltas_by_client(df, client_id=client_id, time_id=time_id)
        self.xiP_array, self.idx_array = optimize_xi_by_deltas_split(
            self.deltas, n_splits
        )
        self.length_dates_by_clients = df.groupby(client_id)[time_id].count().values - 1
        self.first_dates_by_clients = df.groupby(client_id)[time_id].first().values


def dclProcess(N, M):
    T = 10
    theta = 15
    delta = 20

    Z1 = np.random.normal(0.0, 1.0, [M, N])
    X = np.zeros([M, N + 1])

    X[:, 0] = np.random.normal(0.0, 0.2, M)

    time = np.zeros([N + 1])
    dt = T / float(N)

    for i in range(0, N):
        X[:, i + 1] = (
            X[:, i]
            - 1 / theta * X[:, i] * dt
            + np.sqrt((1 - (X[:, i]) ** 2) / (theta * (delta + 1)))
            * np.sqrt(dt)
            * Z1[:, i]
        )

        if (X[:, i + 1] > 1).any():
            X[np.where(X[:, i + 1] >= 1)[0], i + 1] = 0.9999

        if (X[:, i + 1] < -1).any():
            X[np.where(X[:, i + 1] <= -1)[0], i + 1] = -0.9999

        time[i + 1] = time[i] + dt

    return X.T


class TimeGenerationConfig(BaseModel):
    n_split: int = 4


class TRGANConfig(BaseModel):
    lr_generator: float = 3e-4
    lr_discriminator: float = 4e-4
    lr_supervisor: float = 3e-4
    lr_discriminator2: float = 4e-4
    noise_dimension: int = 20
    hidden_dimension: int = 2**6
    num_blocks_gen: int = 2
    num_blocks_dis: int = 2
    gauss_filter_dim: int = 20
    lambda1: float = 3
    alpha: float = 0.75
    batch_size: int = 64
    num_epochs: int = 1
    time_generation: TimeGenerationConfig = TimeGenerationConfig()
    feature_extraction: FeatureExtractorConfig = default_feature_extractor_config()


class TRGAN:
    def __init__(
        self,
        metadata: TRGANMetadata,
        *,
        config: Optional[TRGANConfig] = None,
        device: str = "cpu"
    ):
        if config is None:
            config = TRGANConfig()

        self.cat_features = metadata.categorical_small_n_unique_features()

        self.cont_features = metadata.continuous_features()

        self.date_feature = metadata.get_time_feature()

        self.client_features = metadata.categorical_biq_n_unique_features()
        self.feature_extractor = FeatureExtractor(
            conf=config.feature_extraction,
            cont_features=self.cont_features,
            cat_features=self.cat_features,
            client_features=self.client_features,
            date_feature=self.date_feature,
        )
        self.device = device
        self.config = config

    def build_GAN(self):
        data_dim = self.feature_extractor.get_data_dimension()
        self.dim_Vc = (
            self.feature_extractor.get_condition_vector_autoencoder_dimension()
        )
        self.generator = Generator(
            self.config.noise_dimension + self.dim_Vc,
            data_dim,
            self.config.hidden_dimension,
            self.config.num_blocks_gen,
            self.config.gauss_filter_dim,
            self.device,
        ).to(self.device)
        self.discriminator = Discriminator(
            data_dim + self.dim_Vc,
            self.config.hidden_dimension,
            self.config.num_blocks_dis,
            self.config.gauss_filter_dim,
            self.device,
        ).to(self.device)
        self.supervisor = Supervisor(
            data_dim + self.dim_Vc,
            data_dim,
            self.config.hidden_dimension,
            self.config.num_blocks_gen,
            self.config.gauss_filter_dim,
            self.device,
        ).to(self.device)
        self.discriminator2 = Discriminator(
            data_dim + self.dim_Vc,
            self.config.hidden_dimension,
            self.config.num_blocks_dis,
            self.config.gauss_filter_dim,
            self.device,
        ).to(self.device)

        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.config.lr_generator,
            betas=(0.9, 0.999),
            amsgrad=True,
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.lr_discriminator,
            betas=(0.9, 0.999),
            amsgrad=True,
        )
        self.optimizer_S = optim.Adam(
            self.supervisor.parameters(),
            lr=self.config.lr_supervisor,
            betas=(0.9, 0.999),
            amsgrad=True,
        )
        self.optimizer_D2 = optim.Adam(
            self.discriminator2.parameters(),
            lr=self.config.lr_discriminator2,
            betas=(0.9, 0.999),
            amsgrad=True,
        )

        self.scheduler_G = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer_G, gamma=0.97
        )
        self.scheduler_D = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer_D, gamma=0.97
        )
        self.scheduler_S = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer_S, gamma=0.97
        )
        self.scheduler_D2 = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer_D2, gamma=0.97
        )

    def train_extractor(self, data: TRGANDataset):
        assert isinstance(data, TRGANDataset)
        self.feature_extractor.train(data.data)

    def train(self, data: TRGANDataset):
        self.train_extractor(data)
        self.build_GAN()
        self.train_GAN(data)

    def train_GAN(self, data: TRGANDataset):
        assert isinstance(data, TRGANDataset)
        X_emb = self.feature_extractor.embed(data.data)
        cond_vec = self.feature_extractor.get_condition_vector(data.data)

        data_with_cv = torch.cat(
            [torch.FloatTensor(X_emb), torch.FloatTensor(cond_vec)], axis=1
        )

        idx_batch_array = np.arange(
            len(data_with_cv) // self.config.batch_size * self.config.batch_size
        )
        last_idx = np.setdiff1d(np.arange(len(data_with_cv)), idx_batch_array)
        split_idx = np.split(idx_batch_array, self.config.batch_size)
        split_idx_perm = np.random.permutation(split_idx)
        split_idx_perm = np.append(split_idx_perm, last_idx)

        loader_g = DataLoader(
            data_with_cv[split_idx_perm],
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        self.loss_array = []

        b_d1 = 0.001
        b_d2 = 0.001
        epochs = tqdm(range(self.config.num_epochs))
        for epoch in epochs:
            for batch_idx, X in enumerate(loader_g):
                loss = torch.nn.MSELoss()
                batch_size = X.size(0)

                Vc = X[:, -self.dim_Vc :].to(self.device)

                # noise = torch.randn(batch_size, dim_noise).to(device)
                noise = torch.FloatTensor(
                    dclProcess(batch_size - 1, self.config.noise_dimension)
                ).to(self.device)
                z = torch.cat([noise, Vc], dim=1).to(self.device)
                fake = self.generator(z).detach()

                X = X.to(self.device)
                self.discriminator.trainable = True
                disc_loss = (
                    -torch.mean(self.discriminator(X))
                    + torch.mean(self.discriminator(torch.cat([fake, Vc], dim=1)))
                ).to(self.device)

                fake_super = self.supervisor(torch.cat([fake, Vc], dim=1)).to(
                    self.device
                )
                disc2_loss = (
                    -torch.mean(self.discriminator2(X))
                    + torch.mean(
                        self.discriminator2(torch.cat([fake_super, Vc], dim=1))
                    )
                ).to(self.device)

                self.optimizer_D.zero_grad()
                disc_loss.backward()
                self.optimizer_D.step()

                self.optimizer_D2.zero_grad()
                disc2_loss.backward()
                self.optimizer_D2.step()

                for dp in self.discriminator.parameters():
                    dp.data.clamp_(-b_d1, b_d1)

                for dp in self.discriminator2.parameters():
                    dp.data.clamp_(-b_d2, b_d2)

                if batch_idx % 2 == 0:
                    self.discriminator.trainable = False

                    gen_loss1 = -torch.mean(
                        self.discriminator(torch.cat([self.generator(z), Vc], dim=1))
                    ).to(self.device)
                    supervisor_loss = (
                        -torch.mean(
                            self.discriminator2(
                                torch.cat(
                                    [
                                        self.supervisor(
                                            torch.cat(
                                                [self.generator(z), Vc], dim=1
                                            ).detach()
                                        ),
                                        Vc,
                                    ],
                                    dim=1,
                                )
                            )
                        )
                        + self.config.lambda1
                        * loss(
                            self.supervisor(
                                torch.cat([self.generator(z), Vc], dim=1).detach()
                            ),
                            X[:, : -self.dim_Vc],
                        )
                    ).to(self.device)

                    gen_loss = (
                        self.config.alpha * gen_loss1
                        + (1 - self.config.alpha) * supervisor_loss
                    )

                    supervisor_loss2 = (
                        (
                            -torch.mean(
                                self.discriminator2(
                                    torch.cat(
                                        [
                                            self.supervisor(
                                                torch.cat(
                                                    [self.generator(z), Vc], dim=1
                                                ).detach()
                                            ),
                                            Vc,
                                        ],
                                        dim=1,
                                    )
                                )
                            )
                        )
                        + self.config.lambda1
                        * loss(
                            self.supervisor(
                                torch.cat([self.generator(z), Vc], dim=1).detach()
                            ),
                            X[:, : -self.dim_Vc],
                        )
                    ).to(self.device)

                    self.optimizer_G.zero_grad()
                    gen_loss.backward()
                    self.optimizer_G.step()
                    self.optimizer_S.zero_grad()
                    supervisor_loss2.backward()
                    self.optimizer_S.step()

            if (
                np.isnan(disc_loss.item())
                or np.isnan(disc2_loss.item())
                or np.isnan(gen_loss.item())
                or np.isnan(supervisor_loss2.item())
            ):
                print(disc_loss.item())
                print(disc2_loss.item())
                print(gen_loss.item())
                print(supervisor_loss2.item())
                break

            epochs.set_description(
                "Discriminator Loss: %.5f || Discriminator 2 Loss: %.5f || Generator Loss: %.5f || Supervisor Loss: %.5f"
                % (
                    disc_loss.item(),
                    disc2_loss.item(),
                    gen_loss.item(),
                    supervisor_loss2.item(),
                )
            )
            self.loss_array.append(
                [
                    disc_loss.item(),
                    disc2_loss.item(),
                    gen_loss.item(),
                    supervisor_loss2.item(),
                ]
            )

    def sample_embedding(self, data: TRGANDataset, n_samples: int):
        synth_time = synthetic_time(data)
        df = data.data.copy()
        df = df.iloc[:n_samples]
        df[self.feature_extractor.date_feature] = synth_time[:n_samples]  #
        cond_vector = self.feature_extractor.get_condition_vector(df)

        noise = torch.FloatTensor(
            dclProcess(n_samples - 1, self.config.noise_dimension)
        ).to(self.device)

        z = torch.cat(
            [noise.to(self.device), torch.FloatTensor(cond_vector).to(self.device)],
            axis=1,
        ).to(self.device)

        d = (
            self.supervisor(
                torch.cat(
                    [
                        self.generator(z).detach(),
                        torch.FloatTensor(cond_vector).to(self.device),
                    ],
                    dim=1,
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
        return d, synth_time

    def sample(self, data: TRGANDataset, n_samples: pd.DataFrame):
        synth_data, synth_time = self.sample_embedding(data, n_samples)
        synth_data = self.feature_extractor.inverse_transformation(synth_data)
        synth_data["date"] = synth_time
        return synth_data


def train_trgan(
    data: pd.DataFrame,
    metadata: TRGANMetadata,
    output_path: Path,
    *,
    epochs: int = 10,
    customer_prefix: str = "customer",
    feature_extractor_config: Optional[FeatureExtractorConfig] = None
):
    model = TRGAN(
        metadata,
        epochs,
        device="cpu",
    )
