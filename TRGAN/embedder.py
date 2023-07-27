from typing import Callable, List

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel
from rdt.transformers.categorical import FrequencyEncoder
from rdt.transformers.numerical import GaussianNormalizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from rdt.transformers.numerical import ClusterBasedNormalizer

from TRGAN.encoders import (
    Decoder_client_emb,
    Decoder_cont_emb,
    Decoder_onehot,
    Encoder_client_emb,
    Encoder_cont_emb,
    Encoder_onehot,
)


class AutoEncoderConfig(BaseModel):
    dimension: int
    learning_rate: float = 3e-4
    batch_size: int = 128
    epochs: int = 1
    eps: float = 0.5
    differential_privacy_sensitivity: float = 4
    differential_privacy: bool = True


class AbstractEmbedder(ABC):
    def preprocess(self, X: pd.DataFrame):
        pass

    @property
    @abstractmethod
    def dimension(self):
        pass

    @abstractmethod
    def inverse_transform(self, data: np.ndarray):
        pass

    @abstractmethod
    def train(self, X):
        pass

    @abstractmethod
    def encode(self, X: pd.DataFrame):
        pass


class AutoEncoder(AbstractEmbedder):
    def __init__(
        self,
        *,
        encoder,
        decoder,
        config: AutoEncoderConfig,
        loss: Callable,
        device,
        name: str = "AutoEncoder",
    ):
        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        if decoder is not None:
            self.decoder = decoder.to(device)
        else:
            self.decoder = None

        self.config = config
        self.device = device
        self.loss = loss
        self.name = name

    def preprocess(self, X: pd.DataFrame):
        return X

    @property
    def dimension(self):
        return self.config.dimension

    def inverse_transform(self, d):
        return d

    def get_schedulers(self, optimizer_Enc, optimizer_Dec):
        scheduler_Enc = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_Enc, gamma=0.98
        )
        scheduler_Dec = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_Dec, gamma=0.98
        )
        return scheduler_Enc, scheduler_Dec

    def get_optimizers(self):
        optimizer_Enc = torch.optim.Adam(
            self.encoder.parameters(), lr=self.config.learning_rate
        )
        optimizer_Dec = torch.optim.Adam(
            self.decoder.parameters(), lr=self.config.learning_rate
        )
        return optimizer_Enc, optimizer_Dec

    def train(self, X):
        X = self.preprocess(X)
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.history = []
        optimizer_Enc, optimizer_Dec = self.get_optimizers()
        scheduler_Enc, scheduler_Dec = self.get_schedulers(optimizer_Enc, optimizer_Dec)

        loader_onehot = DataLoader(
            torch.FloatTensor(X), self.config.batch_size, shuffle=True
        )

        epochs = tqdm(range(self.config.epochs))

        # privacy parameters
        q = self.config.batch_size / X.shape[1]
        alpha = 2
        delta = 0.1  # or 1/batch_size
        n_iter = X.shape[1] // self.config.batch_size
        sensitivity = (
            self.config.differential_privacy_sensitivity / self.config.batch_size
        )
        std = np.sqrt(
            (
                4
                * q
                * alpha**2
                * (sensitivity) ** 2
                * np.sqrt(2 * n_iter * np.log(1 / delta))
            )
            / (2 * (alpha - 1) * self.config.eps)
        )

        print(f"E_oh with {self.config, delta}-Differential Privacy")

        for epoch in epochs:
            for batch_idx, X in enumerate(loader_onehot):
                H = self.encoder(X.float().to(self.device))
                X_tilde = self.decoder(H.to(self.device))

                loss_value = self.loss(X_tilde, X.to(self.device)).to(self.device)
                if self.config.differential_privacy:
                    criterion = (loss_value + np.random.normal(0, std)).to(self.device)
                else:
                    criterion = loss_value.to(self.device)

                optimizer_Enc.zero_grad()
                optimizer_Dec.zero_grad()

                criterion.backward()

                optimizer_Enc.step()
                optimizer_Dec.step()

            scheduler_Enc.step()
            scheduler_Dec.step()
            self.history.append(loss_value.item())
            # print(f'epoch {epoch}: Loss = {criterion:.8f}')
            epochs.set_description(f"Loss {self.name}: %.9f " % loss_value.item())

    def encode(self, X: pd.DataFrame):
        self.encoder.eval()
        self.decoder.eval()

        X = self.preprocess(X)
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.encoder(torch.FloatTensor(X).to(self.device)).detach().cpu().numpy()


def undummify(df, prefix_sep="_"):
 
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df


class CategoricalEmbedder(AutoEncoder):
    def __init__(
        self, *, config: AutoEncoderConfig, device, name: str, columns: List[str]
    ):
        super().__init__(
            encoder=None,
            decoder=None,
            config=config,
            loss=torch.nn.BCELoss(),
            device=device,
            name=name,
        )
        self.one_hot_encoder = None
      

    def get_schedulers(self, optimizer_Enc, optimizer_Dec):
        scheduler_Enc = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_Enc, gamma=0.98
        )
        scheduler_Dec = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_Dec, gamma=0.98
        )
        return scheduler_Enc, scheduler_Dec

    def preprocess(self, X: pd.DataFrame):
        self.columns = X.columns
        if self.one_hot_encoder is None:
            self.one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            self.one_hot_encoder.set_output(transform="pandas")
            self.one_hot_encoder.fit(X)
        X_oh = self.one_hot_encoder.transform(X)
        if self.encoder is None:
            self.encoder = Encoder_onehot(X_oh.shape[1], self.config.dimension).to(
                self.device
            )
            self.decoder = Decoder_onehot(self.config.dimension, X_oh.shape[1]).to(
                self.device
            )
        self.one_hot_columns = X_oh.columns.tolist()
        return X_oh

    def inverse_transform(self, data: np.ndarray):
        synth_cat_decoded = (
            self.decoder(torch.FloatTensor(data).to(self.device))
                .detach()
                .cpu()
                .numpy()
            )
        
        synth_df_cat = pd.DataFrame(synth_cat_decoded, columns=self.one_hot_columns)
        synth_df_cat_feat_undum = undummify(synth_df_cat)
        return synth_df_cat_feat_undum


class CBNContinuosEmbedderConfig(BaseModel):
    max_clusters: float = 15
    weight_threshold: float = 0.005


class CBNContinuosEmbedder(AbstractEmbedder):
    def __init__(
        self, *, config: CBNContinuosEmbedderConfig = CBNContinuosEmbedderConfig()
    ):
        self.config = config

    @property
    def dimension(self):
        return len(self.cont_features) 

    def inverse_transform(self, data: np.ndarray):
        out = []
        for i in range(len(self.cont_features)):
            cbn, components = self.scaler[i]
            df = pd.DataFrame({
                self.cont_features[i]+'.component': components,
                self.cont_features[i]+'.normalized': data[:, i]
            })
            out.append(cbn.reverse_transform(df))
        return pd.concat(out, axis=1)

    def train(self, data: pd.DataFrame):
        self.scaler = []
        self.cont_features = data.columns
        for i in range(len(self.cont_features)):
            cbn = ClusterBasedNormalizer(
                learn_rounding_scheme=True,
                enforce_min_max_values=True,
                max_clusters=self.config.max_clusters,
                weight_threshold=self.config.weight_threshold,
            )
            cbn.fit(data, column=self.cont_features[i])
            data1 = cbn.transform(data[[self.cont_features[i]]])
            self.scaler.append((cbn, data1[self.cont_features[i]+'.component']))

    def encode(self, data: pd.DataFrame):
        data_normalized :List[np.ndarray]= []
         
        cont_features = data.columns
        for i in range(len(cont_features)):
            cbn, _ = self.scaler[i]
            transformed = cbn.transform(data[[cont_features[i]]])
            data_normalized.append(
              np.expand_dims(transformed[cont_features[i]+'.normalized'], -1)
            
            )
 
        return np.concatenate(data_normalized, axis=1)




class AutoEncoderContinuosEmbedder(AutoEncoder):
    def __init__(self, *, ncols: int, config: AutoEncoderConfig, device, name: str):
        super().__init__(
            encoder=Encoder_cont_emb(ncols, ncols * config.dimension),
            decoder=Decoder_cont_emb(ncols * config.dimension, ncols),
            config=config,
            loss=torch.nn.MSELoss(),
            device=device,
            name=name,
        )
        self.gaussian_normalizer = None
        self.min_max_scaler = None

    def get_optimizers(self):
        optimizer_Enc_cont_emb = torch.optim.Adam(
            self.encoder.parameters(),
            self.config.learning_rate,
            betas=(0.9, 0.999),
            amsgrad=True,
        )
        optimizer_Dec_cont_emb = torch.optim.Adam(
            self.decoder.parameters(),
            self.config.learning_rate,
            betas=(0.9, 0.999),
            amsgrad=True,
        )
        return optimizer_Enc_cont_emb, optimizer_Dec_cont_emb

    def get_schedulers(self, optimizer_Enc, optimizer_Dec):
        scheduler_Enc = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_Enc, gamma=0.98
        )
        scheduler_Dec = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_Dec, gamma=0.98
        )
        return scheduler_Enc, scheduler_Dec

    def preprocess(self, X: pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns.tolist()

        if self.gaussian_normalizer is None:
            self.gaussian_normalizer = GaussianNormalizer(
                enforce_min_max_values=True, learn_rounding_scheme=True
            )
            self.gaussian_normalizer.reset_randomization()
            self.gaussian_normalizer.fit(X, column=self.columns)

        X = self.gaussian_normalizer.transform(X)
        if self.min_max_scaler is None:
            self.min_max_scaler = MinMaxScaler((-1, 1))
            self.min_max_scaler.fit(X)
        return pd.DataFrame(self.min_max_scaler.transform(X), columns=self.columns)

    def inverse_transform(self, data: np.ndarray):
        d = self.decoder(torch.FloatTensor(data).to(self.device)).detach().cpu().numpy()
        d = self.min_max_scaler.inverse_transform(d)
        self.gaussian_normalizer.reset_randomization()
      
        d = self.gaussian_normalizer.reverse_transform(
            pd.DataFrame(d, columns=self.columns)
        )
        return pd.DataFrame(d, columns=self.columns)


class ClientEmbedder(AutoEncoder):
    def __init__(self, *, ncols: int, config: AutoEncoderConfig, device, name: str):
        super().__init__(
            encoder=Encoder_client_emb(ncols, config.dimension),
            decoder=Decoder_client_emb(config.dimension, ncols),
            config=config,
            device=device,
            loss=torch.nn.MSELoss(),
            name=name,
        )
        self.frequency_encoders = None
        self.scaler_minmax_cl_emb = None

    def preprocess(self, X: pd.DataFrame):
        X = X.copy()
        self.columns = X.columns
        if self.frequency_encoders is None:
            self.frequency_encoders = {i: FrequencyEncoder() for i in X.columns}
            for c in X.columns:
                X[c] = self.frequency_encoders[c].fit(X[[c]], column=c)
        for c in X.columns:
            X[c] = self.frequency_encoders[c].transform(X[[c]])

        if self.scaler_minmax_cl_emb is None:
            self.scaler_minmax_cl_emb = MinMaxScaler((-1, 1))
            self.scaler_minmax_cl_emb.fit(X)
        return pd.DataFrame(
            self.scaler_minmax_cl_emb.transform(X), columns=self.columns
        )

    def inverse_transform(self, data: np.ndarray):
        d = self.decoder(torch.FloatTensor(data).to(self.device)).detach().cpu().numpy()
        d = self.scaler_minmax_cl_emb.inverse_transform(d)

        customer_dec_array = []
        for i, column_name in enumerate(self.columns):
            df = pd.DataFrame(d[:, i], columns=[column_name])
            t = self.frequency_encoders[column_name].reverse_transform(df)
            customer_dec_array.append(t.values)

        customer_dec_array = np.array(customer_dec_array)
        return pd.DataFrame(customer_dec_array.T[0], columns=self.columns)


class ConditionVectorEmbedder(AutoEncoder):
    def __init__(self, *, ncols: int, config: AutoEncoderConfig, device, name: str):
        super().__init__(
            encoder=Encoder_cont_emb(ncols, config.dimension),
            decoder=Decoder_cont_emb(config.dimension, ncols),
            config=config,
            device=device,
            loss=torch.nn.MSELoss(),
            name=name,
        )
