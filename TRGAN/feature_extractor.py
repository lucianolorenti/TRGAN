from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler

from TRGAN.embedder import (
    AbstractEmbedder,
    AutoEncoderConfig,
    CBNContinuosEmbedder,
    CBNContinuosEmbedderConfig,
    CategoricalEmbedder,
    ClientEmbedder,
    ConditionVectorEmbedder,
    AutoEncoderContinuosEmbedder,
)
from TRGAN.time_generation import preprocessing_date


class FeatureExtractorConfig(BaseModel):
    categorical_embedder: AutoEncoderConfig
    continuous_embedder: Union[AutoEncoderConfig, CBNContinuosEmbedderConfig]
    client_embedder: AutoEncoderConfig
    condition_vector_embedder: AutoEncoderConfig


def default_feature_extractor_config(epochs: int = 10) -> FeatureExtractorConfig:
    return FeatureExtractorConfig(
        #continuous_embedder=CBNContinuosEmbedderConfig(),
        continuous_embedder=AutoEncoderConfig(
            dimension=12,
            epochs=epochs,
            learning_rate=8e-4,
       
            eps=0.01,
        ),
        categorical_embedder=AutoEncoderConfig(
            dimension=30,
            epochs=epochs,
            learning_rate=5e-5,
            differential_privacy_sensitivity=300,
            eps=0.01,
            batch_size=128,
        ),
        client_embedder=AutoEncoderConfig(
            dimension=4,
            epochs=epochs,
            learning_rate=1e-3,
            eps=0.01,
        ),
        condition_vector_embedder=AutoEncoderConfig(
            dimension=10,
            epochs=epochs,
            learning_rate=1e-3,
            eps=0.01,
        ),
    )


class FeatureExtractor:
    categorical_embedder: AbstractEmbedder
    continous_embedder: AbstractEmbedder
    client_embedder: AbstractEmbedder
    condition_vector_embedded: AbstractEmbedder
    scaler: MinMaxScaler
    config: FeatureExtractorConfig

    def __init__(
        self,
        *,
        conf,
        cont_features: List[str],
        cat_features: List[str],
        client_features: List[str],
        date_feature: str
    ):
        self.continous_features = cont_features
        self.categorical_features = cat_features
        self.client_features = client_features
        self.config = conf
        self.date_feature = date_feature

    def train(self, data: pd.DataFrame):
        self.train_categorical(data)
        self.train_continuous(data)
        self.train_client_features(data)
        self.train_embedded_scaler(data)

        data_transformed = self.embed(data)

        self.train_condition_vector_embedder(data_transformed)

    def get_data_dimension(self) -> int:
        return (
            self.categorical_embedder.dimension
            + self.continous_embedder.dimension
            + self.client_embedder.dimension
        )

    def get_condition_vector_autoencoder_dimension(self) -> int:
        return self.condition_vector_embedded.dimension + 4

    def get_condition_vector(self, data: pd.DataFrame):
        data_transformed = self.embed(data)
        data_encoded = self.condition_vector_embedded.encode(data_transformed)
        date_transformations = preprocessing_date(data[self.date_feature])
        return np.concatenate([data_encoded, date_transformations], axis=1)

    def train_condition_vector_embedder(self, data: np.ndarray):
        self.condition_vector_embedded = ConditionVectorEmbedder(
            ncols=data.shape[1],
            config=self.config.condition_vector_embedder,
            device="cpu",
            name="Condition vector embedder",
        )
        self.condition_vector_embedded.train(data)

    def train_embedded_scaler(self, data: pd.DataFrame):
        X_oh_emb = self.categorical_embedder.encode(data[self.categorical_features])
        X_cont_emb = self.continous_embedder.encode(data[self.continous_features])
        data_transformed = np.concatenate([X_cont_emb, X_oh_emb], axis=1)
        self.scaler = MinMaxScaler((-1, 1))
        self.scaler.fit(data_transformed)

    def inverse_transformation(self, data: np.ndarray):
        client_dimension = self.client_embedder.config.dimension
        X_cl_emb, data_transformed = (
            data[:, :client_dimension],
            data[:, client_dimension:],
        )

        X_cl = self.client_embedder.inverse_transform(X_cl_emb)

        data_transformed = self.scaler.inverse_transform(data_transformed)
        continuous_dimension = self.continous_embedder.dimension
        X_cont_emb, X_oh_emb = (
            data_transformed[:, :continuous_dimension],
            data_transformed[:, continuous_dimension:],
        )

        X_cont_emb = self.continous_embedder.inverse_transform(X_cont_emb)
        X_oh_emb = self.categorical_embedder.inverse_transform(X_oh_emb)

        return pd.concat((X_cl, X_cont_emb, X_oh_emb), axis=1)

    def embed(self, data: pd.DataFrame):
        X_cont_emb = self.continous_embedder.encode(data[self.continous_features])
        X_oh_emb = self.categorical_embedder.encode(data[self.categorical_features])
        X_cl_emb = self.client_embedder.encode(data[self.client_features])
        data_transformed = np.concatenate([X_cont_emb, X_oh_emb], axis=1)
        data_transformed = self.scaler.transform(data_transformed)
        return np.concatenate([X_cl_emb, data_transformed], axis=1)

    def train_categorical(self, data: pd.DataFrame):
        self.categorical_embedder = CategoricalEmbedder(
            config=self.config.categorical_embedder,
            device="cpu",
            name="Categorical features embedder",
            columns=self.categorical_features,
        )
        self.categorical_embedder.train(data[self.categorical_features])

    def train_continuous(self, data: pd.DataFrame):
        if isinstance(self.config.continuous_embedder, CBNContinuosEmbedderConfig):                    
            self.continous_embedder = CBNContinuosEmbedder(
                config=self.config.continuous_embedder
            )
        elif isinstance(self.config.continuous_embedder, AutoEncoderConfig):

            self.continous_embedder =  AutoEncoderContinuosEmbedder(
                ncols=len(self.continous_features),
                config=self.config.continuous_embedder,
                device="cpu",
                name="Continuous features embedder",
            )

        self.continous_embedder.train(
            data[self.continous_features],
        )

    def train_client_features(self, data: pd.DataFrame):
        self.client_embedder = ClientEmbedder(
            ncols=len(self.client_features),
            config=self.config.client_embedder,
            device="cpu",
            name="Client features embedder",
        )
        self.client_embedder.train(
            data[self.client_features],
        )
