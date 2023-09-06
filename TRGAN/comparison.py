import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from TRGAN.TRGAN import TRGAN


def get_cashflow( df: pd.DataFrame, customer:Optional[pd.Series]=None):
    
    df_month = df.transaction_date.dt.month


    
    if customer is not None:
        df_customers = customer
    else:
        df_customers = df.customer
    
    cash_flow = (df.groupby(by=[df_customers, df_month])['amount'].apply(lambda x: x - x.shift())).dropna()
    return  cash_flow
        

def compare_cashflow(real:  pd.DataFrame, synth: pd.DataFrame, *, ax: Optional[plt.Axes]=None, label:str="TRGAN", min_length: Optional[int] = None ):
    cash_flow_real = get_cashflow(real)
    cash_flow_synth = get_cashflow(synth, real.customer)
    
    if min_length:
        q = np.quantile(cash_flow_real, 0.95)
        cash_flow_real = cash_flow_real.iloc[np.where((cash_flow_real <= q) & (cash_flow_real >= -q))[0]]
        cash_flow_synth = cash_flow_synth.iloc[np.where((cash_flow_synth <= q) & (cash_flow_synth >= -q))[0]]
        cash_flow_real = pd.Series(random.sample(cash_flow_real.values.tolist(), min_length))
        cash_flow_synth = pd.Series(random.sample(cash_flow_synth.values.tolist(), min_length))

    hist_real = np.histogram(cash_flow_real, normed=True, bins=40)
    if ax is None:
        fig, ax = plt.subplots(2, 2, figsize=(20, 10), dpi=100)

    ax.hist(cash_flow_real, bins=40, label='Real', alpha=0.7, density=True, color='black')
    ax.hist(cash_flow_synth, bins=40, label=label, alpha=0.7, density=True, color='red')
    ax.legend(fontsize=18)
    ax.set_xlabel('Monthly cash flow', fontsize=22)
    ax.set_ylabel('Density', fontsize=22)
    ax.set_xlim((-np.max(hist_real[1]), np.max(hist_real[1])))
    ax.set_ylim((0, np.max(hist_real[0]) + 3e-4))
    ax.tick_params(labelsize=20)


def compare_amount(real: pd.Series, synth: pd.Series, *, 
                   ax: Optional[plt.Axes]=None,
                   label:str="TRGAN"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

    ax.hist(np.log1p(abs(real)), bins=40, label="Real", alpha=0.7)
    ax.hist(np.log1p(abs(synth)), bins=40, label=label, alpha=0.7)

    ax.legend()
    ax.set_title("Numerical distribution of the amount")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Density")
    return ax


def compare_categorical(real: pd.Series, synth: pd.Series, *, ax :Optional[plt.Axes]=None,label:str="TRGAN"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    
   
    real_data = real.value_counts() / (np.sum(real.value_counts()))
    synth_data = synth.value_counts() / (np.sum(synth.value_counts()))
    
    real_data.index = real_data.index.astype(str)
    synth_data.index = synth_data.index.astype(str)
    indices = real_data.sort_values(ascending=False).index
    synth_indices = [i for i in indices if i in synth_data.index]
    
    ax.bar(
        indices,
        real_data[indices],
        color="black",
        alpha=1,
        label="Real",
    )
    ax.bar(
        synth_indices,
        synth_data[synth_indices],
        color="red",
        alpha=0.6,
        label=label,
    )

    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.legend()
    ax.set_title("Caregorical distribution")
    ax.set_ylabel("Density")
    ax.set_xlabel("MCC")
    return ax


def plot_embeddings_categorical(ax: plt.Axes, model: TRGAN, real:pd.DataFrame, synth:pd.DataFrame):
    pca1 = PCA(n_components=2)
    X_oh = model.feature_extractor.categorical_embedder.encode(real[model.cat_features])
    synth_df_cat = model.feature_extractor.categorical_embedder.encode(synth[model.cat_features])
    data_transformed_pca = pca1.fit_transform(X_oh)
    synth_pca = pca1.transform(synth_df_cat)
    ax.scatter(
        data_transformed_pca.T[0], data_transformed_pca.T[1], label="Real", alpha=0.4, s=20
    )
    ax.scatter(synth_pca.T[0], synth_pca.T[1], label="Synth", alpha=0.4, s=10)

    ax.legend()
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")
    ax.set_title("PCA")

def plot_tsne(ax:plt.Axes, model: TRGAN, real:pd.DataFrame, synth:pd.DataFrame):
    X_emb = model.feature_extractor.embed(real)
    synth_emb = model.feature_extractor.embed(synth)

    tsne1 = TSNE(n_components=2, perplexity=80)
    tsne2 = TSNE(n_components=2, perplexity=80)

    idx_random = np.random.randint(0, len(X_emb), 5000)

    data_transformed_tsne = tsne1.fit_transform(X_emb[idx_random])
    synth_tsne = tsne2.fit_transform(synth_emb[idx_random])

    ax.scatter(
        data_transformed_tsne.T[0], data_transformed_tsne.T[1], label="Real", s=3, alpha=1
    )
    ax.scatter(synth_tsne.T[0], synth_tsne.T[1], label="Synth", s=3, alpha=1)

    ax.legend()
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")
    ax.set_title("t-SNE")


def plot_embeddings(model: TRGAN, real:pd.DataFrame, synth:pd.DataFrame):
    figure, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=100)
    plot_embeddings_categorical(axs[0], model, real, synth)
    plot_tsne(axs[1], model, real, synth)
    return axs