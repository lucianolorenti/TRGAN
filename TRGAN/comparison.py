import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compare_amount(real: pd.Series, synth: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

    ax.hist(np.log1p(abs(real)), bins=40, label="Real", alpha=0.7)
    ax.hist(np.log1p(abs(synth)), bins=40, label="Synth", alpha=0.7)

    ax.legend()
    ax.set_title("Numerical distribution of the amount")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Density")
    return ax


def compare_categorical(real: pd.Series, synth: pd.Series):
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
        label="TRGAN",
    )

    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.legend()
    ax.set_title("Caregorical distribution")
    ax.set_ylabel("Density")
    ax.set_xlabel("MCC")
    return ax
