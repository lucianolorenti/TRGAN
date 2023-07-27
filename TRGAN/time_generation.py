from typing import List, Iterable, Tuple
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
import pandas as pd


def optimize_xi(xiP, delta, k, n):
    return mean_squared_error(
        np.mean(np.random.poisson(xiP, size=(k, n)), axis=0), delta
    )


def optimize_xi_by_deltas_split(
    deltas: List[Iterable[float]], n_splits: int
) -> Tuple[List[float], List[int]]:
    """Get best poisson parameter for each split of deltas.

    Args:
        deltas (List[Iterable[float]]): _description_
        n_splits (int): _description_

    Returns:
        _type_: _description_
    """
    xiP_array = []
    idx_array = []
    k = 200

    for delta in tqdm(deltas):
        quantiles = [np.quantile(delta, n * 1 / n_splits) for n in range(n_splits + 1)]
        idx_quantiles = []

        for i in range(n_splits):
            idx_quantiles.append(
                list(np.where((quantiles[i] <= delta) & (delta < quantiles[i + 1]))[0])
            )

        idx_quantiles.append(list(np.where(quantiles[i + 1] <= delta)[0]))
        delta_array = []
        xi0_array = []
        bnds_array = []

        for i in idx_quantiles:
            if i != []:
                delta_array.append(delta[i])
                xi0_array.append(np.median(delta[i]))
                if np.median(delta[i]) == 0:
                    bnds_array.append([(0.0, 0.0)])
                else:
                    bnds_array.append([(np.min(delta[i]), 2 * np.median(delta[i]))])
            else:
                delta_array.append(np.zeros(3))
                xi0_array.append(1)
                bnds_array.append([(0, 1)])

        xiP_array_delta = []

        for i in range(len(delta_array)):
            x_opt = minimize(
                lambda x: optimize_xi(x, delta_array[i], k, len(delta_array[i])),
                x0=[xi0_array[i]],
                tol=1e-6,
                bounds=bnds_array[i],
                options={"maxiter": 3000},
                method="L-BFGS-B",
            ).x
            xiP_array_delta.append(x_opt)
        xiP_array.append(xiP_array_delta)
        idx_array.append(idx_quantiles)

    return xiP_array, idx_array


def synthetic_deltas(data: "TRGANDataset"):
    deltas = data.deltas.values
    synth_deltas = []
    for i in range(len(deltas)):
        synth_deltas_primary = []
        for j in range(len(data.xiP_array[i])):
            if data.idx_array[i][j] != []:
                synth_deltas_primary.append(
                    np.around(
                        np.mean(
                            np.random.poisson(
                                data.xiP_array[i][j][0],
                                (200, len(data.idx_array[i][j])),
                            ),
                            axis=0,
                        )
                    )
                )
            else:
                continue

        synth = sorted(
            list(zip(np.hstack(data.idx_array[i]), np.hstack(synth_deltas_primary))),
            key=lambda x: x[0],
        )

        synth_deltas.append(np.array(synth).T[1])
    
    return np.hstack(synth_deltas).astype(int)

def synthetic_time(data: "TRGANDataset"):

    synth_deltas = synthetic_deltas(data)
    splitted_synth_deltas = np.split(
            synth_deltas.astype("timedelta64[D]"),
            np.cumsum(data.length_dates_by_clients),
        )[:-1]
    
    synth_dates_by_clients = list(map(list, data.first_dates_by_clients.reshape(-1, 1)))

    for i in range(len(splitted_synth_deltas)):
        for j in range(len(splitted_synth_deltas[i])):
            synth_dates_by_clients[i].append(
                splitted_synth_deltas[i][j] + synth_dates_by_clients[i][j]
            )

    synth_time = (
        pd.Series(np.hstack(synth_dates_by_clients)).sort_values().reset_index(drop=True)
    )

    return synth_time


def preprocessing_date(date_feature: pd.Series):
    min_year = date_feature.dt.year.min()
    max_year = date_feature.dt.year.max()
    date_transformations = date_feature.apply(
        lambda x: np.array(
            [
                np.cos(2 * np.pi * x.day / 365),
                np.sin(2 * np.pi * x.day / 365),
                np.cos(2 * np.pi * x.month / 12),
                np.sin(2 * np.pi * x.month / 12),
                (x.year - min_year) / (max_year - min_year + 1e-7),
            ]
        )
    ).values

    date_transformations = np.vstack(date_transformations)
    date_transformations = date_transformations[
        :, :-1
    ]  # временно пока не придумаем что делать с годом

    return date_transformations
