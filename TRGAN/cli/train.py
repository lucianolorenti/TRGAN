from pathlib import Path
import typer
from enum import Enum
import pandas as pd
from TRGAN.TRGAN import train_trgan 
from TRGAN.dataset import DATASETS
from TRGAN.methods_comparison import train_copulagan, train_ctgan, train_tvae 

class ModelType(Enum):
    TRGAN = "TRGAN"
    TVAE = "TVAE"
    CTGAN = "CTGAN"
    CopulaGAN = "CopulaGAN"


def main(input_file: str, model: ModelType, output_path:Path, epochs:int = 10):
    if input_file in DATASETS:
        df, metadata = DATASETS[input_file]()
    else:
        df, metadata = pd.read_csv(input_file)
    if model == ModelType.TRGAN:
        train_trgan(df, metadata,  output_path / "TRGAN", epochs)
    elif model == ModelType.TVAE:
        train_tvae(df, metadata, output_path / "TVAE", epochs)
    elif model == ModelType.CTGAN:
        train_ctgan(df, metadata, output_path / "CTGAN", epochs)
    elif model == ModelType.CopulaGAN:
        train_copulagan(df, metadata, output_path / "CopulaGAN", epochs)


if __name__ == "__main__":
    typer.run(main)