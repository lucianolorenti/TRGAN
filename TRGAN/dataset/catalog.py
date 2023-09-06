from pathlib import Path
from typing import Tuple
from TRGAN import DATA_PATH
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from TRGAN.dataset.metadata import  TRGANMetadata, TRGANType
import gdown
from datetime import datetime

def czech_bank() -> Tuple[pd.DataFrame, TRGANMetadata]:
    def build_metadata():
        metadata = TRGANMetadata()
        metadata.add_column(
            column_name="amount", type="numerical", computer_representation="Float"
        )

        metadata.add_column(
            column_name="mcc",
            type="categorical",
            trgan_type=TRGANType.SMALL_N_UNIQUE.value,
        )

        metadata.add_column(
            column_name="customer",
            type="categorical",
            trgan_type=TRGANType.BIG_N_UNIQUE_ID.value,
        )

        metadata.add_column(
            column_name="transaction_date",
            type="datetime",
            datetime_format="%Y %m %d",
        )
        return metadata
    
    if not (DATA_PATH / "data_czech.csv").is_file():
        URL = "https://drive.google.com/uc?id=1fBBx_5_P4CA6yYIi-5pBd_bTbgZo61V3"
        gdown.download(URL, str(DATA_PATH / "data_czech.csv"), quiet=False)
    data = pd.read_csv(DATA_PATH / "data_czech.csv")
    
    czech_date_parser = lambda x: datetime.strptime(str(x), "%y%m%d")
    data["datetime"] = data["date"].apply(czech_date_parser)
    data = data.rename(
        columns={
            "account_id": 'customer', 
            'tcode': 'mcc', 
            'datetime': 'transaction_date'
        }
    )

    data = data.drop(
        ['date', 'Unnamed: 0', 'type',
          'operation', 'k_symbol', 'balance', 'age', 'column_a'], 
        axis=1)

   
    idx_customer = (data['customer'].value_counts().loc[(data['customer'].value_counts() > 20) == True]).index.tolist()
    data = data[data['customer'].isin(idx_customer)]
    data['transaction_date'] = pd.to_datetime(data['transaction_date'])
    data = data.sort_values(by='transaction_date')
    data = data.reset_index(drop=True)
    return data, build_metadata()


def uk_bank() -> Tuple[pd.DataFrame, TRGANMetadata]:
    def build_metadata():

        metadata = TRGANMetadata()
        metadata.add_column(
            column_name="amount", type="numerical", computer_representation="Float"
        )

        metadata.add_column(
            column_name="mcc",
            type="categorical",
            trgan_type=TRGANType.SMALL_N_UNIQUE.value,
        )

        metadata.add_column(
            column_name="customer",
            type="categorical",
            trgan_type=TRGANType.BIG_N_UNIQUE_ID.value,
        )

        metadata.add_column(
            column_name="transaction_date",
            type="datetime",
            datetime_format="%Y %m %d",
        )
        return metadata

    data = pd.read_csv(DATA_PATH / "data_uk_clean.csv")
    data = data.rename(
        columns={
            "account_id": "customer",
            "tcode": "mcc",
            "datetime": "transaction_date",
        }
    )
    data = data[["customer", "mcc", "transaction_date", "amount"]]
    data["transaction_date"] = pd.to_datetime(data["transaction_date"], format="mixed")
    #data["transaction_date"] = pd.to_datetime(data["transaction_date"].dt.date)
    data = data.sort_values(by="transaction_date")
    data = data.reset_index(drop=True)
    le = LabelEncoder()
    data["mcc"] = le.fit_transform(data["mcc"])
    return data, build_metadata()


DATASETS = {
    "uk_bank": uk_bank,
}


def get_dataset(name: str) -> Tuple[pd.DataFrame, TRGANMetadata]:
    return DATASETS[name]()
