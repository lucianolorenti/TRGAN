
from enum import Enum
from typing import Optional


class TRGANType(Enum):
    BIG_N_UNIQUE = "categorical_big_n_unique"
    SMALL_N_UNIQUE = "categorical_small_n_unique"
    BIG_N_UNIQUE_ID = "categorical_big_n_unique_id"


class TRGANMetadata:
    def __init__(self):
        self.columns = {}

    def add_column(
        self,
        column_name: str,
        type: str,
        trgan_type: Optional[str] = None,
        datetime_format: Optional[str] = None,
        computer_representation:  Optional[str] = None,
    ):
        self.columns[column_name] = (
            {
                "column_name": column_name,
                "type": type,
                "trgan_type": trgan_type,
                "datetime_format": datetime_format,
                "computer_representation": computer_representation
            }
        )
     

    def get_big_n_unique_id_feature(self) -> str:
        return [
            k
            for k in self.columns.keys()
            if (self.columns[k]["type"] == "categorical")
            and (self.columns[k]["trgan_type"] == TRGANType.BIG_N_UNIQUE_ID.value)
        ][0]

    def get_time_feature(self) -> str:
        return [
            k for k in self.columns.keys() if self.columns[k]["type"] == "datetime"
        ][0]

    def categorical_small_n_unique_features(self):
        return [
            k
            for k in self.columns.keys()
            if (self.columns[k]["type"] == "categorical")
            and (self.columns[k]["trgan_type"] == TRGANType.SMALL_N_UNIQUE.value)
        ]

    def continuous_features(self):
        return [
            k for k in self.columns.keys() if self.columns[k]["type"] == "numerical"
        ]

    def categorical_features(self):
        return [
            k for k in self.columns.keys() if self.columns[k]["type"] == "categorical"
        ]

    def categorical_biq_n_unique_features(self):
        return [
            k
            for k in self.columns.keys()
            if (self.columns[k]["type"] == "categorical")
            and (
                self.columns[k]["trgan_type"]
                in (TRGANType.BIG_N_UNIQUE.value, TRGANType.BIG_N_UNIQUE_ID.value)
            )
        ]
