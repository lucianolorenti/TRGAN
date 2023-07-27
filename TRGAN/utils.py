import pandas as pd
import numpy as np

def deltas_by_client(data:pd.DataFrame, client_id:str, time_id:str):

    return data.groupby(client_id)[time_id].apply(lambda x: (x - x.shift()).dt.days.values[1:])

