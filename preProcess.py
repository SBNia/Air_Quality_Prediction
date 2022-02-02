# preprocessing dataset
import pandas as pd
import numpy as np
from config import *

# loading csv dataset
def load_data():
    return pd.read_csv(DATA_DIR+DATA_NAME)


def preprocessing_data(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S')
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour

    df = df.set_index('timestamp')
    return df

def creating_lagged_variables(df, columns, lag_count):

    for c, c_name in enumerate(columns):
        for i in range(1, lag_count+1):
            column_name = c_name + '_+' + str(i)
            df[column_name] = df[c_name].shift(i)
    return df

def normalizing_data(df):

    df = df.apply(pd.to_numeric)
    df = (df - df.mean()) / (df.max() - df.min())       # normalizing data
    return df

def create_model_data(df, input_label):

    x_global_column = df.loc[:, 'year':'hour']
    x_column = df.loc[:, [col for col in df.columns if input_label + '_' in col]]
    x_column = pd.concat([x_global_column, x_column], axis=1, sort=False)
    x_column = normalizing_data(x_column)
    y_column = df[input_label]
    return x_column, y_column

def data_labaling(data):

    if data <= 55:
        return 'normal'
    elif 55 < data and data <= 150:
        return 'elevated'
    elif 150 < data and data <= 250:
        return 'high'
    else:
        return 'veryhigh'
