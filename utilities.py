"""
Module with various utilities for Behavioral Cloning
"""

import pandas as pd
import numpy as np


def get_dataframe(samples):
    columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
    data = pd.DataFrame(samples, columns=columns)
    return data


def convert_to_float(df):
    df['steering'] = df['steering'].astype(float)
    df['throttle'] = df['throttle'].astype(float)
    df['brake'] = df['brake'].astype(float)
    df['speed'] = df['speed'].astype(float)
    return df

def data_augmentatation(df):
    """
    指定したデータだけをaugmentationする
    """
    df['augment'] = np.zeros(len(df))
    target_data = df.loc[(df.steering >= 0.04) | (df.steering <= -0.12)].copy()
    target_data['augment'] = 1

    augmented_data = df.append(target_data, ignore_index=True)
    return augmented_data
