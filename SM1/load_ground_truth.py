import pandas as pd
import numpy as np

def load_data(file="ground-truth.csv"):

    df = pd.read_csv(file)

    def convert_time(time):
        if isinstance(time, str):
            minute = int(time[0:2])
            second = int(time[3:5])
            return minute*60 + second
        else:
            return -1

    j1 = df["Jump #1"].apply(convert_time)
    j2 = df["Jump #2"].apply(convert_time)

    df = pd.DataFrame({"name": df["Filename"], "j1": j1, "j2": j2}, columns=("name", "j1", "j2"))

    return df

print(load_data())