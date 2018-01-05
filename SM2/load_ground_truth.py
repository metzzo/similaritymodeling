import pandas as pd
import numpy as np

def load_data(file):

    df = pd.read_csv(file)

    def convert_time(time):
        if isinstance(time, str):
            minute = int(time[0:2])
            second = int(time[3:5])
            return minute*60 + second
        else:
            return -1

    w1 = df["Winch #1"].apply(convert_time)
    w2 = df["Winch #2"].apply(convert_time)

    df = pd.DataFrame({"name": df["Filename"], "w1": w1, "w2": w2}, columns=("name", "w1", "w2"))

    return df

# print(load_data())