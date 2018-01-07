import pandas as pd

def load_data(file):

    df = pd.read_csv(file)

    def convert_time(time):
        if isinstance(time, str):
            time_comps = time.split(':')

            minute = int(time_comps[0])
            second = int(time_comps[1])
            return minute*60 + second
        else:
            return -1

    j1 = df["Jump #1"].apply(convert_time)
    j2 = df["Jump #2"].apply(convert_time)
    w1 = df["Winch #1"].apply(convert_time)
    w2 = df["Winch #2"].apply(convert_time)

    df = pd.DataFrame({"name": df["Filename"], "j1": j1, "j2": j2, "w1": w1, "w2": w2}, columns=("name", "j1", "j2", "w1", "w2"))

    return df
