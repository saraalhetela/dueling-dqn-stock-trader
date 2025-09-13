# data_preprocessing.py
import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)
    del df["volume"], df["date"], df["Name"]
    df["remove"] = df.apply(lambda x: all([abs(i - x.iloc[0]) < 1e-8 for i in x]), axis=1)
    df = df.query("remove == False").reset_index(drop=True)
    del df["remove"]

    # Normalize relative to open
    df["high"] = (df["high"] - df["open"]) / df["open"]
    df["low"] = (df["low"] - df["open"]) / df["open"]
    df["close"] = (df["close"] - df["open"]) / df["open"]
    return df
