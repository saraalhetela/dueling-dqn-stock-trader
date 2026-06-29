import pandas as pd

def load_and_preprocess(path, train_ratio=0.70, val_ratio=0.15):
    df = pd.read_csv(path)
    df = df.drop(columns=["volume", "date", "Name"], errors="ignore")
    mask = df.apply(lambda row: all(abs(v - row.iloc[0]) < 1e-8 for v in row), axis=1)
    df = df[~mask].reset_index(drop=True)
    df["high"]  = (df["high"]  - df["open"]) / df["open"]
    df["low"]   = (df["low"]   - df["open"]) / df["open"]
    df["close"] = (df["close"] - df["open"]) / df["open"]
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df   = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df  = df.iloc[val_end:].reset_index(drop=True)
    print(f"Data split — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} bars")
    return train_df, val_df, test_df
