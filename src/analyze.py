from pathlib import Path

import pandas as pd


def analyze_fight_result(path: str):
    dfs = []
    for f in Path(path).iterdir():
        df = pd.read_csv(f, header=None)
        dfs.append(df)
    data_all = pd.concat(dfs, axis=0, ignore_index=True)
    data_all.columns = ['Round', 'P1', 'P2', 'Time']
    win_ratio = sum(data_all.P1 > data_all.P2) / data_all.shape[0]
    hp_diff = sum(data_all.P1 - data_all.P2) / data_all.shape[0]
    return win_ratio, hp_diff
