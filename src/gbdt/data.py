import pandas as pd


def load_data(path):
    raw = pd.read_excel(path)
    date_col = raw.columns[0]
    target_col = raw.columns[1]
    raw = raw.sort_values(date_col).reset_index(drop=True)
    raw = raw.rename(columns={date_col: "date", target_col: "y"})
    return raw
