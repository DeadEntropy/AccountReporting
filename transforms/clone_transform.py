import pandas as pd

expected_columns = ["Date", "Account", "Amount", "Subcategory", "Memo", "Currency"]


def can_handle(path_in):
    df = pd.read_csv(path_in, nrows=1)
    return set(df.columns) == set(expected_columns)


def load(path_in):
    df = pd.read_csv(path_in)
    assert set(df.columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (Clone)'

    df.Date = pd.to_datetime(df.Date)
    return df
