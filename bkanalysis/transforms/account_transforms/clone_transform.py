import pandas as pd
from bkanalysis.transforms.account_transforms import static_data as sd


def can_handle(path_in, *args):
    if not path_in.endswith('csv'):
        return False
    df = pd.read_csv(path_in, nrows=1)
    return (set(df.columns) == set(sd.target_columns)) or (set(df.columns) == set(sd.target_columns + ['AccountType']))


def load(path_in, *args):
    df = pd.read_csv(path_in)
    assert (set(df.columns) == set(sd.target_columns)) or (set(df.columns) == set(sd.target_columns + ['AccountType'])), \
        f'Was expecting [{", ".join(sd.target_columns)}] but file columns ' \
        f'are [{", ".join(df.columns)}]. (Clone)'

    df.Date = pd.to_datetime(df.Date, format='%d/%m/%Y')
    df.Date = pd.to_datetime(df.Date)

    if 'AccountType' not in df.columns:
        df['AccountType'] = 'flat'
    return df
