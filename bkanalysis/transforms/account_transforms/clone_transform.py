import pandas as pd
from bkanalysis.transforms.account_transforms import static_data as sd


def can_handle(path_in):
    df = pd.read_csv(path_in, nrows=1)
    return set(df.columns) == set(sd.target_columns)


def load(path_in):
    df = pd.read_csv(path_in)
    assert set(df.columns) == set(sd.target_columns), f'Was expecting [{", ".join(sd.target_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (Clone)'

    df.Date = pd.to_datetime(df.Date, format='%d/%m/%Y')
    df.Date = pd.to_datetime(df.Date)

    df['AccountType'] = 'flat'
    return df
