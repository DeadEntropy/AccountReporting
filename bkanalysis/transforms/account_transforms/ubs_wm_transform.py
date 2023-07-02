import configparser

import pandas as pd
import glob
import os
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd
from bkanalysis.config import config_helper as ch


def can_handle(path_in, config, sep=','):
    if not path_in.endswith('csv'):
        return False
    df = pd.read_csv(path_in, sep=sep, nrows=1)
    expected_columns = parse_list(config['expected_columns'], False)

    return set(df.columns) == set(expected_columns)


def load(path_in, config, sep=','):
    df = pd.read_csv(path_in, sep=sep)
    df.columns = [s.strip() for s in df.columns]
    expected_columns = parse_list(config['expected_columns'], False)

    assert set(df.columns) == set(expected_columns), \
        f'Was expecting [{", ".join(expected_columns)}] but file columns are [{", ".join(df.columns)}]. (UBS WM)'

    df_out = pd.DataFrame(columns=sd.target_columns)
    df_out.Date = pd.to_datetime(df["Date"].str.strip(), format='%m/%d/%Y')
    df_out.Account = df['Account Number']
    df_out.Currency = df['Symbol'].fillna('USD')
    df_out.Amount = [q if t =='Investment' else a for t, q, a in zip(df['Type'], df['Quantity'], df['Amount'])]
    df_out.Memo = [f'{a}: {d}' for a, d in zip(df['Activity'], df['Description'])]
    df_out.Subcategory = df['Activity']
    df_out['AccountType'] = config['account_type']

    return df_out


def load_save(config):
    files = glob.glob(os.path.join(config['folder_in'], '*.csv'))
    print(f"found {len(files)} CSV files in {config['folder_in']}.")
    if len(files) == 0:
        return

    df_list = [load(f, config) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(sd.target_columns).cumcount()
    df = pd.concat(df_list)
    df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False).to_csv(config['path_out'], index=False)


def load_save_default():
    config = configparser.ConfigParser()
    if len(config.read(ch.source)) != 1:
        raise OSError(f'no config found in {ch.source}')

    load_save(config['UbsWealthManagement'])
