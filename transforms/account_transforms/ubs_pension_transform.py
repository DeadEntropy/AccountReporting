import configparser

import pandas as pd
import glob
import os
from config.config_helper import parse_list
from account_transforms import static_data as sd


def can_handle(path_in, config):
    df = pd.read_csv(path_in, nrows=1)
    expected_columns = parse_list(config['expected_columns'])
    return set(df.columns) == set(expected_columns)


def load(path_in, config):
    df = pd.read_csv(path_in)
    expected_columns = parse_list(config['expected_columns'])
    assert set(df.columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (Lloyds Mortgage)'

    df_out = pd.DataFrame(columns=sd.target_columns)

    df_out.Date = pd.to_datetime(df["Effective Date"], format='%d/%m/%Y')
    df_out.Account = config['account_name']
    df_out.Currency = df["Transaction Currency"]
    df_out.Amount = df.Amount
    df_out.Subcategory = df["Transaction Type"]
    df_out.Memo = df["Transaction Type"]
    df_out['AccountType'] = config['account_type']

    return df_out


def load_save(config):
    files = glob.glob(os.path.join(config['folder_in'], '*.csv'))
    print(f"found {len(files)} CSV files.")
    if len(files) == 0:
        return

    df_list = [load(f, config['account_name']) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(sd.target_columns).cumcount()
    df = pd.concat(df_list)
    df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False).to_csv(config['path_out'], index=False)


def load_save_default():
    config = configparser.ConfigParser()
    config.read('../config/config.ini')

    load_save(config)
