# coding=utf8
import configparser

import pandas as pd
import glob
import os
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd
from bkanalysis.config import config_helper as ch


def can_handle(path_in, config, *args):
    if not path_in.lower().endswith('csv'):
        return False
    df = pd.read_csv(path_in, nrows=1)
    expected_columns = parse_list(config['expected_columns'])
    return set(df.columns) == set(expected_columns)


def try_get_account_name(file_name, config):
    try:
        return ' '.join([s.capitalize() for s in file_name.split('-')])
    except:
        return config['account_name']


def load(path_in, config, *args):
    df = pd.read_csv(path_in)

    account_name = try_get_account_name(os.path.basename(path_in).split('.')[0], config)

    expected_columns = parse_list(config['expected_columns'])
    assert set(df.columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (Chase)'

    df_out = pd.DataFrame(columns=sd.target_columns)
    df_out.Date = pd.to_datetime(df['Posting Date'], format='%m/%d/%Y')
    df_out.Account = account_name
    df_out.Currency = "USD"
    df_out.Memo = df['Description']
    df_out.Subcategory = df["Type"]
    df_out['AccountType'] = config['account_type']
    df_out.Amount = df['Amount']

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

    load_save(config['ChaseBusiness'])
