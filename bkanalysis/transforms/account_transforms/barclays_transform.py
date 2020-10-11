import configparser
import ast

import pandas as pd
import glob
import os
import re
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd


def can_handle(path_in, config):
    df = pd.read_csv(path_in, nrows=1)
    expected_columns = parse_list(config['expected_columns'])
    return set(df.columns) == set(expected_columns)


def load(path_in, config):
    df = pd.read_csv(path_in)
    expected_columns = parse_list(config['expected_columns'])
    assert set(df.columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (Barclays)'
    
    df_out = df.drop('Number', axis=1)
    
    df_out.Date = pd.to_datetime(df_out.Date, format='%d/%m/%Y')
    account_currencies = ast.literal_eval(config['account_currencies'])
    df_out['Currency'] = [account_currencies[acc] for acc in df_out.Account]
    df_out['Memo'] = [re.sub(' +', ' ', memo) for memo in df_out.Memo]
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
    config.read('config/config.ini')

    load_save(config['Barclays'])