import configparser

import pandas as pd
import glob
import os
from config.config_helper import parse_list
from transforms import static_data as sd


def can_handle(path_in, config):
    df = pd.read_csv(path_in, nrows=1)
    expected_columns = parse_list(config['expected_columns'])
    return set(df.columns) == set(expected_columns)


def load(path_in, config):
    df = pd.read_csv(path_in, sep=',')
    expected_columns = parse_list(config['expected_columns'])
    assert set(df.columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (Nutmeg)'

    df_out = pd.DataFrame(columns=sd.target_columns)
    df_out.Date = pd.to_datetime(df["Date"], format='%d-%b-%y')
    df_out.Account = 'Nutmeg: ' + df['Pot']
    df_out.Currency = config['default_currency']
    df_out.Amount = df["Amount (£)"]
    df_out.Subcategory = df["Description"]
    df_out.Memo = 'Nutmeg: ' + df['Pot']

    return df_out


def load_save(config):
    files = glob.glob(os.path.join(config['default_folder_in'], '*.csv'))
    print(f"found {len(files)} CSV files in {config['default_folder_in']}.")
    if len(files) == 0:
        return

    df_list = [load(f, config['default_currency']) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(sd.target_columns).cumcount()
    df = pd.concat(df_list)
    df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False).to_csv(config['default_path_out'], index=False)


def load_save_default():
    config = configparser.ConfigParser()
    config.read('../config/config.ini')

    load_save(config['Nutmeg'])
