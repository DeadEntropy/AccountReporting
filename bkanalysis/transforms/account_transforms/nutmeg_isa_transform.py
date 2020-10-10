import configparser
import ast
import pandas as pd
import glob
import os
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd
import re


regex = re.compile('\((.*?)\)')


def can_handle(path_in, config):
    df = pd.read_csv(path_in, nrows=1)

    expected_columns = [re.sub(regex, '', s) for s in parse_list(config['expected_columns'])]
    columns = [re.sub(regex, '', s) for s in df.columns]
    return set(columns) == set(expected_columns)


def load(path_in, config):
    df = pd.read_csv(path_in, sep=',')
    try:
        expected_columns = [re.sub(regex, '', s) for s in parse_list(config['expected_columns'])]
    except Exception:
        print(config)
        print(config['expected_columns'])
        raise
    columns = [re.sub(regex, '', s) for s in df.columns]
    assert set(columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (Nutmeg)'

    df_out = pd.DataFrame(columns=sd.target_columns)
    df_out.Date = pd.to_datetime(df["Date"], format='%d-%b-%y')
    df_out.Account = 'Nutmeg: ' + df['Pot']
    df_out.Currency = config['currency']
    df_out.Amount = df["Amount (£)"]
    df_out.Subcategory = df["Description"]
    df_out.Memo = 'Nutmeg: ' + df['Pot']
    account_types = ast.literal_eval(config['account_types'])
    df_out['AccountType'] = [account_types[acc] for acc in df_out.Account]

    return df_out


def load_save(config):
    files = glob.glob(os.path.join(config['folder_in'], '*.csv'))
    print(f"found {len(files)} CSV files in {config['folder_in']}.")
    if len(files) == 0:
        return

    df_list = [load(f, config['currency']) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(sd.target_columns).cumcount()
    df = pd.concat(df_list)
    df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False).to_csv(config['path_out'], index=False)


def load_save_default():
    config = configparser.ConfigParser()
    config.read('config/config.ini')

    load_save(config['Nutmeg'])
