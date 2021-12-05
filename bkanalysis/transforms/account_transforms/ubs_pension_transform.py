import configparser
import pandas as pd
import glob
import os
from bkanalysis.market.market import Market
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd
from bkanalysis.config import config_helper as ch
from bkanalysis.transforms.account_transforms import transformation_helper as helper


def can_handle(path_in, config):
    if not path_in.endswith('csv'):
        return False
    df = pd.read_csv(path_in, nrows=1)
    expected_columns = parse_list(config['expected_columns'])
    return set(df.columns) == set(expected_columns)


def load(path_in, config, market: Market, ref_currency: str):
    df = pd.read_csv(path_in)
    expected_columns = parse_list(config['expected_columns'])
    assert set(df.columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (UBS Pensio Mortgage)'
    df["Effective Date"] = pd.to_datetime(df["Effective Date"], format='%d/%m/%Y')

    df_out = pd.DataFrame(columns=sd.target_columns)

    df_out.Currency = df["Transaction Currency"]
    df_out.Amount = df.Amount
    df_out.Date = df["Effective Date"]
    df_out.Subcategory = df["Transaction Type"]
    df_out.Memo = df["Transaction Type"]
    df_out['AccountType'] = config['account_type']
    df_out.Account = config['account_name']

    if market is not None:
        df_out = helper.get_transaction(df_out,
                                        market,
                                        parse_list(config['proportion'], False),
                                        parse_list(config['switch'], False),
                                        ref_currency,
                                        'UBS Pension')

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
    if len(config.read(ch.source)) != 1:
        raise OSError(f'no config found in {ch.source}')

    load_save(config)
