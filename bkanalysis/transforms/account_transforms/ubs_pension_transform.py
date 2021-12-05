import configparser
import pandas as pd
import glob
import os
from bkanalysis.market.market import Market
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd
from bkanalysis.config import config_helper as ch
import datetime as dt


def can_handle(path_in, config):
    if not path_in.endswith('csv'):
        return False
    df = pd.read_csv(path_in, nrows=1)
    expected_columns = parse_list(config['expected_columns'])
    return set(df.columns) == set(expected_columns)


def get_key(proportion, date):
    return max([v for v in list(proportion.keys()) if date > pd.to_datetime(v)])


def __to_float(s):
    try:
        return s.str.replace('£', '').str.replace(',', '').astype('float64')
    except AttributeError:
        return s


def get_transaction(path_in: str, market: Market, proportion: {}, switch: {}, ref_currency: str):
    df_transaction = pd.read_csv(path_in).fillna(0.0)

    df_transaction['Effective Date'] = pd.to_datetime(df_transaction['Effective Date'], format='%d/%m/%Y')
    df_transaction['Amount'] = __to_float(df_transaction['Amount']).fillna(0.0)
    df_transaction = df_transaction.set_index('Effective Date')

    list_of_funds = list(set([item for sublist in proportion.values() for item in sublist]))
    list_of_dfs = [df_transaction]

    for fund_name in list_of_funds:
        df_fund = pd.DataFrame(columns=df_transaction.columns, index=df_transaction.index)
        df_fund['Price'] = [market.get_price_in_currency(fund_name, date, ref_currency) for date in df_transaction.index]
        df_fund['Amount'] = df_transaction[f'Amount'] / df_fund['Price'] * [proportion[get_key(proportion, date)][fund_name] for date in df_transaction.index]
        df_fund.drop('Price', axis=1, inplace=True)
        df_fund['Transaction Currency'] = fund_name
        df_fund['Transaction Type'] = 'UBS Pension Fund Purchase'

        df_cash_impact = pd.DataFrame(columns=df_transaction.columns, index=df_transaction.index)

        df_cash_impact['Amount'] = -df_transaction[f'Amount'] * [proportion[get_key(proportion, date)][fund_name] for date in df_transaction.index]
        df_cash_impact['Transaction Currency'] = ref_currency
        df_cash_impact['Transaction Type'] = 'UBS Pension Fund Purchase'

        list_of_dfs.append(df_fund)
        list_of_dfs.append(df_cash_impact)

        result = pd.concat(list_of_dfs)
        result = result[result.Amount!=0].reset_index()

    for k, v in switch.items():
        for fund_name, fund_units in v.items():
            result = result.append(pd.DataFrame([[dt.datetime.strptime(k, '%Y-%m-%d'), 'UBS Pension Fund Switch', fund_name, fund_units]], columns=result.columns))

    result['Effective Date'] = pd.to_datetime(result['Effective Date'], format='%d/%m/%Y')
    result = result.sort_values('Effective Date', ascending=False).reset_index(drop=True)

    return result


def load(path_in, config, market: Market, ref_currency: str):
    if market is not None:
        df = get_transaction(path_in,
                             market,
                             parse_list(config['proportion'], False),
                             parse_list(config['switch'], False),
                             ref_currency)
    else:
        df = pd.read_csv(path_in)
        expected_columns = parse_list(config['expected_columns'])
        assert set(df.columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                         f'are [{", ".join(df.columns)}]. (Lloyds Mortgage)'
        df["Effective Date"] = pd.to_datetime(df["Effective Date"], format='%d/%m/%Y')

    df_out = pd.DataFrame(columns=sd.target_columns)

    df_out.Currency = df["Transaction Currency"]
    df_out.Amount = df.Amount
    df_out.Date = df["Effective Date"]
    df_out.Subcategory = df["Transaction Type"]
    df_out.Memo = df["Transaction Type"]
    df_out['AccountType'] = config['account_type']
    df_out.Account = config['account_name']

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
