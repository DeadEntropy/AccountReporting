import configparser
from datetime import timedelta
import pandas as pd
import glob
import os
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd
from bkanalysis.config import config_helper as ch


def can_handle(path_in, config):
    if not path_in.endswith('csv'):
        return False
    df = pd.read_csv(path_in, nrows=1)
    expected_columns = parse_list(config['expected_columns'])
    return set(df.columns) == set(expected_columns)


def get_key(proportion, date):
    return max([v for v in list(proportion.keys()) if date > pd.to_datetime(v)])


def fund_price(df_unit_price, date, fund_name, recursif=0):
    try:
        if recursif == 0:
            return list(df_unit_price[(df_unit_price['Unit Price Date'] == date) & (df_unit_price['Fund Name'] == fund_name)]['Unit Price'])[0]
        elif recursif<5:
            return list(df_unit_price[(df_unit_price['Unit Price Date'] == date - timedelta(days=recursif)) & (df_unit_price['Fund Name'] == fund_name)]['Unit Price'])[0]
        else:
            raise Exception(f'Failed with fallback on {date}\t{fund_name}')
    except IndexError:
        if recursif < 5:
            return fund_price(df_unit_price, date, fund_name, recursif+1)
        else:
            raise Exception(f'Failed on {date}\t{fund_name}')


def __to_float(s):
    try:
        return s.str.replace('£', '').str.replace(',', '').astype('float64')
    except AttributeError:
        return s


def get_transaction(path_in, path_unit_price, proportion: {}, switch: {}):
    df_transaction = pd.read_csv(path_in).fillna(0.0)
    df_unit_price = pd.read_csv(path_unit_price).dropna()

    df_transaction['Effective Date'] = pd.to_datetime(df_transaction['Effective Date'], format='%d/%m/%Y')
    df_transaction['Amount'] = __to_float(df_transaction['Amount']).fillna(0.0)
    df_transaction = df_transaction.set_index('Effective Date')
    df_unit_price['Unit Price Date'] = pd.to_datetime(df_unit_price['Unit Price Date'], format='%d/%m/%Y')

    for fund_name in df_unit_price["Fund Name"].unique():
        df_transaction[f'Price {fund_name}'] = [fund_price(df_unit_price, date, fund_name) for date in df_transaction.index]
        df_transaction[f'Unit {fund_name}'] = df_transaction[f'Amount'] / df_transaction[f'Price {fund_name}'] * [proportion[get_key(proportion, date)][fund_name] for date in df_transaction.index]

    for k,v in switch.items():
        df_transaction.loc[k, "Unit Global Equity (Voluntary)"] = v['Global Equity (Voluntary)']
        df_transaction.loc[k, "Unit Lifestyle (Voluntary)"] = v['Lifestyle (Voluntary)']

    for fund_name in df_unit_price["Fund Name"].unique():
        df_transaction[f'Cumulated Unit {fund_name}'] = df_transaction[f'Unit {fund_name}'][::-1].cumsum()[::-1]
    df_transaction['Total Value'] = df_transaction['Price Global Equity (Voluntary)'] * df_transaction['Cumulated Unit Global Equity (Voluntary)'] + df_transaction['Price Lifestyle (Voluntary)'] * df_transaction['Cumulated Unit Lifestyle (Voluntary)']
    df_transaction['Cumulated Capital Gain'] = df_transaction['Total Value'] - df_transaction['Amount'][::-1].cumsum()[::-1]
    df_transaction['Incremental Capital Gain'] = df_transaction['Cumulated Capital Gain'][::-1].rolling(window=2).apply(
        lambda x: x.iloc[1] - x.iloc[0]).fillna(0.0)[::-1]

    return df_transaction


def simplify(df_transaction):
    df_gain = df_transaction[['Transaction Type', 'Transaction Currency', 'Incremental Capital Gain']].reset_index().rename(
        columns={'Incremental Capital Gain': 'Amount'})
    df_gain['Transaction Type'] = 'Capital Gain'
    df_transfer = df_transaction[['Transaction Type', 'Transaction Currency', 'Amount']].reset_index()
    return df_transfer.append(df_gain).sort_values('Effective Date', ascending=False).reset_index(drop=True)


def load(path_in, config):

    if 'path_unit_price' in config:
        df = simplify(get_transaction(path_in,
                                      config['path_unit_price'],
                                      parse_list(config['proportion'], False),
                                      parse_list(config['switch'], False)))
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
