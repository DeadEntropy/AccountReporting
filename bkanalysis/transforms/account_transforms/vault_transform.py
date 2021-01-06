import configparser
from re import sub

import numpy as np
import pandas as pd
import glob
import os
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd
import datetime


def can_handle(path_in, config):
    df = pd.read_csv(path_in, nrows=1)
    expected_columns = parse_list(config['expected_columns'])
    return set(df.columns) == set(expected_columns)


def as_string(v):
    if v == 0:
        return "0"
    return v


def get_product_name(s):
    names = []
    for e in list(s.unique()):
        if e is not np.nan:
            names.append(e)

    if len(names) != 1:
        raise Exception(f'Found more than one Produce Names: {names}')

    return names[0]


def  get_year(s):
    # if int(datetime.datetime.now().year) != 2020:
    #     raise Exception('This function only works for 2020!')

    return s


def get_dates_from_description(df, fallback_year):
    results = []
    year = None
    for index, row in df.iterrows():
        description = row['Description']
        date = row['Completed Date']
        if '\n' in description:
            if len(description.split('\n')[1].split(' ')) == 3:
                try:
                    current_date = pd.to_datetime(description.split('\n')[1].split(' ')[2], format='%Y/%M/%d')
                    results.append(current_date)
                    year = current_date.year
                except:
                    results.append(pd.to_datetime(f'{date}, {fallback_year if (year is None) else year}', format='%b %d, %Y'))
            else:
                results.append(pd.to_datetime(f'{date}, {fallback_year if (year is None) else year}', format='%b %d, %Y'))
        else:
            results.append(pd.to_datetime(f'{date}, {fallback_year if (year is None) else year}', format='%b %d, %Y'))

    return results


def load(path_in, config):
    df = pd.read_csv(path_in)
    expected_columns = parse_list(config['expected_columns'])
    assert set(df.columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (Lloyds Current)'

    df["Money in (GBP)"] = df["Money in (GBP)"].fillna(0)
    df["Money out (GBP)"] = df["Money out (GBP)"].fillna(0)
    df["Interest rate (AER)"] = df["Interest rate (AER)"].fillna("")

    df["Money in (GBP)"] = [float(sub(r'[^\d\-.]', '', as_string(x))) for x in df["Money in (GBP)"]]
    df["Money out (GBP)"] = [float(sub(r'[^\d\-.]', '', as_string(x))) for x in df["Money out (GBP)"]]

    df_out = pd.DataFrame(columns=sd.target_columns)

    df_out.Date = get_dates_from_description(df[["Description", "Completed Date"]], config['year'])
    # df_out.Date = get_year(pd.to_datetime(df["Completed Date"] + ", " + config['year'], format='%b %d, %Y'))

    df_out.Account = get_product_name(df["Product name"])
    df_out.Currency = config['currency']
    df_out.Amount = df["Money in (GBP)"] + df["Money out (GBP)"]
    df_out.Subcategory = df["Description"].str.split('\n').str[0].str.strip()
    df_out.Memo = (df["Description"].str.split('\n').str[0].str.strip() + " " + df["Interest rate (AER)"].astype(str)).str.strip()
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
    df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False).to_csv(config['path_out'],
                                                                                             index=False)


def load_save_default():
    config = configparser.ConfigParser()
    config.read('config/config.ini')

    load_save(config['Vault'])
