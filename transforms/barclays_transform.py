import pandas as pd
import glob
import os
import re

account_currency = {'20-26-77 47500711': 'EUR', '20-26-77 13105881': 'GBP', '20-26-77 83568083': 'GBP'}
default_folder_in = r'D:\NicoFolder\BankAccount\BarclaysData\RawData'
default_path_out = r'D:\NicoFolder\BankAccount\BarclaysData\Barclays.csv'

expected_columns = ["Number", "Date", "Account", "Amount", "Subcategory", "Memo"]
target_columns = ["Date", "Account", "Amount", "Subcategory", "Memo", "Currency"]


def can_handle(path_in):
    df = pd.read_csv(path_in, nrows=1)
    return set(df.columns) == set(expected_columns)


def load(path_in):
    df = pd.read_csv(path_in)
    assert set(df.columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (Barclays)'
    
    df_out = df.drop('Number', axis=1)
    
    df_out.Date = pd.to_datetime(df_out.Date, format='%d/%m/%Y')
    df_out['Currency'] = [account_currency[acc] for acc in df_out.Account]
    df_out['Memo'] = [re.sub(' +', ' ', memo) for memo in df_out.Memo]
    
    return df_out


def load_save(folder_in, path_out):
    files = glob.glob(os.path.join(folder_in, '*.csv'))
    print(f"found {len(files)} CSV files.")
    if len(files) == 0:
        return

    df_list = [load(f) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(target_columns).cumcount()
    df = pd.concat(df_list)
    df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False).to_csv(path_out, index=False)
    

def load_save_default():
    load_save(default_folder_in, default_path_out)