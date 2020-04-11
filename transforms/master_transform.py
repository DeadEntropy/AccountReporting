import glob
import os
import pandas as pd

from transforms import barclays_transform as barc
from transforms import lloyds_current_transform as lloyds_curr
from transforms import lloyds_mortgage_transform as lloyds_mort
from transforms import nutmeg_isa_transform as nut_transform
from transforms import revolut_transform as rev_transform
from transforms import clone_transform as clone_transform

default_path_in = r'D:\NicoFolder\BankAccount\Lake'
default_path_out = r'D:\NicoFolder\BankAccount\lake_result.csv'
target_columns = ["Date", "Account", "Amount", "Subcategory", "Memo", "Currency"]


def load(file):
    print(f'Loading {file}')
    if barc.can_handle(file):
        return barc.load(file)
    elif lloyds_curr.can_handle(file):
        return lloyds_curr.load(file)
    elif lloyds_mort.can_handle(file):
        return lloyds_mort.load(file)
    elif nut_transform.can_handle(file):
        return nut_transform.load(file)
    elif rev_transform.can_handle(file):
        return rev_transform.load(file)
    elif clone_transform.can_handle(file):
        return clone_transform.load(file)
    rev_transform.can_handle(file)

    raise ValueError(f'file {file} could not be processed by any of the loaders.')


def load_all(folder_in=default_path_in):
    files = glob.glob(os.path.join(folder_in, '*.csv'))
    print(f"found {len(files)} CSV files.")
    if len(files) == 0:
        return

    df_list = [load(f) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(target_columns).cumcount()
    df = pd.concat(df_list)
    return df.drop_duplicates().drop(['count'], axis=1)


def load_save(folder_in, path_out):
    files = glob.glob(os.path.join(folder_in, '*.csv'))
    print(f"found {len(files)} CSV files.")
    if len(files) == 0:
        return

    df_list = [load(f) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(target_columns).cumcount()
    df = pd.concat(df_list)
    df = df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False)
    df.Date = df.Date.apply(lambda x: x.strftime("%d-%b-%Y"))
    df.to_csv(path_out, index=False)


def load_save_default():
    load_save(default_path_in, default_path_out)
