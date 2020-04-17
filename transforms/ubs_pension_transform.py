import pandas as pd
import glob
import os

default_account_name = 'UBS Pension'
default_folder_in = r''
default_path_out = r''

expected_columns = ['Effective Date', 'Transaction Type', 'Transaction Currency', 'Amount']
target_columns = ["Date", "Account", "Amount", "Subcategory", "Memo", "Currency"]


def can_handle(path_in):
    df = pd.read_csv(path_in, nrows=1)
    return set(df.columns) == set(expected_columns)


def load(path_in, account_name = default_account_name):
    df = pd.read_csv(path_in)
    assert set(df.columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (Lloyds Mortgage)'

    df_out = pd.DataFrame(columns=target_columns)

    df_out.Date = pd.to_datetime(df["Effective Date"], format='%d/%m/%Y')
    df_out.Account = account_name
    df_out.Currency = df["Transaction Currency"]
    df_out.Amount = df.Amount
    df_out.Subcategory = df["Transaction Type"]
    df_out.Memo = df["Transaction Type"]

    return df_out


def load_save(folder_in, account_name, path_out):
    files = glob.glob(os.path.join(folder_in, '*.csv'))
    print(f"found {len(files)} CSV files.")
    if len(files) == 0:
        return

    df_list = [load(f, account_name) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(target_columns).cumcount()
    df = pd.concat(df_list)
    df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False).to_csv(path_out, index=False)


def load_save_default():
    load_save(default_folder_in, default_account_name, default_path_out)
