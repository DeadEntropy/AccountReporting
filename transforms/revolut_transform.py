import pandas as pd
import glob
import os

possible_currencies = ['GBP', 'EUR', 'USD']
default_account_name = 'Revolut'
default_folder_in = r'D:\NicoFolder\BankAccount\RevolutData\RawData'
default_path_out = r'D:\NicoFolder\BankAccount\RevolutData\Revolut.csv'

expected_columns = ['Completed Date ', ' Description ', ' Paid Out (CCY) ', ' Paid In (CCY) ', ' Exchange Out',
                    ' Exchange In', ' Balance (CCY)', ' Category', ' Notes']
target_columns = ["Date", "Account", "Amount", "Subcategory", "Memo", "Currency"]


def can_handle(path_in):
    df = pd.read_csv(path_in, sep=';', nrows=1)

    if len(df.columns) < 2:
        return False

    for ccy in possible_currencies:
        if ccy in df.columns[2]:
            currency = ccy

    return set(df.columns) == set(set([e.replace('CCY', currency) for e in expected_columns]))


def load(path_in, account_name=default_account_name):
    df = pd.read_csv(path_in, sep=';')

    for ccy in possible_currencies:
        if ccy in df.columns[2]:
            currency = ccy

    assert set(df.columns) == set(set([e.replace('CCY', currency) for e in expected_columns])), \
        f'Was expecting [{", ".join(expected_columns)}] but file columns are [{", ".join(df.columns)}]. (Nutmeg)'

    df[f" Paid In ({currency}) "] = pd.to_numeric(df[f" Paid In ({currency}) "].str.replace(',', ''),
                                                  errors='coerce').fillna(0)
    df[f" Paid Out ({currency}) "] = pd.to_numeric(df[f" Paid Out ({currency}) "].str.replace(',', ''),
                                                   errors='coerce').fillna(0)

    df_out = pd.DataFrame(columns=target_columns)
    df_out.Date = pd.to_datetime(df["Completed Date "].str.strip(), format='%d %b %Y')
    df_out.Account = account_name + " " + currency
    df_out.Currency = currency
    df_out.Amount = df[f" Paid In ({currency}) "] - df[f" Paid Out ({currency}) "]
    df_out.Subcategory = df[" Category"]
    df_out.Memo = df[" Description "] + df[" Notes"] + df[" Exchange Out"] + df[" Exchange In"]
    df_out.Memo = df_out.Memo.str.replace('£', 'GBP')
    df_out.Memo = df_out.Memo.str.replace('€', 'EUR')
    df_out.Memo = df_out.Memo.str.replace(',', '')

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
