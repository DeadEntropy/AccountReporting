import pandas as pd
import glob
import os

default_currency = 'GBP'
default_folder_in = r'D:\NicoFolder\BankAccount\LloydsData\RawData\CurrentAccount'
default_path_out = r'D:\NicoFolder\BankAccount\LloydsData\Lloyds_Current.csv'

expected_columns = ['Transaction Date', 'Transaction Type', 'Sort Code', 'Account Number', 'Transaction Description',
                    'Debit Amount', 'Credit Amount', 'Balance']
target_columns = ["Date", "Account", "Amount", "Subcategory", "Memo", "Currency"]


def can_handle(path_in):
    df = pd.read_csv(path_in, nrows=1)
    return set(df.columns) == set(expected_columns)


def load(path_in, currency = default_currency):
    df = pd.read_csv(path_in)
    assert set(df.columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (Lloyds Current)'
    
    df["Debit Amount"] = df["Debit Amount"].fillna(0)
    df["Credit Amount"] = df["Credit Amount"].fillna(0)
    
    df_out = pd.DataFrame(columns=target_columns)
    
    df_out.Date = pd.to_datetime(df["Transaction Date"], format='%d/%m/%Y')
    df_out.Account = df["Sort Code"].astype(str) + " " + df["Account Number"].astype(str)
    df_out.Currency = currency
    df_out.Amount = df["Credit Amount"] - df["Debit Amount"]
    df_out.Subcategory = df["Transaction Type"]
    df_out.Memo = df["Transaction Description"]
    
    return df_out


def load_save(folder_in, currency, path_out):
    files = glob.glob(os.path.join(folder_in, '*.csv'))
    print(f"found {len(files)} CSV files.")
    if len(files) == 0:
        return

    df_list = [load(f, currency) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(target_columns).cumcount()
    df = pd.concat(df_list)
    df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False).to_csv(path_out, index=False)
    

def load_save_default():
    load_save(default_folder_in, default_currency, default_path_out)
