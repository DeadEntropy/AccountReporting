import pandas as pd
import glob
import os

default_currency = 'GBP'
default_folder_in = r'D:\NicoFolder\BankAccount\NutmegData\RawData'
default_path_out = r'D:\NicoFolder\BankAccount\NutmegData\Nutmeg_ISAs.csv'

expected_columns = ['Date', 'Description', 'Pot', 'Amount (£)']
target_columns = ["Date", "Account", "Amount", "Subcategory", "Memo", "Currency"]

def load(path_in, currency):
    df = pd.read_csv(path_in, sep=',', parse_dates=[1])
    assert set(df.columns) == set(expected_columns)

    df_out = pd.DataFrame(columns=target_columns)
    df_out.Date = pd.to_datetime(df["Date"])
    df_out.Account = 'Nutmeg: ' + df['Pot'] 
    df_out.Currency = default_currency
    df_out.Amount = df["Amount (£)"]
    df_out.Subcategory = df["Description"]
    df_out.Memo = 'Nutmeg: ' + df['Pot']
    
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
    
load_save_default()