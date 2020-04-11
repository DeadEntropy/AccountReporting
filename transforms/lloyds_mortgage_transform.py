import pandas as pd
import glob
import os

default_currency = 'GBP'
default_account_name = 'Lloyds Mortgage'
default_folder_in = r'D:\NicoFolder\BankAccount\LloydsData\RawData\MortgageAccount'
default_path_out = r'D:\NicoFolder\BankAccount\LloydsData\Lloyds_Mortgage.csv'

expected_columns = ['DATE', 'TRANSACTION', 'OUT(£)', 'IN(£)', 'BALANCE(£)']
target_columns = ["Date", "Account", "Amount", "Subcategory", "Memo", "Currency"]

def to_memo(row):
    if row['TRANSACTION'] == 'Interest':
        return 'Mortgage Interest'
    elif row['TRANSACTION'] == 'Bank payment':
        return 'Mortgage Repayment'
    elif row['TRANSACTION'] == 'Direct Debit':
        return 'Mortgage Repayment'
    return row['TRANSACTION']

    
def load(path_in, account_name, currency):
    df = pd.read_csv(path_in, parse_dates=[1])
    assert set(df.columns) == set(expected_columns)
    
    df["OUT(£)"] = df["OUT(£)"].fillna(0)
    df["IN(£)"] = df["IN(£)"].fillna(0)
    
    df_out = pd.DataFrame(columns=target_columns)
    
    df_out.Date = pd.to_datetime(df.DATE)
    df_out.Account = account_name
    df_out.Currency = currency
    df_out.Amount = df["IN(£)"] - df["OUT(£)"]
    df_out.Subcategory = df.TRANSACTION
    df_out.Memo = df.apply (lambda row: to_memo(row), axis=1)
    
    return df_out


def load_save(folder_in, account_name, currency, path_out):
    files = glob.glob(os.path.join(folder_in, '*.csv'))
    print(f"found {len(files)} CSV files.")
    if len(files) == 0:
        return

    df_list = [load(f, account_name, currency) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(target_columns).cumcount()
    df = pd.concat(df_list)
    df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False).to_csv(path_out, index=False)


def load_save_default():
    load_save(default_folder_in, default_account_name, default_currency, default_path_out)
    
load_save_default()