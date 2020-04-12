import pandas as pd
import datetime
import re

default_path_in = r'D:\NicoFolder\BankAccount\lake_result.csv'
default_path_out = r'D:\NicoFolder\BankAccount\lake_result_processed.csv'
new_columns = ['YearToDate', 'FiscalYear', 'AdjustedYear', 'AdjustedMonth', 'Year', 'Month', 'Day', 'Date', 'Account',
               'Amount', 'Subcategory', 'Memo', 'Currency', 'MemoSimple', 'MemoMapped']
expected_columns = ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency']
default_map_path = r'D:\NicoFolder\BankAccount\Utils\MemoMapping.csv'
default_map_missing_path = r'D:\NicoFolder\BankAccount\Utils\MemoMappingMissing.csv'


def get_adjusted_month(dt):
    if dt.day > 20:
        if dt.month < 12:
            return dt.month + 1
        return 1
    return dt.month


def get_adjusted_year(dt):
    if dt.day > 20 and dt.month < 12:
        return dt.year + 1
    return dt.year


def get_fiscal_year(dt):
    if dt < datetime.datetime(dt.year, 4, 5):
        return dt.year - 1
    return dt.year


def get_year_to_date(dt):
    return int((datetime.date.today() - dt.date()).days / 365.25)


def map_memo(memo_series, map_path=default_map_path):
    mapping_df = pd.read_csv(map_path)

    mapping_df['Memo Mapped'] = mapping_df['Memo Mapped'].str.strip()
    mapping_df['Memo Simple'] = mapping_df['Memo Simple'].str.strip()

    mapping = pd.Series(mapping_df['Memo Mapped'].values, index=mapping_df['Memo Simple']).to_dict()
    memo_mapped = []
    map_missing = []
    for memo in memo_series:
        if memo in mapping.keys():
            memo_mapped.append(mapping[memo].strip())
        else:
            memo_mapped.append('N/A')
            map_missing.append(memo)
    return memo_mapped, list(set(map_missing))


def process(path_in=default_path_in):
    df = pd.read_csv(path_in, parse_dates=[0])
    assert set(df.columns) == set(expected_columns), 'Columns do not match expectation.'

    df_out = pd.DataFrame(columns=new_columns)

    df_out.Date = df.Date
    df_out.Account = df.Account.str.strip()
    df_out.Amount = df.Amount
    df_out.Subcategory = df.Subcategory.str.strip()
    df_out.Memo = df.Memo.str.strip()
    df_out.Currency = df.Currency

    df_out.Day = df.Date.dt.day
    df_out.Month = df.Date.dt.month
    df_out.Year = df.Date.dt.year
    df_out.AdjustedMonth = [get_adjusted_month(dt) for dt in df.Date]
    df_out.AdjustedYear = [get_adjusted_year(dt) for dt in df.Date]
    df_out.FiscalYear = [get_fiscal_year(dt) for dt in df.Date]
    df_out.YearToDate = [get_year_to_date(dt) for dt in df.Date]
    df_out.MemoSimple = [re.sub('\*', '', re.sub(' +', ' ', s.split(' ON')[0])).replace(',', '').strip() for s in df.Memo]

    memo_mapped, map_missing = map_memo(df_out.MemoSimple, default_map_path)

    df_out.MemoMapped = memo_mapped

    return df_out, map_missing


def process_save(path_in=default_path_in, path_out=default_path_out):
    df, map_missing = process(path_in)
    df.to_csv(path_out, index=False)

    with open(default_map_missing_path, 'w') as outfile:
        outfile.write("\n".join(map_missing))
