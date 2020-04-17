import pandas as pd
import datetime
import re

default_path_in = r'D:\NicoFolder\BankAccount\lake_result.csv'
default_path_out = r'D:\NicoFolder\BankAccount\lake_result_processed.csv'

new_columns = ['YearToDate', 'FiscalYear', 'AdjustedYear', 'AdjustedMonth', 'Year', 'Month', 'Day', 'Date', 'Account',
               'Amount', 'Subcategory', 'Memo', 'Currency', 'MemoSimple', 'MemoMapped', 'Type', 'FullType', 'SubType',
               'FullSubType', 'Week', 'MasterType', 'FullMasterType']
expected_columns = ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency']

default_map_path = r'D:\NicoFolder\BankAccount\Utils\MemoMapping.csv'
default_map_missing_path = r'D:\NicoFolder\BankAccount\Utils\MemoMappingMissing.csv'
default_type_missing_path = r'D:\NicoFolder\BankAccount\Utils\MemoTypeMissing.csv'

default_type_map_path = r'D:\NicoFolder\BankAccount\Utils\TypeMapping.csv'
default_full_type_map_path = r'D:\NicoFolder\BankAccount\Utils\FullTypeMapping.csv'
default_full_subtype_map_path = r'D:\NicoFolder\BankAccount\Utils\FullSubTypeMapping.csv'
default_full_master_type_map_path = r'D:\NicoFolder\BankAccount\Utils\FullMasterTypeMapping.csv'
default_override_path = r'D:\NicoFolder\BankAccount\Utils\TypeOverrides.csv'


def get_adjusted_month(dt):
    if dt.day > 20:
        if dt.month < 12:
            return dt.month + 1
        return 1
    return dt.month


def get_adjusted_year(dt):
    if dt.day > 20 and dt.month == 12:
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

    mapping_df['Memo Mapped'] = mapping_df['Memo Mapped'].str.strip().str.upper()
    mapping_df['Memo Simple'] = mapping_df['Memo Simple'].str.strip().str.upper()

    mapping = pd.Series(mapping_df['Memo Mapped'].values, index=mapping_df['Memo Simple']).to_dict()
    memo_mapped = []
    map_missing = []
    for memo in memo_series.str.upper():
        if memo in mapping.keys():
            memo_mapped.append(mapping[memo].strip())
        else:
            memo_mapped.append('N/A')
            map_missing.append(memo)
    return memo_mapped, list(set(map_missing))


def get_full_type(type, type_mapping):
    if type in type_mapping.keys():
        return type_mapping[type].strip()
    else:
        return ''


def get_master_type(type, master_type_mapping, full_master_type_mapping):
    if type in master_type_mapping:
        try:
            master_type = master_type_mapping[type].strip()
            if master_type_mapping[type] in full_master_type_mapping.keys():
                return master_type, full_master_type_mapping[master_type_mapping[type]].strip()
            else:
                return master_type, 'N/A'
        except AttributeError as e:
            raise AttributeError(f'failed on: {type} {master_type_mapping[type]}',
                                 e)
    else:
        return 'N/A', 'N/A'


def map_type(memo_series, map_type_path=default_type_map_path, map_full_type_path=default_full_type_map_path,
             map_full_subtype_path=default_full_subtype_map_path,
             map_full_master_type_path=default_full_master_type_map_path):
    mapping_df = pd.read_csv(map_type_path)

    mapping_df["Memo Mapped"] = mapping_df["Memo Mapped"].fillna('').str.strip().str.upper()
    mapping_df["Type"] = mapping_df["Type"].fillna('').str.strip()
    mapping_df["SubType"] = mapping_df["SubType"].fillna('').str.strip()

    type_mapping_df = pd.read_csv(map_full_type_path)
    type_mapping_df['MasterType'] = type_mapping_df['MasterType'].fillna('').str.strip()
    subtype_mapping_df = pd.read_csv(map_full_subtype_path)
    master_type_mapping_df = pd.read_csv(map_full_master_type_path)

    mapping_a = pd.Series(mapping_df['Type'].values, index=mapping_df['Memo Mapped']).to_dict()
    mapping_b = pd.Series(mapping_df['SubType'].values, index=mapping_df['Memo Mapped']).to_dict()
    type_mapping = pd.Series(type_mapping_df['FullType'].values, index=type_mapping_df['Type']).to_dict()
    master_type_mapping = pd.Series(type_mapping_df['MasterType'].values, index=type_mapping_df['Type']).to_dict()
    subtype_mapping = pd.Series(subtype_mapping_df['FullSubType'].values, index=subtype_mapping_df['SubType']).to_dict()
    full_master_type_mapping = pd.Series(master_type_mapping_df['FullMasterType'].values,
                                         index=master_type_mapping_df['MasterType']).to_dict()

    type = []
    full_type = []
    subtype = []
    full_subtype = []
    master_type = []
    full_master_type = []
    type_missing = []
    for memo in memo_series.str.upper():
        if memo in mapping_a.keys():
            try:
                type.append(mapping_a[memo].strip())
                subtype.append(mapping_b[memo].strip())

                full_type.append(get_full_type(mapping_a[memo], type_mapping))
                full_subtype.append(get_full_type(mapping_b[memo], subtype_mapping))
            except AttributeError as e:
                raise AttributeError(f'failed on: {memo} {mapping_a[memo]} {mapping_b[memo]}', e)

            m, full_m = get_master_type(mapping_a[memo], master_type_mapping, full_master_type_mapping)
            master_type.append(m)
            full_master_type.append(full_m)
        else:
            type.append('N/A')
            subtype.append('N/A')
            full_type.append('N/A')
            full_subtype.append('N/A')
            master_type.append('N/A')
            full_master_type.append('N/A')
            type_missing.append(memo)

    return type, full_type, subtype, full_subtype, master_type, full_master_type, list(set(type_missing))


def apply_overrides(df, override_path=default_override_path, map_full_type_path=default_full_type_map_path,
                    map_full_subtype_path=default_full_subtype_map_path,
                    map_full_master_type_path=default_full_master_type_map_path):
    mapping_df = pd.read_csv(override_path, parse_dates=[0])
    mapping_df['Date'] = mapping_df['Date'].apply(lambda x: x.strftime('%d-%b-%Y'))
    mapping_df['MemoMapped'] = mapping_df['MemoMapped'].fillna('').str.strip().str.upper()
    mapping_df['Account'] = mapping_df['Account'].fillna('').str.strip().str.upper()
    mapping_df['OverridesType'] = mapping_df['OverridesType'].fillna('').str.strip().str.upper()
    mapping_df['OverrideSubType'] = mapping_df['OverrideSubType'].fillna('').str.strip().str.upper()

    subset_values = mapping_df[['OverridesType', 'OverrideSubType']]
    subset_keys = mapping_df[['Date', 'Account', 'MemoMapped']]
    dict_overrides = pd.Series([tuple(x) for x in subset_values.to_numpy()],
                               index=[tuple(x) for x in subset_keys.to_numpy()]).to_dict()

    type_mapping_df = pd.read_csv(map_full_type_path)
    subtype_mapping_df = pd.read_csv(map_full_subtype_path)
    master_type_mapping_df = pd.read_csv(map_full_master_type_path)
    type_mapping_df['MasterType'] = type_mapping_df['MasterType'].fillna('').str.strip()
    type_mapping = pd.Series(type_mapping_df['FullType'].values, index=type_mapping_df['Type']).to_dict()
    subtype_mapping = pd.Series(subtype_mapping_df['FullSubType'].values, index=subtype_mapping_df['SubType']).to_dict()
    master_type_mapping = pd.Series(type_mapping_df['MasterType'].values, index=type_mapping_df['Type']).to_dict()
    full_master_type_mapping = pd.Series(master_type_mapping_df['FullMasterType'].values,
                                         index=master_type_mapping_df['MasterType']).to_dict()

    for index, row in df.iterrows():
        key = (row['Date'].strftime("%d-%b-%Y"), row['Account'].strip(), row['MemoMapped'].strip())
        if key in dict_overrides.keys():
            if dict_overrides[key][0] != '':
                df.loc[index, 'Type'] = dict_overrides[key][0]
                df.loc[index, 'FullType'] = get_full_type(dict_overrides[key][0], type_mapping)
                df.loc[index, 'MasterType'], df.loc[index, 'FullMasterType'] = get_master_type(dict_overrides[key][0],
                                                                                               master_type_mapping,
                                                                                               full_master_type_mapping)
            if dict_overrides[key][1] != '':
                df.loc[index, 'SubType'] = dict_overrides[key][1]
                df.loc[index, 'FullSubType'] = get_full_type(dict_overrides[key][1], subtype_mapping)

    return df


def process(path_in=default_path_in, ignore_overrides=False):
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
    df_out.MemoSimple = [re.sub('\*', '', re.sub(' +', ' ', s.split(' ON')[0])).replace(',', '').strip() for s in
                         df.Memo]

    memo_mapped, map_missing = map_memo(df_out.MemoSimple, default_map_path)
    df_out.MemoMapped = memo_mapped

    type, full_type, subtype, full_subtype, master_type, full_master_type, type_missing = \
        map_type(df_out.MemoMapped, default_type_map_path, default_full_type_map_path, default_full_subtype_map_path,
                 default_full_master_type_map_path)

    df_out.Type = type
    df_out.FullType = full_type
    df_out.SubType = subtype
    df_out.FullSubType = full_subtype

    if not ignore_overrides:
        apply_overrides(df_out, default_override_path)

    df_out.MasterType = master_type
    df_out.FullMasterType = full_master_type

    return df_out, map_missing, type_missing


def process_save(path_in=default_path_in, path_out=default_path_out):
    df, map_missing, type_missing = process(path_in)
    df.to_csv(path_out, index=False)

    with open(default_map_missing_path, 'w') as outfile:
        outfile.write("\n".join(map_missing))

    with open(default_type_missing_path, 'w') as outfile:
        outfile.write("\n".join(type_missing))
