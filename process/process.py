import pandas as pd
import re
from process.process_helper import get_adjusted_month, get_adjusted_year, get_fiscal_year, get_year_to_date
import configparser
import ast


def map_memo(memo_series, map_path):
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


def map_type(memo_series, map_type_path, map_full_type_path, map_full_subtype_path, map_full_master_type_path):
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


def apply_overrides(df, override_path, map_full_type_path, map_full_subtype_path, map_full_master_type_path):
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


def process(config, ignore_overrides=False):
    df = pd.read_csv(config['IO']['path_aggregated'], parse_dates=[0])
    expected_columns = [n.strip() for n in ast.literal_eval(config['Mapping']['expected_columns'])]
    assert set(df.columns) == set(expected_columns), 'Columns do not match expectation.'

    new_columns = [n.strip() for n in ast.literal_eval(config['Mapping']['new_columns'])]
    df_out = pd.DataFrame(columns=list(new_columns))

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

    memo_mapped, map_missing = map_memo(df_out.MemoSimple, config['Mapping']['path_map'])
    df_out.MemoMapped = memo_mapped

    type, full_type, subtype, full_subtype, master_type, full_master_type, type_missing = \
        map_type(df_out.MemoMapped,
                 config['Mapping']['path_map_type'],
                 config['Mapping']['path_map_full_type'],
                 config['Mapping']['path_map_full_subtype'],
                 config['Mapping']['path_map_full_master_type'])

    df_out.Type = type
    df_out.FullType = full_type
    df_out.SubType = subtype
    df_out.FullSubType = full_subtype

    if not ignore_overrides:
        apply_overrides(df_out, config['Mapping']['path_override'],
                        config['Mapping']['path_map_full_type'],
                        config['Mapping']['path_map_full_subtype'],
                        config['Mapping']['path_map_full_master_type'])

    df_out.MasterType = master_type
    df_out.FullMasterType = full_master_type

    return df_out, map_missing, type_missing


def process_save():
    config = configparser.ConfigParser()
    config.read('../config/config.ini')

    df, map_missing, type_missing = process(config)
    df.to_csv(config['IO']['path_processed'], index=False)

    with open(config['IO']['path_missing_map'], 'w') as outfile:
        outfile.write("\n".join(map_missing))

    with open(config['IO']['path_missing_type'], 'w') as outfile:
        outfile.write("\n".join(type_missing))
