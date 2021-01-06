# coding=utf8
import pandas as pd
import re
from bkanalysis.process.process_helper import get_adjusted_month, get_adjusted_year, get_fiscal_year, get_year_to_date, \
    get_missing_map, get_missing_type
from bkanalysis.process.iat_identification import IatIdentification
import configparser
from bkanalysis.config import config_helper as ch
import ast
import os


class Process:
    @staticmethod
    def __initialise_map(config, path, root='folder_root', default_columns=None, parse_dates=None):
        try:
            return pd.read_csv(ch.get_path(config, path, root), parse_dates=parse_dates)
        except:
            print(f'Could not find mapping file in: {ch.get_path(config, path, root)}.')
            try:
                pd.DataFrame(columns=default_columns).to_csv(ch.get_path(config, path, root), index=False)
            except:
                raise Exception(f'Couldnt access {os.path.abspath(ch.get_path(config, path))}')
            return pd.DataFrame(columns=default_columns)

    @staticmethod
    def __save_to_csv(config, df, key, root='folder_root'):
        if root in config:
            path = os.path.join(config[root], config[key])
        else:
            path = config[key]
        df.to_csv(path, index=False)

    def __init__(self, config=None):
        if config is None:
            self.config = configparser.ConfigParser()
            self.config.read('config/config.ini')
        else:
            self.config = config

        self.map_simple = Process.__initialise_map(self.config['Mapping'], 'path_map', default_columns=['Memo Simple', 'Memo Mapped'])
        self.map_main = Process.__initialise_map(self.config['Mapping'],'path_map_type', default_columns=['Memo Mapped', 'Type', 'SubType'])
        self.map_full_type = Process.__initialise_map(self.config['Mapping'],'path_map_full_type', default_columns=['Type', 'FullType', 'MasterType'])
        self.map_full_subtype = Process.__initialise_map(self.config['Mapping'],'path_map_full_subtype', default_columns=['SubType', 'FullSubType'])
        self.map_master = Process.__initialise_map(self.config['Mapping'],'path_map_full_master_type', default_columns=['MasterType', 'FullMasterType'])
        self.mapping_override_df = Process.__initialise_map(self.config['Mapping'],'path_override', parse_dates=[0], \
                                                            default_columns=['Date', 'Account', 'MemoMapped', 'OverridesType','OverrideSubType'])

    def __del__(self):
        self.__save_to_csv(self.config['Mapping'], self.map_simple, 'path_map')
        self.__save_to_csv(self.config['Mapping'], self.map_main, 'path_map_type')
        self.__save_to_csv(self.config['Mapping'], self.map_full_type, 'path_map_full_type')
        self.__save_to_csv(self.config['Mapping'], self.map_full_subtype, 'path_map_full_subtype')
        self.__save_to_csv(self.config['Mapping'], self.map_master, 'path_map_full_master_type')

    def map_memo(self, memo_series):
        self.map_simple['Memo Mapped'] = self.map_simple['Memo Mapped'].str.strip().str.upper()
        self.map_simple['Memo Simple'] = self.map_simple['Memo Simple'].str.strip().str.upper()

        mapping = pd.Series(self.map_simple['Memo Mapped'].values, index=self.map_simple['Memo Simple']).to_dict()
        memo_mapped = []
        for memo in memo_series.str.upper():
            if memo in mapping.keys():
                try:
                    memo_mapped.append(mapping[memo].strip())
                except AttributeError:
                    raise AttributeError(
                        f'Failed to strip {memo}: {mapping[memo]}. Please ensure the value is a string.')
            else:
                memo_mapped.append(get_missing_map(memo, mapping))

        self.map_simple = pd.DataFrame(mapping.items(), columns=['Memo Simple', 'Memo Mapped'])
        return memo_mapped

    @staticmethod
    def get_full_type(type, type_mapping):
        if type in type_mapping.keys():
            return type_mapping[type].strip()
        else:
            return ''

    @staticmethod
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

    def map_type(self, memo_series):
        self.map_main["Memo Mapped"] = self.map_main["Memo Mapped"].fillna('').str.strip().str.upper()
        self.map_main["Type"] = self.map_main["Type"].fillna('').str.strip().str.upper()
        self.map_main["SubType"] = self.map_main["SubType"].fillna('').str.strip().str.upper()

        self.map_full_type['MasterType'] = self.map_full_type['MasterType'].fillna('').str.strip()

        mapping_a = pd.Series(self.map_main['Type'].values, index=self.map_main['Memo Mapped']).to_dict()
        mapping_b = pd.Series(self.map_main['SubType'].values, index=self.map_main['Memo Mapped']).to_dict()
        type_mapping = pd.Series(self.map_full_type['FullType'].values, index=self.map_full_type['Type']).to_dict()
        master_type_mapping = pd.Series(self.map_full_type['MasterType'].values,
                                        index=self.map_full_type['Type']).to_dict()
        subtype_mapping = pd.Series(self.map_full_subtype['FullSubType'].values,
                                    index=self.map_full_subtype['SubType']).to_dict()
        full_master_type_mapping = pd.Series(self.map_master['FullMasterType'].values,
                                             index=self.map_master['MasterType']).to_dict()

        type = []
        full_type = []
        subtype = []
        full_subtype = []
        master_type = []
        full_master_type = []
        for memo in memo_series.str.upper():
            if memo in mapping_a.keys():
                try:
                    type.append(mapping_a[memo].strip())
                    subtype.append(mapping_b[memo].strip())

                    full_type.append(self.get_full_type(mapping_a[memo], type_mapping))
                    full_subtype.append(self.get_full_type(mapping_b[memo], subtype_mapping))
                except AttributeError as e:
                    raise AttributeError(f'failed on: {memo} {mapping_a[memo]} {mapping_b[memo]}', e)

                m, full_m = self.get_master_type(mapping_a[memo], master_type_mapping, full_master_type_mapping)
                master_type.append(m)
                full_master_type.append(full_m)
            else:
                t, st = get_missing_type(memo, mapping_a, mapping_b)
                type.append(t)
                subtype.append(st)

                new_row = pd.DataFrame([[memo, t, st]], columns=['Memo Mapped', 'Type', 'SubType'])
                self.map_main = pd.concat([self.map_main, new_row], ignore_index=True)

                full_type.append(self.get_full_type(t, type_mapping))
                full_subtype.append(self.get_full_type(st, subtype_mapping))
                m, full_m = self.get_master_type(t, master_type_mapping, full_master_type_mapping)
                master_type.append(m)
                full_master_type.append(full_m)

        return type, full_type, subtype, full_subtype, master_type, full_master_type

    def apply_overrides(self, df):
        self.mapping_override_df['Date'] = self.mapping_override_df['Date'].apply(lambda x: x.strftime('%d-%b-%Y'))
        self.mapping_override_df['MemoMapped'] = self.mapping_override_df['MemoMapped'].fillna(
            '').str.strip().str.upper()
        self.mapping_override_df['Account'] = self.mapping_override_df['Account'].fillna('').str.strip().str.upper()
        self.mapping_override_df['OverridesType'] = self.mapping_override_df['OverridesType'].fillna(
            '').str.strip().str.upper()
        self.mapping_override_df['OverrideSubType'] = self.mapping_override_df['OverrideSubType'].fillna(
            '').str.strip().str.upper()

        subset_values = self.mapping_override_df[['OverridesType', 'OverrideSubType']]
        subset_keys = self.mapping_override_df[['Date', 'Account', 'MemoMapped']]
        dict_overrides = pd.Series([tuple(x) for x in subset_values.to_numpy()],
                                   index=[tuple(x) for x in subset_keys.to_numpy()]).to_dict()

        self.map_full_type['MasterType'] = self.map_full_type['MasterType'].fillna('').str.strip()
        type_mapping = pd.Series(self.map_full_type['FullType'].values, index=self.map_full_type['Type']).to_dict()
        subtype_mapping = pd.Series(self.map_full_subtype['FullSubType'].values,
                                    index=self.map_full_subtype['SubType']).to_dict()
        master_type_mapping = pd.Series(self.map_full_type['MasterType'].values,
                                        index=self.map_full_type['Type']).to_dict()
        full_master_type_mapping = pd.Series(self.map_master['FullMasterType'].values,
                                             index=self.map_master['MasterType']).to_dict()

        for index, row in df.iterrows():
            try:
                key = (row['Date'].strftime("%d-%b-%Y"), row['Account'].strip(), row['MemoMapped'].strip())
            except AttributeError:
                print(row)
                raise
            if key in dict_overrides.keys():
                if dict_overrides[key][0] != '':
                    df.loc[index, 'Type'] = dict_overrides[key][0]
                    df.loc[index, 'FullType'] = self.get_full_type(dict_overrides[key][0], type_mapping)
                    df.loc[index, 'MasterType'], df.loc[index, 'FullMasterType'] = self.get_master_type(
                        dict_overrides[key][0],
                        master_type_mapping,
                        full_master_type_mapping)
                if dict_overrides[key][1] != '':
                    df.loc[index, 'SubType'] = dict_overrides[key][1]
                    df.loc[index, 'FullSubType'] = self.get_full_type(dict_overrides[key][1], subtype_mapping)

        return df

    def remove_offsetting(self, df):
        iat = IatIdentification(self.config)
        df_out = iat.remove_duplicate(df)
        df_out = iat.map_iat(df_out)
        df_out = iat.map_iat_fx(df_out)

        return df_out

    @staticmethod
    def __clean_memo(s):
        if isinstance(s, str):
            return re.sub('\*', '', re.sub(' +', ' ', s.split(' ON')[0])).replace(',', '').strip()
        return s

    def extend(self, df, ignore_overrides=False):
        expected_columns = [n.strip() for n in ast.literal_eval(self.config['Mapping']['expected_columns'])]
        assert set(df.columns) == set(expected_columns), 'Columns do not match expectation.'

        new_columns = [n.strip() for n in ast.literal_eval(self.config['Mapping']['new_columns'])]
        df_out = pd.DataFrame(columns=list(new_columns))

        df_out.Date = df.Date
        df_out.Account = df.Account.str.strip()
        df_out.AccountType = df.AccountType.str.strip()
        df_out.Amount = df.Amount
        df_out.Subcategory = df.Subcategory.str.strip()
        df_out.Memo = df.Memo.str.strip()
        df_out.Currency = df.Currency
        df_out.SourceFile = df.SourceFile

        df_out.Day = df.Date.dt.day
        df_out.Month = df.Date.dt.month
        df_out.Year = df.Date.dt.year
        df_out.AdjustedMonth = [get_adjusted_month(dt) for dt in df.Date]
        df_out.AdjustedYear = [get_adjusted_year(dt) for dt in df.Date]
        df_out.FiscalYear = [get_fiscal_year(dt) for dt in df.Date]
        df_out.YearToDate = [get_year_to_date(dt) for dt in df.Date]
        df_out.MemoSimple = [self.__clean_memo(s) for s in df.Memo]

        memo_mapped = self.map_memo(df_out.MemoSimple)
        df_out.MemoMapped = memo_mapped

        type, full_type, subtype, full_subtype, master_type, full_master_type = \
            self.map_type(df_out.MemoMapped)

        df_out.Type = type
        df_out.FullType = full_type
        df_out.SubType = subtype
        df_out.FullSubType = full_subtype

        if not ignore_overrides:
            self.apply_overrides(df_out)

        df_out.MasterType = master_type
        df_out.FullMasterType = full_master_type

        df_out.FacingAccount = ''

        return df_out

    def process(self, df, ignore_overrides=False):
        df_out = self.extend(df, ignore_overrides)
        df_out = self.remove_offsetting(df_out)
        return df_out

    def save(self, df):
        df_copy = df.copy(True)
        df_copy.Date = df_copy.Date.apply(lambda x: x.strftime("%d-%b-%Y"))
        df_copy.to_csv(self.config['IO']['path_processed'], index=False)

    def process_save(self):
        df_raw = pd.read_csv(self.config['IO']['path_aggregated'], parse_dates=[0])
        print(f'Process {df_raw.shape[0]} line(s).')
        df = self.process(df_raw)
        self.save(df)
