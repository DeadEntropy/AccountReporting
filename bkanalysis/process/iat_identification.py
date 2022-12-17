# coding=utf8
import ast
import configparser
import pandas as pd
from bkanalysis.config import config_helper as ch


class IatIdentification:
    iat_types = ['SA', 'IAT', 'W_IN', 'W_OUT', 'SC', 'R', 'MC', 'O', 'FR', 'TAX', 'FPC', 'FLC', 'FLL', 'FSC']
    iat_full_types = ['Savings', 'Intra-Account Transfert', 'Wire In', 'Wire Out', 'Service Charge', 'Rent/Mortgage', 'Others', 'Tax', 'Flat Capital', 'Flat Living Cost']
    iat_fx_types = ['FX']
    relative_tolerance: float = 0.0

    def __init__(self, config=None):
        if config is None:
            self.config = configparser.ConfigParser()
            if len(self.config.read(ch.source)) != 1:
                raise OSError(f'no config found in {ch.source}')
        else:
            self.config = config

    def remove_duplicate(self, df):
        # print('removing offsetting transactions...')
        new_columns = [n.strip() for n in ast.literal_eval(self.config['Mapping']['new_columns'])]
        assert set(list(df.columns)) == set(new_columns), f'columns do not match expectation. Expected : [{new_columns}]'

        df['IDX'] = df.index
        ndf = pd.merge(left=df, right=df, on=('Account', 'Memo', 'AccountType', 'Currency'), how='inner')
        out = ndf[(abs(ndf.Amount_x + ndf.Amount_y) < ndf.Amount_x * self.relative_tolerance)
                  & (ndf.Date_x - ndf.Date_y < pd.Timedelta(7, 'D'))
                  & (ndf.IDX_x < ndf.IDX_y)
                  & (ndf.Amount_x != 0)]

        duplicate_couples = list(zip(list(out.IDX_x.values), list(out.IDX_y.values)))
        df.drop('IDX', axis=1, inplace=True)

        offsetting_rows = []
        for dup in duplicate_couples:
            if dup[0] in offsetting_rows:
                continue
            if dup[1] in offsetting_rows:
                continue
            offsetting_rows.append(dup[0])
            offsetting_rows.append(dup[1])

        return df.drop(offsetting_rows)

    def map_iat(self, df, iat_value_col='Amount', adjust_dates:bool=False):
        # print('mapping transactions between accounts...')
        columns_req = ['Currency', 'FullType', 'Date', 'Account'] + [iat_value_col]
        assert all([col_req in df.columns for col_req in columns_req]), f'columns do not match expectation. Expected : {columns_req} but receive {df.columns}'

        df['IDX'] = df.index
        df_mini = df[df.FullType.str.upper().isin([t.upper() for t in self.iat_full_types])][columns_req + ['IDX']]
        ndf = pd.merge(left=df_mini, right=df_mini, on='Currency', how='inner')
        out = ndf[(ndf[f'{iat_value_col}_x'] == (-1) * ndf[f'{iat_value_col}_y'])
                  & (abs(ndf.Date_x - ndf.Date_y) < pd.Timedelta(7, 'D'))
                  & (ndf.Account_x != ndf.Account_y)
                  & (ndf[f'{iat_value_col}_x'] != 0)]
        iat_transfers = list(zip(list(out.IDX_x.values), list(out.IDX_y.values)))
        df.drop('IDX', axis=1, inplace=True)

        iat_transfers_unique = []
        offsetting_rows = []
        for dup in iat_transfers:
            if dup[0] in offsetting_rows:
                continue
            if dup[1] in offsetting_rows:
                continue
            iat_transfers_unique.append(dup)
            offsetting_rows.append(dup[0])
            offsetting_rows.append(dup[1])

        for dup in iat_transfers_unique:
            if adjust_dates:
                adjusted_date = max(df.loc[dup[0], 'Date'], df.loc[dup[1], 'Date'])
                df.loc[dup[0], 'Date'] = adjusted_date
                df.loc[dup[1], 'Date'] = adjusted_date

            df.loc[dup[0], 'FacingAccount'] = df.loc[dup[1], 'Account']
            df.loc[dup[1], 'FacingAccount'] = df.loc[dup[0], 'Account']

        return df

    def map_iat_fx(self, df):
        # print('mapping transactions between accounts...')
        new_columns = [n.strip() for n in ast.literal_eval(self.config['Mapping']['new_columns'])]
        assert set(list(df.columns)) == set(new_columns), 'columns do not match expectation.'

        df['IDX'] = df.index
        df_mini = df[df.Type.str.upper().isin(self.iat_fx_types)][
            ['AccountType', 'Currency', 'Type', 'IDX', 'Date', 'Account', 'Amount']]
        df_mini['dummy_key'] = 1
        ndf = pd.merge(left=df_mini, right=df_mini, how='inner',on='dummy_key').drop('dummy_key', axis=1)
        out = ndf[(abs(ndf.Date_x - ndf.Date_y) < pd.Timedelta(7, 'D'))
                  & (ndf.Account_x != ndf.Account_y)
                  & (ndf.Currency_x != ndf.Currency_y)
                  & (ndf.Amount_x != 0)]
        iat_transfers = list(zip(list(out.IDX_x.values), list(out.IDX_y.values)))
        df.drop('IDX', axis=1, inplace=True)

        iat_transfers_unique = []
        offsetting_rows = []
        for dup in iat_transfers:
            if dup[0] in offsetting_rows:
                continue
            if dup[1] in offsetting_rows:
                continue
            iat_transfers_unique.append(dup)
            offsetting_rows.append(dup[0])
            offsetting_rows.append(dup[1])

        for dup in iat_transfers_unique:
            df.loc[dup[0], 'FacingAccount'] = df.loc[dup[1], 'Account']
            df.loc[dup[1], 'FacingAccount'] = df.loc[dup[0], 'Account']

        return df
