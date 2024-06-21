# coding=utf8
import ast
import configparser
import pandas as pd
from bkanalysis.config import config_helper as ch


class IatIdentification:
    _use_old = False
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
        # Ensure columns match the expected structure
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

        offsetting_rows = set()
        for dup in duplicate_couples:
            offsetting_rows.update(dup)

        return df.drop(offsetting_rows)
    
    
    @staticmethod
    def mark_facing_accounts(grp, adjust_dates: bool = False):
        if len(grp) == 1 or grp.Amount.iloc[0] == 0:
            return grp
        
        pairs = []
        i = 1
        while i < len(grp):
            if grp.Amount.iloc[i-1] == -grp.Amount.iloc[i] \
                and abs(grp.Date.iloc[i-1] - grp.Date.iloc[i]) < pd.Timedelta(7, "d") \
                and grp.Account.iloc[i-1] != grp.Account.iloc[i]:
                pairs.append((i-1, i))
                if adjust_dates:
                    adjusted_date = max(grp.Date.iloc[i-1], grp.Date.iloc[i])
                    grp.at[i-1, 'Date'] = adjusted_date
                    grp.at[i, 'Date'] = adjusted_date
                i += 2
            else:
                i += 1

        for i, j in pairs:
            grp.at[i, 'FacingAccount'] = grp.Account.iloc[j]
            grp.at[j, 'FacingAccount'] = grp.Account.iloc[i]

        return grp
    
    def map_iat(self, df, iat_value_col='Amount', adjust_dates: bool = False):
        # Ensure columns match the expected structure
        columns_req = ['Currency', 'FullType', 'Date', 'Account'] + [iat_value_col]
        assert all([col_req in df.columns for col_req in columns_req]), f'columns do not match expectation. Expected: {columns_req} but received: {df.columns}'

        # Filter the DataFrame to include only relevant rows
        df['IDX'] = df.index
        df_mini = df[df.FullType.str.upper().isin([t.upper() for t in self.iat_full_types])][columns_req + ['IDX']]

        # Perform the merge operation
        ndf = pd.merge(df_mini, df_mini, on='Currency', suffixes=('_x', '_y'))

        # Filter the merged DataFrame to find matching transactions
        out = ndf[(ndf[f'{iat_value_col}_x'] == -ndf[f'{iat_value_col}_y'])
                & (abs(ndf.Date_x - ndf.Date_y) < pd.Timedelta(7, 'D'))
                & (ndf.Account_x != ndf.Account_y)
                & (ndf[f'{iat_value_col}_x'] != 0)]

        # Identify the unique transaction pairs
        iat_transfers = list(zip(out.IDX_x.values, out.IDX_y.values))
        df.drop('IDX', axis=1, inplace=True)

        # Keep track of processed rows to avoid duplicates
        processed_rows = set()
        for dup in iat_transfers:
            if dup[0] in processed_rows or dup[1] in processed_rows:
                continue
            processed_rows.update(dup)

            if adjust_dates:
                adjusted_date = max(df.loc[dup[0], 'Date'], df.loc[dup[1], 'Date'])
                df.at[dup[0], 'Date'] = adjusted_date
                df.at[dup[1], 'Date'] = adjusted_date

            df.at[dup[0], 'FacingAccount'] = df.loc[dup[1], 'Account']
            df.at[dup[1], 'FacingAccount'] = df.loc[dup[0], 'Account']

        return df

    
    def map_iat_fx(self, df):
        if self._use_old:
            return self.map_iat_fx_OLD(df)
        new_columns = [n.strip() for n in ast.literal_eval(self.config['Mapping']['new_columns'])]
        assert set(list(df.columns)) == set(new_columns), 'columns do not match expectation.'

        df['IDX'] = df.index
        df_mini = df[df.Type.str.upper().isin(self.iat_fx_types)][['AccountType', 'Currency', 'Type', 'IDX', 'Date', 'Account', 'Amount']]
        df_mini['dummy_key'] = 1
        ndf = pd.merge(left=df_mini, right=df_mini, how='inner', on='dummy_key').drop('dummy_key', axis=1)
        out = ndf[(abs(ndf.Date_x - ndf.Date_y) < pd.Timedelta(7, 'D'))
                & (ndf.Account_x != ndf.Account_y)
                & (ndf.Currency_x != ndf.Currency_y)
                & (ndf.Amount_x != 0)]
        iat_transfers = list(zip(list(out.IDX_x.values), list(out.IDX_y.values)))
        df.drop('IDX', axis=1, inplace=True)

        offsetting_rows = set()
        for dup in iat_transfers:
            offsetting_rows.update(dup)

        for dup in iat_transfers:
            df.at[dup[0], 'FacingAccount'] = df.loc[dup[1], 'Account']
            df.at[dup[1], 'FacingAccount'] = df.loc[dup[0], 'Account']

        return df
