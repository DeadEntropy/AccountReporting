import ast
import configparser
import pandas as pd


class IatIdentification:
    iat_types = ['SA', 'IAT', 'W_IN', 'W_OUT', 'SC', 'R', 'MC', 'O', 'FR', 'TAX', 'FPC', 'FLC', 'FLL']
    iat_fx_types = ['FX']

    def __init__(self, config=None):
        if config is None:
            self.config = configparser.ConfigParser()
            self.config.read('../config/config.ini')
        else:
            self.config = config

    def remove_duplicate(self, df):
        print('removing offsetting transactions...')
        new_columns = [n.strip() for n in ast.literal_eval(self.config['Mapping']['new_columns'])]
        assert set(list(df.columns)) == set(new_columns), 'columns do not match expectation.'

        df['IDX'] = df.index
        ndf = pd.merge(left=df, right=df, on=('Account', 'Memo', 'AccountType', 'Currency'), how='inner')
        out = ndf[(ndf.Amount_x == (-1) * ndf.Amount_y)
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

    def map_iat(self, df):
        print('mapping transactions between accounts...')
        new_columns = [n.strip() for n in ast.literal_eval(self.config['Mapping']['new_columns'])]
        assert set(list(df.columns)) == set(new_columns), 'columns do not match expectation.'

        df['IDX'] = df.index
        df_mini = df[df.Type.str.upper().isin(self.iat_types)][
            ['AccountType', 'Currency', 'Type', 'IDX', 'Date', 'Account', 'Amount']]
        ndf = pd.merge(left=df_mini, right=df_mini, on='Currency', how='inner')
        out = ndf[(ndf.Amount_x == (-1) * ndf.Amount_y)
                  & (abs(ndf.Date_x - ndf.Date_y) < pd.Timedelta(7, 'D'))
                  & (ndf.Account_x != ndf.Account_y)
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

    def map_iat_fx(self, df):
        print('mapping transactions between accounts...')
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
