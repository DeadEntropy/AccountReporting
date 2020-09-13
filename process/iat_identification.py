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

        smallest_elt = 0
        indices_to_remove = []
        for index, row in df.iterrows():
            df_date_match = df[(df['Account'] == row['Account'])]
            df_date_match = df_date_match[(df_date_match['Amount'] == -row['Amount'])]
            df_date_match = df_date_match[(abs(df_date_match['Date'] - row['Date']) < pd.Timedelta(7, 'D'))]
            df_offsetting = df_date_match[(df_date_match['MemoMapped'] == row['MemoMapped'])
                                          & (df_date_match.index < index)
                                          & (df_date_match.index > smallest_elt)]

            if len(df_offsetting) >= 1:  # There is one or more offsetting transaction
                indices_to_remove.append(df_offsetting.index[0])
                indices_to_remove.append(index)
                smallest_elt = df_offsetting.index[0]

        return df.drop(indices_to_remove)

    def map_iat(self, df):
        print('mapping transactions between accounts...')
        new_columns = [n.strip() for n in ast.literal_eval(self.config['Mapping']['new_columns'])]
        assert set(list(df.columns)) == set(new_columns), 'columns do not match expectation.'

        for index, row in df[df.Type.str.upper().isin(self.iat_types)].iterrows():
            if df.loc[index, 'FacingAccount'] != '':
                continue

            df_to_explore = df[(abs(df['Date'] - row['Date']) < pd.Timedelta(7, 'D'))
                               & (df['Account'] != row['Account'])
                               & (df['Currency'] == row['Currency'])
                               & (df['FacingAccount'] == '')
                               & (df['Type'].str.upper().isin(self.iat_types))]

            df_to_explore_offsetting = df_to_explore[df_to_explore['Amount'] == -row['Amount']]
            if len(df_to_explore_offsetting) >= 1:  # There is one or more offsetting transaction
                df.loc[df_to_explore_offsetting.index[0], 'FacingAccount'] = row['Account']
                df.loc[index, 'FacingAccount'] = df.loc[df_to_explore_offsetting.index[0], 'Account']

        return df

    def map_iat_fx(self, df):
        print('mapping fx transactions between accounts...')
        new_columns = [n.strip() for n in ast.literal_eval(self.config['Mapping']['new_columns'])]
        assert set(list(df.columns)) == set(new_columns), 'columns do not match expectation.'

        for index, row in df[df.Type.str.upper().isin(self.iat_fx_types)].iterrows():
            if df.loc[index, 'FacingAccount'] != '':
                continue

            df_to_explore = df[(df['Date'] == row['Date'])
                               & (df['Account'] != row['Account'])
                               & (df['Currency'] != row['Currency'])
                               & (df['FacingAccount'] == '')
                               & (df['Type'].str.upper().isin(self.iat_fx_types))]

            if len(df_to_explore) >= 1:  # There is one or more offsetting transaction
                df.loc[df_to_explore.index[0], 'FacingAccount'] = row['Account']
                df.loc[index, 'FacingAccount'] = df.loc[df_to_explore.index[0], 'Account']

        return df
