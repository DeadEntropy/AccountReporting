import ast
import configparser
import pandas as pd


class IatIdentification:
    iat_types = ['SA', 'IAT', 'W_IN', 'W_OUT', 'SC', 'R', 'MC', 'O', 'FR', 'TAX', 'FPC']

    def __init__(self, config=None):
        if config is None:
            self.config = configparser.ConfigParser()
            self.config.read('../config/config.ini')
        else:
            self.config = config

    def map_iat(self, df):
        new_columns = [n.strip() for n in ast.literal_eval(self.config['Mapping']['new_columns'])]
        assert set(list(df.columns)) == set(new_columns), 'columns do not match expectation.'

        for index, row in df[df.Type.str.upper().isin(self.iat_types)].iterrows():
            if df.loc[index, 'FacingAccount'] != '':
                continue

            df_to_explore = df[(abs(df['Date'] - row['Date']) < pd.Timedelta(7, 'D'))
                               & (df['Account'] != row['Account'])
                               & (df['Currency'] == row['Currency'])
                               & (df['FacingAccount'] == '')
                               & (df['Type'].str.upper().isin(self.iat_types) )]

            df_to_explore_offsetting = df_to_explore[df_to_explore['Amount'] == -row['Amount']]
            if len(df_to_explore_offsetting) == 1:  # There is exactly one offsetting transaction
                df.loc[df_to_explore_offsetting.index[0], 'FacingAccount'] = row['Account']
                df.loc[index, 'FacingAccount'] = df.loc[df_to_explore_offsetting.index[0], 'Account']
            elif len(df_to_explore_offsetting) > 1:  # There are more than one offsetting transactions
                df.loc[df_to_explore_offsetting.index[0], 'FacingAccount'] = row['Account']
                df.loc[index, 'FacingAccount'] = df.loc[df_to_explore_offsetting.index[0], 'Account']

        return df
