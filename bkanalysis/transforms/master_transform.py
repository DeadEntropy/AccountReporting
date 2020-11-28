# coding=utf8
import glob
import os
import pandas as pd
import configparser

from bkanalysis.transforms.account_transforms import barclays_transform as barc, clone_transform, citi_transform, \
    lloyds_mortgage_transform as lloyds_mort, revolut_transform as rev_transform, \
    lloyds_current_transform as lloyds_curr, nutmeg_isa_transform as nut_transform, ubs_pension_transform, \
    static_data as sd, vault_transform


class Loader:

    def __init__(self, config=None):
        if config is None:
            self.config = configparser.ConfigParser()
            self.config.read('config/config.ini')
        else:
            self.config = config

    def load(self, file):
        df = self.load_internal(file)
        df['SourceFile'] = os.path.basename(file)
        return df

    def load_internal(self, file):
        # print(f'Loading {file}')
        if barc.can_handle(file, self.config['Barclays']):
            return barc.load(file, self.config['Barclays'])
        elif lloyds_curr.can_handle(file, self.config['LloydsCurrent']):
            return lloyds_curr.load(file, self.config['LloydsCurrent'])
        elif lloyds_mort.can_handle(file, self.config['LloydsMortgage']):
            return lloyds_mort.load(file, self.config['LloydsMortgage'])
        elif nut_transform.can_handle(file, self.config['Nutmeg']):
            return nut_transform.load(file, self.config['Nutmeg'])
        elif rev_transform.can_handle(file, self.config['Revolut']):
            return rev_transform.load(file, self.config['Revolut'])
        elif citi_transform.can_handle(file, self.config['Citi']):
            return citi_transform.load(file, self.config['Citi'])
        elif clone_transform.can_handle(file):
            return clone_transform.load(file)
        elif ubs_pension_transform.can_handle(file, self.config['UbsPension']):
            return ubs_pension_transform.load(file, self.config['UbsPension'])
        elif vault_transform.can_handle(file, self.config['Vault']):
            return vault_transform.load(file, self.config['Vault'])

        raise ValueError(f'file {file} could not be processed by any of the loaders.')

    @staticmethod
    def get_files(folder_lake, root=None):
        print(f'Loading files from {os.path.abspath(folder_lake)}.')
        if root is None:
            files = glob.glob(os.path.join(folder_lake, '*.csv'))
        else:
            files = glob.glob(os.path.join(root, folder_lake, '*.csv'))

        print(f"Loading {len(files)} CSV file(s).")
        return files

    def load_all(self):
        if 'folder_root' in self.config['IO']:
            files = self.get_files(self.config['IO']['folder_lake'], self.config['IO']['folder_root'])
        else:
            files = self.get_files(self.config['IO']['folder_lake'])
        if len(files) == 0:
            return

        df_list = [self.load(f) for f in files]
        for df_temp in df_list:
            df_temp['count'] = df_temp.groupby(sd.target_columns).cumcount()
        df = pd.concat(df_list)
        df = df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False)
        return df.reset_index(drop=True)

    def load_save(self):
        df = self.load_all()
        df.Date = df.Date.apply(lambda x: x.strftime("%d-%b-%Y"))
        df.to_csv(self.config['IO']['path_aggregated'], index=False)
