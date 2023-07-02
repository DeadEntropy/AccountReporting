# coding=utf8
import glob
import os
import ast
import pandas as pd
import configparser

from bkanalysis.transforms.account_transforms import barclays_transform as barc, clone_transform, citi_transform, \
    lloyds_mortgage_transform as lloyds_mort, revolut_transform as rev_transform, \
    lloyds_current_transform as lloyds_curr, nutmeg_isa_transform as nut_transform, ubs_pension_transform, \
    static_data as sd, vault_transform, coinbase_transform, coinbase_pro_transform, bnp_stock_transform, \
    bnp_cash_transform, chase_transform, revolut_transform_2 as rev_transform_2, fidelity_transform, discover_transform, \
    marcus_transform, discover_credit_transform, ubs_us_pension_transform, capital_one_transform, \
    first_republic_transform, first_republic_mtg_transform, nutmeg_transaction_transform, ubs_wm_transform
from bkanalysis.config import config_helper as ch
from bkanalysis.market.market import Market
from bkanalysis.market import market_loader as ml


class Loader:

    def __init__(self, config=None, include_market=True, ref_currency='GBP'):
        if config is None:
            self.config = configparser.ConfigParser()
            if len(self.config.read(ch.source)) != 1:
                raise OSError(f'no config found in {ch.source}')
        else:
            self.config = config

        self.market = None
        if 'Market' in self.config and include_market:
            if 'instr_to_preload' in self.config['Market']:
                self.market = Market(ml.MarketLoader().load(\
                    ast.literal_eval(self.config['Market']['instr_to_preload']), ref_currency, '10y'))

    def load(self, file):
        df = self.load_internal(file)
        df['SourceFile'] = os.path.basename(file)
        return df

    def load_internal(self, file, ref_currency='GBP'):
        # print(f'Loading {file}')
        if 'Barclays' in self.config and barc.can_handle(file, self.config['Barclays']):
            return barc.load(file, self.config['Barclays'])
        elif 'LloydsCurrent' in self.config and lloyds_curr.can_handle(file, self.config['LloydsCurrent']):
            return lloyds_curr.load(file, self.config['LloydsCurrent'])
        elif 'LloydsMortgage' in self.config and lloyds_mort.can_handle(file, self.config['LloydsMortgage']):
            return lloyds_mort.load(file, self.config['LloydsMortgage'])
        elif 'Nutmeg' in self.config and nut_transform.can_handle(file, self.config['Nutmeg']):
            return nut_transform.load(file, self.config['Nutmeg'], self.market, ref_currency)
        elif 'Revolut' in self.config and rev_transform.can_handle(file, self.config['Revolut'], ';'):
            return rev_transform.load(file, self.config['Revolut'], ';')
        elif 'Revolut' in self.config and rev_transform.can_handle(file, self.config['Revolut'], ','):
            return rev_transform.load(file, self.config['Revolut'], ',')
        elif 'Revolut2' in self.config and rev_transform_2.can_handle(file, self.config['Revolut2'], ','):
            return rev_transform_2.load(file, self.config['Revolut2'], ',')
        elif 'Citi' in self.config and citi_transform.can_handle(file, self.config['Citi']):
            return citi_transform.load(file, self.config['Citi'])
        elif clone_transform.can_handle(file):
            return clone_transform.load(file)
        elif 'UbsPension' in self.config and ubs_pension_transform.can_handle(file, self.config['UbsPension']):
            return ubs_pension_transform.load(file, self.config['UbsPension'], self.market, ref_currency)
        elif 'UbsUsPension' in self.config and ubs_us_pension_transform.can_handle(file, self.config['UbsUsPension']):
            return ubs_us_pension_transform.load(file, self.config['UbsUsPension'], None, ref_currency)
        elif 'Vault' in self.config and vault_transform.can_handle(file, self.config['Vault']):
            return vault_transform.load(file, self.config['Vault'])
        elif 'CoinbasePro' in self.config and coinbase_pro_transform.can_handle(file, self.config['CoinbasePro']):
            return coinbase_pro_transform.load(file, self.config['CoinbasePro'])
        elif 'Coinbase' in self.config and coinbase_transform.can_handle(file, self.config['Coinbase']):
            return coinbase_transform.load(file, self.config['Coinbase'])
        elif 'BnpStocks' in self.config and bnp_stock_transform.can_handle(file, self.config['BnpStocks']):
            return bnp_stock_transform.load(file, self.config['BnpStocks'])
        elif 'BnpCash' in self.config and bnp_cash_transform.can_handle(file, self.config['BnpCash']):
            return bnp_cash_transform.load(file, self.config['BnpCash'])
        elif 'Chase' in self.config and chase_transform.can_handle(file, self.config['Chase']):
            return chase_transform.load(file, self.config['Chase'])
        elif 'Fidelity' in self.config and fidelity_transform.can_handle(file, self.config['Fidelity']):
            return fidelity_transform.load(file, self.config['Fidelity'])
        elif 'Discover' in self.config and discover_transform.can_handle(file, self.config['Discover']):
            return discover_transform.load(file, self.config['Discover'])
        elif 'Marcus' in self.config and marcus_transform.can_handle(file, self.config['Marcus']):
            return marcus_transform.load(file, self.config['Marcus'])
        elif 'DiscoverCredit' in self.config and discover_credit_transform.can_handle(file, self.config['DiscoverCredit']):
            return discover_credit_transform.load(file, self.config['DiscoverCredit'])
        elif 'CapitalOne' in self.config and capital_one_transform.can_handle(file, self.config['CapitalOne']):
            return capital_one_transform.load(file, self.config['CapitalOne'])
        elif 'FirstRepublic' in self.config and first_republic_transform.can_handle(file, self.config['FirstRepublic']):
            return first_republic_transform.load(file, self.config['FirstRepublic'])
        elif 'FirstRepublicMortgage' in self.config and first_republic_mtg_transform.can_handle(file, self.config['FirstRepublicMortgage']):
            return first_republic_mtg_transform.load(file, self.config['FirstRepublicMortgage'])
        elif 'NutmegInvestment' in self.config and nutmeg_transaction_transform.can_handle(file, self.config['NutmegInvestment']):
            return nutmeg_transaction_transform.load(file, self.config['NutmegInvestment'])
        elif 'UbsWealthManagement' in self.config and ubs_wm_transform.can_handle(file, self.config['UbsWealthManagement']):
            return ubs_wm_transform.load(file, self.config['UbsWealthManagement'])

        raise ValueError(f'file {file} could not be processed by any of the loaders.')

    @staticmethod
    def get_files(folder_lake, root=None, include_xls=True):
        print(f'Loading files from {os.path.abspath(folder_lake)}.')
        if root is None:
            csv_files = glob.glob(os.path.join(folder_lake, '*.csv'))
        else:
            csv_files = glob.glob(os.path.join(root, folder_lake, '*.csv'))

        if include_xls:
            if root is None:
                xls_files = glob.glob(os.path.join(folder_lake, '*.xls'))
            else:
                xls_files = glob.glob(os.path.join(root, folder_lake, '*.xls'))
        else:
            xls_files = []

        print(f"Loading {len(csv_files)} CSV file(s) and {len(xls_files)} XLS file(s).")
        return csv_files + xls_files

    def load_all(self, include_xls=True):
        if 'folder_root' in self.config['IO']:
            files = self.get_files(self.config['IO']['folder_lake'], self.config['IO']['folder_root'], include_xls)
        else:
            files = self.get_files(self.config['IO']['folder_lake'], include_xls=include_xls)
        if len(files) == 0:
            return

        df_list = [self.load(f) for f in files]
        for df_temp in df_list:
            df_temp['count'] = df_temp.groupby(sd.target_columns).cumcount()
        df = pd.concat(df_list)
        df = df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False)
        return df.reset_index(drop=True)

    def save(self, df):
        df_copy = df.copy(True)
        df_copy.Date = df_copy.Date.apply(lambda x: x.strftime("%d-%b-%Y"))
        df_copy.to_csv(self.config['IO']['path_aggregated'], index=False)

    def load_save(self):
        df = self.load_all()
        self.save(df)
