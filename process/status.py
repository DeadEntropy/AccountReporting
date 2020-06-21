import pandas as pd
import configparser


class LastUpdate:

    def __init__(self, config=None):
        if config is None:
            self.config = configparser.ConfigParser()
            self.config.read('../config/config.ini')

    @staticmethod
    def get_link(config, account):
        return config['Status'][account.replace(' ', '').replace(':', '')]

    def last_update(self, df_input):
        dic_last_update = {}
        df_input['Date'] = pd.to_datetime(df_input['Date'])
        for bank_acc in df_input.Account.unique():
            dic_last_update[bank_acc] = [df_input[df_input.Account == bank_acc].Date.max(),
                                         self.get_link(self.config, bank_acc)]

        return pd.DataFrame.from_dict(dic_last_update, orient='index', columns=['LastUpdate', 'Link'])

    def last_update_save(self):
        df_input = pd.read_csv(self.config['IO']['path_aggregated'])
        self.last_update(df_input).to_csv(self.config['IO']['path_last_updated'])
