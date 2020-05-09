import pandas as pd
import configparser


def last_update(input_path):
    df_input = pd.read_csv(input_path)
    dic_last_update = {}
    df_input['Date'] = pd.to_datetime(df_input['Date'])
    for bank_acc in df_input.Account:
        dic_last_update[bank_acc] = df_input[df_input.Account == bank_acc].Date.max()

    return pd.DataFrame.from_dict(dic_last_update, orient='index', columns=['LastUpdate'])


def last_update_save():
    config = configparser.ConfigParser()
    config.read('../config/config.ini')

    last_update(config['IO']['path_aggregated']).to_csv(config['IO']['path_processed'])