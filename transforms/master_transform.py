import glob
import os
import pandas as pd
import configparser

from transforms import barclays_transform as barc
from transforms import lloyds_current_transform as lloyds_curr
from transforms import lloyds_mortgage_transform as lloyds_mort
from transforms import nutmeg_isa_transform as nut_transform
from transforms import revolut_transform as rev_transform
from transforms import clone_transform
from transforms import citi_transform
from transforms import ubs_pension_transform
from transforms import static_data as sd


def load(file, config):
    print(f'Loading {file}')
    if barc.can_handle(file, config['Barclays']):
        return barc.load(file, config['Barclays'])
    elif lloyds_curr.can_handle(file, config['LloydsCurrent']):
        return lloyds_curr.load(file, config['LloydsCurrent'])
    elif lloyds_mort.can_handle(file, config['LloydsMortgage']):
        return lloyds_mort.load(file, config['LloydsMortgage'])
    elif nut_transform.can_handle(file, config['Nutmeg']):
        return nut_transform.load(file, config['Nutmeg'])
    elif rev_transform.can_handle(file, config['Revolut']):
        return rev_transform.load(file, config['Revolut'])
    elif citi_transform.can_handle(file, config['Citi']):
        return citi_transform.load(file, config['Citi'])
    elif clone_transform.can_handle(file):
        return clone_transform.load(file)
    elif ubs_pension_transform.can_handle(file, config['UbsPension']):
        return ubs_pension_transform.load(file, config['UbsPension'])
    rev_transform.can_handle(file, config['Revolut'])

    raise ValueError(f'file {file} could not be processed by any of the loaders.')


def load_all(config):
    files = glob.glob(os.path.join(config['IO']['folder_lake'], '*.csv'))
    print(f"found {len(files)} CSV files.")
    if len(files) == 0:
        return

    df_list = [load(f, config) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(sd.target_columns).cumcount()
    df = pd.concat(df_list)
    return df.drop_duplicates().drop(['count'], axis=1)


def load_save():
    config = configparser.ConfigParser()
    config.read('../config/config.ini')

    files = glob.glob(os.path.join(config['IO']['folder_lake'], '*.csv'))
    print(f"found {len(files)} CSV files.")
    if len(files) == 0:
        return

    df_list = [load(f, config) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(sd.target_columns).cumcount()
    df = pd.concat(df_list)
    df = df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False)
    df.Date = df.Date.apply(lambda x: x.strftime("%d-%b-%Y"))
    df.to_csv(config['IO']['path_aggregated'], index=False)
