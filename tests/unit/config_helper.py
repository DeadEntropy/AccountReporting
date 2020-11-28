import configparser
import os
from pathlib import Path

__config_path = os.path.join(Path(__file__).parent, "config.ini")


def get_config_path():
    if os.path.exists(__config_path):
        return __config_path
    raise Exception(f'No Valid Config Path found. tried: \r\n{os.path.abspath(__config_path)}')


def get_config():
    config = configparser.ConfigParser()
    if len(config.read(os.path.abspath(get_config_path()))) != 1:
        raise Exception(f'did not successfully load the config. please make sure the config path is correct.')

    config['IO']['folder_root'] = os.path.join(Path(__file__).parent.parent, "test_data")
    config['Mapping']['folder_root'] = os.path.join(Path(__file__).parent.parent, "test_data")
    return config
