import ast
import os


def parse_list(item, strip=True):
    if strip:
        return [n.strip() for n in ast.literal_eval(item)]
    return ast.literal_eval(item)


def get_path(config, key, root='folder_root'):
    if root in config:
        return os.path.join(root, config[key])
    return config[key]
