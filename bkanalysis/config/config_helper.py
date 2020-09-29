import ast


def parse_list(item, strip = True):
    if strip:
        return [n.strip() for n in ast.literal_eval(item)]
    return ast.literal_eval(item)