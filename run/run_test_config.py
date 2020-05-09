import ast
import configparser


config = configparser.ConfigParser()
config.read('../config/config.ini')
print(config['Mapping']['expected_columns'])
expected_columns = ast.literal_eval(config['Mapping']['expected_columns'])
print(type(expected_columns))
for v in expected_columns:
    print(v)
print(expected_columns)
