import ast
import configparser


config = configparser.ConfigParser()
config.read('../config/config.ini')
print(config['Process']['expected_columns'])
expected_columns = ast.literal_eval(config['Process']['expected_columns'])
print(type(expected_columns))
for v in expected_columns:
    print(v)
print(expected_columns)
