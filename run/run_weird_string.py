import configparser
from chardet import detect

config = configparser.ConfigParser()

expected_columns = ['DATE', 'TRANSACTION', 'OUT(£)', 'IN(£)', 'BALANCE(£)']

config['DEFAULT'] = {'expected_columns': expected_columns}

with open('example.ini', 'w') as configfile:
    config.write(configfile)

encoding = lambda x: detect(x)['encoding']

f = open('example.ini','r', encoding='ANSI')
for line in f:
    print(line)


