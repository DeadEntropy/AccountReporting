import json
import os


class CacheDict:
    def __init__(self, path):
        self.__path = path

        if os.path.exists(self.__path):
            try:
                with open(self.__path, 'r') as json_file:
                    self.__cache = json.load(json_file)
            except TypeError:
                raise TypeError(f"Failed to deserialise jSon file '{self.__path}' {json_file}")
        else:
            self.__cache = {}

    def __update(self, key, value):
        self.__cache[key] = value

        with open(self.__path, 'w') as outfile:
            json.dump(self.__cache, outfile)

    def get(self, key, func):
        if key in self.__cache:
            return self.__cache[key]

        value = func(key)
        self.__update(key, value)

        return value
