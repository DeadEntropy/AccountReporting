import datetime
import difflib


def get_adjusted_month(dt):
    if dt.day > 20:
        if dt.month < 12:
            return dt.month + 1
        return 1
    return dt.month


def get_adjusted_year(dt):
    if dt.day > 20 and dt.month == 12:
        return dt.year + 1
    return dt.year


def get_fiscal_year(dt):
    if dt < datetime.datetime(dt.year, 4, 5):
        return dt.year - 1
    return dt.year


def get_year_to_date(dt):
    return int((datetime.date.today() - dt.date()).days / 365.25)


def get_missing_map(memo, mapping):
    if memo == '' or memo == '000001':
        return ''
    suggested_keys = difflib.get_close_matches(memo, list(mapping.keys()))
    suggestions = list(set([mapping[x] for x in suggested_keys]))
    if len(suggestions) > 0:
        value = input(f'Please enter the mapping for "{memo}". Suggestions:\n {suggestions}\n '
                      f'Enter the Index of the suggestion if relevant:').upper()
        try:
            index = int(value)
            value = suggestions[index-1]
        except ValueError:
            pass
            # Check if something very similar already exists
            suggested_values = difflib.get_close_matches(value.upper(), list(mapping.values()), 1)
            if len(suggested_values)>0 and suggested_values.count(value) == 0:
                replace = input(f'this is similar to already existing value: "{suggested_values[0]}", '
                                f'use that instead? (y/n)')
                if replace.upper() == 'Y':
                    value = suggested_values[0]
    else:
        value = input(f'Please enter the mapping for "{memo}:').upper()
        # Check if something very similar already exists
        suggested_values = difflib.get_close_matches(value, list(mapping.values()), 1)
        if len(suggested_values)>0 and suggested_values.count(value) == 0:
            replace = input(f'this is similar to already existing value: "{suggested_values[0]}", '
                            f'use that instead? (y/n)')
            if replace.upper() == 'Y':
                value = suggested_values[0]

    mapping[memo] = value
    return value.strip().upper()


def get_missing_type(memo, mapping_a, mapping_b):
    suggested_keys = difflib.get_close_matches(memo, list(mapping_a.keys()))
    suggestions = [f"{x}: {mapping_a[x]}, {mapping_b[x]}" for x in suggested_keys]
    t = input(f"Please Enter a type and subtype (space-separated) to associate to {memo}\nSuggestions: {suggestions}:")

    try:
        index = int(t)
        mapping_a[memo] = mapping_a[suggested_keys[index - 1]]
        mapping_b[memo] = mapping_b[suggested_keys[index - 1]]
        return mapping_a[suggested_keys[index - 1]], mapping_b[suggested_keys[index - 1]]
    except ValueError:
        pass

    if ' ' not in t:
        mapping_a[memo] = t
        mapping_b[memo] = ''
        return t, ''

    mapping_a[memo] = t.split(' ')[0]
    mapping_b[memo] = t.split(' ')[1]

    return t.split(' ')[0].upper(), t.split(' ')[1].upper()

