# Created by lan at 2021/11/9

date_patterns = ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%Y/%m/%d', '%y/%m/%d', '%d/%m/%y']
import datetime


def is_numeric(value):
    """
    check whether the given value is a numeric value. Numbers with decimal point or thousand separator can be properly determined.
    :param value: the given value to be verified
    :return: True if the given value is numeric
    """
    value_str = str(value).replace(',', '')
    try:
        float(value_str)
        return True
    except:
        return False

def to_numeric(value):
    if not is_numeric(value):
        raise Exception('Value is not numeric.')
    value_str = str(value).replace(',', '')
    try:
        return float(value_str)
    except:
        raise Exception('Conversion error.')


def isDate(value):
    _isDate = False
    for date_pattern in date_patterns:
        try:
           datetime.datetime.strptime(value, date_pattern)
           _isDate = True
           break
        except:
            pass
    return _isDate


def detect_datatype(value):
    """
    :param value: the value whose data type is to be checked.
    :return: 0 - 'date', 1 - 'float', 2 - 'int', 3 - 'empty', 4 - 'string', 5 - 'null'
    """
    if isDate(value):
        return 0
    try:
        float(value)
        return 1 if '.' in value else 2
    except:
        if len(value) == 0:
            return 3
        else:
            return 4