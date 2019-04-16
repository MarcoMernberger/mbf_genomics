import pandas as pd
from frozendict import frozendict


def read_pandas(filename):
    filename = str(filename)
    if filename.endswith(".xls") or filename.endswith(".xlsx"):
        from xlrd import XLRDError

        try:
            filein = pd.read_excel(filename)
        except XLRDError:
            filein = pd.read_csv(filename, sep="\t")
        return filein

    elif filename.endswith(".tsv"):
        return pd.read_csv(filename, sep="\t")
    elif filename.endswith(".csv"):
        return pd.read_csv(filename)
    else:
        raise ValueError("Unknown filetype: %s" % filename)

def freeze(obj):
    ''' Turn dicts into frozendict,
        lists into tuples, and sets
        into frozensets, recursively - usefull
        to get a hash value..
    '''

    try:
        hash(obj)
        return obj
    except TypeError:
        pass

    if isinstance(obj, dict):
        frz = {k: freeze(obj[k]) for k in obj}
        return frozendict(frz)
    elif isinstance(obj, list):
        return tuple(obj)
    elif isinstance(obj, set):
        return frozenset(obj)
    else:
        msg = 'Unsupported type: %r' % type(obj).__name__
        raise TypeError(msg)
