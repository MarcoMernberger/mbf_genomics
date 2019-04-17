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
    """ Turn dicts into frozendict,
        lists into tuples, and sets
        into frozensets, recursively - usefull
        to get a hash value..
    """

    try:
        hash(obj)
        return obj
    except TypeError:
        pass

    if isinstance(obj, dict):
        frz = {k: freeze(obj[k]) for k in obj}
        return frozendict(frz)
    elif isinstance(obj, (list, tuple)):
        return tuple([freeze(x) for x in obj])

    elif isinstance(obj, set):
        return frozenset(obj)
    else:
        msg = "Unsupported type: %r" % type(obj).__name__
        raise TypeError(msg)


def parse_a_or_c_to_column(k):
    """Parse an annotator + column spec to the column name.
        Input may be
            a str - column name
            an annotaator -> anno.columns[0]
            an (annotator, str) tuple -> str
            an (annotator, int(i)) tuple -> annotator.columns[i]
    """
    from mbf_genomics.annotator import Annotator
    if isinstance(k, str):
        return k
    elif isinstance(k, Annotator):
        return k.columns[0]
    elif isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], Annotator):
        if isinstance(k[1], int):
            return k[0].columns[k[1]]
        else:
            if not k[1] in k[0].columns:
                raise KeyError(
                    "Invalid column name, %s -annotator had %s", (k[1], k[0].columns)
                )
            return k[1]
    else:
        raise ValueError("parse_a_or_c_to_column could not parse %s" % (k,))


def parse_a_or_c_to_anno(k):
    """Parse an annotator + column spec to the annotator (or None)
        Input may be
            a str - column name
            an annotaator -> anno.columns[0]
            an (annotator, str) tuple -> str
            an (annotator, int(i)) tuple -> annotator.columns[i]
    """
    from mbf_genomics.annotator import Annotator
    if isinstance(k, str):
        return None
    elif isinstance(k, Annotator):
        return k
    elif isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], Annotator):
        if isinstance(k[1], int):
            k[0].columns[k[1]]  # check for exception...
            return k[0]
        else:
            if not k[1] in k[0].columns:
                raise KeyError(
                    "Invalid column name, %s -annotator had %s", (k[1], k[0].columns)
                )
            return k[0]
    else:
        raise ValueError("parse_a_or_c_to_column could not parse %s" % (k,))


def parse_a_or_c_to_plot_name(k):
    """Parse an annotator + column spec to a plot name
    See parse_a_or_c_to_column

    """
    from mbf_genomics.annotator import Annotator
    if isinstance(k, str):
        return k
    elif isinstance(k, Annotator):
        return getattr(k, "plot_name", k.columns[0])
    elif isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], Annotator):
        if isinstance(k[1], int):
            return getattr(k[0], "plot_name", k[0].columns[k[1]])
        else:
            if not k[1] in k[0].columns:
                raise KeyError(
                    "Invalid column name, %s -annotator had %s", (k[1], k[0].columns)
                )
            return getattr(k[0], "plot_name", k[1])
    else:
        raise ValueError("parse_a_or_c_to_column could not parse %s" % (k,))
