import pandas as pd


def read_pandas(filename):
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
