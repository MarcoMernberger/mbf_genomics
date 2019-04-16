import pandas as pd
import pytest
from mbf_genomics.util import read_pandas, freeze
from pandas.testing import assert_frame_equal


def test_read_pandas_csv_in_xls(new_pipegraph):
    df = pd.DataFrame({"x": ["a", "b", "c"], "y": [1, 2.5, 3]})
    df.to_excel("shu.xls", index=False)
    assert_frame_equal(df, read_pandas("shu.xls"))
    df.to_csv("shu.xls", sep="\t", index=False)
    assert_frame_equal(df, read_pandas("shu.xls"))
    df.to_csv("shu.tsv", sep="\t", index=False)
    assert_frame_equal(df, read_pandas("shu.tsv"))
    df.to_csv("shu.csv", index=False)
    assert_frame_equal(df, read_pandas("shu.csv"))
    df.to_csv("shu.something", index=False)
    with pytest.raises(ValueError):
        read_pandas("shu.something")

def test_freeze():
    a = {'a': [1,2,3], 'b': {'c': set([2,3,5])}}
    with pytest.raises(TypeError):
        hash(a)
    assert hash(freeze(a))
    assert freeze(a) == freeze(freeze(a))
    class Nohash:
        def __hash__(self):
            raise NotImplemented
    with pytest.raises(TypeError):
        freeze(Nohash())
