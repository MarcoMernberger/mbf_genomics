from abc import ABC
import pandas as pd
import pypipegraph as ppg
from .util import freeze

annotator_singletons = {}


class Annotator(ABC):
    def __new__(cls, *args, **kwargs):
        cn = cls.__name__
        if ppg.util.global_pipegraph:
            if not hasattr(ppg.util.global_pipegraph, "_annotator_singleton_dict"):
                ppg.util.global_pipegraph._annotator_singleton_dict = {}
            singleton_dict = ppg.util.global_pipegraph._annotator_singleton_dict
        else:
            singleton_dict = annotator_singletons
        if not cn in singleton_dict:
            singleton_dict[cn] = {}
        key = {}
        for ii in range(0, len(args)):
            key["arg_%i" % ii] = args[ii]
        key.update(kwargs)
        for k, v in key.items():
            key[k] = freeze(v)
        key = tuple(sorted(key.items()))
        if not key in singleton_dict[cn]:
            singleton_dict[cn][key] = object.__new__(cls)
        return singleton_dict[cn][key]

    def __hash__(self):
        return hash(self.get_cache_name())

    def __str__(self):
        return "Annotator %s" % self.columns[0]

    def __repr__(self):
        return "Annotator(%s)" % self.columns[0]

    def get_cache_name(self):
        if hasattr(self, "cache_name"):
            return self.cache_name
        else:
            return self.columns[0]

    def calc(self, df):
        raise NotImplementedError()  # pragma: no cover

    def deps(self, ddf):
        """Return ppg.jobs"""
        return []

    def dep_annos(self):
        """Return other annotators"""
        return []


class Constant(Annotator):
    def __init__(self, column_name, value):
        self.columns = [column_name]
        self.value = value

    def calc(self, df):
        return pd.DataFrame({self.columns[0]: self.value}, index=df.index)
