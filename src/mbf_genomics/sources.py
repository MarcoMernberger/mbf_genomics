"""A source is a job aware generator for objects of the same kind,
used for example with an 'misnamed' L{OverlapScorer}.

Basically, it has an __iter__() function that yields objects (GenomicRegions,
Motifs, ChIP-chip datasets) that you then can use in further jobs.
This is done to decouple the listing of the objects from their use, which is necessary
for example for de-novo searched motifs.

__iter__() is only available after Source.deps() has been satisfied,
the objects returns still need their load() jobs executed.
"""
import pypipegraph as ppg
from abc import ABC


class Source:

    name = "DefaultSource"  # all sources must have a name!

    def __iter__(self):
        """Iterate over the available GenomicRegions.
        This is called after all jobs in deps() have been satisfied"""
        raise NotImplementedError()
        # return iter(self.list)

    def deps(self):
        """Jobs returned here must be completed
        before Source.__iter__ is accessed"""
        raise NotImplementedError()
        # return [ jobs...  ]

    def load(self):
        return self.deps()


class CombinedSource(Source):
    def __init__(self, list_of_sources, name=None):
        self.sources = list_of_sources
        if name is None:
            self.name = ", ".join(sorted(x.name for x in list_of_sources))
        else:
            self.name = name

    def deps(self):
        deps = []
        for source in self.sources:
            deps.extend(source.deps())
        deps.append(
            ppg.ParameterInvariant("motif_source" + self.name, (1))
        )  # which encodes the list of sources
        return deps

    def __iter__(self):
        for source in self.sources:
            for motif in source:
                yield motif
