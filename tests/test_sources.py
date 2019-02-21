from .test_common import (
    new_pipeline,
    run_pipeline,
    DummyGenome,
    dummyGenome,
    ConstantAnnotator,
)
import unittest
from genomics import sources
from genomics import motifs
import pypipegraph as ppg
import pytest


@pytest.mark.usefixtures("new_pipegraph")
class SourceIsAbstractTests:
    def test(self):
        s = sources.Source()

        def call_iter():
            iter(s)

        with pytest.raises(NotImplementedError):
            call_iter()

        def call_get_dependencies():
            s.get_dependencies()

        with pytest.raises(NotImplementedError):
            call_get_dependencies()


class DummySource(motifs.sources.ManualMotifSource):
    def deps(self):
        return [ppg.JobGeneratingJob(self.name + "_load", lambda: None)]


@pytest.mark.usefixtures("new_pipegraph")
class CombinedSourceTests(unittest.TestCase):
    def test_combined_source(self):
        new_pipeline()
        sourceA = DummySource(
            "A", [motifs.PWMMotif_FromText("aa", "AGTC", dummyGenome)]
        )
        sourceB = DummySource(
            "B", [motifs.PWMMotif_FromText("ba", "TGTC", dummyGenome)]
        )
        cb = sources.CombinedSource([sourceA, sourceB])
        assert cb.name.find(sourceA.name) != -1
        assert cb.name.find(sourceB.name) != -1
        deps = cb.get_dependencies()
        for s in sourceA, sourceB:
            for dep in s.get_dependencies():
                assert dep in deps
        all = list(cb)
        assert len(all) == 2
