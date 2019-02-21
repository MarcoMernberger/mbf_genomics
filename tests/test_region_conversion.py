import pytest
import pandas as pd
import pypipegraph as ppg
from mbf_genomes.example_genomes import get_Candidatus_carsonella_ruddii_pv

from mbf_genomics import regions
from mbf_genomics.annotator import Constant

dummy_genome = get_Candidatus_carsonella_ruddii_pv()

@pytest.mark.usefixtures("new_pipegraph")
class TestGenomicRegionConvertTests:
    def test_random_same_number(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1"],
                    "start": [10, 100, 1000],
                    "stop": [12, 110, 1110],
                    "column_that_will_disappear": ["A", "b", "c"],
                }
            )

        def convert(df):
            res = df[["chr", "start", "stop"]]
            res["start"] += 1
            return res

        job = ppg.ParameterInvariant("shuParam", ("hello"))
        a = regions.GenomicRegions("sharum", sample_data, [], dummy_genome)
        a.add_annotator(Constant('Constant', 5))
        a.annotate()
        b = a.convert("a+1", convert, dependencies=[job])
        b.load()
        assert job in b.load().prerequisites
        ppg.run_pipegraph()
        assert len(a) == len(b)
        assert (a.df["start"] == b.df["start"] - 1).all()
        assert "column_that_will_disappear" in a.df.columns
        assert not ("column_that_will_disappear" in b.df.columns)

    def test_raises_on_conversion_function_not_returning_df(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1"],
                    "start": [10, 100, 1000],
                    "stop": [12, 110, 1110],
                }
            )

        def convert(df):
            return None

        a = regions.GenomicRegions("sharum", sample_data, [], dummy_genome)
        b = a.convert("a+1", convert)
        with pytest.raises(ppg.RuntimeError) as e:
            ppg.run_pipegraph()
            assert len(e.value.exceptions) == 1

    def test_raises_on_non_genome(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1"],
                    "start": [10, 100, 1000],
                    "stop": [12, 110, 1110],
                }
            )

        def convert(df):
            res = df[:]
            res["start"] += 1
            return res

        a = regions.GenomicRegions("sharum", sample_data, [], dummy_genome)
        b = a.convert("a+1", convert, dummy_genome)

        def inner():
            c = a.convert("a+1b", convert, "hello")

        with pytest.raises(ValueError):
            inner()

    def test_grow(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "1", "2"],
                    "start": [10, 100, 1000],
                    "stop": [12, 110, 1110],
                }
            )

        a = regions.GenomicRegions("sharum", sample_data, [], dummy_genome)
        b = a.convert("agrown", regions.convert.grow(12))
        b.load()
        ppg.run_pipegraph()
        assert (b.df["start"] == [0, 100 - 12, 1000 - 12]).all()
        assert (b.df["stop"] == [24, 110 + 12, 1110 + 12]).all()

    def test_grow2(self):
        def sample_data():
            return pd.DataFrame(
                [
                    {"chr": "1", "start": 7774885, "stop": 7791673},
                    {"chr": "1", "start": 8286026, "stop": 8298500},
                    {"chr": "1", "start": 8323232, "stop": 8342008},
                ]
            )

        a = regions.GenomicRegions("sharum", sample_data, [], dummy_genome)
        b = a.convert("agrown", regions.convert.grow(50000), on_overlap="merge")
        b.load()
        ppg.run_pipegraph()
        assert (b.df["start"] == [7724885, 8236026]).all()
        assert (b.df["stop"] == [7841673, 8392008]).all()

    def test_promotorize(self):
        genome = DummyGenome(
            pd.DataFrame(
                [
                    {
                        "stable_id": "fake1",
                        "chr": "1",
                        "strand": 1,
                        "tss": 5000,
                        "tes": 5500,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake2",
                        "chr": "2",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake3",
                        "chr": "3",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                ]
            )
        )
        g = genes.Genes(genome)
        b = g.convert("b", regions.convert.promotorize(444))
        b.load()
        ppg.run_pipegraph()
        assert len(g) > 0
        assert len(g) == len(b)
        for ii in range(0, len(g)):
            if g.df.get_value(ii, "strand") == 1:
                assert b.df.get_value(ii, "start") == g.df.get_value(ii, "tss") - 444
                assert b.df.get_value(ii, "stop") == g.df.get_value(ii, "tss")
            else:
                assert b.df.get_value(ii, "start") == g.df.get_value(ii, "tss")
                assert b.df.get_value(ii, "stop") == g.df.get_value(ii, "tss") + 444

    def test_merge_connected(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "1", "1", "2", "2"],
                    "start": [10, 13, 102, 5, 6000],
                    "stop": [12, 100, 1000, 5000, 6010],
                }
            )

        a = regions.GenomicRegions("sharum", sample_data, [], dummy_genome)
        b = a.convert("agrown", regions.convert.merge_connected())
        b.load()
        ppg.run_pipegraph()
        assert (b.df["start"] == [10, 102, 5, 6000]).all()
        assert (b.df["stop"] == [100, 1000, 5000, 6010]).all()

    def test_merge_connected_2(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "1", "2"],
                    "start": [27897200, 27898600, 0],
                    "stop": [27897300, 27898700, 100],
                }
            )

        a = regions.GenomicRegions("sharum", sample_data, [], dummy_genome)
        b = a.convert("agrown", regions.convert.merge_connected())
        b.load()
        ppg.run_pipegraph()
        assert (b.df["start"] == [27897200, 27898600, 0]).all()
        assert (b.df["stop"] == [27897300, 27898700, 100]).all()

    def test_liftover(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "1", "2"],
                    "start": [27897200, 27898600, 100000],
                    "stop": [27897300, 27898700, 100100],
                }
            )

        a = regions.GenomicRegions("sharum", sample_data, [], dummy_genome)
        b = a.convert("hg38", regions.convert.hg19_to_hg38())
        b.load()
        ppg.run_pipegraph()
        # made these with the ucsc web liftover utility
        # http://genome.ucsc.edu/cgi-bin/hgLiftOver
        assert (b.df["start"] == [27570689, 27572089, 100000]).all()
        assert (b.df["stop"] == [27570789, 27572189, 100100]).all()

    def test_windows(self):
        def sample_data():
            return pd.DataFrame(
                {"chr": ["1", "2"], "start": [0, 0], "stop": [200, 99 * 3]}
            )

        a = regions.GenomicRegions("sharum", sample_data, [], dummy_genome)
        b = a.convert("ndiwo", regions.convert.windows(99, False))
        c = a.convert("ndiwo2", regions.convert.windows(99, True))
        b.load()
        c.load()
        ppg.run_pipegraph()
        assert (b.df["start"] == [0, 99, 99 * 2, 0, 99, 99 * 2]).all()
        assert (b.df["stop"] == [99, 99 * 2, 200, 99, 99 * 2, 99 * 3]).all()
        assert (c.df["start"] == [0, 99, 0, 99, 99 * 2]).all()
        assert (c.df["stop"] == [99, 99 * 2, 99, 99 * 2, 99 * 3]).all()


