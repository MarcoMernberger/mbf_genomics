import pytest
import pandas as pd
import pypipegraph as ppg
import os
from mbf_genomes.example_genomes import get_Candidatus_carsonella_ruddii_pv

from mbf_genomics import regions
dummy_genome = get_Candidatus_carsonella_ruddii_pv()

@pytest.mark.usefixtures("new_pipegraph")
class TestGenomicRegionRandomizerTests:
    def test_random_same_number(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1"],
                    "start": [10, 100, 1000],
                    "stop": [12, 110, 1110],
                }
            )

        a = regions.GenomicRegions("sharum", sample_data, [], dummy_genome)
        b = a.randomize(regions.random.Somewhere())
        b.load()
        ppg.run_pipegraph()
        assert len(a) == len(b)

    # no longer true since we have a summit annotator anyhow.
    def __test_randomized_has_no_cache_dir(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1"],
                    "start": [10, 100, 1000],
                    "stop": [12, 110, 1110],
                }
            )

        a = regions.GenomicRegions("sharum", sample_data, [], dummy_genome)
        assert not os.path.exists("cache/GenomicRegions/sharumrandom_genome_position0")
        b = a.randomize(regions.random.Somewhere())
        assert not os.path.exists(b.cache_dir)
        b.load()  # this forces a chache dir right now
        ppg.run_pipegraph()
        assert not os.path.exists(b.cache_dir)

    def test_random_can_be_called_twice(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1"],
                    "start": [10, 100, 1000],
                    "stop": [12, 110, 1110],
                }
            )

        a = regions.GenomicRegions("sharum", sample_data, [], dummy_genome)
        b = a.randomize(regions.random.Somewhere())
        c = a.randomize(regions.random.Somewhere())
        assert b.name != c.name

    def test_random_same_tss_distance(self):
        genome = DummyGenome(
            pd.DataFrame(
                [
                    {
                        "stable_id": "fake1",
                        "chr": "3",
                        "strand": 1,
                        "tss": 5000,
                        "tes": 5500,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake2",
                        "chr": "3",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake3",
                        "chr": "4",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                ]
            )
        )

        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["3", "4", "3"],
                    "start": [10, 100, 1000],
                    "stop": [12, 110, 1110],
                    "summit": [1, 2, 3],
                }
            )

        numpy.random.seed(5)
        a = regions.GenomicRegions("sharum", sample_data, [], genome)
        randomizer = regions.random.SameTSSDistance()
        b = a.randomize(randomizer)
        c = a.randomize(randomizer)
        b.add_annotator(regions.annotators.NextGenes(still_ok=True))
        c.add_annotator(regions.annotators.NextGenes(still_ok=True))
        b.annotate()
        c.annotate()
        ppg.run_pipegraph()
        for x in [b, c]:
            assert len(a.df) == len(x.df)
            for dummy_idx, row in x.df.iterrows():
                # I abuse the summit as an id to do this here..
                should_distance = a.df[a.df["summit"] == row["summit"]][
                    "Primary gene distance"
                ].iloc[0]
                summit_pos = row["start"] + row["summit"]
                gs = genome.get_genes_with_closest_start(row["chr"], summit_pos, 5)
                found = False
                for name, distance, strand, desc in gs:
                    if distance == should_distance:
                        found = True
                assert found

    def test_random_same_tss_distance_works_without_summit(self):
        genome = DummyGenome(
            pd.DataFrame(
                [
                    {
                        "stable_id": "fake1",
                        "chr": "3",
                        "strand": 1,
                        "tss": 5000,
                        "tes": 5500,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake1a",
                        "chr": "1",
                        "strand": 1,
                        "tss": 6000,
                        "tes": 6600,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake1b",
                        "chr": "1",
                        "strand": 1,
                        "tss": 16000,
                        "tes": 16600,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake2",
                        "chr": "3",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake3",
                        "chr": "4",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                ]
            )
        )

        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["3", "4", "3"],
                    "start": [10, 100, 1000],
                    "stop": [12, 110, 1110],
                }
            )

        numpy.random.seed(5)
        a = regions.GenomicRegions("sharum", sample_data, [], genome)
        randomizer = regions.random.SameTSSDistance()
        b = a.randomize(randomizer)
        c = a.randomize(randomizer)
        b.add_annotator(regions.annotators.NextGenes(still_ok=True))
        c.add_annotator(regions.annotators.NextGenes(still_ok=True))
        b.annotate()
        c.annotate()
        g = genes.Genes(genome)
        g.load()
        ppg.run_pipegraph()
        for x in [b]:
            assert len(a.df) == len(x.df)
            # we just check whether ther's a gene at that distance...
            for dummy_idx, row in x.df.iterrows():
                tss_pos = (
                    row["start"]
                    + (row["stop"] - row["start"]) / 2
                    - row["Primary gene distance"]
                )
                # for a gene on the negative strand, we'll have to add the gene distance
                # but I don't see the way to calculate this right now
                # but on the other hand, being negative when you're in front of
                # a gene is correct
                rev_tss_pos = (
                    row["start"]
                    + (row["stop"] - row["start"]) / 2
                    + row["Primary gene distance"]
                )
                if not (
                    (tss_pos in g.df["tss"].values)
                    or (rev_tss_pos in g.df["tss"].values)
                ):
                    raise ValueError("Gene distance was not kept")

    def test_random_same_tss_distance_raises_on_unexpectedly_large_distances(self):
        genome = DummyGenome(
            pd.DataFrame(
                [
                    {
                        "stable_id": "fake1",
                        "chr": "3",
                        "strand": 1,
                        "tss": 5000,
                        "tes": 5500,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake2",
                        "chr": "3",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake3",
                        "chr": "4",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                ]
            )
        )

        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["3", "4", "3"],
                    "start": [10, 100, 1000],
                    "stop": [12, 110, 1110],
                    "summit": [1, 2, 3],
                    "Primary gene distance": [500, 360, int(3e9)],
                }
            )

        numpy.random.seed(5)
        a = regions.GenomicRegions("sharum", sample_data, [], genome)
        randomizer = regions.random.SameTSSDistance()
        randomizer.get_annotators = lambda self: []
        b = a.randomize(randomizer)
        b.load()
        with pytest.raises(ppg.RuntimeError) as e:
            ppg.run_pipegraph()
            assert len(e.value.exceptions) == 1
            assert str(e.value.exceptions).find("too large") != -1

