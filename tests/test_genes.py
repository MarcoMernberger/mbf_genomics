import pypipegraph as ppg
import numpy as np
import pandas as pd
import pytest
from mbf_fileformats.bed import read_bed
import mbf_genomics.regions as regions
import mbf_genomics.genes as genes
from mbf_genomics.annotator import Constant
from mbf_genomes import HardCodedGenome

from .shared import (
    get_genome,
    get_genome_chr_length,
    force_load,
    run_pipegraph,
    RaisesDirectOrInsidePipegraph,
)


def DummyGenome(df_genes, df_transcripts=None):
    chr_lengths = {"1": 100_000, "2": 200_000, "3": 300_000, "4": 400_000, "5": 500_000}

    df_genes = df_genes.rename(columns={"stable_id": "gene_stable_id"})
    if not "start" in df_genes.columns:
        starts = []
        stops = []
        for idx, row in df_genes.iterrows():
            if row["strand"] == 1:
                starts.append(row["tss"])
                stops.append(row["tes"])
            else:
                starts.append(row["tes"])
                stops.append(row["tss"])
        df_genes = df_genes.assign(start=starts, stop=stops)
    if not "biotype" in df_genes.columns:
        df_genes = df_genes.assign(biotype="protein_coding")
    df_genes = df_genes.sort_values(["chr", "start"])
    df_genes = df_genes.set_index("gene_stable_id")
    if df_transcripts is not None:
        if not "biotype" in df_transcripts.columns:
            df_transcripts = df_transcripts.assign(biotype="protein_coding")
        if "exons" in df_transcripts.columns:
            if len(df_transcripts["exons"].iloc[0]) == 3:
                df_transcripts = df_transcripts.assign(
                    exons=[(x[0], x[1]) for x in df_transcripts["exons"]]
                )
        df_transcripts = df_transcripts.set_index("transcript_stable_id")
    return HardCodedGenome("dummy", chr_lengths, df_genes, df_transcripts, None)

@pytest.mark.usefixtures("new_pipegraph")
class TestGenesLoadingPPGOnly:
    def test_loading_from_genome_is_singletonic(self):
        genome = get_genome()
        genesA = genes.Genes(genome)
        genesB = genes.Genes(genome)
        assert genesA is genesB
        filterA = genesA.filter("fa", lambda df: df.index[:10])
        filterAa = genesA.filter("faa", lambda df: df.index[:10])
        filterB = genesB.filter("fab", lambda df: df.index[:10])
        assert not (filterA is genesA)
        assert not (filterAa is filterA)
        assert not (filterAa is filterB)
        with pytest.raises(ValueError):  # can't have a different loading func
            filterB = genesB.filter("fab", lambda df: df.index[:15])
        force_load(filterA.load)
        ppg.run_pipegraph()
        assert len(filterA.df) == 10

@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestGenesLoading:
    def test_basic_loading_from_genome(self):
        g = genes.Genes(get_genome())
        force_load(g.load())
        run_pipegraph()
        assert len(g.df) == 246
        assert (g.df["gene_stable_id"][:3] == ["CRP_001", "CRP_002", "CRP_003"]).all()
        assert g.df["gene_stable_id"].iloc[-1] == "CRP_182"
        assert g.df["start"].iloc[-1] == 158_649 - 1
        assert g.df["stop"].iloc[-1] == 159_662
        assert g.df["strand"].iloc[-1] == -1

    def test_alternative_loading_raises_on_non_df(self):
        with RaisesDirectOrInsidePipegraph(ValueError):
            g = genes.Genes(get_genome_chr_length(), lambda: None, "myname")
            force_load(g.load())

    def test_alternative_loading_raises_on_missing_column(self):
        df = pd.DataFrame(
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
                    "chr": "1",
                    "strand": -1,
                    "tss": 5400,
                    "tes": 4900,
                    "description": "bla",
                },
                {
                    "stable_id": "fake3",
                    "chr": "2",
                    "strand": -1,
                    "tss": 5400,
                    "tes": 4900,
                    "description": "bla",
                },
            ]
        )

        def inner_tss():
            df2 = df.copy()
            df2 = df2.drop("tss", axis=1)
            g = genes.Genes(get_genome(), lambda: df2, name="sha")
            g.load()
            run_pipegraph()

        def inner_chr():
            df2 = df.copy()
            df2 = df2.drop("chr", axis=1)
            g = genes.Genes(get_genome(), lambda: df2, name="shu")
            g.load()
            run_pipegraph()

        def inner_tes():
            df2 = df.copy()
            df2 = df2.drop("tes", axis=1)
            g = genes.Genes(get_genome(), lambda: df2, name="shi")
            g.load()
            run_pipegraph()

        with pytest.raises(ValueError):
            inner_tss()
        with pytest.raises(ValueError):
            inner_tes()
        with pytest.raises(ValueError):
            inner_chr()

    def test_alternative_loading_raises_on_missing_name(self):
        df = pd.DataFrame(
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
                    "chr": "1",
                    "strand": -1,
                    "tss": 5400,
                    "tes": 4900,
                    "description": "bla",
                },
                {
                    "stable_id": "fake3",
                    "chr": "2",
                    "strand": -1,
                    "tss": 5400,
                    "tes": 4900,
                    "description": "bla",
                },
            ]
        )

        with pytest.raises(ValueError):
            genes.Genes(get_genome(), lambda: df)

    def test_alternative_loading_raises_on_invalid_chromosome(self):
        df = pd.DataFrame(
            [
                {
                    "stable_id": "fake1",
                    "chr": "1b",
                    "strand": 1,
                    "tss": 5000,
                    "tes": 5500,
                    "description": "bla",
                },
                {
                    "stable_id": "fake2",
                    "chr": "1",
                    "strand": -1,
                    "tss": 5400,
                    "tes": 4900,
                    "description": "bla",
                },
                {
                    "stable_id": "fake3",
                    "chr": "2",
                    "strand": -1,
                    "tss": 5400,
                    "tes": 4900,
                    "description": "bla",
                },
            ]
        )

        with pytest.raises(ValueError):
            g = genes.Genes(get_genome(), lambda: df, name="shu")
            force_load(g.load())
            run_pipegraph()

    def test_alternative_loading_raises_on_non_int_tss(self):
        df = pd.DataFrame(
            [
                {
                    "stable_id": "fake1",
                    "chr": "1",
                    "strand": 1,
                    "tss": 5000.5,
                    "tes": 5500,
                    "description": "bla",
                },
                {
                    "stable_id": "fake2",
                    "chr": "1",
                    "strand": -1,
                    "tss": 5400,
                    "tes": 4900,
                    "description": "bla",
                },
                {
                    "stable_id": "fake3",
                    "chr": "2",
                    "strand": -1,
                    "tss": 5400,
                    "tes": 4900,
                    "description": "bla",
                },
            ]
        )

        with pytest.raises(ValueError):
            g = genes.Genes(get_genome(), lambda: df, name="shu")
            force_load(g.load())
            run_pipegraph()

    def test_alternative_loading_raises_on_non_int_tes(self):
        df = pd.DataFrame(
            [
                {
                    "stable_id": "fake1",
                    "chr": "1",
                    "strand": 1,
                    "tss": 5000,
                    "tes": "",
                    "description": "bla",
                },
                {
                    "stable_id": "fake2",
                    "chr": "1",
                    "strand": -1,
                    "tss": 5400,
                    "tes": 4900,
                    "description": "bla",
                },
                {
                    "stable_id": "fake3",
                    "chr": "2",
                    "strand": -1,
                    "tss": 5400,
                    "tes": 4900,
                    "description": "bla",
                },
            ]
        )

        with pytest.raises(ValueError):
            g = genes.Genes(get_genome(), lambda: df, name="shu")
            force_load(g.load())
            run_pipegraph()

    def test_do_load_only_happens_once(self):
        df = pd.DataFrame(
            [
                {
                    "gene_stable_id": "fake1",
                    "chr": "1",
                    "strand": 1,
                    "tss": 5000,
                    "tes": 5500,
                    "description": "bla",
                }
            ]
        )
        counter = [0]

        def load():
            counter[0] += 1
            return df

        g = genes.Genes(get_genome_chr_length(), load, name="shu")
        if ppg.inside_ppg():
            assert counter[0] == 0
            g.load()
            assert counter[0] == 0
            g.load()
            assert counter[0] == 0
            ppg.run_pipegraph()
        else:
            assert counter[0] == 1
            g.load()
            assert counter[0] == 1

    def test_filtering_away_works(self):
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
                        "chr": "1",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake3",
                        "chr": "2",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                ]
            )
        )
        g = genes.Genes(genome)
        filtered = g.filter("nogenes", lambda df: df["chr"] == "4")
        force_load(filtered.load())
        run_pipegraph()
        assert len(filtered.df) == 0
        assert "start" in filtered.df.columns
        assert "stop" in filtered.df.columns
        assert "tss" in filtered.df.columns
        assert "tes" in filtered.df.columns
        assert "gene_stable_id" in filtered.df.columns

    def test_annotators_are_kept_on_filtering(self):
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
                        "chr": "1",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake3",
                        "chr": "2",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                ]
            )
        )
        g = genes.Genes(genome)
        ca = Constant("shu", 5)
        g.add_annotator(ca)
        filtered = g.filter("nogenes", lambda df: df["chr"] == "4")
        assert filtered.has_annotator(ca)

    
    def test_filtering_returns_genes(self):
        g = genes.Genes(get_genome())
        on_chr_1 = g.filter("on_1", lambda df: df["chr"] == "1")
        assert g.__class__ == on_chr_1.__class__

    def test_overlap_genes_requires_two_genes(self):
        genome = get_genome()
        a = genes.Genes(genome)

        def sample_data():
            return pd.DataFrame(
                {"chr": ["Chromosome"], "start": [1000], "stop": [1100]}
            )

        b = regions.GenomicRegions("sha", sample_data, [], genome)
        force_load(a.load())
        force_load(b.load())
        run_pipegraph()

        with pytest.raises(ValueError):
            a.overlap_genes(b)

    def test_overlap_genes_raises_on_unequal_genomes(self):
        genome = get_genome("A")
        genomeB = get_genome("B")
        a = genes.Genes(genome)
        b = genes.Genes(genomeB)

        with pytest.raises(ValueError):
            a.overlap_genes(b)

    def test_overlap(self):
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
                        "chr": "1",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake3",
                        "chr": "2",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                ]
            )
        )
        g = genes.Genes(genome)
        on_chr_1 = g.filter("on_1", lambda df: df["chr"] == "1")
        on_chr_2 = g.filter("on_2", lambda df: df["chr"] == "2")
        one = g.filter("one", lambda df: df["gene_stable_id"] == "fake1")
        force_load(on_chr_1.load())
        force_load(on_chr_2.load())
        force_load(one.load())
        run_pipegraph()
        assert len(on_chr_1.df) == 2
        assert len(on_chr_2.df) == 1
        assert len(one.df) == 1
        assert g.overlap_genes(on_chr_1) == len(on_chr_1.df)
        assert on_chr_1.overlap_genes(g) == len(on_chr_1.df)
        assert on_chr_1.overlap_genes(on_chr_1) == len(on_chr_1.df)
        assert g.overlap_genes(on_chr_2) == len(on_chr_2.df)
        assert on_chr_2.overlap_genes(g) == len(on_chr_2.df)
        assert on_chr_2.overlap_genes(on_chr_2) == len(on_chr_2.df)
        assert g.overlap_genes(one) == len(one.df)
        assert one.overlap_genes(g) == len(one.df)
        assert one.overlap_genes(one) == len(one.df)

        assert on_chr_1.overlap_genes(one) == 1
        assert one.overlap_genes(on_chr_1) == 1

        assert on_chr_1.overlap_genes(on_chr_2) == 0
        assert on_chr_2.overlap_genes(on_chr_1) == 0

    def test_get_tss_regions(self):
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
                        "chr": "1",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake3",
                        "chr": "2",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                ]
            )
        )
        g = genes.Genes(genome)
        tss = g.regions_tss()
        force_load(tss.load())
        run_pipegraph()
        assert len(tss.df) == 3
        assert (tss.df["start"] == [5000, 5400, 5400]).all()
        assert (tss.df["stop"] == tss.df["start"] + 1).all()
        assert (tss.df["chr"] == ["1", "1", "2"]).all()

    def test_get_tes_regions(self):
        genome = DummyGenome(
            pd.DataFrame(
                [
                    {
                        "stable_id": "fake1",
                        "chr": "1",
                        "strand": 1,
                        "tss": 3000,
                        "tes": 4900,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake2",
                        "chr": "1",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake3",
                        "chr": "2",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                ]
            )
        )
        g = genes.Genes(genome)
        tes = g.regions_tes()
        force_load(tes.load())
        run_pipegraph()
        assert len(tes.df) == 2
        assert (tes.df["start"] == [4900, 4900]).all()
        assert (tes.df["stop"] == tes.df["start"] + 1).all()
        assert (tes.df["chr"] == ["1", "2"]).all()

    def test_get_exons_regions_overlapping(self):
        genome = DummyGenome(
            pd.DataFrame(
                [
                    {
                        "stable_id": "fake1",
                        "chr": "1",
                        "strand": 1,
                        "tss": 3000,
                        "tes": 4900,
                        "description": "bla",
                        "name": "bla1",
                    },
                    {
                        "stable_id": "fake2",
                        "chr": "1",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                        "name": "bla2",
                    },
                    {
                        "stable_id": "fake3",
                        "chr": "2",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                        "name": "bla3",
                    },
                ]
            ),
            # {transcript_stable_id, gene_stable_id, strand, start, end, exons},
            df_transcripts=pd.DataFrame(
                {
                    "transcript_stable_id": ["trans1a", "trans1b", "trans2", "trans3"],
                    "gene_stable_id": ["fake1", "fake1", "fake2", "fake3"],
                    "chr": ["1", "1", "1", "2"],
                    "strand": [1, 1, -1, -1],
                    "start": [3100, 3000, 4910, 4900],
                    "stop": [4900, 4000, 5400, 5400],
                    "exons": [
                        [(3100, 4900, 0)],
                        [(3000, 3500, 0), (3300, 3330, 0), (3750, 4000, 0)],
                        [(4910, 5000, 0), (5100, 5400, 0)],
                        [(4900, 5400, 0)],
                    ],
                }
            ),
        )
        g = genes.Genes(genome)
        exons = g.regions_exons_overlapping()
        force_load(exons.load())
        run_pipegraph()
        assert (exons.df["start"] == [3000, 3100, 3300, 3750, 4910, 5100, 4900]).all()
        assert (exons.df["stop"] == [3500, 4900, 3330, 4000, 5000, 5400, 5400]).all()
        assert (exons.df["chr"] == np.array(["1", "1", "1", "1", "1", "1", "2"])).all()

    def test_get_exons_regions_merging(self):
        genome = DummyGenome(
            pd.DataFrame(
                [
                    {
                        "stable_id": "fake1",
                        "chr": "1",
                        "strand": 1,
                        "tss": 3000,
                        "tes": 4900,
                        "description": "bla",
                        "name": "bla1",
                    },
                    {
                        "stable_id": "fake2",
                        "chr": "1",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                        "name": "bla2",
                    },
                    {
                        "stable_id": "fake3",
                        "chr": "2",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                        "name": "bla3",
                    },
                ]
            ),
            # {transcript_stable_id, gene_stable_id, strand, start, end, exons},
            df_transcripts=pd.DataFrame(
                {
                    "transcript_stable_id": ["trans1a", "trans1b", "trans2", "trans3"],
                    "gene_stable_id": ["fake1", "fake1", "fake2", "fake3"],
                    "chr": ["1", "1", "1", "2"],
                    "strand": [1, 1, -1, -1],
                    "start": [3100, 3000, 4910, 4900],
                    "stop": [4900, 4000, 5400, 5400],
                    "exons": [
                        [(3100, 4900, 0)],
                        [(3000, 3500, 0), (3300, 3330, 0), (3750, 4000, 0)],
                        [(4910, 5000, 0), (5100, 5400, 0)],
                        [(4900, 5400, 0)],
                    ],
                }
            ),
        )
        g = genes.Genes(genome)
        exons = g.regions_exons_merged()
        force_load(exons.load())
        run_pipegraph()
        assert (exons.df["start"] == [3000, 4910, 5100, 4900]).all()
        assert (exons.df["stop"] == [4900, 5000, 5400, 5400]).all()
        assert (exons.df["chr"] == ["1", "1", "1", "2"]).all()

    def test_get_intron_regions(self):
        genome = DummyGenome(
            pd.DataFrame(
                [
                    {
                        "stable_id": "fake1",
                        "chr": "1",
                        "strand": 1,
                        "tss": 3000,
                        "tes": 4900,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake2",
                        "chr": "1",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake3",
                        "chr": "2",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                ]
            ),
            # {transcript_stable_id, gene_stable_id, strand, start, end, exons},
            df_transcripts=pd.DataFrame(
                {
                    "transcript_stable_id": ["trans1a", "trans1b", "trans2", "trans3"],
                    "gene_stable_id": ["fake1", "fake1", "fake2", "fake3"],
                    "chr": ["1", "1", "1", "2"],
                    "strand": [1, 1, -1, -1],
                    "start": [3100, 3000, 4900, 4900],
                    "stop": [4900, 4000, 5400, 5400],
                    "exons": [
                        [(3100, 4900, 0)],
                        [(3000, 3500, 0), (3750, 4000, 0)],
                        [(4900, 5000, 0), (5100, 5400, 0)],
                        [(4900, 5400, 0)],
                    ],
                }
            ),
        )
        g = genes.Genes(genome)
        introns = g.regions_introns()
        force_load(introns.load())
        run_pipegraph()
        assert (introns.df["start"] == [3000, 3500, 4000, 5000]).all()
        assert (introns.df["stop"] == [3100, 3750, 4900, 5100]).all()
        # no intronic region on chr 2
        assert (introns.df["chr"] == ["1", "1", "1", "1"]).all()


@pytest.mark.usefixtures("new_pipegraph")
class TestGenes:
    def test_write_bed(self):
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
                        "chr": "1",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake3",
                        "chr": "2",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                ]
            )
        )
        g = genes.Genes(genome)
        sample_filename = "genes.bed"
        g.write_bed(sample_filename)
        run_pipegraph()
        assert len(g.df) > 0
        read = read_bed(g.result_dir / sample_filename)
        assert len(read) == len(g.df)
        assert read[0].refseq == b"1"
        assert read[1].refseq == b"1"
        assert read[2].refseq == b"2"
        assert read[0].position == 4900
        assert read[1].position == 5000
        assert read[2].position == 4900
        assert read[0].length == 500
        assert read[1].length == 500
        assert read[2].length == 500
        assert read[0].name == b"fake2"
        assert read[1].name == b"fake1"
        assert read[2].name == b"fake3"

    def test_write_bed_auto_filename(self):
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
                        "chr": "1",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake3",
                        "chr": "2",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                ]
            )
        )
        g = genes.Genes(genome)
        sample_filename = g.write_bed().job_id  # the jobs output filename
        run_pipegraph()
        assert len(g.df) > 0
        read = read_bed(sample_filename)
        assert len(read) == len(g.df)
        assert read[0].refseq == b"1"
        assert read[1].refseq == b"1"
        assert read[2].refseq == b"2"
        assert read[0].position == 4900
        assert read[1].position == 5000
        assert read[2].position == 4900
        assert read[0].length == 500
        assert read[1].length == 500
        assert read[2].length == 500
        assert read[0].name == b"fake2"
        assert read[1].name == b"fake1"
        assert read[2].name == b"fake3"

    # def test_annotation_keeps_row_names(self):
    # g = genes.Genes(dummyGenome)
    # g.do_load()
    # row_names = g.df.row_names
    # g.annotate()
    # run_pipegraph()
    # self.assertTrue((row_names == g.df.row_names).all())
