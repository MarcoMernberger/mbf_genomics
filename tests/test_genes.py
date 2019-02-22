import pypipegraph as ppg
import numpy as np
import pandas as pd
import pytest
from mbf_fileformats.bed import read_bed
import mbf_genomics.regions as regions
import mbf_genomics.genes as genes
from mbf_genomics.annotator import Constant

from .shared import get_genome, force_load


def DummyGenome(df_genes, df_transcripts=None):
    g = get_genome()
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
    g.df_genes = df_genes
    if df_transcripts:
        g.df_transcripts = df_transcripts
        if not "biotype" in df_transcripts.columns:
            df_transcripts = df_genes.assign(biotype="protein_coding")
        if 'exons' in df_transcripts.columns:
            if len(df_transcripts['exons'].iloc[0]) == 3:
                df_transcripts = df_transcripts.assign(exons = [
                    (x[0], x[1]) for x in df_transcripts['exons']])


    return g


@pytest.mark.usefixtures("newpipegraph")
class GenesLoadingTests:
    def test_basic_loading_from_genome(self):
        g = genes.Genes(get_genome())
        force_load(g.load())
        ppg.run_pipegraph()
        assert len(g.df) == 246
        assert (g.df["stable_id"][:3] == ["CRP_001", "CRP_002", "CRP_003"]).all()
        assert g.df["stable_id"].iloc[-1] == "CRP_182"
        assert g.df["start"].iloc[-1] == 158649
        assert g.df["stop"].iloc[-1] == 159662
        assert g.df["strand"].iloc[-1] == -1

    def test_alternative_loading_raises_on_non_df(self):
        g = genes.Genes(get_genome(), lambda: None, "myname")
        force_load(g.load())

        with pytest.raises(ValueError):
            ppg.run_pipegraph()

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
            ppg.run_pipegraph()

        def inner_chr():
            df2 = df.copy()
            df2 = df2.drop("chr", axis=1)
            g = genes.Genes(get_genome(), lambda: df2, name="shu")
            g.load()
            ppg.run_pipegraph()

        def inner_tes():
            df2 = df.copy()
            df2 = df2.drop("tes", axis=1)
            g = genes.Genes(get_genome(), lambda: df2, name="shi")
            g.load()
            ppg.run_pipegraph()

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
            g = genes.Genes(get_genome(), lambda: df)

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
            ppg.run_pipegraph()

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
            ppg.run_pipegraph()

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
            ppg.run_pipegraph()

    def test_do_load_only_happens_once(self):
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
        counter = [0]

        def load():
            counter[0] += 1
            return df

        g = genes.Genes(get_genome(), load, name="shu")
        assert counter[0] == 0
        g.do_load()
        assert counter[0] == 1
        g.do_load()
        assert counter[0] == 1  # still one

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
        ppg.run_pipegraph()
        assert len(filtered.df) == 0
        assert "start" in filtered.df.columns
        assert "stop" in filtered.df.columns
        assert "tss" in filtered.df.columns
        assert "tes" in filtered.df.columns
        assert "stable_id" in filtered.df.columns

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

    def test_loading_from_genome_is_singletonic(self):
        genome = get_genome()
        genesA = genes.Genes(genome)
        genesB = genes.Genes(genome)
        assert genesA is genesB
        filterA = genesA.filter("fa", select_top_k=10)
        filterAa = genesA.filter("faa", select_top_k=10)
        filterB = genesB.filter("fab", select_top_k=10)
        assert not (filterA is genesA)
        assert not (filterAa is filterA)
        assert not (filterAa is filterB)

    def test_filtering_returns_genes(self):
        g = genes.Genes(get_genome()())
        on_chr_1 = g.filter("on_1", lambda df: df["chr"] == "1")
        assert g.__class__ == on_chr_1.__class__

    def test_overlap_genes_requires_two_genes(self):
        genome = get_genome()
        a = genes.Genes(genome)

        def sample_data():
            return pd.DataFrame({"chr": ["1"], "start": [1000], "stop": [1100]})

        b = regions.GenomicRegions("sha", sample_data, [], genome)
        force_load(a.load())
        force_load(b.load())
        ppg.run_pipegraph()

        with pytest.raises(ValueError):
            a.overlap_genes(b)

    def test_overlap_genes_raises_on_unequal_genomes(self):
        genome = get_genome('A')
        genomeB = get_genome('B')
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
        one = g.filter("one", lambda df: df["stable_id"] == "fake1")
        force_load(on_chr_1.load())
        force_load(on_chr_2.load())
        force_load(one.load())
        ppg.run_pipegraph()
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

        assert on_chr_1.overlap_genes(one.df) == 1
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
        tss = g.get_tss_regions()
        force_load(tss.load())
        ppg.run_pipegraph()
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
        tes = g.get_tes_regions()
        force_load(tes.load())
        ppg.run_pipegraph()
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
        exons = genome.get_exon_regions_overlapping()
        force_load(exons.load())
        ppg.run_pipegraph()
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
        exons = genome.get_exon_regions_merged()
        force_load(exons.load())
        ppg.run_pipegraph()
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
        introns = g.get_intron_regions()
        force_load(introns.load())
        ppg.run_pipegraph()
        assert (introns.df["start"] == [3000, 3500, 4000, 5000]).all()
        assert (introns.df["stop"] == [3100, 3750, 4900, 5100]).all()
        # no intronic region on chr 2
        assert (introns.df["chr"] == ["1", "1", "1", "1"]).all()

    def test_intronify_more_complex(self):
        transcript = {
            u"chr": "2R",
            u"exons": [
                (14243005, 14244766, 0),
                (14177040, 14177355, 0),
                (14176065, 14176232, 0),
                (14175632, 14175893, 0),
                (14172742, 14175244, 0),
                (14172109, 14172226, 0),
                (14170836, 14172015, 0),
                (14169750, 14170749, 0),
                (14169470, 14169683, 0),
                (14169134, 14169402, 0),
                (14167751, 14169018, 0),
                (14166570, 14167681, 0),
            ],
            u"gene_stable_id": "FBgn0010575",
            u"start": 14166570,
            u"stop": 14244766,
            u"strand": -1,
            u"transcript_stable_id": "FBtr0301547",
        }
        gene = {
            u"biotype": "protein_coding",
            u"chr": "2R",
            u"description": "CG5580 [Source:FlyBase;GeneId:FBgn0010575]",
            u"name": "sbb",
            u"stable_id": "FBgn0010575",
            u"strand": -1,
            u"tes": 14166570,
            u"tss": 14244766,
        }

        g = genes.Genes(get_genome())
        introns = g._intron_intervals(transcript, gene)
        assert (
            np.array(introns)
            == [
                (14167681, 14167751),
                (14169018, 14169134),
                (14169402, 14169470),
                (14169683, 14169750),
                (14170749, 14170836),
                (14172015, 14172109),
                (14172226, 14172742),
                (14175244, 14175632),
                (14175893, 14176065),
                (14176232, 14177040),
                (14177355, 14243005),
            ]
        ).all()

    def test_intron_intervals_raises_on_inverted(self):
        transcript = {
            u"chr": "2R",
            u"exons": [
                (14243005, 14244766, 0),
                (14177040, 14177355, 0),
                (14176065, 14176232, 0),
                (14175632, 14175893, 0),
                (14172742, 14175244, 0),
                (14172109, 14172226, 0),
                (14172015, 14170836, 0),  # inverted
                (14169750, 14170749, 0),
                (14169470, 14169683, 0),
                (14169134, 14169402, 0),
                (14167751, 14169018, 0),
                (14166570, 14167681, 0),
            ],
            u"gene_stable_id": "FBgn0010575",
            u"start": 14166570,
            u"stop": 14244766,
            u"strand": -1,
            u"transcript_stable_id": "FBtr0301547",
        }
        gene = {
            u"biotype": "protein_coding",
            u"chr": "2R",
            u"description": "CG5580 [Source:FlyBase;GeneId:FBgn0010575]",
            u"name": "sbb",
            u"stable_id": "FBgn0010575",
            u"strand": -1,
            u"tes": 14166570,
            u"tss": 14244766,
        }
        g = genes.Genes(get_genome())

        with pytest.raises(ValueError):
            introns = g._intron_intervals(transcript, gene)

    def test_get_gene_exons(self):
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
        force_load(g.load())
        ppg.run_pipegrap()
        two = g.get_gene_exons("fake2")
        assert (two["start"] == [4910, 5100]).all()
        assert (two["stop"] == [5000, 5400]).all()

    def test_get_gene_introns(self):
        genome = DummyGenome()(
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
                        "tss": 5500,
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
        force_load(g.load())
        ppg.run_pipegraph()
        one = g.get_gene_introns("fake1")
        print("one", one)
        assert len(one) == 0

        two = g.get_gene_introns("fake2")
        assert (two["start"] == [4900, 5000, 5400]).all()
        assert (two["stop"] == [4910, 5100, 5500]).all()


@pytest.mark.usefixtures("newpipegraph")
class GenesTests:
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
        sample_filename = "cache/genes.bed"
        g.write_bed(sample_filename)
        ppg.run_pipegraph()
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
        ppg.run_pipegraph()
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
    # ppg.run_pipegraph()
    # self.assertTrue((row_names == g.df.row_names).all())
