import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from mbf_genomics.genes.anno_tag_counts import IntervalStrategyGene
from mbf_genomics.genes import Genes, anno_tag_counts
from .shared import MockGenome, force_load, run_pipegraph, RaisesDirectOrInsidePipegraph
import pysam


def MockBam(chr, length, mode=1):
    """a number of reads of length 1 at any position!"""
    chrs = [chr]
    fn = "mock.bam"
    bam = pysam.Samfile(
        fn, "wb", reference_names=chrs, reference_lengths=[100_000] * len(chrs)
    )

    def write_read(pos, strand, name, nh=1):
        al = pysam.AlignedSegment()
        al.pos = pos
        al.reference_id = 0
        al.is_reverse = strand == -1
        al.query_name = name
        al.seq = "A"
        al.cigar = ((0, 1),)
        al.tags = (("NH", nh),)
        bam.write(al)

    for ii in range(0, length):
        if mode == 0:
            write_read(ii, 1, "fw_%i" % ii)
        elif mode == 1:
            write_read(ii, 1, "fw_%i" % ii)
            write_read(ii, -1, "rv_%i" % ii)
        elif mode == 2:
            write_read(ii, 1, "fw_%i" % ii)
            write_read(ii, -1, "rv_%i" % ii)
            write_read(ii, -1, "rv2_%i" % ii)
        elif mode == 3:  # good for testing the 'count one read in one gene once'
            write_read(ii, 1, "fw_%i" % ii, 4)
            write_read(ii, 1, "fw_%i" % ii, 4)
            write_read(ii, 1, "fw_%i" % ii, 4)
            write_read(ii, 1, "fw_%i" % ii, 4)
            write_read(ii, -1, "rv2_%i" % ii)
        elif mode == 4:
            write_read(ii, 1, "fw_%i_a" % ii)
            write_read(ii, 1, "fw_%i_b" % ii)
            write_read(ii, 1, "fw_%i_c" % ii)
            write_read(ii, 1, "fw_%i_d" % ii)
            write_read(ii, -1, "rv2_%i_a" % ii)
        else:
            raise ValueError("Invalid mode")
    bam.close()
    pysam.sort("-o", "mock_sorted.bam", fn)
    pysam.index("mock_sorted.bam")
    res = pysam.Samfile("mock_sorted.bam", "rb")
    return res


def MockBamFixed(reads):
    """Each read is a
            dict: strand, regions, tags, qname, is_read1
    """
    ii = 0
    for read in reads:
        if "qname" not in read:
            name = "read_%i" % ii
            ii = +1
            read["qname"] = name
        if "chr" not in read:
            read["chr"] = "1"
        if "tags" not in read:
            read["tags"] = {}
        if "is_read1" not in read:
            read["is_read1"] = True
    chrs = sorted(set([r["chr"] for r in reads]))
    fn = "mock.bam"
    if Path(fn).exists():
        Path(fn).unlink()
    bam = pysam.Samfile(
        fn, "wb", reference_names=chrs, reference_lengths=[100_000] * len(chrs)
    )
    for read in reads:
        al = pysam.AlignedSegment()
        al.query_name = read["qname"]
        read_len = sum([x[1] - x[0] for x in read["regions"]])
        al.query_sequence = "A" * read_len
        al.reference_id = bam.references.index(read["chr"])
        al.is_read1 = read["is_read1"]
        al.is_reverse = read["strand"] == -1
        t = [tuple(x) for x in read["tags"].items()]
        al.tags = t
        al.pos = read["regions"][0][0]
        last_stop = None
        cigar = []
        for start, stop in read["regions"]:
            if last_stop:
                cigar.append((2, start - last_stop))
            last_stop = stop
            cigar.append((0, stop - start))
        al.cigar = tuple(cigar)
        bam.write(al)
    bam.close()
    pysam.sort("-o", "mock_sorted.bam", fn)
    pysam.index("mock_sorted.bam")
    res = pysam.Samfile("mock_sorted.bam", "rb")
    return res


def FakePaireEndBam(chr, length, reverse=False):
    chrs = [chr]
    fn = "mock.bam"
    bam = pysam.Samfile(
        fn, "wb", reference_names=chrs, reference_lengths=[100_000] * len(chrs)
    )

    def write_read(pos, strand, name):
        al = pysam.AlignedSegment()
        al.pos = pos
        al.reference_id = 0
        al.is_reverse = strand == -1
        al.query_name = name
        al.seq = "A"
        al.cigar = ((0, 1),)
        al.tags = (("NH", 1),)
        bam.write(al)

    for ii in range(0, length):
        if not reverse:
            write_read(ii, 1, "read_%i" % ii, is_read1=True)
            write_read(ii, -1, "read_%i" % ii, is_read1=False)
        else:
            write_read(ii, 1, "read_%i" % ii, is_read1=False)
            write_read(ii, -1, "read_%i" % ii, is_read1=True)

    bam.close()
    pysam.sort("-o", "mock_sorted.bam", fn)
    pysam.index("mock_sorted.bam")
    res = pysam.Samfile("mock_sorted.bam", "rb")
    return res


class MockLane(object):
    def __init__(self, bam, genome, name="fakelane1"):
        self.name = name
        self.genome = genome
        self.bam = bam
        self.vid = "FAkeVID"

    def get_bam(self):
        return self.bam

    def load(self):
        pass


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestGeneCount:
    def test_unstranded(self):

        bam = MockBam("1", 10000)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 1000,
                    "tes": 2000,
                    "chr": "1",
                }
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        genes = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.GeneUnstranded(lane)
        genes += anno
        force_load(genes.annotate)
        run_pipegraph()
        df = genes.df
        assert (df[anno.columns[0]] == np.array([2000])).all()

    def test_stranded_raises_on_too_many_reverse_reads(self):

        bam = MockBam("1", 10000, mode=2)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 1000,
                    "tes": 2000,
                    "chr": "1",
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 2010,
                    "tes": 1910,
                    "chr": "1",
                },
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        genes = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.GeneStranded(lane)
        with RaisesDirectOrInsidePipegraph(ValueError):
            force_load(genes.add_annotator(anno))

    def test_stranded(self):

        bam = MockBam("1", 10000, mode=2)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 2000,
                    "tes": 1000,
                    "chr": "1",
                    "strand": -1,
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 2010,
                    "tes": 1910,
                    "chr": "1",
                    "strand": -1,
                },
                {
                    "gene_stable_id": "FakeC",
                    "name": "C",
                    "tss": 400,
                    "tes": 410,
                    "chr": "1",
                    "strand": 1,
                },
                # to get around the too many reverse reads threshold
                {
                    "gene_stable_id": "FakeD",
                    "name": "D",
                    "tss": 2010,
                    "tes": 1910,
                    "chr": "1",
                    "strand": -1,
                },
                {
                    "gene_stable_id": "FakeD2",
                    "name": "D2",
                    "tss": 2010,
                    "tes": 1910,
                    "chr": "1",
                    "strand": -1,
                },
                {
                    "gene_stable_id": "FakeD3",
                    "name": "D3",
                    "tss": 2010,
                    "tes": 1910,
                    "chr": "1",
                    "strand": -1,
                },
                {
                    "gene_stable_id": "FakeD4",
                    "name": "D4",
                    "tss": 2010,
                    "tes": 1910,
                    "chr": "1",
                    "strand": -1,
                },
                {
                    "gene_stable_id": "FakeD5",
                    "name": "D5",
                    "tss": 2010,
                    "tes": 1910,
                    "chr": "1",
                    "strand": -1,
                },
                {
                    "gene_stable_id": "FakeD6",
                    "name": "D6",
                    "tss": 2010,
                    "tes": 1910,
                    "chr": "1",
                    "strand": -1,
                },
                {
                    "gene_stable_id": "FakeD7",
                    "name": "D7",
                    "tss": 2010,
                    "tes": 1910,
                    "chr": "1",
                    "strand": -1,
                },
            ]
        ).sort_values("gene_stable_id")

        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        genes = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.GeneStranded(lane)
        force_load(genes.add_annotator(anno))
        run_pipegraph()
        assert list(genes.df.index) == list(
            range(len(genes.df))
        ), (
            "make sure the Genes.df has an index 0..n"
        )  # make sure that the Genes have an index that goes 0..n

        assert (
            genes.df.gene_stable_id
            == [
                "FakeC",
                "FakeA",
                "FakeB",
                "FakeD",
                "FakeD2",
                "FakeD3",
                "FakeD4",
                "FakeD5",
                "FakeD6",
                "FakeD7",
            ]
        ).all()

        assert (
            genes.df[anno.columns[0]]
            == np.array([10, 2000, 200, 200, 200, 200, 200, 200, 200, 200])
        ).all()


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestExonSmartCount:
    def test_validation(self):
        genome = MockGenome(
            pd.DataFrame(
                [
                    {
                        "gene_stable_id": "FakeA",
                        "name": "A",
                        "tss": 500,
                        "tes": 1000,
                        "chr": "1",
                        "strand": 1,
                    }
                ]
            ),
            None,
            {"1": 10000},
        )
        lane = object()
        with pytest.raises(ValueError):
            anno_tag_counts.ExonSmartUnstranded(lane)

    def test_unstranded(self):

        bam = MockBam("1", 10000, mode=2)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 500,
                    "tes": 1000,
                    "chr": "1",
                    "strand": 1,
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 4000,
                    "tes": 2000,
                    "chr": "1",
                    "strand": -1,
                },
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "transcript_stable_id": "fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb1",
                    "transcript_stable_id": "fakeb1",
                    "chr": "1",
                    "exons": [(3900, 4000), (2900, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
                {
                    "name": "Fakeb2",
                    "transcript_stable_id": "fakeb2",
                    "chr": "1",
                    "exons": [(3900, 4000), (2000, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb3",
                    "transcript_stable_id": "fakeab3",
                    "chr": "1",
                    "exons": [(3900, 4000), (2850, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonSmartUnstranded(lane)
        genes = Genes(genome)
        force_load(genes.add_annotator(anno))
        run_pipegraph()
        df = genes.df.sort_values("name")
        print(df)
        print(genome.genes["FakeA"].exons_protein_coding_overlapping)
        assert (df[anno.columns[0]] == np.array([300, (100 + 150) * 3])).all()

    def test_stranded(self):

        bam = MockBam("1", 10000, mode=4)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 500,
                    "tes": 600,
                    "chr": "1",
                    "strand": 1,
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 4000,
                    "tes": 2000,
                    "chr": "1",
                    "strand": -1,
                },
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb1",
                    "chr": "1",
                    "exons": [(3900, 4000), (2900, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
                {
                    "name": "Fakeb2",
                    "chr": "1",
                    "exons": [(3900, 4000), (2000, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb3",
                    "chr": "1",
                    "exons": [(3900, 4000), (2850, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonSmartStranded(lane)
        anno.count_strategy.disable_sanity_check = True
        force_load(gr.add_annotator(anno))
        run_pipegraph()
        df = gr.df.sort_values("name")
        print(df)
        assert (df[anno.columns[0]] == np.array([100 * 4, (100 + 150) * 1])).all()

    def test_stranded_weighted(self):

        bam = MockBam("1", 10000, mode=4)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 500,
                    "tes": 600,
                    "chr": "1",
                    "strand": 1,
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 4000,
                    "tes": 2000,
                    "chr": "1",
                    "strand": -1,
                },
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb1",
                    "chr": "1",
                    "exons": [(3900, 4000), (2900, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
                {
                    "name": "Fakeb2",
                    "chr": "1",
                    "exons": [(3900, 4000), (2000, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb3",
                    "chr": "1",
                    "exons": [(3900, 4000), (2850, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonSmartWeightedStranded(lane)
        anno.count_strategy.disable_sanity_check = True
        force_load(gr.add_annotator(anno))
        run_pipegraph()
        df = gr.df.sort_values("name")
        print(df)
        assert (df[anno.columns[0]] == np.array([100 * 4, (100 + 150) * 1])).all()

    def test_stranded_dedup(self):

        bam = MockBam("1", 10000, mode=3)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 500,
                    "tes": 700,
                    "chr": "1",
                    "strand": 1,
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 4000,
                    "tes": 2000,
                    "chr": "1",
                    "strand": -1,
                },
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb1",
                    "chr": "1",
                    "exons": [(3900, 4000), (2900, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
                {
                    "name": "Fakeb2",
                    "chr": "1",
                    "exons": [(3900, 4000), (2000, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb3",
                    "chr": "1",
                    "exons": [(3900, 4000), (2850, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonSmartStranded(lane)
        force_load(gr.add_annotator(anno))
        run_pipegraph()
        assert (gr.df[anno.columns[0]] == np.array([100 * 1, (100 + 150) * 1])).all()


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestExonCount:
    def test_unstranded(self):

        bam = MockBam("1", 10000, mode=2)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 500,
                    "tes": 2000,
                    "chr": "1",
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 4000,
                    "tes": 2000,
                    "chr": "1",
                },
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb1",
                    "chr": "1",
                    "exons": [(3900, 4000), (2900, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
                {
                    "name": "Fakeb2",
                    "chr": "1",
                    "exons": [(3900, 4000), (2000, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb3",
                    "chr": "1",
                    "exons": [(3900, 4000), (2850, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonUnstranded(lane)
        force_load(gr.add_annotator(anno))
        run_pipegraph()
        df = gr.df.sort_values("gene_stable_id")
        assert (df[anno.columns[0]] == np.array([300, (1100) * 3])).all()

    def test_stranded(self):

        bam = MockBam("1", 10000, mode=4)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 0,
                    "tes": 2000,
                    "chr": "1",
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 4000,
                    "tes": 2000,
                    "chr": "1",
                },
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb1",
                    "chr": "1",
                    "exons": [(3900, 4000), (2900, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
                {
                    "name": "Fakeb2",
                    "chr": "1",
                    "exons": [(3900, 4000), (2000, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb3",
                    "chr": "1",
                    "exons": [(3900, 4000), (2850, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonStranded(lane)
        anno.count_strategy.disable_sanity_check = True
        force_load(gr.add_annotator(anno))
        run_pipegraph()
        df = gr.df.sort_values("gene_stable_id")
        assert (df[anno.columns[0]] == np.array([100 * 4, (1100) * 1])).all()


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestCPM:
    def test_stranded(self):

        bam = MockBam("1", 10000, mode=4)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 0,
                    "tes": 2000,
                    "chr": "1",
                    "strand": 1,
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 4000,
                    "tes": 2000,
                    "chr": "1",
                    "strand": -1,
                },
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb1",
                    "chr": "1",
                    "exons": [(3900, 4000), (2900, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
                {
                    "name": "Fakeb2",
                    "chr": "1",
                    "exons": [(3900, 4000), (2000, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb3",
                    "chr": "1",
                    "exons": [(3900, 4000), (2850, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonSmartStranded(lane)
        anno.count_strategy.disable_sanity_check = True
        anno2 = anno_tag_counts.NormalizationCPM(anno)
        force_load(gr.add_annotator(anno2))
        run_pipegraph()
        df = gr.df.sort_values("gene_stable_id")
        assert (df[anno.columns[0]] == np.array([100 * 4, (250) * 1])).all()
        assert (
            df[anno2.columns[0]] == np.array([400 * 1e6 / 650.0, 250 * 1e6 / 650.0])
        ).all()


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestTPM:
    def test_stranded(self):

        bam = MockBam("1", 10000, mode=4)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 0,
                    "tes": 2000,
                    "chr": "1",
                    "strand": 1,
                },  # length: 1000
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 4000,
                    "tes": 2000,
                    "chr": "1",
                    "strand": -1,
                },  # length 2000
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                },
                # length 100
                {
                    "name": "Fakeb1",
                    "chr": "1",
                    "exons": [(3900, 4000), (2900, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },  # length = 200
                {
                    "name": "Fakeb2",
                    "chr": "1",
                    "exons": [(3900, 4000), (2000, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "misc_RNA",
                },  # 1100...
                {
                    "name": "Fakeb3",
                    "chr": "1",
                    "exons": [(3900, 4000), (2850, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },  # length: 250
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonSmartStranded(lane)
        anno.count_strategy.disable_sanity_check = True
        anno2 = anno_tag_counts.NormalizationTPM(anno)

        force_load(gr.add_annotator(anno2))
        run_pipegraph()
        df = gr.df.sort_values("gene_stable_id")
        tpm_factor = 1e6 / (400.0 / 100 + 250.0 / 250)

        assert (
            df[anno2.columns[0]]
            == np.array([400 / 100.0 * tpm_factor, 250 / 250.0 * tpm_factor])
        ).all()


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestCPMBiotypes:
    def test_stranded(self):

        bam = MockBam("1", 10000, mode=4)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 0,
                    "tes": 2000,
                    "chr": "1",
                    "biotype": "protein_coding",
                    "strand": 1,
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 4000,
                    "tes": 2000,
                    "chr": "1",
                    "biotype": "misc_RNA",
                    "strand": -1,
                },
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb1",
                    "chr": "1",
                    "exons": [(3900, 4000), (2900, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
                {
                    "name": "Fakeb2",
                    "chr": "1",
                    "exons": [(3900, 4000), (2000, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb3",
                    "chr": "1",
                    "exons": [(3900, 4000), (2850, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonSmartStranded(lane)
        anno.count_strategy.disable_sanity_check = True
        anno2 = anno_tag_counts.NormalizationCPMBiotypes(anno, ("protein_coding",))
        force_load(gr.add_annotator(anno2))
        run_pipegraph()
        df = gr.df.sort_values("gene_stable_id")

        assert df[anno2.columns[0]][0] == 1e6
        assert np.isnan(df[anno2.columns[0]][1])

    def test_stranded2(self):

        bam = MockBam("1", 10000, mode=4)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 0,
                    "tes": 2000,
                    "chr": "1",
                    "biotype": "protein_coding",
                    "strand": 1,
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 4000,
                    "tes": 2000,
                    "chr": "1",
                    "biotype": "misc_RNA",
                    "strand": -1,
                },
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb1",
                    "chr": "1",
                    "exons": [(3900, 4000), (2900, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
                {
                    "name": "Fakeb2",
                    "chr": "1",
                    "exons": [(3900, 4000), (2000, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb3",
                    "chr": "1",
                    "exons": [(3900, 4000), (2850, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonSmartStranded(lane)
        anno.count_strategy.disable_sanity_check = True
        anno2 = anno_tag_counts.NormalizationCPMBiotypes(
            anno, ("protein_coding", "misc_RNA")
        )
        force_load(gr.add_annotator(anno2))
        run_pipegraph()
        df = gr.df.sort_values("gene_stable_id")

        assert (
            df[anno2.columns[0]] == np.array([400 * 1e6 / 650, 250 * 1e6 / 650])
        ).all()

    def test_validation(self):
        bam = MockBam("1", 10000, mode=4)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 0,
                    "tes": 2000,
                    "chr": "1",
                    "biotype": "protein_coding",
                    "strand": 1,
                }
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                }
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonSmartStranded(lane)
        anno.count_strategy.disable_sanity_check = True
        with pytest.raises(ValueError):
            anno_tag_counts.NormalizationCPMBiotypes(anno, "protein_coding")


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestTPMBiotypes:
    def test_stranded(self):

        bam = MockBam("1", 10000, mode=4)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 0,
                    "tes": 2000,
                    "chr": "1",
                    "biotype": "protein_coding",
                    "strand": 1,
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 4000,
                    "tes": 2000,
                    "chr": "1",
                    "biotype": "misc_RNA",
                    "strand": -1,
                },
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb1",
                    "chr": "1",
                    "exons": [(3900, 4000), (2900, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
                {
                    "name": "Fakeb2",
                    "chr": "1",
                    "exons": [(3900, 4000), (2000, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb3",
                    "chr": "1",
                    "exons": [(3900, 4000), (2850, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonSmartStranded(lane)
        anno.count_strategy.disable_sanity_check = True
        anno2 = anno_tag_counts.NormalizationTPMBiotypes(anno, ("protein_coding",))

        force_load(gr.add_annotator(anno2))
        run_pipegraph()
        df = gr.df.sort_values("gene_stable_id")
        # length_by_gene = anno.interval_strategy.get_interval_lengths_by_gene(genome)
        counts = np.array([400, 250.0])
        lengths = [100.0, 250]
        xi = counts / lengths
        # tpms = xi * (1e6 / xi.sum())
        assert df[anno2.columns[0]][0] == 1e6
        assert np.isnan(df[anno2.columns[0]][1])

    def test_stranded2(self):

        bam = MockBam("1", 10000, mode=4)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 0,
                    "tes": 2000,
                    "chr": "1",
                    "biotype": "protein_coding",
                    "strand": 1,
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 4000,
                    "tes": 2000,
                    "chr": "1",
                    "biotype": "misc_RNA",
                    "strand": -1,
                },
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb1",
                    "chr": "1",
                    "exons": [(3900, 4000), (2900, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
                {
                    "name": "Fakeb2",
                    "chr": "1",
                    "exons": [(3900, 4000), (2000, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb3",
                    "chr": "1",
                    "exons": [(3900, 4000), (2850, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonSmartStranded(lane)
        anno.count_strategy.disable_sanity_check = True
        anno2 = anno_tag_counts.NormalizationTPMBiotypes(
            anno, ("protein_coding", "misc_RNA")
        )
        force_load(gr.add_annotator(anno2))
        run_pipegraph()
        df = gr.df.sort_values("gene_stable_id")

        length_by_gene = anno.interval_strategy.get_interval_lengths_by_gene(genome)
        tpm_factor = 1e6 / (400.0 / 100 + 250.0 / 250)
        print(length_by_gene)
        assert (
            df[anno2.columns[0]]
            == np.array([400 / 100.0 * tpm_factor, 250 / 250.0 * tpm_factor])
        ).all()

    def test_validation(self):
        bam = MockBam("1", 10000, mode=4)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 0,
                    "tes": 2000,
                    "chr": "1",
                    "biotype": "protein_coding",
                    "strand": 1,
                }
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                }
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonSmartStranded(lane)
        anno.count_strategy.disable_sanity_check = True
        with pytest.raises(ValueError):
            anno_tag_counts.NormalizationTPMBiotypes(anno, "protein_coding")


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestFPKM:
    def test_stranded_exon(self):

        bam = MockBam("1", 10000, mode=4)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 0,
                    "tes": 2000,
                    "chr": "1",
                    "strand": 1,
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 4000,
                    "tes": 2000,
                    "chr": "1",
                    "strand": -1,
                },
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb1",
                    "chr": "1",
                    "exons": [(3900, 4000), (2900, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
                {
                    "name": "Fakeb2",
                    "chr": "1",
                    "exons": [(3900, 4000), (2000, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb3",
                    "chr": "1",
                    "exons": [(3900, 4000), (2850, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonSmartStranded(lane)
        anno.count_strategy.disable_sanity_check = True
        anno2 = anno_tag_counts.NormalizationFPKM(anno)
        force_load(gr.add_annotator(anno2))
        run_pipegraph()
        df = gr.df.sort_values("gene_stable_id")
        assert (
            np.abs(
                df[anno2.columns[0]]
                - np.array([400 * 1e6 / 650.0 / 0.1, 250 * 1e6 / 650 / 0.25])
            )
            < 0.00001
        ).all()

    def test_stranded_gene(self):

        bam = MockBam("1", 10000, mode=4)  # 4 forward, one reverse...
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 1000,
                    "tes": 2000,
                    "chr": "1",
                    "strand": 1,
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 4000,
                    "tes": 2000,
                    "chr": "1",
                    "strand": -1,
                },
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(1500, 1600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb1",
                    "chr": "1",
                    "exons": [(3900, 4000), (2900, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
                {
                    "name": "Fakeb2",
                    "chr": "1",
                    "exons": [(3900, 4000), (2000, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb3",
                    "chr": "1",
                    "exons": [(3900, 4000), (2850, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.GeneStranded(lane)
        anno.count_strategy.disable_sanity_check = True
        anno2 = anno_tag_counts.NormalizationFPKM(anno)
        force_load(gr.add_annotator(anno2))
        run_pipegraph()
        df = gr.df.sort_values("gene_stable_id")

        print(df)
        print(np.array([4 * 1000 * 1e6 / 6000 / 1.0, 1 * 2000 * 1e6 / 6000 / 2.0]))
        assert (
            np.abs(
                df[anno2.columns[0]]
                - np.array([4 * 1000 * 1e6 / 6000 / 1.0, 1 * 2000 * 1e6 / 6000 / 2.0])
            )
            < 0.00001
        ).all()


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestFPKMBiotypes:
    def test_stranded_exon(self):

        bam = MockBam("1", 10000, mode=4)
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "biotype": "non_coding",
                    "name": "A",
                    "tss": 0,
                    "tes": 2000,
                    "chr": "1",
                    "strand": 1,
                },
                {
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                    "name": "B",
                    "tss": 4000,
                    "tes": 2000,
                    "chr": "1",
                    "strand": -1,
                },
            ]
        )
        transcripts = pd.DataFrame(
            [
                {
                    "name": "Fakea1",
                    "chr": "1",
                    "exons": [(500, 600)],
                    "strand": 1,
                    "gene_stable_id": "FakeA",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb1",
                    "chr": "1",
                    "exons": [(3900, 4000), (2900, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
                {
                    "name": "Fakeb2",
                    "chr": "1",
                    "exons": [(3900, 4000), (2000, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "misc_RNA",
                },
                {
                    "name": "Fakeb3",
                    "chr": "1",
                    "exons": [(3900, 4000), (2850, 3000)],
                    "strand": -1,
                    "gene_stable_id": "FakeB",
                    "biotype": "protein_coding",
                },
            ]
        )
        genome = MockGenome(genes, transcripts, {"1": 10000})
        gr = Genes(genome)
        lane = MockLane(bam, genome)
        anno = anno_tag_counts.ExonSmartStranded(lane)
        anno.count_strategy.disable_sanity_check = True
        anno2 = anno_tag_counts.NormalizationFPKMBiotypes(anno, ("protein_coding",))
        with pytest.raises(ValueError):
            anno_tag_counts.NormalizationFPKMBiotypes(anno, "protein_coding")

        force_load(gr.add_annotator(anno2))
        run_pipegraph()
        df = gr.df.sort_values("gene_stable_id")
        should = np.array([np.nan, 400 * 1e6 / 400.0 / 0.1])
        delta = df[anno2.columns[0]] - should
        assert ((np.abs(delta) < 0.00001) | (np.isnan(delta) == np.isnan(should))).all()


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestWeigthedCounts:

    # i need to test
    # a) reads only partialy mapping a gene
    # b) a read that only maps one gene
    # c) a read that's mapped twice, to multiple genes
    # d) a read that's mapped twice to one gene
    # e) a read that's mapped twice, second time outside of a gene (how do
    # we count that?! - very carefully, we only consider reads within our
    # reference intervals!)

    # f) a read that's mappend once, but there are two genes there
    # g) a read that's mapped multiple times, once to two genes
    # f) a read that's mapped multiple times, both times within overlapping genes
    # g) a read that's spliced over two genes

    def test_partial_mapping(self):
        # handles a and b

        bam = MockBamFixed(
            [
                # matches geneA, partial -> 1 count
                {"strand": 1, "regions": [(10, 20)], "tags": {"NH": 1}}
            ]
        )
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 15,
                    "tes": 300,
                    "chr": "1",
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 300,
                    "tes": 15,
                    "chr": "1",
                },
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        count_strategy = anno_tag_counts.CounterStrategyWeightedStranded()
        interval_strategy = IntervalStrategyGene()
        lookup = count_strategy.count_reads(interval_strategy, genome, bam)
        assert lookup["FakeA"] == 1
        assert lookup["FakeB"] == 0

    def test_no_tag_raises(self):
        # handles c

        bam = MockBamFixed(
            [
                # same read, just a different alignment
                {"strand": 1, "regions": [(1010, 1020)], "tags": {}, "qname": "read0"}
            ]
        )
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 0,
                    "tes": 1200,
                    "chr": "1",
                }
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        count_strategy = anno_tag_counts.CounterStrategyWeightedStranded()
        interval_strategy = IntervalStrategyGene()

        with pytest.raises(KeyError):
            count_strategy.count_reads(interval_strategy, genome, bam)

        # self.assertEqual(lookup['FakeA'], 1)  # no tag -> count them both
        # self.assertEqual(lookup['FakeB'], 1)

    def test_unaccounted_reads_raise(self):

        bam = MockBamFixed(
            [
                # guess the second one newer shows up
                {
                    "strand": 1,
                    "regions": [(1010, 1020)],
                    "tags": {"NH": 2},
                    "qname": "read0",
                }
            ]
        )
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 1000,
                    "tes": 1200,
                    "chr": "1",
                }
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        count_strategy = anno_tag_counts.CounterStrategyWeightedStranded()
        interval_strategy = IntervalStrategyGene()

        with pytest.raises(ValueError):
            count_strategy.count_reads(interval_strategy, genome, bam)

    def test_one_too_many_raises_nh_1(self):

        bam = MockBamFixed(
            [
                # guess the second one newer shows up
                {
                    "strand": 1,
                    "regions": [(1010, 1020)],
                    "tags": {"NH": 1},
                    "qname": "read0",
                },
                # guess the second one newer shows up
                {
                    "strand": 1,
                    "regions": [(1010, 1020)],
                    "tags": {"NH": 1},
                    "qname": "read0",
                },
            ]
        )
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 1000,
                    "tes": 1200,
                    "chr": "1",
                }
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        count_strategy = anno_tag_counts.CounterStrategyWeightedStranded()
        interval_strategy = IntervalStrategyGene()

        def inner():
            count_strategy.count_reads(interval_strategy, genome, bam)

        try:
            inner
        except ValueError:
            self.fail("One too many was detected when it shouldn't have been possible")

    def one_too_many_raises_nh_larger_than_1(self):

        bam = MockBamFixed(
            [
                # guess the second one newer shows up
                {
                    "strand": 1,
                    "regions": [(1010, 1020)],
                    "tags": {"NH": 2},
                    "qname": "read0",
                },
                # guess the second one newer shows up
                {
                    "strand": 1,
                    "regions": [(1010, 1020)],
                    "tags": {"NH": 2},
                    "qname": "read0",
                },
                # guess the second one newer shows up
                {
                    "strand": 1,
                    "regions": [(1010, 1020)],
                    "tags": {"NH": 2},
                    "qname": "read0",
                },
            ]
        )
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 1000,
                    "tes": 1200,
                    "chr": "1",
                }
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        count_strategy = anno_tag_counts.CounterStrategyWeightedStranded()
        interval_strategy = IntervalStrategyGene()

        def inner():
            count_strategy.count_reads(interval_strategy, genome, bam)

        with pytest.raises(ValueError):
            inner()

    def test_read_mapped_twice_different_genes_with_tag(self):
        # handles c

        bam = MockBamFixed(
            [
                {
                    "strand": 1,
                    "regions": [(10, 20)],
                    "tags": {"NH": 2},
                    "qname": "read0",
                },
                # same read, just a different alignment
                {
                    "strand": 1,
                    "regions": [(1010, 1020)],
                    "tags": {"NH": 2},
                    "qname": "read0",
                },
            ]
        )
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 15,
                    "tes": 300,
                    "chr": "1",
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 1000,
                    "tes": 1200,
                    "chr": "1",
                },
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        count_strategy = anno_tag_counts.CounterStrategyWeightedStranded()
        interval_strategy = IntervalStrategyGene()
        lookup = count_strategy.count_reads(interval_strategy, genome, bam)
        assert lookup["FakeA"] == 0.5  #
        assert lookup["FakeB"] == 0.5

    def test_read_mapped_twice_to_the_same_gene(self):
        # handles c

        bam = MockBamFixed(
            [
                {
                    "strand": 1,
                    "regions": [(10, 20)],
                    "tags": {"NH": 2},
                    "qname": "read0",
                },
                # same read, just a different alignment, same gene!
                {
                    "strand": 1,
                    "regions": [(30, 40)],
                    "tags": {"NH": 2},
                    "qname": "read0",
                },
            ]
        )
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 15,
                    "tes": 300,
                    "chr": "1",
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 1000,
                    "tes": 1200,
                    "chr": "1",
                },
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        count_strategy = anno_tag_counts.CounterStrategyWeightedStranded()
        interval_strategy = IntervalStrategyGene()
        lookup = count_strategy.count_reads(interval_strategy, genome, bam)
        # how am I ever going to achieve this?
        assert lookup["FakeA"] == 1
        assert lookup["FakeB"] == 0

    def test_read_mapped_twice_once_outside(self):
        # handles c

        bam = MockBamFixed(
            [
                {
                    "strand": 1,
                    "regions": [(10, 20)],
                    "tags": {"NH": 2},
                    "qname": "read0",
                },
                {
                    "strand": 1,
                    "regions": [(5000, 5010)],
                    "tags": {"NH": 2},
                    "qname": "read0",
                },  # not in a gene!
            ]
        )
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 15,
                    "tes": 300,
                    "chr": "1",
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 1000,
                    "tes": 1200,
                    "chr": "1",
                },
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        count_strategy = anno_tag_counts.CounterStrategyWeightedStranded()
        interval_strategy = IntervalStrategyGene()
        lookup = count_strategy.count_reads(interval_strategy, genome, bam)
        assert lookup["FakeA"] == 1
        assert lookup["FakeB"] == 0

    def test_read_mapped_twice_to_the_same_gene_once_to_another(self):

        bam = MockBamFixed(
            [
                # repeated maps to the same gene count as one!
                {
                    "strand": 1,
                    "regions": [(10, 20)],
                    "tags": {"NH": 3},
                    "qname": "read0",
                },
                {
                    "strand": 1,
                    "regions": [(30, 40)],
                    "tags": {"NH": 3},
                    "qname": "read0",
                },
                {
                    "strand": 1,
                    "regions": [(1000, 1010)],
                    "tags": {"NH": 3},
                    "qname": "read0",
                },
            ]
        )
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 15,
                    "tes": 300,
                    "chr": "1",
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 1000,
                    "tes": 1200,
                    "chr": "1",
                },
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        count_strategy = anno_tag_counts.CounterStrategyWeightedStranded()
        interval_strategy = IntervalStrategyGene()
        lookup = count_strategy.count_reads(interval_strategy, genome, bam)
        assert lookup["FakeA"] == 0.5
        assert lookup["FakeB"] == 0.5

    def test_read_mapped_thrice(self):

        bam = MockBamFixed(
            [
                {
                    "strand": 1,
                    "regions": [(10, 20)],
                    "tags": {"NH": 3},
                    "qname": "read0",
                },
                {
                    "strand": 1,
                    "regions": [(1000, 1010)],
                    "tags": {"NH": 3},
                    "qname": "read0",
                },
                {
                    "strand": 1,
                    "regions": [(2000, 2040)],
                    "tags": {"NH": 3},
                    "qname": "read0",
                },
            ]
        )
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 15,
                    "tes": 300,
                    "chr": "1",
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 1000,
                    "tes": 1200,
                    "chr": "1",
                },
                {
                    "gene_stable_id": "FakeC",
                    "name": "C",
                    "tss": 2000,
                    "tes": 2020,
                    "chr": "1",
                },
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        count_strategy = anno_tag_counts.CounterStrategyWeightedStranded()
        interval_strategy = IntervalStrategyGene()
        lookup = count_strategy.count_reads(interval_strategy, genome, bam)
        assert lookup["FakeA"] == 1 / 3.0
        assert lookup["FakeB"] == 1 / 3.0
        assert lookup["FakeC"] == 1 / 3.0

    def test_two_genes_one_read(self):

        bam = MockBamFixed(
            [{"strand": 1, "regions": [(10, 20)], "tags": {"NH": 1}, "qname": "read0"}]
        )
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 15,
                    "tes": 300,
                    "chr": "1",
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 15,
                    "tes": 1200,
                    "chr": "1",
                },
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        count_strategy = anno_tag_counts.CounterStrategyWeightedStranded()
        interval_strategy = IntervalStrategyGene()
        lookup = count_strategy.count_reads(interval_strategy, genome, bam)
        assert lookup["FakeA"] == 0.5
        assert lookup["FakeB"] == 0.5

    def test_two_genes_one_read_mapped_multitimes(self):

        bam = MockBamFixed(
            [
                {
                    "strand": 1,
                    "regions": [(10, 20)],
                    "tags": {"NH": 3},
                    "qname": "read0",
                },
                {
                    "strand": 1,
                    "regions": [(30, 40)],
                    "tags": {"NH": 3},
                    "qname": "read0",
                },
                {
                    "strand": 1,
                    "regions": [(310, 320)],
                    "tags": {"NH": 3},
                    "qname": "read0",
                },
            ]
        )
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 15,
                    "tes": 300,
                    "chr": "1",
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 15,
                    "tes": 1200,
                    "chr": "1",
                },
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        count_strategy = anno_tag_counts.CounterStrategyWeightedStranded()
        interval_strategy = IntervalStrategyGene()
        lookup = count_strategy.count_reads(interval_strategy, genome, bam)
        assert lookup["FakeA"] == 0.5
        assert lookup["FakeB"] == 0.5

    def test_two_genes_one_read_mapped_multitimes2(self):

        bam = MockBamFixed(
            [
                {
                    "strand": 1,
                    "regions": [(10, 20)],
                    "tags": {"NH": 4},
                    "qname": "read0",
                },  # FakeA + FakeB
                {
                    "strand": 1,
                    "regions": [(30, 40)],
                    "tags": {"NH": 4},
                    "qname": "read0",
                },  # FakeA + FakeB
                {
                    "strand": 1,
                    "regions": [(310, 320)],
                    "tags": {"NH": 4},
                    "qname": "read0",
                },  # Fake B
                {
                    "strand": 1,
                    "regions": [(315, 320)],
                    "tags": {"NH": 4},
                    "qname": "read0",
                },  # Fake B
                # different read...
                {
                    "strand": 1,
                    "regions": [(315, 320)],
                    "tags": {"NH": 1},
                    "qname": "read1",
                },
            ]
        )
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 15,
                    "tes": 300,
                    "chr": "1",
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 15,
                    "tes": 1200,
                    "chr": "1",
                },
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        count_strategy = anno_tag_counts.CounterStrategyWeightedStranded()
        interval_strategy = IntervalStrategyGene()
        lookup = count_strategy.count_reads(interval_strategy, genome, bam)
        assert lookup["FakeA"] == 0.5  #
        assert lookup["FakeB"] == 0.5 + 1

    def test_reads_reverse(self):

        bam = MockBamFixed(
            [
                {
                    "strand": 1,
                    "regions": [(10, 20)],
                    "tags": {"NH": 2},
                    "qname": "read0",
                },  # FakeA + FakeB
                # FakeA + FakeB
                {
                    "strand": -1,
                    "regions": [(500, 505), (1150, 1155)],
                    "tags": {"NH": 2},
                    "qname": "read0",
                },
                # different read...
                {
                    "strand": -1,
                    "regions": [(1000, 1010)],
                    "tags": {"NH": 1},
                    "qname": "read1",
                },
            ]
        )
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 15,
                    "tes": 300,
                    "chr": "1",
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 1200,
                    "tes": 1000,
                    "chr": "1",
                },
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        count_strategy = anno_tag_counts.CounterStrategyWeightedStranded()
        interval_strategy = IntervalStrategyGene()
        lookup = count_strategy.count_reads(interval_strategy, genome, bam)
        assert lookup["FakeA"] == 0.5  #
        assert lookup["FakeB"] == 0.5 + 1

    def test_first_read_does_not_hit_a_gene(self):

        bam = MockBamFixed(
            [
                # nothing...
                {"strand": 1, "regions": [(0, 1)], "tags": {"NH": 1}, "qname": "read1"},
                {
                    "strand": 1,
                    "regions": [(10, 20)],
                    "tags": {"NH": 2},
                    "qname": "read0",
                },  # FakeA + FakeB
                # FakeA + FakeB
                {
                    "strand": -1,
                    "regions": [(500, 505), (1150, 1155)],
                    "tags": {"NH": 2},
                    "qname": "read0",
                },
                # different read...
                {
                    "strand": -1,
                    "regions": [(1000, 1010)],
                    "tags": {"NH": 1},
                    "qname": "read1",
                },
            ]
        )
        genes = pd.DataFrame(
            [
                {
                    "gene_stable_id": "FakeA",
                    "name": "A",
                    "tss": 15,
                    "tes": 300,
                    "chr": "1",
                },
                {
                    "gene_stable_id": "FakeB",
                    "name": "B",
                    "tss": 1200,
                    "tes": 1000,
                    "chr": "1",
                },
            ]
        )
        transcripts = pd.DataFrame({"name": [], "chr": [], "exons": [], "strand": []})
        genome = MockGenome(genes, transcripts, {"1": 10000})
        count_strategy = anno_tag_counts.CounterStrategyWeightedStranded()
        interval_strategy = IntervalStrategyGene()
        lookup = count_strategy.count_reads(interval_strategy, genome, bam)
        assert lookup["FakeA"] == 0.5  #
        assert lookup["FakeB"] == 0.5 + 1

    def test_end_to_end(self):
        pass


@pytest.mark.usefixtures("new_pipegraph")
class TestQC:
    def test_biotypes_plot(self):
        from mbf_sampledata import get_human_22_fake_genome, get_sample_path
        from mbf_align.lanes import AlignedSample
        from mbf_genomics.genes.anno_tag_counts import GeneStranded
        from mbf_qualitycontrol import do_qc
        from mbf_qualitycontrol.testing import assert_image_equal

        genome = get_human_22_fake_genome()
        aligned = AlignedSample(
            "rnaseq22", get_sample_path("mbf_align/rnaseq_spliced_chr22.bam"),
            is_paired=False, vid=None, genome=genome
        )
        genes = Genes(genome)
        tc = GeneStranded(aligned)
        genes += tc
        jobs = do_qc(lambda name: "reads_per_biotype_" in str(name))
        assert len(jobs) == 1
        run_pipegraph()
        p = aligned.result_dir / f"reads_per_biotype_{genes.name}.png"
        assert_image_equal(p)
