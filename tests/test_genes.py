from .test_common import new_pipeline, run_pipeline, DummyGenome, dummyGenome, ConstantAnnotator, functional_available
import unittest
import genomics.regions as regions
import genomics.genes as genes
import numpy
import pandas as pd
import fileformats
import pytest



class GenesLoadingTests(unittest.TestCase):

    def setUp(self):
        new_pipeline()

    def test_basic_loading_from_genome(self):
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ]))
        g = genes.Genes(genome)
        g.load()
        run_pipeline()
        assert len(g.df) == 3
        assert (g.df['stable_id'] == ['fake2', 'fake1', 'fake3']).all()

    def test_alternative_loading_raises_on_non_df(self):
        def inner():
            g = genes.Genes(dummyGenome, lambda: None, 'myname')
            g.load()
            run_pipeline()
        with pytest.raises(ValueError):
            inner()

    def test_alternative_loading_raises_on_missing_column(self):
        df = pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ])

        def inner_tss():
            new_pipeline()
            df2 = df.copy()
            df2 = df2.drop('tss', axis=1)
            g = genes.Genes(dummyGenome, lambda: df2, name='sha')
            g.load()
            run_pipeline()

        def inner_chr():
            new_pipeline()
            df2 = df.copy()
            df2 = df2.drop('chr', axis=1)
            g = genes.Genes(dummyGenome, lambda: df2, name='shu')
            g.load()
            run_pipeline()

        def inner_tes():
            new_pipeline()
            df2 = df.copy()
            df2 = df2.drop('tes', axis=1)
            g = genes.Genes(dummyGenome, lambda: df2, name='shi')
            g.load()
            run_pipeline()
        with pytest.raises(ValueError):
            inner_tss()
        with pytest.raises(ValueError):
            inner_tes()
        with pytest.raises(ValueError):
            inner_chr()

    def test_alternative_loading_raises_on_missing_name(self):
        df = pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ])

        def inner():
            g = genes.Genes(dummyGenome, lambda: df)
        with pytest.raises(ValueError):
            inner()

    def test_alternative_loading_raises_on_invalid_chromosome(self):
        new_pipeline(quiet=True)
        df = pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1b', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ])

        def inner():
            g = genes.Genes(dummyGenome, lambda: df, name='shu')
            g.load()
            run_pipeline()
        with pytest.raises(ValueError):
            inner()

    def test_alternative_loading_raises_on_non_int_tss(self):
        df = pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000.5, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ])

        def inner():
            g = genes.Genes(dummyGenome, lambda: df, name='shu')
            g.load()
            run_pipeline()
        with pytest.raises(ValueError):
            inner()

    def test_alternative_loading_raises_on_non_int_tes(self):
        df = pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': '', 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ])

        def inner():
            g = genes.Genes(dummyGenome, lambda: df, name='shu')
            g.load()
            run_pipeline()
        with pytest.raises(ValueError):
            inner()

    def test_do_load_only_happens_once(self):
        df = pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ])
        new_pipeline()
        counter = [0]

        def load():
            counter[0] += 1
            return df
        g = genes.Genes(dummyGenome, load, name='shu')
        assert counter[0] == 0
        g.do_load()
        assert counter[0] == 1
        g.do_load()
        assert counter[0] == 1  # still one

    def test_filtering_away_works(self):
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ]))
        g = genes.Genes(genome)
        filtered = g.filter('nogenes', lambda df: df['chr'] == '4')
        filtered.load()
        run_pipeline()
        assert len(filtered.df) == 0
        assert 'start' in filtered.df.columns
        assert 'stop' in filtered.df.columns
        assert 'tss' in filtered.df.columns
        assert 'tes' in filtered.df.columns
        assert 'stable_id' in filtered.df.columns

    def test_annotators_are_kept_on_filtering(self):
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ]))
        g = genes.Genes(genome)
        ca = ConstantAnnotator()
        g.add_annotator(ca)
        filtered = g.filter('nogenes', lambda df: df['chr'] == '4')
        assert filtered.has_annotator(ca)

    def test_loading_from_genome_is_singletonic(self):
        genesA = genes.Genes(dummyGenome)
        genesB = genes.Genes(dummyGenome)
        assert genesA is genesB
        filterA = genesA.filter('fa', select_top_k=10)
        filterAa = genesA.filter('faa', select_top_k=10)
        filterB = genesB.filter('fab', select_top_k=10)
        assert not (filterA is genesA)
        assert not (filterAa is filterA)
        assert not (filterAa is filterB)

    def test_homology(self):
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ]))
        genomeB = DummyGenome()
        genomeB.name = 'genomeB'
        a = genes.Genes(genome)
        b = a.homology(genomeB, False)
        b.load()
        c = a.homology(genomeB, True)
        d = c.homology(genome, True)
        c.load()
        d.load()
        run_pipeline()
        assert len(b) == 3  # homo1, homo2a, homo2b
        assert len(c) == 1  # homo1
        assert len(d) == 1  # fake1
        assert (b.df['stable_id'] == ['homo2a', 'homo1', 'homo2b']).all()
        assert (c.df['stable_id'] == ['homo1']).all()
        assert (d.df['stable_id'] == ['fake1']).all()

    def test_homology_works_on_empty_sets(self):
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ]))
        genomeB = DummyGenome()
        genomeB.name = 'genomeB'
        g = genes.Genes(genome)
        filtered = g.filter('nogenes', lambda df: df['chr'] == '4')
        hom = filtered.homology(genomeB, True)
        hom2 = filtered.homology(genomeB, False)
        run_pipeline()
        assert len(hom.df) == 0
        assert len(hom2.df) == 0

    def test_homology_conserves_annotators(self):
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ]))
        genomeB = DummyGenome()
        genomeB.name = 'genomeB'
        a = genes.Genes(genome)
        con = ConstantAnnotator()
        a.add_annotator(con)
        b = a.homology(genomeB, False)
        assert b.has_annotator(con)

    def test_filtering_returns_genes(self):
        g = genes.Genes(DummyGenome())
        on_chr_1 = g.filter('on_1', lambda df: df['chr'] == '1')
        assert g.__class__ == on_chr_1.__class__

    def test_overlap_genes_requires_two_genes(self):
        a = genes.Genes(dummyGenome)

        def sample_data():
            return pd.DataFrame({"chr": ['1'], 'start': [1000], 'stop': [1100]})
        b = regions.GenomicRegions('sha', sample_data, [], dummyGenome)
        a.load()
        b.load()
        run_pipeline()

        def inner():
            a.overlap_genes(b)
        with pytest.raises(ValueError):
            inner()

    def test_overlap_genes_raises_on_unequal_genomes(self):
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ]))
        genomeB = DummyGenome()
        genomeB.name = 'genomeB'
        a = genes.Genes(genome)
        b = genes.Genes(genomeB)

        def inner():
            a.overlap_genes(b)
        with pytest.raises(ValueError):
            inner()

    def test_overlap(self):
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ]))
        g = genes.Genes(genome)
        on_chr_1 = g.filter('on_1', lambda df: df['chr'] == '1')
        on_chr_2 = g.filter('on_2', lambda df: df['chr'] == '2')
        one = g.filter('one', lambda df: df['stable_id'] == 'fake1')
        on_chr_1.load()
        on_chr_2.load()
        one.load()
        run_pipeline()
        assert len(on_chr_1) == 2
        assert len(on_chr_2) == 1
        assert len(one) == 1
        assert g.overlap_genes(on_chr_1) == len(on_chr_1)
        assert on_chr_1.overlap_genes(g) == len(on_chr_1)
        assert on_chr_1.overlap_genes(on_chr_1) == len(on_chr_1)
        assert g.overlap_genes(on_chr_2) == len(on_chr_2)
        assert on_chr_2.overlap_genes(g) == len(on_chr_2)
        assert on_chr_2.overlap_genes(on_chr_2) == len(on_chr_2)
        assert g.overlap_genes(one) == len(one)
        assert one.overlap_genes(g) == len(one)
        assert one.overlap_genes(one) == len(one)

        assert on_chr_1.overlap_genes(one) == 1
        assert one.overlap_genes(on_chr_1) == 1

        assert on_chr_1.overlap_genes(on_chr_2) == 0
        assert on_chr_2.overlap_genes(on_chr_1) == 0

    def test_get_tss_regions(self):
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ]))
        g = genes.Genes(genome)
        tss = g.get_tss_regions()
        tss.load()
        run_pipeline()
        assert len(tss.df) == 3
        assert (tss.df['start'] == [5000, 5400, 5400]).all()
        assert (tss.df['stop'] == tss.df['start'] + 1).all()
        assert (tss.df['chr'] == ['1', '1', '2']).all()

    def test_get_tes_regions(self):
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 3000, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ]))
        g = genes.Genes(genome)
        tes = g.get_tes_regions()
        tes.load()
        run_pipeline()
        assert len(tes.df) == 2
        assert (tes.df['start'] == [4900, 4900]).all()
        assert (tes.df['stop'] == tes.df['start'] + 1).all()
        assert (tes.df['chr'] == ['1', '2']).all()

    def test_get_exons_regions_overlapping(self):
        new_pipeline(quiet=False)
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1, 'tss': 3000,
                    'tes': 4900, 'description': 'bla', 'name': 'bla1'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1, 'tss': 5400,
                    'tes': 4900, 'description': 'bla', 'name': 'bla2'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1, 'tss': 5400,
                    'tes': 4900, 'description': 'bla', 'name': 'bla3'},
            ]),
            #{transcript_stable_id, gene_stable_id, strand, start, end, exons},
            all_transcripts=pd.DataFrame({
                'transcript_stable_id': ['trans1a', 'trans1b', 'trans2', 'trans3'],
                'gene_stable_id': ['fake1', 'fake1', 'fake2', 'fake3'],
                'chr': ['1', '1', '1', '2'],
                'strand': [1, 1, -1, -1],
                'start': [3100, 3000, 4910, 4900],
                'stop': [4900, 4000, 5400, 5400],
                'exons': [[(3100, 4900, 0)], [(3000, 3500, 0), (3300, 3330, 0), (3750, 4000, 0)], [(4910, 5000, 0), (5100, 5400, 0)], [(4900, 5400, 0)]]
            }
        ))
        g = genes.Genes(genome)
        exons = genome.get_exon_regions_overlapping()
        print('called')
        exons.load()
        print('loaded')
        run_pipeline()
        assert (exons.df[
            'start'] == [3000, 3100, 3300, 3750, 4910, 5100, 4900]).all()
        assert (exons.df[
            'stop'] == [3500, 4900, 3330, 4000, 5000, 5400, 5400]).all()
        assert (exons.df['chr'] == numpy.array(
            ['1', '1', '1', '1', '1', '1', '2'])).all()

    def test_get_exons_regions_merging(self):
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1, 'tss': 3000,
                    'tes': 4900, 'description': 'bla', 'name': 'bla1'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1, 'tss': 5400,
                    'tes': 4900, 'description': 'bla', 'name': 'bla2'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1, 'tss': 5400,
                    'tes': 4900, 'description': 'bla', 'name': 'bla3'},
            ]),
            #{transcript_stable_id, gene_stable_id, strand, start, end, exons},
            all_transcripts=pd.DataFrame({
                'transcript_stable_id': ['trans1a', 'trans1b', 'trans2', 'trans3'],
                'gene_stable_id': ['fake1', 'fake1', 'fake2', 'fake3'],
                'chr': ['1', '1', '1', '2'],
                'strand': [1, 1, -1, -1],
                'start': [3100, 3000, 4910, 4900],
                'stop': [4900, 4000, 5400, 5400],
                'exons': [[(3100, 4900, 0)], [(3000, 3500, 0), (3300, 3330, 0), (3750, 4000, 0)], [(4910, 5000, 0), (5100, 5400, 0)], [(4900, 5400, 0)]]
            }
        ))
        g = genes.Genes(genome)
        exons = genome.get_exon_regions_merged()
        exons.load()
        run_pipeline()
        assert (exons.df['start'] == [3000, 4910, 5100, 4900]).all()
        assert (exons.df['stop'] == [4900, 5000, 5400, 5400]).all()
        assert (exons.df['chr'] == ['1', '1', '1', '2']).all()

    def test_get_intron_regions(self):
        new_pipeline(False)
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 3000, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ]),
            #{transcript_stable_id, gene_stable_id, strand, start, end, exons},
            all_transcripts=pd.DataFrame({
                'transcript_stable_id': ['trans1a', 'trans1b', 'trans2', 'trans3'],
                'gene_stable_id': ['fake1', 'fake1', 'fake2', 'fake3'],
                'chr': ['1', '1', '1', '2'],
                'strand': [1, 1, -1, -1],
                'start': [3100, 3000, 4900, 4900],
                'stop': [4900, 4000, 5400, 5400],
                'exons': [[(3100, 4900, 0)], [(3000, 3500, 0), (3750, 4000, 0)], [(4900, 5000, 0), (5100, 5400, 0)], [(4900, 5400, 0)]]
            }
        ))
        g = genes.Genes(genome)
        introns = g.get_intron_regions()
        introns.load()
        run_pipeline()
        assert (introns.df['start'] == [3000, 3500, 4000, 5000]).all()
        assert (introns.df['stop'] == [3100, 3750, 4900, 5100]).all()
        # no intronic region on chr 2
        assert (introns.df['chr'] == ['1', '1', '1', '1']).all()

    def test_intronify_more_complex(self):
        transcript = {u'chr': '2R',
                      u'exons': [(14243005, 14244766, 0),
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
                                 (14166570, 14167681, 0)],
                      u'gene_stable_id': 'FBgn0010575',
                      u'start': 14166570,
                      u'stop': 14244766,
                      u'strand': -1,
                      u'transcript_stable_id': 'FBtr0301547'}
        gene = {u'biotype': 'protein_coding',
                u'chr': '2R',
                u'description': 'CG5580 [Source:FlyBase;GeneId:FBgn0010575]',
                u'name': 'sbb',
                u'stable_id': 'FBgn0010575',
                u'strand': -1,
                u'tes': 14166570,
                u'tss': 14244766}

        g = genes.Genes(dummyGenome)
        introns = g._intron_intervals(transcript, gene)
        assert (numpy.array(introns) == [
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
            ]).all()

    def test_intron_intervals_raises_on_inverted(self):
        transcript = {u'chr': '2R',
                      u'exons': [(14243005, 14244766, 0),
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
                                 (14166570, 14167681, 0)],
                      u'gene_stable_id': 'FBgn0010575',
                      u'start': 14166570,
                      u'stop': 14244766,
                      u'strand': -1,
                      u'transcript_stable_id': 'FBtr0301547'}
        gene = {u'biotype': 'protein_coding',
                u'chr': '2R',
                u'description': 'CG5580 [Source:FlyBase;GeneId:FBgn0010575]',
                u'name': 'sbb',
                u'stable_id': 'FBgn0010575',
                u'strand': -1,
                u'tes': 14166570,
                u'tss': 14244766}
        g = genes.Genes(dummyGenome)

        def inner():
            introns = g._intron_intervals(transcript, gene)
        with pytest.raises(ValueError):
            inner()

    def test_get_gene_exons(self):
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1, 'tss': 3000,
                    'tes': 4900, 'description': 'bla', 'name': 'bla1'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1, 'tss': 5400,
                    'tes': 4900, 'description': 'bla', 'name': 'bla2'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1, 'tss': 5400,
                    'tes': 4900, 'description': 'bla', 'name': 'bla3'},
            ]),
            #{transcript_stable_id, gene_stable_id, strand, start, end, exons},
            all_transcripts=pd.DataFrame({
                'transcript_stable_id': ['trans1a', 'trans1b', 'trans2', 'trans3'],
                'gene_stable_id': ['fake1', 'fake1', 'fake2', 'fake3'],
                'chr': ['1', '1', '1', '2'],
                'strand': [1, 1, -1, -1],
                'start': [3100, 3000, 4910, 4900],
                'stop': [4900, 4000, 5400, 5400],
                'exons': [[(3100, 4900, 0)], [(3000, 3500, 0), (3300, 3330, 0), (3750, 4000, 0)], [(4910, 5000, 0), (5100, 5400, 0)], [(4900, 5400, 0)]]
            }
        ))
        g = genes.Genes(genome)
        g.do_load()
        two = g.get_gene_exons('fake2')
        assert (two['start'] == [4910, 5100]).all()
        assert (two['stop'] == [5000, 5400]).all()

    def test_get_gene_introns(self):
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1, 'tss': 3000,
                    'tes': 4900, 'description': 'bla', 'name': 'bla1'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1, 'tss': 5500,
                    'tes': 4900, 'description': 'bla', 'name': 'bla2'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1, 'tss': 5400,
                    'tes': 4900, 'description': 'bla', 'name': 'bla3'},
            ]),
            #{transcript_stable_id, gene_stable_id, strand, start, end, exons},
            all_transcripts=pd.DataFrame({
                'transcript_stable_id': ['trans1a', 'trans1b', 'trans2', 'trans3'],
                'gene_stable_id': ['fake1', 'fake1', 'fake2', 'fake3'],
                'chr': ['1', '1', '1', '2'],
                'strand': [1, 1, -1, -1],
                'start': [3100, 3000, 4910, 4900],
                'stop': [4900, 4000, 5400, 5400],
                'exons': [[(3100, 4900, 0)], [(3000, 3500, 0), (3300, 3330, 0), (3750, 4000, 0)], [(4910, 5000, 0), (5100, 5400, 0)], [(4900, 5400, 0)]]
            }
        ))
        g = genes.Genes(genome)
        g.do_load()
        one = g.get_gene_introns('fake1')
        print('one', one)
        assert len(one) == 0

        two = g.get_gene_introns('fake2')
        assert (two['start'] == [4900, 5000, 5400]).all()
        assert (two['stop'] == [4910, 5100, 5500]).all()


class GenesTests(unittest.TestCase):

    def test_write_bed(self):
        new_pipeline()
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ]))
        g = genes.Genes(genome)
        sample_filename = 'cache/genes.bed'
        g.write_bed(sample_filename)
        run_pipeline()
        assert len(g.df) > 0
        read = fileformats.read_bed(sample_filename)
        assert len(read) == len(g.df)
        assert read[0].refseq == b'1'
        assert read[1].refseq == b'1'
        assert read[2].refseq == b'2'
        assert read[0].position == 4900
        assert read[1].position == 5000
        assert read[2].position == 4900
        assert read[0].length == 500
        assert read[1].length == 500
        assert read[2].length == 500
        assert read[0].name == b'fake2'
        assert read[1].name == b'fake1'
        assert read[2].name == b'fake3'

    def test_write_bed_auto_filename(self):
        new_pipeline()
        genome = DummyGenome(pd.DataFrame(
            [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ]))
        g = genes.Genes(genome)
        sample_filename = g.write_bed().job_id  # the jobs output filename
        run_pipeline()
        assert len(g.df) > 0
        read = fileformats.read_bed(sample_filename)
        assert len(read) == len(g.df)
        assert read[0].refseq == b'1'
        assert read[1].refseq == b'1'
        assert read[2].refseq == b'2'
        assert read[0].position == 4900
        assert read[1].position == 5000
        assert read[2].position == 4900
        assert read[0].length == 500
        assert read[1].length == 500
        assert read[2].length == 500
        assert read[0].name == b'fake2'
        assert read[1].name == b'fake1'
        assert read[2].name == b'fake3'

    # def test_annotation_keeps_row_names(self):
        # new_pipeline()
        # g = genes.Genes(dummyGenome)
        # g.do_load()
        # row_names = g.df.row_names
        # g.annotate()
        # run_pipeline()
        # self.assertTrue((row_names == g.df.row_names).all())


if functional_available():
    class HasPeakInRegulatoryRegionTests(unittest.TestCase):

        def setUp(self):
            new_pipeline(verbose=False)
            genome = DummyGenome(pd.DataFrame(
                [
                    {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                        'tss': 5000, 'tes': 5500, 'description': 'bla'},
                    {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                        'tss': 5400, 'tes': 4900, 'description': 'bla'},
                    {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                        'tss': 5400, 'tes': 4900, 'description': 'bla'},
                ]))
            self.genes = genes.Genes(genome)

        def test_simple(self):
            new_pipeline()
            genome = DummyGenome(pd.DataFrame(
                [
                    {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                        'tss': 5000, 'tes': 5500, 'description': 'bla'},
                    {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                        'tss': 5400, 'tes': 4900, 'description': 'bla'},
                    {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                        'tss': 5400, 'tes': 4900, 'description': 'bla'},
                ]))
            # so regulatory regionsa re from 0...6000, 5400..100000 and 0...200000,
            # I guess...
            g = genes.Genes(genome)
            # s
            gr_in_both = regions.GenomicRegions("in_both", lambda: pd.DataFrame(
                {'chr': ['1'], 'start': [5500], 'stop': [5600]}), [], genome)
            gr_in_fake_1 = regions.GenomicRegions("gr_in_fake_1", lambda: pd.DataFrame(
                {'chr': ['1'], 'start': [3500], 'stop': [3600]}), [], genome)
            anno_both = genes.annotators.HasPeakInRegulatoryRegion(gr_in_both)
            anno_in_fake_1 = genes.annotators.HasPeakInRegulatoryRegion(
                gr_in_fake_1, 'infake1')
            g.add_annotator(anno_both)
            g.add_annotator(anno_in_fake_1)
            g.annotate()
            run_pipeline()
            # self.assertTrue(g.df.row_names is not None)
            assert anno_both.column_name in g.df.columns
            assert "infake1" in g.df.columns
            gdf = g.df.set_index('stable_id')
            assert gdf.ix['fake1'][anno_both.column_name] == True
            assert gdf.ix['fake2'][anno_both.column_name] == True
            assert gdf.ix['fake3'][anno_both.column_name] == False

            assert gdf.ix['fake1']['infake1'] == True
            assert gdf.ix['fake2']['infake1'] == False
            assert gdf.ix['fake3']['infake1'] == False

        def test_annotation(self):
            def sample_data():
                return pd.DataFrame({"chr": ['1'], 'start': [1100], 'stop': [1200]})
            a = regions.GenomicRegions('sha', sample_data, [], self.genes.genome)
            b = regions.GenomicRegions('shb', sample_data, [], self.genes.genome)
            anno = genes.annotators.HasPeakInRegulatoryRegion(
                a, 'my favourite column')
            anno2 = genes.annotators.HasPeakInRegulatoryRegion(b)
            assert anno.column_name == 'my favourite column'
            self.genes.add_annotator(anno)
            self.genes.add_annotator(anno2)
            self.genes.annotate()
            run_pipeline()
            assert 'my favourite column' in self.genes.df.columns
            # print self.genes.df
            assert (self.genes.df[
                'my favourite column'] == [False, True, False]).all()
            assert (self.genes.df[anno2.column_name] == [False, True, False]).all()

        def test_bannotation(self):
            def sample_data():
                return pd.DataFrame({"chr": ['1'], 'start': [4500], 'stop': [4600]})
            a = regions.GenomicRegions('sha', sample_data, [], self.genes.genome)

            anno = genes.annotators.HasPeakInRegulatoryRegion(
                a, 'my favourite column')
            assert anno.column_name == 'my favourite column'
            self.genes.add_annotator(anno)
            self.genes.annotate()
            run_pipeline()
            assert 'my favourite column' in self.genes.df.columns
            assert (self.genes.df[
                'my favourite column'] == [True, True, False]).all()


    class PeakInRegulatoryRegionTests(unittest.TestCase):

        def setUp(self):
            new_pipeline(verbose=False)
            genome = DummyGenome(pd.DataFrame(
                [
                    {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                        'tss': 5000, 'tes': 5500, 'description': 'bla'},
                    {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                        'tss': 5400, 'tes': 4900, 'description': 'bla'},
                    {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                        'tss': 5400, 'tes': 4900, 'description': 'bla'},
                ]))
            self.genes = genes.Genes(genome)

        def test_annotation(self):
            def sample_data():
                return pd.DataFrame({"chr": ['1'], 'start': [1100], 'stop': [1200]})
            a = regions.GenomicRegions('sha', sample_data, [], self.genes.genome)
            b = regions.GenomicRegions('shb', sample_data, [], self.genes.genome)
            anno = genes.annotators.PeakInRegulatoryRegion(
                a, 'my favourite column')
            anno2 = genes.annotators.PeakInRegulatoryRegion(b)
            assert anno.column_name == 'my favourite column'
            self.genes.add_annotator(anno)
            self.genes.add_annotator(anno2)
            self.genes.annotate()
            run_pipeline()
            assert 'my favourite column' in self.genes.df.columns
            assert (self.genes.df[
                'my favourite column'] == ["", "1100...1200", ""]).all()
            assert (self.genes.df[
                anno2.column_name] == ["", "1100...1200", ""]).all()


class HasPeakCloseToTSSTests(unittest.TestCase):

    def setUp(self):
        new_pipeline(verbose=False)
        gens = [
                {'stable_id': 'fake1', 'chr': '1', 'strand': 1,
                    'tss': 5000, 'tes': 5500, 'description': 'bla'},
                {'stable_id': 'fake2', 'chr': '1', 'strand': -1,
                    'tss': 6400, 'tes': 5900, 'description': 'bla'},
                {'stable_id': 'fake3', 'chr': '2', 'strand': -1,
                    'tss': 5400, 'tes': 4900, 'description': 'bla'},
            ]
        transcripts = []
        for row in gens:
            transcripts.append({
                'transcript_stable_id': 'tr-%i' % len(transcripts),
                'gene_stable_id': row['stable_id'],
                'start': min(row['tss'], row['tes']),
                'stop': max(row['tss'], row['tes']),
                'strand': row['strand'],
            })
        genome = DummyGenome(pd.DataFrame(
            gens),
            pd.DataFrame(transcripts)
        )
        self.genes = genes.Genes(genome)

    def test_assoc_any_tss_raises_on_negative(self):
        def a():
            genes.peak_annotators.Assoc_AnyTSS(-100, 100)
        def b():
            genes.peak_annotators.Assoc_AnyTSS(100, -100)
        with pytest.raises(ValueError):
            a()
        with pytest.raises(ValueError):
            b()
        # must not raise
        genes.peak_annotators.Assoc_AnyTSS(100, 100)
        genes.peak_annotators.Assoc_AnyTSS(100, 0)
        genes.peak_annotators.Assoc_AnyTSS(0, 100)



    def test_annotation(self):
        def sample_data():
            return pd.DataFrame({"chr": ['1', '2'], 'start': [4900, 5410], 'stop': [4950, 5460]})
        a = regions.GenomicRegions('sha', sample_data, [], self.genes.genome)
        b = regions.GenomicRegions('shb', sample_data, [], self.genes.genome)
        anno = genes.peak_annotators.HasPeak(a, genes.peak_annotators.Assoc_AnyTSS(100, 100))
        anno2 = genes.peak_annotators.HasPeak(a, genes.peak_annotators.Assoc_AnyTSS(49, 49))
        anno3 = genes.peak_annotators.HasPeak(b, genes.peak_annotators.Assoc_AnyTSS(100, 100), 'colSHU')
        assert anno3.column_name == 'colSHU'
        self.genes.add_annotator(anno)
        self.genes.add_annotator(anno2)
        self.genes.add_annotator(anno3)
        self.genes.annotate()
        run_pipeline()
        print(self.genes.df)
        assert 'colSHU' in self.genes.df.columns
        assert (self.genes.df[anno.column_name] == [True, False, True]).all()
        assert (self.genes.df[anno2.column_name] == [False, False, True]).all()
        assert (self.genes.df[
            anno.column_name] == self.genes.df[anno3.column_name]).all()

if __name__ == '__main__':
    unittest.main()
