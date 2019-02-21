import pytest
import os
import numpy as np
import pandas as pd
import scipy.stats
import pypipegraph as ppg
from pathlib import Path

from mbf_genomics.motifs import (
    PWMMotif_FromText,
    MisMatchMotif,
    DifferentialMisMatchMotif,
    denovo,
    sources,
    PWMMotif,
)
from mbf_genomics.motifs.motifs import (
    completeReverseIupac,
    reverse_complement,
    MotifBase,
    hamming_distance,
)
import mbf_genomics.sequences as sequences

from mbf_genomes.example_genomes import get_Candidatus_carsonella_ruddii_pv

meme_available = False  # TODO
dummy_genome = get_Candidatus_carsonella_ruddii_pv()

bg = dummy_genome


@pytest.mark.usefixtures("new_pipegraph")
class TestMotifBase:
    def test_scan_is_abstract(self):
        motif = MotifBase("test")

        with pytest.raises(NotImplementedError):
            motif.scan(None, None, None)

    def test_roc_auc(self):
        motif = PWMMotif_FromText("Test", "AGGTCA", bg)
        foreground = ["AGGTCA", "AGGTTA"]
        background = ["AGGTGA", "AGGGAAAAA"]
        assert (
            round(abs(motif.calculate_roc_auc(foreground, background) - 0.75), 7) == 0
        )

    def test_roc_auc_without_background(self):
        motif = PWMMotif_FromText("Test", "AGGTCA", bg)
        foreground = ["AGGTCA", "AGGTTA"]

        with pytest.raises(ZeroDivisionError):
            assert round(abs(motif.calculate_roc_auc(foreground, []) - 0.75), 7) == 0

    def test_roc_auc_pre_scored(self):
        motif = PWMMotif_FromText("Test", "AGGTCA", bg)
        foreground = ["AGGTCA", "AGGTTA"]
        background = ["AGGTGA", "AGGGAAAAA"]
        assert (
            round(
                abs(
                    motif.calculate_roc_auc(foreground, background, ([10, 9], [9, 0]))
                    - 0.75
                ),
                7,
            )
            == 0
        )

    def test_mncp(self):
        motif = PWMMotif_FromText("Test", "AGGTCA", bg)
        foreground = ["AGGTCA", "AGGTTA"]
        background = ["AGGTGA", "AGGGAAAAA"]
        assert round(abs(motif.calculate_mncp(foreground, background) - 1.5), 7) == 0

    def test_roc_auc_and_mncp(self):
        motif = PWMMotif_FromText("Test", "AGGTCA", bg)
        foreground = ["AGGTCA", "AGGTTA"]
        background = ["AGGTGA", "AGGGAAAAA"]
        roc, mncp = motif.calculate_roc_auc_and_mncp(foreground, background)
        assert round(abs(roc - 0.75), 7) == 0
        assert round(abs(mncp - 1.5), 7) == 0

    def test_plot_roc_returns_plotjob(self):
        np.random.seed(500)
        motif = PWMMotif(
            "Test",
            [
                {"A": 10, "C": 0, "G": 0, "T": 90},
                {"A": 30, "C": 0, "G": 30, "T": 40},
                {"A": 0, "C": 1, "G": 99, "T": 0},
                {"A": 0, "C": 12, "G": 12, "T": 76},
                {"A": 24, "C": 0, "G": 0, "T": 76},
                {"A": 33, "C": 33, "G": 34, "T": 0},
                {"A": 25, "C": 25, "G": 50, "T": 0},
                {"A": 40, "C": 30, "G": 20, "T": 10},
                {"A": 0, "C": 0, "G": 0, "T": 100},
            ],
            bg,
        )
        np.random.seed(500)
        foreground = motif.bogus_kmers(200)
        background = motif.bogus_kmers(200)
        pj = motif.plot_roc(
            foreground, background, "cache/shu_test_plot_returns_plotjob.png"
        )
        assert isinstance(pj, ppg.PlotJob)
        assert not os.path.exists("cache/shu_test_plot_returns_plotjob.png")
        ppg.run_pipegraph()
        assert os.path.exists("cache/shu_test_plot_returns_plotjob.png")

    def test_plot_roc_draws_something(self):
        motif = PWMMotif_FromText("Test", "AGGTCA", bg)
        foreground = ["AGGTCA", "AGGTTA"]
        background = ["AGGTGA", "AGGGAAAAA"]
        try:
            os.unlink("cache/shu.png")
        except OSError:
            pass
        pj = motif.plot_roc(foreground, background, "cache/shu.png")
        assert isinstance(pj, ppg.PlotJob)
        ppg.run_pipegraph()
        assert Path("cache/shu.png").exists()


@pytest.mark.usefixtures("new_pipegraph")
class TestHammingDistance:
    def test_identical(self):
        assert hamming_distance("shu", "shu") == 0

    def test_rising(self):
        assert hamming_distance("shu", "shu") == 0
        assert hamming_distance("shu", "sha") == 1
        assert hamming_distance("shu", "saa") == 2
        assert hamming_distance("shu", "sau") == 1
        assert hamming_distance("shu", "xxx") == 3

    def test_requires_same_length(self):

        with pytest.raises(ValueError):
            hamming_distance("shu", "sham")


@pytest.mark.usefixtures("new_pipegraph")
class TestPWM:
    def test_simple(self):
        pwm = PWMMotif_FromText("dummy", "AGC", bg)
        seq = "TTTAGCTTGCT"
        df, cum_score, max_score = pwm.scan(seq, pwm.max_score)
        assert len(df) == 3
        assert (df["start"] == [3, 7, 11]).all()
        assert (df["stop"] == [6, 4, 8]).all()
        for s in df["score"]:
            assert round(abs(s - pwm.max_score), 4) == 0
        assert round(abs(sum(df["score"]) - cum_score), 7) == 0
        assert round(abs(max(df["score"]) - max_score), 7) == 0

    def test_twice_gives_same_object(self):
        pwm = PWMMotif_FromText("dummy", "AGC", bg)
        pwm2 = PWMMotif_FromText("dummy", "AGC", bg)
        assert pwm is pwm2

    def test_raises_on_twice_with_different_matrices(self):
        PWMMotif_FromText("dummy", "AGC", bg)

        with pytest.raises(ValueError):
            PWMMotif_FromText("dummy", "AGT", bg)
   
    def test_bogus_kmers(self):
        np.random.seed(500)
        motif_seq = "TGANR"
        motif = PWMMotif_FromText("dummy", motif_seq, bg)
        a_s = []
        c_s = []
        g_s = []
        t_s = []
        for i in range(0, 100):
            total = 2000
            kmers = motif.bogus_kmers(total)
            counts = {"A": 0, "C": 0, "G": 0, "T": 0}
            for seq in kmers:
                counts[seq[3]] += 1
            a_s.append(counts["A"])
            c_s.append(counts["C"])
            g_s.append(counts["G"])
            t_s.append(counts["T"])
        assert scipy.stats.ttest_1samp(np.array(a_s, dtype=float), total / 4)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(c_s, dtype=float), total / 4)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(g_s, dtype=float), total / 4)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(t_s, dtype=float), total / 4)[1] >= 0.05

        a_s = []
        c_s = []
        g_s = []
        t_s = []
        for i in range(0, 100):
            total = 200
            kmers = motif.bogus_kmers(total)
            counts = {"A": 0, "C": 0, "G": 0, "T": 0}
            for seq in kmers:
                counts[seq[4]] += 1
            a_s.append(counts["A"])
            c_s.append(counts["C"])
            g_s.append(counts["G"])
            t_s.append(counts["T"])
        print(a_s)
        print(c_s)
        print(g_s)
        print(t_s)
        print(scipy.stats.ttest_1samp(np.array(c_s, dtype=float), 0)[1])

        assert scipy.stats.ttest_1samp(np.array(a_s, dtype=float), total / 2)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(c_s, dtype=float), 0)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(g_s, dtype=float), total / 2)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(t_s, dtype=float), 0)[1] > 0.05

    def test_raises_on_invalid_background_distribution(self):
        with pytest.raises(ValueError):
            motif_seq = "TGANR"

            broken = get_Candidatus_carsonella_ruddii_pv()
            broken.get_base_distribution = lambda _self: {
                "A": 0.25,
                "C": 0.25,
                "G": 0.25,
                "T": 0.3,
            }

            motif = PWMMotif_FromText("dummy", motif_seq, broken)

    def test_raisse_on_invalid_counts_if_not_renormalize(self):
        def inner():
            counts = [
                {"A": 1, "C": 0, "G": 0, "T": 0},
                {"A": 1, "C": 1, "G": 0, "T": 0},
            ]
            motif = PWMMotif("dummy", counts, bg)

        with pytest.raises(ValueError):
            inner()

    def test_len(self):
        motif_seq = "TGANR"
        motif = PWMMotif_FromText("dummy", motif_seq, bg)
        assert len(motif_seq) == len(motif)

    def test_magic(self):
        motif_seq = "TGANR"
        motif = PWMMotif_FromText("dummy", motif_seq, bg)
        assert "PWMMotif" in str(motif)
        assert "PWMMotif" in repr(motif)

    def test_score_requires_same_length(self):
        def inner():
            motif_seq = "TGANR"
            motif = PWMMotif_FromText("dummy", motif_seq, bg)
            motif.score(motif_seq + "A")

        with pytest.raises(ValueError):
            inner()

    def test_plot_logo(self):
        motif_seq = "TGANR"
        motif = PWMMotif_FromText("dummy", motif_seq, bg)
        output_filename = os.path.join("results", "logo_test.png")
        try:
            os.unlink(output_filename)
        except OSError:
            pass
        try:
            motif.plot_logo(output_filename)
            ppg.run_pipegraph()
            assert Path(output_filename).exists()
        finally:
            try:
                os.unlink(output_filename)
            except OSError:
                pass

    def test_complete_iupac(self):
        motif_seq = "ACGTNRYMKWSBDHV"
        motif = PWMMotif_FromText("dummy", motif_seq, bg, n_seq=120)
        for pos, count_A, count_C, count_G, count_T in [
            (0, 120, 0, 0, 0),
            (1, 0, 120, 0, 0),
            (2, 0, 0, 120, 0),
            (3, 0, 0, 0, 120),
            (4, 30, 30, 30, 30),
            (5, 60, 0, 60, 0),
            (6, 0, 60, 0, 60),
            (7, 60, 60, 0, 0),
            (8, 0, 0, 60, 60),
            (9, 60, 0, 0, 60),
            (10, 0, 60, 60, 0),
            (11, 0, 40, 40, 40),
            (12, 40, 0, 40, 40),
            (13, 40, 40, 0, 40),
            (14, 40, 40, 40, 0),
        ]:
            assert motif.counts[pos]["A"] == count_A
            assert motif.counts[pos]["C"] == count_C
            assert motif.counts[pos]["G"] == count_G
            assert motif.counts[pos]["T"] == count_T
        if pos < len(motif) - 1:
            raise ValueError("Did not check all")

    def test_raises_on_non_iupac(self):
        pwm = PWMMotif_FromText("dummyA", "AGC", bg)

        def inner():
            pwm = PWMMotif_FromText("dummyB", "AGX", bg)

        with pytest.raises(ValueError):
            inner()
        pwm = PWMMotif_FromText("dummyC", "AGC", bg)


@pytest.mark.usefixtures("new_pipegraph")
class TestMismatch:
    def test_simple(self):
        seq = "AGTCAC"
        motif = MisMatchMotif("dummy", "A")
        df, cum_score, max_score = motif.scan(seq, 1)
        assert max_score == 1
        assert len(df) == 3
        assert (df["start"] == [0, 4, 3]).all()

    def test_empty_sequence_raises_on_scan(self):
        motif = MisMatchMotif("dummy", "A")

        def inner():
            df, cum_score, max_score = motif.scan("", 1)

        with pytest.raises(ValueError):
            inner()

    def test_assumes_half_as_minimum_threshold(self):
        seq = "AGTCAC"
        motif = MisMatchMotif("dummy", "AA")
        df, cum_score, max_score = motif.scan(seq)
        assert max_score == 1
        assert len(df) == 5
        # self.assertTrue((df['start'] == [0, 4, 3]).all())

    def test_scores_on_not_found_are_0(self):
        seq = "AGTCAC"
        motif = MisMatchMotif("dummy", "AAGGGG")
        cum_score, max_score = motif.scan(seq, len(motif), keep_list=False)
        assert max_score == 0
        assert cum_score == 0

    def test_magic(self):
        seq = "AGTCAC"
        motif = MisMatchMotif("dummy", seq)
        str(motif)
        repr(motif)
        assert len(motif) == len("AGTCAC")

    def test_longfish(self):
        seq = "ACGAGGCCAAGATGAACCAGAGCGGGGTGTCAGTGGACGAGGAGCAGGACATTAGCCAGGAGGAGGAGCGG"
        "GACGGCCTGTACTTTGAGCCTGCGGTTCCCCTGCCAGACCTGGTAGAGATCTCCACTGGAGAGGAGAATGAAAATGT"
        "GTGTTTCAGCCACAGGGCAAAGCTGGATCGCTATGACAAAGACCTGAACCAGTGGAAGGAGAGGGGCATCGGAGACC"
        "TCAAGATACTGCAGAACTACAACACCAAACGAGGCAGACTCATCATGAGGAGAGACCAAGTCCTGAAA"
        motif_seq = "TGAAACANNSYWT"
        motif = MisMatchMotif("dummy", motif_seq)
        max_hamming = 3
        df, cum_score, max_score = motif.scan(seq, len(motif_seq) - max_hamming)
        for dummy_idx, row in df[["start", "stop"]].iterrows():
            if row["start"] < row["stop"]:
                s = seq[row["start"] : row["stop"]]
            else:
                s = reverse_complement(seq[row["stop"] : row["start"]])
            hamming = 0
            for ii in range(0, len(motif_seq)):
                if not s[ii] in completeReverseIupac[motif_seq[ii]]:
                    hamming += 1
            assert hamming <= max_hamming
        assert len(df) == 6

    def test_ambigious_in_seq(self):
        seq = "TGAANCAGTCTTT"
        motif_seq = "TGAAACAGTCTTT"
        motif = MisMatchMotif("dummy", motif_seq)
        max_hamming = 1
        df, cum_score, max_score = motif.scan(seq, len(motif_seq) - max_hamming)
        for dummy_idx, row in df[["start", "stop"]].iterrows():
            if row["start"] < row["stop"]:
                s = seq[row["start"] : row["stop"]]
            else:
                s = reverse_complement(seq[row["stop"] : row["start"]])
            hamming = 0
            for ii in range(0, len(motif_seq)):
                if not s[ii] in completeReverseIupac[motif_seq[ii]]:
                    if not motif_seq[ii] in completeReverseIupac[s[ii]]:
                        hamming += 1
            if not hamming <= max_hamming:
                print("m", motif_seq)
                print("s", s)
            assert hamming <= max_hamming
        assert len(df) == 1
        assert df.get_value(0, "score") == len(motif_seq) - 1

    def test_ambigious_in_both(self):
        seq = "TGAAWCAGTCTTT"
        motif_seq = "TGAABCAGTCTTT"
        motif = MisMatchMotif("dummy", motif_seq)
        max_hamming = 0
        df, cum_score, max_score = motif.scan(seq, len(motif_seq) - max_hamming)
        assert len(df) == 1

    def test_position(self):
        seq = "A" * 100 + "TCCA"
        motif_seq = "TCCA"
        motif = MisMatchMotif("dummy", motif_seq)
        df, cum_score, max_score = motif.scan(seq, len(motif_seq))
        assert len(df) == 1
        assert df.get_value(0, "start") == 100

    def test_bogus_kmers(self):
        np.random.seed(500)
        motif_seq = "TGANR"
        motif = MisMatchMotif("dumm", motif_seq)
        a_s = []
        c_s = []
        g_s = []
        t_s = []
        for i in range(0, 100):
            total = 2000
            kmers = motif.bogus_kmers(total)
            counts = {"A": 0, "C": 0, "G": 0, "T": 0}
            for seq in kmers:
                counts[seq[3]] += 1
            a_s.append(counts["A"])
            c_s.append(counts["C"])
            g_s.append(counts["G"])
            t_s.append(counts["T"])

        assert scipy.stats.ttest_1samp(np.array(a_s, dtype=float), total / 4)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(c_s, dtype=float), total / 4)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(g_s, dtype=float), total / 4)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(t_s, dtype=float), total / 4)[1] >= 0.05

        a_s = []
        c_s = []
        g_s = []
        t_s = []
        for i in range(0, 100):
            total = 2000
            kmers = motif.bogus_kmers(total)
            counts = {"A": 0, "C": 0, "G": 0, "T": 0}
            for seq in kmers:
                counts[seq[4]] += 1
            a_s.append(counts["A"])
            c_s.append(counts["C"])
            g_s.append(counts["G"])
            t_s.append(counts["T"])
        a_s.append(1)  # prevent a nan in ttest_1samp
        c_s.append(1)
        g_s.append(1)
        t_s.append(1)
        assert scipy.stats.ttest_1samp(np.array(a_s, dtype=float), total / 2)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(c_s, dtype=float), 0)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(g_s, dtype=float), total / 2)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(t_s, dtype=float), 0)[1] >= 0.05


@pytest.mark.usefixtures("new_pipegraph")
@pytest.mark.usefixtures("clear_motif_cache")
class TestDifferentialMismatch:
    def test_simple(self):
        seq = "AAAGGT"
        motif = DifferentialMisMatchMotif("dummyA", "AAACGT", [0.3, 0.3, 0.3, 1, 1, 1])
        df, cum_score, best_score = motif.scan(seq, 4)
        for dummy_idx, row in df[["start", "stop"]].iterrows():
            if row["start"] < row["stop"]:
                s = seq[row["start"] : row["stop"]]
            else:
                s = reverse_complement(seq[row["stop"] : row["start"]])
            # print row, s
        assert best_score == 5
        motif = DifferentialMisMatchMotif("dummyB", "CAGGGT", [0.3, 0.3, 0.3, 1, 1, 1])
        cum_score, best_score = motif.scan(seq, 4, keep_list=False)
        assert round(abs(best_score - 6 - 0.6), 4) == 0

    def test_penality_must_have_same_length(self):
        seq = "AAAGGTCC"

        def inner():
            motif = DifferentialMisMatchMotif(
                "dummy", "AAACGT", [0.3, 0.3, 0.3, 1, 1, 1, 1]
            )

        with pytest.raises(ValueError):
            inner()

    # FF 20170405 - fail to see how this was ever sinsible
    # def test_penality_must_be_less_than_one(self):
    #    seq = "AAAGGTCC"
    #    def inner():
    #        motif = DifferentialMisMatchMotif(
    #            'dummy', 'AAACGT', [0.3, 0.3, 0.3, 1, 1, 2])
    #    self.assertRaises(ValueError, inner)

    def test_bogus_kmers(self):  # make sure the kmers are evenly distributed?
        np.random.seed(500)
        motif_seq = "TGANR"
        motif = DifferentialMisMatchMotif("dumm", motif_seq, [1] * len(motif_seq))
        counts = {"A": 0, "C": 0, "G": 0, "T": 0}
        a_s = []
        c_s = []
        g_s = []
        t_s = []
        for i in range(0, 100):
            counts = {"A": 0, "C": 0, "G": 0, "T": 0}
            total = 2000
            kmers = motif.bogus_kmers(total)
            for seq in kmers:
                counts[seq[3]] += 1  # so that's the 'N' position'
            a_s.append(counts["A"])
            c_s.append(counts["C"])
            g_s.append(counts["G"])
            t_s.append(counts["T"])
        assert scipy.stats.ttest_1samp(np.array(a_s, dtype=float), total / 4)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(c_s, dtype=float), total / 4)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(g_s, dtype=float), total / 4)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(t_s, dtype=float), total / 4)[1] >= 0.05

        a_s = []
        c_s = []
        g_s = []
        t_s = []
        for i in range(0, 100):
            counts = {"A": 0, "C": 0, "G": 0, "T": 0}
            total = 2000
            kmers = motif.bogus_kmers(total)
            for seq in kmers:
                counts[seq[4]] += 1
            a_s.append(counts["A"])
            c_s.append(counts["C"])
            g_s.append(counts["G"])
            t_s.append(counts["T"])
        a_s.append(1)
        c_s.append(1)
        g_s.append(1)
        t_s.append(1)
        assert scipy.stats.ttest_1samp(np.array(a_s, dtype=float), total / 2)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(c_s, dtype=float), 0)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(g_s, dtype=float), total / 2)[1] >= 0.05
        assert scipy.stats.ttest_1samp(np.array(t_s, dtype=float), 0)[1] >= 0.05

    def test_magic(self):
        seq = "AGTCAC"
        motif = DifferentialMisMatchMotif("dummy", seq, [0.3] * len(seq))
        str(motif)
        repr(motif)
        assert len(motif) == len("AGTCAC")

    def test_assumes_half_as_minimum_threshold(self):
        seq = "AGTCAC"
        motif = DifferentialMisMatchMotif("dummy", "AA", [0.5, 1])
        df, cum_score, max_score = motif.scan(seq)
        assert max_score == 1.5
        assert len(df) == 5
        # self.assertTrue((df['start'] == [0, 4, 3]).all())

    def test_scores_on_not_found_are_0(self):
        seq = "AGTCAC"
        motif = DifferentialMisMatchMotif("dummy", "AAGGGG", [1] * 6)
        cum_score, max_score = motif.scan(seq, len(motif), keep_list=False)
        assert max_score == 0
        assert cum_score == 0


@pytest.mark.usefixtures("new_pipegraph")
class TestDeNovo_search:
    def test_fake_searcher(self):
        seqs = [
            "CATCTCCTGCTTTTATAAATTAATTTGTTCGTAACGTATCA",
            "AGTC",
        ]  # must be more than 1 seq

        def sample_data():
            return pd.DataFrame(
                {"name": ["spiked_" + str(x) for x in range(0, len(seqs))], "seq": seqs}
            )

        class FakeAlgo(object):
            name = "FAKE"

            @staticmethod
            def run(dummy_foreground_fasta, dummy_background_fasta, dummy_cache_dir):
                return "this would be parsed"

            @staticmethod
            def parse(output):
                results = []
                results.append(
                    {
                        "sequence": "ACT",
                        "strand": 1,
                        "start": 0,
                        "end": 4,
                        "motif_no": 0,
                    }
                )
                results.append(
                    {
                        "sequence": "ACG",
                        "strand": 1,
                        "start": 10,
                        "end": 14,
                        "motif_no": 1,
                    }
                )
                return pd.DataFrame(results)

            def get_dependencies(self):
                return []

            def get_parameters(self):
                return ()

        foreground = sequences.Sequences(
            "spiked", sample_data, [], dummy_genome, do_cache=False
        )
        searcher = FakeAlgo()
        denovosource = sources.DeNovoMotifSource(searcher, foreground, None)

        def dummy():
            pass

        job = ppg.JobGeneratingJob("shu", dummy)
        job.depends_on(denovosource.get_dependencies())
        ppg.run_pipegraph()
        assert len(denovosource) == 2
        found = list(denovosource)[0]  # first motif...
        assert len(found) == 3


if meme_available:

    @pytest.mark.usefixtures("new_pipegraph")
    class TestDeNovo_search_using_meme:
        def test_simple(self):
            seqs = [
                "CATCTCCTGCTTTTATAAATTAATTTGTTCGTAACGTATCA",
                "TTTCATGCTTGCTTTCTTTTCCCCTTACCCTCCTCTTGCTTCGTGTTCAGAGGAATTATTTTTTTCGTTC",
                "GCACTAGCACCTAGTCGTTTTGTTTTGTGACAACTATCTCTCTGATTTTCCTCTTGTCTAGTTTTGTCC"
                "TTGTCAATCTTT",
                "ATCGAGTTTGTTCAACGCGGGATAGGCAGTTGTCTTTCCCCGAGTTCCTAGAGTTTG",
                "GTGCTATTGGGTATCTCTCCTAGTCGTCCCAAAAACTATTTTGGAACATTATTTTT",
                "TTACTTCGTAGTGAGACCGTTTCCCTCAGTGGGCTCATTTTTTAAAGTTGTTCGTGCTTTGTATTTGGG"
                "GGTTGTCAG",
                "TCTTTCCGTTCCATCGCACTCTTTTGTTTTCTCGTCACAGTTAGTCTGTTTAAAGACATAAATACGTGTC"
                "TAGTTCCTGATA",
                "CCTCCCACAAAGTCGTTTTATTAAGTGCTTTGTGGTTATGGGTATCTATTTATATCGTTACGTTTTGTTG"
                "TCTCGATGTTTTTTTCTTTT",
                "GGGGTTTTATTTGTGGACAACCAAGTATATGGTTGAAATTCTATCTAGGCT",
                "TAAGTAGTTTGTCACTGGTTTATTTATGGTGGTTGGGTGGCGCTATGGGGCCTGTGG",
                "GGATTTTCTATGCCGTTTTGTGTGCTCTAGGGCTAATTGCATGCGCATGCTCTGTATGGTATTTATGGCG"
                "GTGCCAAATC",
                "CTATATCTCGGTATCTTGTAAGTTCAGTTCTTTCCCTTGTCCCATTCTTTCTGATGGCTAGTCT",
                "TGTTCCGGGTGGTTCATAGTCCTCGTTGGATTTTTCTGCATGGAGATTACCCTGAAGTTAGTATTAGTC",
                "CTAGTTATGGTGGATTTCACCTCATTTGCCATCCTCGTTTGTCAACTTTAGTTTCCTCTTTTTTCTCGTT",
                "CAGCTGTAATAGCCTCCGTTTCCGAATACCTTGAGTTACGGTTTGAGTATCACT",
            ]
            motif = "AGGTCA"
            spiked = []
            for s in seqs:
                spiked.append(s[:20] + motif + s[20:])

            def sample_data():
                return pd.DataFrame(
                    {
                        "name": ["spiked_" + str(x) for x in range(0, len(spiked))],
                        "seq": spiked,
                    }
                )

            foreground = sequences.Sequences(
                "spiked", sample_data, [], dummy_genome, do_cache=False
            )
            searcher = denovo.Meme(minw=len(motif), maxw=len(motif), nmotifs=1)
            denovosource = sources.DeNovoMotifSource(searcher, foreground, None)

            def dummy():
                pass

            job = ppg.JobGeneratingJob("shu", dummy)
            job.depends_on(denovosource.get_dependencies())
            ppg.run_pipegraph()
            assert len(denovosource) >= 1
            found = list(denovosource)[0]
            assert found.score(motif) == found.max_score
            assert found == denovosource[0]

        def test_de_novo_source_throws_on_get_item_before_run(self):
            foreground = sequences.Sequences(
                "spiked", lambda: [], [], dummy_genome, do_cache=False
            )
            searcher = denovo.Meme(minw=5, maxw=5, nmotifs=1)
            denovosource = sources.DeNovoMotifSource(searcher, foreground, None)

            def inner():
                denovosource["shu"]

            with pytest.raises(ValueError):
                inner()

        def test_with_background(self):
            seqs = [
                "CATCTCCTGCTTTTATAAATTAATTTGTTCGTAACGTATCA",
                "TTTCATGCTTGCTTTCTTTTCCCCTTACCCTCCTCTTGCTTCGTGTTCAGAGGAATTATTTTTTTCGTTC",
                "GCACTAGCACCTAGTCGTTTTGTTTTGTGACAACTATCTCTCTGATTTTCCTCTTGTCTAGTTTTGTCC"
                "TTGTCAATCTTT",
                "ATCGAGTTTGTTCAACGCGGGATAGGCAGTTGTCTTTCCCCGAGTTCCTAGAGTTTG",
                "GTGCTATTGGGTATCTCTCCTAGTCGTCCCAAAAACTATTTTGGAACATTATTTTT",
                "TCTTTCCGTTCCATCGCACTCTTTTGTTTTCTCGTCACAGTTAGTCTGTTTAAAGACATAAATACGTGT"
                "CTAGTTCCTGATA",
            ]
            background_seqs = [
                "CCTCCCACAAAGTCGTTTTATTAAGTGCTTTGTGGTTATGGGTATCTATTTATATCGTTACGTTTTGTT"
                "GTCTCGATGTTTTTTTCTTTT",
                "GGGGTTTTATTTGTGGACAACCAAGTATATGGTTGAAATTCTATCTAGGCT",
                "TAAGTAGTTTGTCACTGGTTTATTTATGGTGGTTGGGTGGCGCTATGGGGCCTGTGG",
                "GGATTTTCTATGCCGTTTTGTGTGCTCTAGGGCTAATTGCATGCGCATGCTCTGTATGGTATTTATGGC"
                "GGTGCCAAATC",
                "CTATATCTCGGTATCTTGTAAGTTCAGTTCTTTCCCTTGTCCCATTCTTTCTGATGGCTAGTCT",
                "TGTTCCGGGTGGTTCATAGTCCTCGTTGGATTTTTCTGCATGGAGATTACCCTGAAGTTAGTATTAGTC",
                "TTACTTCGTAGTGAGACCGTTTCCCTCAGTGGGCTCATTTTTTAAAGTTGTTCGTGCTTTGTATTTGGG"
                "GGTTGTCAG",
                "CTAGTTATGGTGGATTTCACCTCATTTGCCATCCTCGTTTGTCAACTTTAGTTTCCTCTTTTTTCTCGTT",
                "CAGCTGTAATAGCCTCCGTTTCCGAATACCTTGAGTTACGGTTTGAGTATCACT",
            ]
            motif = "AGGTCA"
            spiked = []
            for s in seqs:
                for i in range(10, 20):
                    spiked.append(s[:i] + motif + s[i:])

            def sample_data():
                return pd.DataFrame(
                    {
                        "name": ["spiked_" + str(x) for x in range(0, len(spiked))],
                        "seq": spiked,
                    }
                )

            def sample_background():
                return pd.DataFrame(
                    {
                        "name": [
                            "bg_" + str(x) for x in range(0, len(background_seqs))
                        ],
                        "seq": background_seqs,
                    }
                )

            foreground = sequences.Sequences(
                "spiked", sample_data, [], dummy_genome, do_cache=False
            )
            background = sequences.Sequences(
                "background", sample_background, [], dummy_genome, do_cache=False
            )
            searcher = denovo.Meme(minw=len(motif), maxw=len(motif), nmotifs=1)
            denovosource = sources.DeNovoMotifSource(searcher, foreground, background)

            def dummy():
                pass

            job = ppg.JobGeneratingJob("shu", dummy)
            job.depends_on(denovosource.get_dependencies())
            ppg.run_pipegraph()
            assert len(denovosource) >= 1
            found = list(denovosource)[0]
            assert found.score(motif) == found.max_score

        def test_with_background_cant_find(self):
            seqs = [
                "CATCTCCTGCTTTTATAAATTAATTTGTTCGTAACGTATCA",
                "TTTCATGCTTGCTTTCTTTTCCCCTTACCCTCCTCTTGCTTCGTGTTCAGAGGAATTATTTTTTTCGTTC",
                "GCACTAGCACCTAGTCGTTTTGTTTTGTGACAACTATCTCTCTGATTTTCCTCTTGTCTAGTTTTGTCC"
                "TTGTCAATCTTT",
                "ATCGAGTTTGTTCAACGCGGGATAGGCAGTTGTCTTTCCCCGAGTTCCTAGAGTTTG",
                "GTGCTATTGGGTATCTCTCCTAGTCGTCCCAAAAACTATTTTGGAACATTATTTTT",
                "TCTTTCCGTTCCATCGCACTCTTTTGTTTTCTCGTCACAGTTAGTCTGTTTAAAGACATAAATACGTGT"
                "CTAGTTCCTGATA",
            ]
            background_seqs = [
                "CCTCCCACAAAGTCGTTTTATTAAGTGCTTTGTGGTTATGGGTATCTATTTATATCGTTACGTTTTGTT"
                "GTCTCGATGTTTTTTTCTTTT",
                "GGGGTTTTATTTGTGGACAACCAAGTATATGGTTGAAATTCTATCTAGGCT",
                "TAAGTAGTTTGTCACTGGTTTATTTATGGTGGTTGGGTGGCGCTATGGGGCCTGTGG",
                "GGATTTTCTATGCCGTTTTGTGTGCTCTAGGGCTAATTGCATGCGCATGCTCTGTATGGTATTTATGGC"
                "GGTGCCAAATC",
                "CTATATCTCGGTATCTTGTAAGTTCAGTTCTTTCCCTTGTCCCATTCTTTCTGATGGCTAGTCT",
                "TGTTCCGGGTGGTTCATAGTCCTCGTTGGATTTTTCTGCATGGAGATTACCCTGAAGTTAGTATTAGTC",
                "TTACTTCGTAGTGAGACCGTTTCCCTCAGTGGGCTCATTTTTTAAAGTTGTTCGTGCTTTGTATTTGGG"
                "GGTTGTCAG",
                "CTAGTTATGGTGGATTTCACCTCATTTGCCATCCTCGTTTGTCAACTTTAGTTTCCTCTTTTTTCTCGTT",
                "CAGCTGTAATAGCCTCCGTTTCCGAATACCTTGAGTTACGGTTTGAGTATCACT",
            ]
            motif = "AGGTCA"
            spiked = []
            for s in seqs:
                spiked.append(s[:20] + motif + s[20:])
            background_spiked = []
            for s in background_seqs:
                background_spiked.append(s[:20] + motif * 3 + s[20:])

            def sample_data():
                return pd.DataFrame(
                    {
                        "name": ["spiked_" + str(x) for x in range(0, len(spiked))],
                        "seq": spiked,
                    }
                )

            def sample_background():
                return pd.DataFrame(
                    {
                        "name": [
                            "bg_" + str(x) for x in range(0, len(background_spiked))
                        ],
                        "seq": background_spiked,
                    }
                )

            foreground = sequences.Sequences(
                "spiked", sample_data, [], dummy_genome, do_cache=False
            )
            background = sequences.Sequences(
                "background_spiked", sample_background, [], dummy_genome, do_cache=False
            )
            searcher = denovo.Meme(minw=len(motif), maxw=len(motif), nmotifs=1)
            denovosource = sources.DeNovoMotifSource(searcher, foreground, background)

            def dummy():
                pass

            job = ppg.JobGeneratingJob("shu", dummy)
            job.depends_on(denovosource.get_dependencies())
            ppg.run_pipegraph()
            assert len(denovosource) >= 1
            found = list(denovosource)[0]
            assert found.score(motif) != found.max_score

        def test_empty_leads_to_empty_sources(self):
            seqs = []
            motif = "AGGTCA"
            spiked = []
            for s in seqs:
                spiked.append(s[:20] + motif + s[20:])

            def sample_data():
                return pd.DataFrame(
                    {
                        "name": ["spiked_" + str(x) for x in range(0, len(spiked))],
                        "seq": spiked,
                    }
                )

            foreground = sequences.Sequences(
                "spiked", sample_data, [], dummy_genome, do_cache=False
            )
            searcher = denovo.Meme(
                minw=len(motif), maxw=len(motif), nmotifs=1, maxsize=10000
            )
            denovosource = sources.DeNovoMotifSource(searcher, foreground, None)

            def dummy():
                pass

            job = ppg.JobGeneratingJob("shu", dummy)
            job.depends_on(denovosource.get_dependencies())
            ppg.run_pipegraph()

        def test_empty_background_works(self):
            seqs = [
                "CATCTCCTGCTTTTATAAATTAATTTGTTCGTAACGTATCA",
                "TTTCATGCTTGCTTTCTTTTCCCCTTACCCTCCTCTTGCTTCGTGTTCAGAGGAATTATTTTTTTCGTTC",
                "GCACTAGCACCTAGTCGTTTTGTTTTGTGACAACTATCTCTCTGATTTTCCTCTTGTCTAGTTTTGTCC"
                "TTGTCAATCTTT",
                "ATCGAGTTTGTTCAACGCGGGATAGGCAGTTGTCTTTCCCCGAGTTCCTAGAGTTTG",
                "GTGCTATTGGGTATCTCTCCTAGTCGTCCCAAAAACTATTTTGGAACATTATTTTT",
                "TCTTTCCGTTCCATCGCACTCTTTTGTTTTCTCGTCACAGTTAGTCTGTTTAAAGACATAAATACGTGT"
                "CTAGTTCCTGATA",
            ]
            background_seqs = []
            motif = "AGGTCA"
            spiked = []
            for s in seqs:
                spiked.append(s[:20] + motif + s[20:])
            background_spiked = []
            for s in background_seqs:
                background_spiked.append(s[:20] + motif * 3 + s[20:])

            def sample_data():
                return pd.DataFrame(
                    {
                        "name": ["spiked_" + str(x) for x in range(0, len(spiked))],
                        "seq": spiked,
                    }
                )

            def sample_background():
                return pd.DataFrame(
                    {
                        "name": [
                            "bg_" + str(x) for x in range(0, len(background_spiked))
                        ],
                        "seq": background_spiked,
                    }
                )

            foreground = sequences.Sequences(
                "spiked", sample_data, [], dummy_genome, do_cache=False
            )
            background = sequences.Sequences(
                "background_spiked", sample_background, [], dummy_genome, do_cache=False
            )
            searcher = denovo.Meme(minw=len(motif), maxw=len(motif), nmotifs=1)
            denovosource = sources.DeNovoMotifSource(searcher, foreground, background)

            def dummy():
                pass

            job = ppg.JobGeneratingJob("shu", dummy)
            job.depends_on(denovosource.get_dependencies())
            ppg.run_pipegraph()
            assert len(denovosource) >= 1


@pytest.mark.usefixtures("new_pipegraph")
class TestManualMotifSource:
    def test_init(self):
        source = sources.ManualMotifSource("shu", ["one", "two"])
        assert source.name == "shu"
        assert len(source.list), 2
        assert source.list == ["one", "two"]

    def test_iter(self):
        source = sources.ManualMotifSource("shu", ["one", "two"])
        assert source.name == "shu"
        assert len(source.list), 2
        assert list(source) == ["one", "two"]

    def test_get_dependencies(self):
        source = sources.ManualMotifSource("shu", ["one", "two"])
        assert source.get_dependencies() == []


@pytest.mark.usefixtures("new_pipegraph")
@pytest.mark.usefixtures("clear_motif_cache")
class TestMatbaseSource:
    # TODO: why is matbase even in genomics, and not a dataset?

    def test_len(self):
        mb = sources.Matbase(dummy_genome)
        assert len(mb)

    def test_get_dependencies(self):
        mb = sources.Matbase(dummy_genome)
        assert mb.get_dependencies() == []

    def test_yields_motifs(self):
        mb = sources.Matbase(dummy_genome)
        count = 0
        for m in mb:
            assert isinstance(m, MotifBase)
            count += 1
        assert count == len(mb)

    def test_getitem(self):
        genome = dummy_genome
        mb = sources.Matbase(genome)
        mot = mb["MBAARE.01"]
        # the _dummy is from the genome
        assert mot.name == "MBAARE.01_dummy"

    def test_getitem_throws_key_error(self):
        mb = sources.Matbase(dummy_genome)

        def inner():
            mot = mb["AARE.01DOESNOTEXISt"]

        with pytest.raises(KeyError):
            inner()
