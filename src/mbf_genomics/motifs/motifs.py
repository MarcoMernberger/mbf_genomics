import itertools
import shutil
import numpy as np
import math
import dppd
import dppd_plotnine
import pypipegraph as ppg
import pandas as pd
from . import plots
from . import _motifs
import common
from mbf_genomes.common import reverse_complement_iupac

dp, X = dppd.dppd()


class MotifBase(object):
    """Common base for Motifs"""

    def __init__(self, name):
        self.name = str(name)
        ppg.assert_uniqueness_of_object(self)

    def scan(self, sequence, minimum_threshold, keep_list=True):
        raise NotImplementedError("Do not use MotifBase directly, overwrite scan")

    def calculate_roc_auc(
        self, foreground_sequences_list, background_sequence_list, pre_scored=None
    ):
        """Calculate the area under the roc curve for two lists (positive/negative) of sequences
        @pre_scored may be a tuple of ([best_score, best_score], [best_score, best_score])
        for the foreground and background best motif intstances respectively (if you have them already calculated...
        """
        raise ValueError("TODO: replace with scipy")
        import croc

        curve_df = self.calculate_roc(
            foreground_sequences_list, background_sequence_list, pre_scored=None
        )
        c = croc.Curve()
        for dummy_idx, row in curve_df.iterrows():
            c.append(row["False positive rate"], row["Support"])
        return c.area()

    def _score_two_lists(self, foreground_sequences_list, background_sequence_list):
        foreground_scores = []
        for seq in foreground_sequences_list:
            (cum_score, best_score) = self.scan(seq, 0, keep_list=False)
            foreground_scores.append((best_score))
        background_scores = []
        for seq in background_sequence_list:
            (cum_score, best_score) = self.scan(seq, 0, keep_list=False)
            background_scores.append((best_score))
        return foreground_scores, background_scores

    def calculate_mncp(
        self, foreground_sequences_list, background_sequence_list, pre_scored=None
    ):
        """Calculate the mean normalized conditional probability (MNCP, see Clarke, 2003)"""
        raise ValueError("todo: inline")
        import mncp

        if pre_scored is None:
            pre_scored = self._score_two_lists(
                foreground_sequences_list, background_sequence_list
            )
        foreground_scores, background_scores = pre_scored
        positive_count = len(foreground_sequences_list)
        return mncp.mncp_from_score_lists(foreground_scores, background_scores, True)

    def calculate_roc_auc_and_mncp(
        self, foreground_sequences_list, background_sequence_list
    ):
        """Calculate both ROC_AUC and MNCP"""
        scores = self._score_two_lists(
            foreground_sequences_list, background_sequence_list
        )
        roc_auc = self.calculate_roc_auc(
            foreground_sequences_list, background_sequence_list, scores
        )
        mncp = self.calculate_mncp(
            foreground_sequences_list, background_sequence_list, scores
        )
        return roc_auc, mncp

    def calculate_roc(
        self, foreground_sequences_list, background_sequence_list, pre_scored=None
    ):
        """Suitable for plotting the roc (receiver operator characteristic)"""
        data = {"Threshold": [], "Support": [], "False positive rate": []}
        class_foreground = 0
        class_background = 1
        best_motifs = []
        if pre_scored is None:
            pre_scored = self._score_two_lists(
                foreground_sequences_list, background_sequence_list
            )
        foreground_scores, background_scores = pre_scored
        for best_score in foreground_scores:
            best_motifs.append((best_score, class_foreground))
        for best_score in background_scores:
            best_motifs.append((best_score, class_background))

        best_motifs.sort()
        best_motifs.reverse()
        true_positive_count = 0
        false_positive_count = 0
        for score, entries in itertools.groupby(
            best_motifs, lambda x: x[0]
        ):  # aggregate on scores..
            entries = list(entries)
            for entry in entries:
                if entry[1] == class_foreground:
                    true_positive_count += 1
                else:
                    false_positive_count += 1
            data["Threshold"].append(score)
            data["Support"].append(
                float(true_positive_count) / len(foreground_sequences_list)
            )
            data["False positive rate"].append(
                float(false_positive_count) / len(background_sequence_list)
            )
        return pd.DataFrame(data)

    def plot_roc(self, sequence_list_positive, sequence_list_negative, output_filename):
        """JOB: Calculate roc and plot it"""
        raise ValueError("Todo scipy rocauc")
        import croc

        def calc():
            return self.calculate_roc(sequence_list_positive, sequence_list_negative)

        def plot(df):
            df = df.copy()  # because we mess it up

            c = croc.Curve()
            for dummy_idx, row in df.iterrows():
                c.append(row["False positive rate"], row["Support"])
            area = c.area()

            df = df.assign(ThresholdX=["%.2f" % x for x in df["Threshold"]])
            df = df.drop("Threshold", axis=1)
            df = df.rename(columns={"ThresholdX": "Threshold"})
            if len(df) > 10:
                df.ix[(np.array(range(0, len(df))) % 300) != 0, "Threshold"] = ""
            return (
                dp(df)
                .p9()
                .add_label(0.1, 1, str(area), size=5)
                .add_line("False positive rate", "Support", color="grey")
                .add_scatter("False positive rate", "Support")
                .add_ab_line(0, 1)
                .add_text(
                    "False positive rate",
                    "Support",
                    label="Threshold",
                    hjust=-1,
                    vjust=1,
                )
                .scale_x_continuous(limits=[0, 1])
                .scale_y_continuous(limits=[0, 1])
                .pd
            )

        return ppg.PlotJob(output_filename, calc, plot, skip_table=True).depends_on(
            self.load()
        )

    def load(self):  # for compability with pipeline
        return []


def hamming_distance(seq_a, seq_b):
    """Calculate the 'letter replacement' Hamming distance"""
    if len(seq_a) != len(seq_b):
        raise ValueError("Hamming distance is only defined on equal length strings")
    distance = 0
    for ii in range(0, len(seq_a)):
        if seq_a[ii] != seq_b[ii]:
            distance += 1
    return distance


def number_to_kmer(number, pad_to_length):
    res = []
    while number > 0:
        if number % 4 == 0:
            res.append("A")
        elif number % 4 == 1:
            res.append("C")
        elif number % 4 == 2:
            res.append("G")
        elif number % 4 == 3:
            res.append("T")
        number = number / 4
    res = "".join(res[::-1])
    while len(res) < pad_to_length:
        res = "A" + res
    return res


def iter_all_kmers(length):
    for ii in range(0, pow(4, length)):
        yield number_to_kmer(ii, length)


class MisMatchMotif(MotifBase):
    """A simple motif model that allows up to N mismatches"""

    def __init__(self, name, iupac_consensus):
        self.iupac_consensus = iupac_consensus
        self.reverse_iupac_consensus = reverse_complement_iupac(iupac_consensus)
        self.max_score = len(self.iupac_consensus)
        self.sensible_minimum_threshold = self.max_score - (max(5, self.max_score / 2))
        MotifBase.__init__(self, name)

    def __str__(self):
        return "mismatch_motif " + self.name + " " + self.iupac_consensus

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.iupac_consensus)

    def bogus_kmers(self, count=200):
        cum_probs = []
        for iupac_letter in self.iupac_consensus:
            letters = completeReverseIupac[iupac_letter]
            p = 1.0 / len(letters)
            c = {}
            total = 0
            for l in "ACGT":
                if l in letters:
                    total += p
                c[l] = total
            cum_probs.append(c)
        random_values = np.random.uniform(size=count * len(cum_probs))
        seqs = []
        r = 0
        for ii in range(0, count):
            s = ""
            for tt in range(0, len(cum_probs)):
                val = random_values[r]
                r += 1
                for letter in "ACGT":
                    if val <= cum_probs[tt][letter]:
                        s += letter
                        break
            seqs.append(s)
        return seqs

    def scan(
        self,
        sequence,
        minimum_threshold=None,  # =motiflen - max_number_of_mismatches
        keep_list=True,
    ):
        if not sequence:
            raise ValueError("Passed empty sequence to %s.scan" % self)
        if not minimum_threshold:
            minimum_threshold = len(self.iupac_consensus) / 2

        max_score = self.max_score
        penalties = np.array([1] * len(self.iupac_consensus))
        starts_forward, ends_forward, scores_forward = _motifs.differential_mismatch_motif_scan(
            common.to_bytes(self.iupac_consensus),
            common.to_bytes(sequence),
            max_score,
            minimum_threshold,
            penalties,
        )
        ends_reverse, starts_reverse, scores_reverse = _motifs.differential_mismatch_motif_scan(
            common.to_bytes(self.reverse_iupac_consensus),
            common.to_bytes(sequence),
            max_score,
            minimum_threshold,
            penalties,
        )
        starts = starts_forward + starts_reverse
        ends = ends_forward + ends_reverse
        scores = scores_forward + scores_reverse
        df = pd.DataFrame(
            {
                "start": np.array(starts, dtype=np.uint32),
                "stop": np.array(ends, dtype=np.uint32),
                "score": scores,
            }
        )
        if scores:
            max_score = max(scores)
            cum_score = sum(scores)
        else:
            max_score = 0
            cum_score = 0
        if keep_list:
            return df, cum_score, max_score
        else:
            return cum_score, max_score


class DifferentialMisMatchMotif(MotifBase):
    """This motif has a list of how much to substract for a mismatch
    in each position.
    Usefull for those flanking - halfsite - halfsite motifs"""

    def __init__(self, name, iupac_consensus, penalties):
        self.iupac_consensus = iupac_consensus
        self.reverse_iupac_consensus = reverse_complement_iupac(iupac_consensus)
        self.max_score = len(self.iupac_consensus)
        self.sensible_minimum_threshold = self.max_score - (max(5, self.max_score / 2))
        self.penalties = np.array(penalties, dtype=np.float)
        if (self.penalties >= 0).all():
            self.penalties *= -1
        elif (self.penalties <= 0).all():
            pass
        else:
            raise ValueError("Penalities must all be negative (or 0)")

        if len(self.penalties) != len(self.iupac_consensus):
            raise ValueError("Penaltiest have the wrong length")
        self.reverse_penalties = np.array(penalties[::-1], dtype=np.float)
        MotifBase.__init__(self, name)

    def __len__(self):
        return len(self.iupac_consensus)

    def bogus_kmers(self, count=200):
        cum_probs = []
        for iupac_letter in self.iupac_consensus:
            letters = completeReverseIupac[iupac_letter]
            p = 1.0 / len(letters)
            c = {}
            total = 0
            for l in "ACGT":
                if l in letters:
                    total += p
                c[l] = total
            cum_probs.append(c)
        random_values = np.random.uniform(size=count * len(cum_probs))
        seqs = []
        r = 0
        for ii in range(0, count):
            s = ""
            for tt in range(0, len(cum_probs)):
                val = random_values[r]
                r += 1
                for letter in "ACGT":
                    if val <= cum_probs[tt][letter]:
                        s += letter
                        break
            seqs.append(s)
        return seqs

    def __str__(self):
        return (
            "DifferentialMisMatchMotif "
            + self.name
            + " "
            + self.iupac_consensus
            + "P "
            + str(self.penalties)
        )

    def __repr__(self):
        return str(self)

    def scan(
        self,
        sequence,
        minimum_threshold=None,  # =motiflen - max_number_of_mismatches
        keep_list=True,
    ):
        if not minimum_threshold:
            minimum_threshold = len(self.iupac_consensus) / 2
        max_number_of_mismatches = len(self.iupac_consensus) - minimum_threshold
        max_score = self.max_score
        if not sequence:
            raise ValueError("Passed empty sequence to %s.scan" % self)
        if not minimum_threshold:
            minimum_threshold = len(self.iupac_consensus) / 2

        max_score = self.max_score
        starts_forward, ends_forward, scores_forward = _motifs.differential_mismatch_motif_scan(
            common.to_bytes(self.iupac_consensus),
            common.to_bytes(sequence),
            max_score,
            minimum_threshold,
            self.penalties * -1,
        )
        ends_reverse, starts_reverse, scores_reverse = _motifs.differential_mismatch_motif_scan(
            common.to_bytes(self.reverse_iupac_consensus),
            common.to_bytes(sequence),
            max_score,
            minimum_threshold,
            self.penalties * -1,
        )
        starts = starts_forward + starts_reverse
        ends = ends_forward + ends_reverse
        scores = scores_forward + scores_reverse
        df = pd.DataFrame(
            {
                "start": np.array(starts, dtype=np.uint32),
                "stop": np.array(ends, dtype=np.uint32),
                "score": scores,
            }
        )
        if scores:
            max_score = max(scores)
            cum_score = sum(scores)
        else:
            max_score = 0
            cum_score = 0
        if keep_list:
            return df, cum_score, max_score
        else:
            return cum_score, max_score


class BagOfSequencesMotif(MotifBase):
    def __init__(self, name, sequences_accepted):
        self.sequences_accepted = [x.upper() for x in sequences_accepted]
        self.reverse_complement = [
            reverse_complement_iupac(x) for x in self.sequences_accepted
        ]
        self.max_score = len(self)
        MotifBase.__init__(self, name)

    def __len__(self):
        return max((len(x) for x in self.sequences_accepted))

    def scan(self, sequence, minimum_threshold=None, keep_list=True):
        hits = set()
        for s in self.sequences_accepted:
            f = sequence.find(s)
            while f != -1:
                hits.add((f, f + len(s)))
                f = sequence.find(s, f + 1)
        for s in self.reverse_complement:
            f = sequence.find(s)
            while f != -1:
                hits.add((f + len(s), f))
                f = sequence.find(s, f + 1)
        starts = []
        ends = []
        scores = []
        for s, e in hits:
            starts.append(s)
            ends.append(e)
            scores.append(len(self))
        df = pd.DataFrame(
            {
                "start": np.array(starts, np.uint32),
                "stop": np.array(ends, np.uint32),
                "score": scores,
            }
        )
        if len(df):
            max_score = np.max(df["score"])
            cum_score = np.sum(df["score"])
        else:
            max_score = 0
            cum_score = 0
        return df, cum_score, max_score

    def score(self, seq):
        return self.scan(seq)[2]


_PWMMotifCache = {}


class PWMMotif(MotifBase):
    """A position weight matrix based motif model"""

    def __new__(cls, name, counts, genome, re_normalize=False):
        """So we can create the exact same motif multiple times"""
        key = (name, str(counts), genome.name, re_normalize)
        if not key in _PWMMotifCache:
            _PWMMotifCache[key] = MotifBase.__new__(
                cls
            )  # , name + '_' + genome.short_name)
        return _PWMMotifCache[key]

    def __getnewargs__(self):
        return (self.name, self.counts, self.genome, self.re_normalize)

    def __init__(self, name, counts, genome, re_normalize=False):
        """Counts is a [ {'A': 5, 'C': 3,...},{...},...] list,
        background distribution is the a priory probability of the single letters (genome.get_base_distribution())
        another dict
        """
        if hasattr(self, "counts"):
            if counts != self.counts:
                raise ValueError(
                    "Trying to create the same PWMMotif(name) with different counts: %s\n%s\n%s"
                    % (name, counts, self.counts)
                )
            if re_normalize != self.re_normalize:
                raise ValueError(
                    "Trying to create the same PWMMotif(name) with re_normalize settings"
                )
        else:
            self.counts = counts
            self.genome = genome
            self.background_distribution = genome.get_base_distribution()
            self.re_normalize = re_normalize
            if sum(self.background_distribution.values()) > 1:
                raise ValueError(
                    "sum(background_distribution) > 1, was %f"
                    % sum(self.background_distribution.values())
                )
            self.loaded = False
            MotifBase.__init__(self, name + "_" + genome.short_name)
            try:
                self.ll, self.max_score = self.calculate_matrix()
            except ValueError:
                if re_normalize:
                    self.normalize_counts()
                    self.ll, self.max_score = self.calculate_matrix()
                else:
                    print("no renormalize")
                    raise
            self.sensible_minimum_threshold = 0

    def __len__(self):
        return len(self.counts)

    def bogus_kmers(self, count=200):
        probability_matrix = []
        letters = "ACGT"
        for col in self.ll:
            prob_col = {}
            total = 0
            for letter in letters:
                b = self.background_distribution[letter]
                prob = pow(2, col[letter]) * b  # / p / b
                total += prob
                prob_col[letter] = total
            probability_matrix.append(prob_col)
        # ok, we now have a matrix of cumulative probabilities...
        random_values = np.random.uniform(size=count * len(probability_matrix))
        seqs = []
        r = 0
        for ii in range(0, count):
            s = ""
            for tt in range(0, len(probability_matrix)):
                val = random_values[r]
                r += 1
                for letter in letters:
                    if val <= probability_matrix[tt][letter]:
                        s += letter
                        break
            seqs.append(s)
        return seqs

    def __str__(self):
        return "PWMMotif " + self.name + "%s" % self.ll

    def __repr__(self):
        return 'PWMMotif("%s", %s, %s, %s)' % (
            self.name,
            self.counts,
            self.background_distribution,
            self.re_normalize,
        )

    def do_load(self):
        if not self.loaded:
            self.forward_matrix = self.fill_scanning_matrix()
            self.reverse_matrix = self.reverse_scanning_matrix()
            self.loaded = True

    def load(self):  # for compability with pipeline
        return [
            ppg.ParameterInvariant(
                "motif %s" % self.name,
                (self.counts, self.background_distribution, self.re_normalize),
            )
        ]  # so that recreation's of the motif with different parameter trigger rebuilds of plots

    def normalize_counts(self):
        """Sometimes the counts in the databases don't add up to the same number in each column. This takes care of that..."""
        max_total = max(sum(col.values()) for col in self.counts)
        max_delta = max(abs(sum(col.values()) - max_total) for col in self.counts)
        # print 'renormalizing', self.name, 'max delta' ,max_delta, 'of',  max_total
        for ii in range(0, len(self.counts)):
            col = self.counts[ii]
            if sum(col.values()) != max_total:
                ratio = sum(col.values()) / float(max_total)
                for letter in col:
                    col[letter] /= ratio

    def calculate_matrix(self):
        """turn the counts into a log likelihood (scoring) matrix"""
        ll_matrix = []
        max_score = 0
        last_total = False
        for col in self.counts:
            total = float(sum(col.values()))
            if last_total and abs(total - last_total) > 0.0001:
                raise ValueError(
                    "Unequal number of bases in different columns, motif %s, should have been %f, was %f: \n%s %s "
                    % (self.name, last_total, total, self.counts, col)
                )
            last_total = total
            pseudo_count = (
                0.04
            )  # follwing the argument in nishada et all, Psedocounts for transcription factor binding sides, we use a very small pseudocount
            ll_col = {}
            for letter in col:
                p = (col[letter] + pseudo_count / 4.0) / (total + pseudo_count)
                b = self.background_distribution[letter]
                ll = math.log(p / b, 2)
                ll_col[letter] = ll
            ll_matrix.append(ll_col)
            max_score += max(ll_col.values())
        return ll_matrix, max_score

    def fill_scanning_matrix(self, iupac=True):
        """Turn a loglikelihood list into a scanning/scoring matrix"""
        matrix = np.zeros((len(self.ll), 256), dtype=np.float32)
        for pos in range(0, len(self.ll)):
            matrix[pos][ord("A")] = self.ll[pos]["A"]
            matrix[pos][ord("C")] = self.ll[pos]["C"]
            matrix[pos][ord("G")] = self.ll[pos]["G"]
            matrix[pos][ord("T")] = self.ll[pos]["T"]
            if iupac:
                matrix[pos][ord("R")] = max(self.ll[pos]["A"], self.ll[pos]["G"])
                matrix[pos][ord("Y")] = max(self.ll[pos]["C"], self.ll[pos]["T"])
                matrix[pos][ord("M")] = max(self.ll[pos]["A"], self.ll[pos]["C"])
                matrix[pos][ord("K")] = max(self.ll[pos]["T"], self.ll[pos]["G"])
                matrix[pos][ord("W")] = max(self.ll[pos]["T"], self.ll[pos]["A"])
                matrix[pos][ord("S")] = max(self.ll[pos]["C"], self.ll[pos]["G"])
                matrix[pos][ord("B")] = max(
                    self.ll[pos]["C"], self.ll[pos]["T"], self.ll[pos]["G"]
                )
                matrix[pos][ord("D")] = max(
                    self.ll[pos]["A"], self.ll[pos]["T"], self.ll[pos]["G"]
                )
                matrix[pos][ord("H")] = max(
                    self.ll[pos]["A"], self.ll[pos]["T"], self.ll[pos]["C"]
                )
                matrix[pos][
                    ord("N")
                ] = (
                    0
                )  # min(motif.ll[pos]['A'],motif.ll[pos]['C'], motif.ll[pos]['G'], motif.ll[pos]['G'])
        return matrix

    def reverse_scanning_matrix(self, iupac=True):
        """reverse complement a scanning matrix"""
        # reverse
        rev_matrix = np.flipud(self.forward_matrix).copy()
        # complement
        for pos in range(0, self.forward_matrix.shape[0]):
            rev_matrix[pos][ord("A")], rev_matrix[pos][ord("T")] = (
                rev_matrix[pos][ord("T")],
                rev_matrix[pos][ord("A")],
            )
            rev_matrix[pos][ord("C")], rev_matrix[pos][ord("G")] = (
                rev_matrix[pos][ord("G")],
                rev_matrix[pos][ord("C")],
            )
            if iupac:
                rev_matrix[pos][ord("R")] = max(
                    rev_matrix[pos][ord("A")], rev_matrix[pos][ord("G")]
                )
                rev_matrix[pos][ord("Y")] = max(
                    rev_matrix[pos][ord("C")], rev_matrix[pos][ord("T")]
                )
                rev_matrix[pos][ord("M")] = max(
                    rev_matrix[pos][ord("A")], rev_matrix[pos][ord("C")]
                )
                rev_matrix[pos][ord("K")] = max(
                    rev_matrix[pos][ord("T")], rev_matrix[pos][ord("G")]
                )
                rev_matrix[pos][ord("W")] = max(
                    rev_matrix[pos][ord("T")], rev_matrix[pos][ord("A")]
                )
                rev_matrix[pos][ord("S")] = max(
                    rev_matrix[pos][ord("C")], rev_matrix[pos][ord("G")]
                )
                rev_matrix[pos][ord("B")] = max(
                    rev_matrix[pos][ord("C")],
                    rev_matrix[pos][ord("T")],
                    rev_matrix[pos][ord("G")],
                )
                rev_matrix[pos][ord("D")] = max(
                    rev_matrix[pos][ord("A")],
                    rev_matrix[pos][ord("T")],
                    rev_matrix[pos][ord("G")],
                )
                rev_matrix[pos][ord("H")] = max(
                    rev_matrix[pos][ord("A")],
                    rev_matrix[pos][ord("T")],
                    rev_matrix[pos][ord("C")],
                )
                rev_matrix[pos][
                    ord("N")
                ] = (
                    0
                )  # min(motif.matrix[pos]['A'],motif.matrix[pos]['C'], motif.matrix[pos]['G'], motif.matrix[pos]['G'])
        return rev_matrix

    def fill_min_needed_matrix(self, matrix, threshold):
        """create a matrix of needed scores at position i to be still able to score a hit"""
        min_needed_matrix = np.zeros((matrix.shape[0],), dtype=np.float32)
        # print matrix
        # print 'max_score', max_score
        min_score_needed = threshold - np.max(matrix[-1, :])
        for ii in range(matrix.shape[0] - 2, -1, -1):
            min_needed_matrix[ii] = min_score_needed
            min_score_needed -= np.max(matrix[ii, :])
        # print min_needed_matrix
        return min_needed_matrix

    def scan(self, sequence, threshold, keep_list=True):
        """scan sequences for hits of motif at >= threshold.
        Result is (dataframe{start, end, score}, cumulative_score, best_score)
        where start < end == forward strand, start > end = backward strand)

        Optionally: just count cumulative and best score (keep_list = False),
        then it returns only (cum_score, best_score)
        """
        sequence = sequence.upper()
        if hasattr(sequence, "encode"):
            sequence = sequence.encode("latin-1")  # motifs are search on bytes!
        self.do_load()
        keep_list = int(keep_list)
        matrix = self.forward_matrix
        revmatrix = self.reverse_matrix
        min_needed_matrix = self.fill_min_needed_matrix(self.forward_matrix, threshold)
        min_rev_needed_matrix = self.fill_min_needed_matrix(
            self.reverse_matrix, threshold
        )
        startpoints = []
        endpoints = []
        scores = []
        other_results = []
        threshold = float(threshold) - 0.0001
        cumscore, maxscore, startpoints, endpoints, scores = _motifs.scan_pwm(
            sequence,
            self.forward_matrix,
            self.reverse_matrix,
            threshold,
            min_needed_matrix,
            min_rev_needed_matrix,
            keep_list,
        )
        if keep_list:
            result = pd.DataFrame(
                {
                    "start": np.array(startpoints, dtype=np.uint32),
                    "stop": np.array(endpoints, dtype=np.uint32),
                    "score": scores,
                }
            )
            return (result, cumscore, maxscore)  # cumulative score  # maximal score
        else:
            return (cumscore, maxscore)

    def score(self, sequence):
        if len(sequence) != len(self.ll):
            raise ValueError(
                "sequence must have same length as motif (mostly an implementation limitation"
            )
        forward_score = 0
        for ii, letter in enumerate(sequence.upper()):
            forward_score += self.ll[ii][letter]
        reverse_score = 0
        for ii, letter in enumerate(reverse_complement_iupac(sequence).upper()):
            reverse_score += self.ll[ii][letter]
        return max(forward_score, reverse_score)

    def plot_logo(
        self, output_filename, kmer_count=200, width=3, height=1.5, reverse=False
    ):
        def do_plot(
            output_filename=output_filename, kmer_count=kmer_count, reverse=reverse
        ):
            seqs = self.bogus_kmers(kmer_count)
            op = open(output_filename + ".seqs.txt", "wb")
            op.write(b"\n".join([common.to_bytes(s) for s in seqs]))
            op.close()
            op = open(output_filename + ".counts.txt", "wb")
            op.write(repr(self.counts).encode("utf-8"))
            op.close()
            if reverse:
                plots.plot_sequences(seqs, output_filename, width=width, height=height)
            else:
                plots.plot_sequences(
                    [reverse_complement_iupac(x) for x in seqs],
                    output_filename,
                    width=width,
                    height=height,
                )

        return ppg.FileGeneratingJob(output_filename, do_plot).depends_on(
            ppg.ParameterInvariant(
                output_filename + "_params", ((kmer_count, width, height, reverse))
            )
        )

    def write_matrix(self, output_filename):
        def do_write():
            op = open(output_filename, "wb")
            op.write("A\tC\tG\tT\n")
            for col in self.ll:
                op.write(
                    "%2.e\t%2.e\t%2.e\t%2.e\t\n"
                    % (col["A"], col["C"], col["G"], col["T"])
                )
            op.close()

        return ppg.FileGeneratingJob(output_filename, do_write)

    def plot_support(self, sequences, output_filename):
        def calc():
            max_score = self.max_score
            best_scores_by_sequence = []
            for s in sequences.df["seq"]:
                cum_score, best_score = self.scan(s, 0, keep_list=False)
                best_scores_by_sequence.append(best_score)
            best_scores_by_sequence.sort()
            best_scores_by_sequence.reverse()
            so_far = 0
            res = {"Threshold": [], "Support": []}
            for threshold, entries in itertools.groupby(best_scores_by_sequence):
                so_far += len(list(entries))
                res["Threshold"].append(threshold)
                res["Support"].append(so_far)
            return pd.DataFrame(res)

        def plot(df):
            return (
                dp(df)
                .p9()
                .add_line("Threshold", "Support")
                .add_scatter("Threshold", "Support")
                .title(
                    "Support for %s\nin %s\n at various thresholds"
                    % (self.name, sequences.name)
                )
                .scale_y_continuous(limits=[0, len(sequences)])
            ).pd

        def plot_tiny(df):
            return (
                dp(df)
                .p9()
                .add_line("Threshold", "Support")
                .set_base_size(5)
                .smaller_margins()
                .hide_x_axis_title()
                .hide_y_axis_title()
                .hide_x_axis_labels()
                .hide_grid_minor()
                .hide_grid()
                .scale_y_continuous(limits=[0, len(sequences)])
            ).pd

        job = (
            ppg.PlotJob(output_filename, calc, plot)
            .depends_on(sequences.load())
            .depends_on(self.load())
        )
        ppg.Job.depends_on(job, sequences.load())
        tiny_job = job.add_another_plot(
            output_filename.replace(".png", ".tiny.png"),
            plot_tiny,
            render_args={"height": 1, "width": 1},
        )
        ppg.Job.depends_on(tiny_job, sequences.load())
        return job

    def tomtom(self, motif_source, output_directory):
        """Call TomTom to score this motif (=query) against all in source (=database)
        and store output in output_directory/tomtom"""
        raise ValueError("TODO What to do about meme")
        import meme

        def call():
            meme.run_tomtom(self, motif_source, output_directory)
            shutil.move(
                os.path.join(output_directory, "tomtom_out", "tomtom.html"),
                os.path.join(output_directory, "tomtom_out", "index.html"),
            )

        return (
            ppg.FileGeneratingJob(
                os.path.join(output_directory, "tomtom_out", "index.html"), call
            )
            .depends_on(self.load())
            .depends_on(motif_source.get_dependencies())
        )

    def plot_acceptance_probability_depending_on_threshold(self, output_filename):
        def calc():
            scores = []
            for sequence in iter_all_kmers(len(self)):
                scores.append(max(self.score(sequence), 0))
            scores.sort()
            res = {"Score": [], "Accepts": []}
            total = len(scores)
            for score, group in itertools.groupby(scores):
                res["Score"].append(score)
                res["Accepts"].append(total)
                total -= len(list(group))
            return pd.DataFrame(res)

        def plot(df):
            return dp(df).p9().add_line("Score", "Accepts").pd

        return ppg.PlotJob(output_filename, calc, plot, skip_table=True)


def PWMMotif_FromSeqs(name, seqs, background_distribution):
    """Create a PWMMotif from a list of equi-length sequences"""
    counts = []
    for ii in range(0, len(seqs[0])):
        counts.append({"A": 0, "C": 0, "G": 0, "T": 0})
    for seq in seqs:
        for ii, letter in enumerate(seq.upper()):
            counts[ii][letter] += 1
    return PWMMotif(name, counts, background_distribution)


def PWMMotif_FromText(name, iupac_motif, genome, n_seq=5 * 3 * 10):
    """Create a PWMMotif from a iupac string"""
    counts = []
    for letter in iupac_motif.upper():
        c = {"A": 0, "C": 0, "G": 0, "T": 0}
        if letter in "ACGT":
            c[letter] += n_seq
        elif letter == "N":
            for letter in c:
                c[letter] += n_seq / 4.0
        elif letter == "R":
            c["A"] += n_seq / 2.0
            c["G"] += n_seq / 2.0
        elif letter == "Y":
            c["C"] += n_seq / 2.0
            c["T"] += n_seq / 2.0
        elif letter == "M":
            c["A"] += n_seq / 2.0
            c["C"] += n_seq / 2.0
        elif letter == "K":
            c["T"] += n_seq / 2.0
            c["G"] += n_seq / 2.0
        elif letter == "W":
            c["A"] += n_seq / 2.0
            c["T"] += n_seq / 2.0
        elif letter == "S":
            c["C"] += n_seq / 2.0
            c["G"] += n_seq / 2.0
        elif letter == "B":
            c["C"] += n_seq / 3.0
            c["G"] += n_seq / 3.0
            c["T"] += n_seq / 3.0
        elif letter == "D":
            c["A"] += n_seq / 3.0
            c["G"] += n_seq / 3.0
            c["T"] += n_seq / 3.0
        elif letter == "H":
            c["A"] += n_seq / 3.0
            c["C"] += n_seq / 3.0
            c["T"] += n_seq / 3.0
        elif letter == "V":
            c["A"] += n_seq / 3.0
            c["C"] += n_seq / 3.0
            c["G"] += n_seq / 3.0
        else:
            raise ValueError(
                "Invalid letter in IUPAC string %s: % s" % (iupac_motif.upper(), letter)
            )
        counts.append(c)
    return PWMMotif(name, counts, genome)
