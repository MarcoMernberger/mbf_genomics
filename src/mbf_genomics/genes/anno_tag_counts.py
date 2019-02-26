"""New style (fast) tag count annos

Use these for new projects.

"""
from mbf_genomics.annotator import Annotator
import collections
import numpy as np
import pypipegraph as ppg
import hashlib
import pandas as pd
from mbf_genomes.intervals import merge_intervals


# ## Base classes and strategies - skip these if you just care about using TagCount annotators


class _CounterStrategyBase:
    def count_reads(self, interval_strategy, genome, bamfile, reverse=False):
        lookup = {}
        for chr in genome.get_chromosome_lengths():
            lookup.update(
                self.count_gene_reads_on_chromosome(
                    interval_strategy, genome, bamfile, chr, False
                )
            )
        self.sanity_check(genome, lookup, bamfile, interval_strategy)
        return lookup


class CounterStrategyStranded(_CounterStrategyBase):
    """This counter fetches() all reads on one chromosome at once, then matches them to the respective intervals
    defined by self.strategy.get_intervals"""

    def __init__(self):
        self.disable_sanity_check = False

    def count_gene_reads_on_chromosome(
        self, interval_strategy, genome, samfile, chr, reverse=False
    ):
        """Return a dict of gene_stable_id -> spliced exon matching read counts"""
        # basic algorithm: for each aligned region, check whether it overlaps a (merged) exon
        # if so, consider this read a hit for the gene with that exon
        # reads may hit multiple genes (think overlapping genes, where each could have generated the read)
        # multi aligned reads may count for multiple genes
        # but not for the same gene multiple times
        tree_forward, tree_reverse, gene_to_no = interval_strategy.get_intervals(
            genome, chr
        )

        counts = {}
        # one multi aligned read should only count once for a gene
        # further alignments into the same gene don't count
        # further alignments into other genes do though.
        # because of that' we need to keep the read names in a set.
        # uargh.
        for read in samfile.fetch(chr, 0, genome.get_chromosome_lengths()[chr]):
            seen = set()
            if not reverse:
                if read.is_reverse:
                    tree = tree_reverse
                else:
                    tree = tree_forward
            else:
                if read.is_reverse:
                    tree = tree_forward
                else:
                    tree = tree_reverse
            for sub_start, sub_stop in read.get_reference_regions():
                for x in tree.find(sub_start, sub_stop):
                    seen.add(
                        x.value
                    )  # one read matches one gene once. It might match multiple genes though.
            for ii in seen:
                if ii not in counts:
                    counts[ii] = set()
                counts[ii].add(read.qname)
                # counts[ii] += 1
        real_counts = {}
        no_to_gene = dict((v, k) for (k, v) in gene_to_no.items())
        for ii in counts:
            gene_stable_id = no_to_gene[ii]
            real_counts[gene_stable_id] = len(counts[ii])
        # if not real_counts and gene_to_no and len(chr) == 1:
        # raise ValueError("Nothing counted %s" % chr)
        return real_counts

    def sanity_check(self, genome, lookup, bam_file, interval_strategy):
        # the sanity check allows the exon tag count annotator to detect if you've reversed your reads
        if self.disable_sanity_check:
            return
        longest_chr = list(
            sorted([(v, k) for (k, v) in genome.get_chromosome_lengths().items()])
        )[-1][1]
        reverse_count = self.count_gene_reads_on_chromosome(
            interval_strategy, genome, bam_file, longest_chr, reverse=True
        )
        error_count = 0
        for gene_stable_id in reverse_count:
            if reverse_count[gene_stable_id] > 100 and reverse_count[
                gene_stable_id
            ] > 1.1 * (lookup[gene_stable_id] if gene_stable_id in lookup else 0):
                error_count += 1

        if error_count > 0.1 * len(reverse_count):
            # import cPickle
            # with open('debug.pickle','wb') as op:
            # cPickle.dump(lookup, op)
            # cPickle.dump(reverse_count, op)
            raise ValueError(
                "Found at least %.2f%% of genes on longest chromosome to have a reverse read count (%s) was above 110%% of the exon read count. This indicates that this lane should have been reversed before alignment. Set reverse_reads=True on your Lane object"
                % (100.0 * error_count / len(reverse_count), self.__class__.__name__)
            )


class CounterStrategyUnstranded(_CounterStrategyBase):
    """This counter fetches() all reads on one chromosome at once, then matches them to the respective intervals
    defined by self.get_intervals"""

    def count_gene_reads_on_chromosome(
        self, interval_strategy, genome, samfile, chr, reverse=False
    ):
        """Return a dict of gene_stable_id -> spliced exon matching read counts"""
        # basic algorithm: for each aligned region, check whether it overlaps a (merged) exon
        # if so, consider this read a hit for the gene with that exon
        # reads may hit multiple genes (think overlapping genes, where each could have generated the read)
        # multi aligned reads may count for multiple genes
        # but not for the same gene multiple times
        tree_forward, tree_reverse, gene_to_no = interval_strategy.get_intervals(
            genome, chr
        )

        counts = {}
        # one multi aligned read should only count once for a gene
        # further alignments into the same gene don't count
        # further alignments into other genes do though.
        # because of that' we need to keep the read names in a set.
        # uargh.
        for read in samfile.fetch(chr, 0, genome.get_chromosome_lengths()[chr]):
            seen = set()
            for sub_start, sub_stop in read.get_reference_regions():
                for t in tree_forward, tree_reverse:
                    for x in t.find(sub_start, sub_stop):
                        seen.add(x.value)
            for ii in seen:
                if ii not in counts:
                    counts[ii] = set()
                counts[ii].add(read.qname)
                # counts[ii] += 1
        real_counts = {}
        no_to_gene = dict((v, k) for (k, v) in gene_to_no.items())
        for ii in counts:
            gene_stable_id = no_to_gene[ii]
            real_counts[gene_stable_id] = len(counts[ii])
        if not real_counts and gene_to_no and len(chr) == 1:
            print(counts)
            print(real_counts)
            print(not real_counts)
        #            raise ValueError("Nothing counted %s" % chr)
        return real_counts

    def sanity_check(self, genome, lookup, bam_file, interval_strategy):
        pass  # no op


class CounterStrategyWeightedStranded(_CounterStrategyBase):
    """Counts reads matching multiple genes as 1/hit_count
    for each gene"""

    def count_reads(self, interval_strategy, genome, bamfile, reverse=False):
        lookup = collections.defaultdict(int)
        read_storage = {}
        for chr, length in genome.get_chromosome_lengths().items():
            tree_forward, tree_reverse, gene_to_no = interval_strategy.get_intervals(
                genome, chr
            )
            no_to_gene = dict((v, k) for (k, v) in gene_to_no.items())

            for read in bamfile.fetch(chr, 0, length):
                if read.is_reverse:
                    tree = tree_reverse
                else:
                    tree = tree_forward
                genes_hit = set()
                for sub_start, sub_stop in read.get_reference_regions():
                    for x in tree.find(sub_start, sub_stop):
                        genes_hit.add(
                            no_to_gene[x.value]
                        )  # this is the gene_to_no encoded number of the gene...
                if genes_hit:
                    count_per_hit = 1.0 / len(genes_hit)
                else:
                    count_per_hit = 1
                nh = read.get_tag("NH")
                if nh == 1:
                    for gene_stable_id in genes_hit:
                        lookup[gene_stable_id] += count_per_hit
                else:
                    if read.qname not in read_storage:
                        read_storage[read.qname] = [(genes_hit, count_per_hit)]
                    else:
                        if read_storage[read.qname] is False:
                            import pprint

                            raise ValueError(
                                "Unaccounted for reads: \n%s"
                                % pprint.pformat(read_storage)[:10000]
                            )
                        read_storage[read.qname].append((genes_hit, count_per_hit))
                        if (
                            len(read_storage[read.qname]) == nh
                        ):  # time to store the values
                            all_hits = {}
                            for genes_hit, count_per_hit in read_storage[read.qname]:
                                for gene_stable_id in genes_hit:
                                    all_hits[gene_stable_id] = count_per_hit
                            per_hit = float(len(all_hits))
                            for gene_stable_id in all_hits:
                                lookup[gene_stable_id] += count_per_hit / per_hit
                            read_storage[read.qname] = False
        for x in read_storage.values():
            if x is not False:
                debug = [(x, y) for (x, y) in read_storage.items() if y is not False]
                import pprint

                raise ValueError(
                    "Unaccounted for reads: \n%s" % pprint.pformat(debug)[:10000]
                )
        return lookup


class _IntervalStrategy:
    def deps(self, genome):
        return [self.load_intervals(genome)]

    def load_intervals(self, genome):
        def load():
            return self.do_load(genome)

        if ppg.inside_ppg():
            res = ppg.DataLoadingJob((genome.name + self.key), load).depends_on(
                [
                    genome.job_genes,
                    genome.job_transcripts,
                    ppg.FunctionInvariant(
                        self.__class__.__name__ + "_load_func", type(self).do_load
                    ),
                ]
            )
            res.ignore_code_changes()  # n o sense in tracing that trivial load above
            return res
        else:

            class Dummy:
                def callback(inner_self):
                    return self.do_load(genome)

            return Dummy()

    def get_intervals(self, genome, chr):
        if not ppg.inside_ppg() and not hasattr(genome, self.key):
            self.do_load(genome)
        return getattr(genome, self.key)[0][chr]

    def get_interval_lengths_by_gene(self, genome):
        return getattr(genome, self.key)[1]

    def do_load(self, genome):
        raise NotImplementedError()


class IntervalStrategyGene(_IntervalStrategy):
    """Count from TSS to TES"""

    key = "_bx_gene_intervals"

    def do_load(self, genome):
        import bx.intervals

        _bx_gene_intervals = {}
        gene_info = genome.df_genes
        length_by_gene = {}
        for chr in genome.get_chromosome_lengths():
            tree_forward = bx.intervals.IntervalTree()
            tree_reverse = bx.intervals.IntervalTree()
            gene_to_no = {}
            ii = 0
            for row in gene_info[gene_info["chr"] == chr].iterrows():
                start = min(row[1]["tss"], row[1]["tes"])
                stop = max(row[1]["tss"], row[1]["tes"])
                strand = 1 if row[1]["tss"] < row[1]["tes"] else -1
                if strand == 1:
                    tree_forward.insert_interval(bx.intervals.Interval(start, stop, ii))
                else:
                    tree_reverse.insert_interval(bx.intervals.Interval(start, stop, ii))
                gene_stable_id = row[0]
                gene_to_no[gene_stable_id] = ii
                length_by_gene[gene_stable_id] = stop - start
                ii += 1
            _bx_gene_intervals[chr] = tree_forward, tree_reverse, gene_to_no
        setattr(genome, self.key, (_bx_gene_intervals, length_by_gene))


class IntervalStrategyExon(_IntervalStrategy):
    """count all exons"""

    key = "_bx_exon_intervals"

    def do_load(self, genome):
        import bx.intervals

        _bx_exon_intervals = {}
        length_by_gene = collections.Counter()
        for chr in genome.get_chromosome_lengths():
            tree_forward = bx.intervals.IntervalTree()
            tree_reverse = bx.intervals.IntervalTree()
            gene_to_no = {}
            ii = -1
            last_stable_id = ""
            exon_info = genome.df_exons
            exon_info = merge_intervals(exon_info)
            if not isinstance(exon_info, pd.DataFrame):
                exon_info = exon_info.to_pandas()
            exon_info = exon_info[exon_info.chr == chr]
            for rowno, row in exon_info.iterrows():
                if last_stable_id != row["gene_stable_id"]:
                    ii += 1
                    last_stable_id = row["gene_stable_id"]
                    gene_to_no[last_stable_id] = ii
                if row["strand"] == 1:
                    tree_forward.insert_interval(
                        bx.intervals.Interval(row["start"], row["stop"], ii)
                    )
                else:
                    tree_reverse.insert_interval(
                        bx.intervals.Interval(row["start"], row["stop"], ii)
                    )
                length_by_gene[row["gene_stable_id"]] += row["stop"] - row["start"]

            _bx_exon_intervals[chr] = tree_forward, tree_reverse, gene_to_no
        setattr(genome, self.key, (_bx_exon_intervals, length_by_gene))


def get_all_gene_exons_protein_coding(genome):
    result = []
    for g in genome.genes:
        exons = g.exons_protein_coding_merged
        exons = exons.assign(gene_stable_id=g.gene_stable_id)
        result.append(exons)
    if len(result) == 0:
        result = pd.DataFrame({})
    else:
        result = pd.concat(result, sort=False)
        return result


class IntervalStrategyExonSmart(_IntervalStrategy):
    """For protein coding genes: count only in exons of protein-coding transcripts.
    For other genes: count all exons"""

    key = "_bx_exon_smart_intervals_protein_coding"

    def do_load(self, genome):
        import bx.intervals

        _bx_exon_intervals = {}
        length_by_gene = collections.Counter()
        exon_info_all = get_all_gene_exons_protein_coding(
            genome
        )  # these are already merged
        for chr in genome.get_chromosome_lengths():
            tree_forward = bx.intervals.IntervalTree()
            tree_reverse = bx.intervals.IntervalTree()
            gene_to_no = {}
            ii = -1
            last_stable_id = ""

            exon_info = exon_info_all[exon_info_all.chr == chr]
            for rowno, row in exon_info.iterrows():
                if last_stable_id != row["gene_stable_id"]:
                    ii += 1
                    last_stable_id = row["gene_stable_id"]
                    gene_to_no[last_stable_id] = ii
                if row["strand"] == 1:
                    tree_forward.insert_interval(
                        bx.intervals.Interval(row["start"], row["stop"], ii)
                    )
                else:
                    tree_reverse.insert_interval(
                        bx.intervals.Interval(row["start"], row["stop"], ii)
                    )
                length_by_gene[row["gene_stable_id"]] += row["stop"] - row["start"]

            _bx_exon_intervals[chr] = tree_forward, tree_reverse, gene_to_no
        setattr(genome, self.key, (_bx_exon_intervals, length_by_gene))


# Now the actual tag count annotators


class _FastTagCounter(Annotator):
    def __init__(
        self, aligned_lane, count_strategy, interval_strategy, column_name, column_desc
    ):
        if not hasattr(aligned_lane, "get_bam"):
            raise ValueError("_FastTagCounter only accepts aligned lanes!")
        self.aligned_lane = aligned_lane
        self.genome = self.aligned_lane.genome
        self.count_strategy = count_strategy
        self.interval_strategy = interval_strategy
        self.columns = [(column_name % (self.aligned_lane.short_name,)).strip()]
        self.cache_name = hashlib.md5(self.columns[0].encode("utf-8")).hexdigest()
        self.column_properties = {self.columns[0]: {"description": column_desc}}
        self.vid = aligned_lane.vid

    def calc(self, df):
        lookup = {}
        bam_file = self.aligned_lane.get_bam()
        lookup = self.count_strategy.count_reads(
            self.interval_strategy, self.genome, bam_file
        )
        result = []
        for gene_stable_id in df["gene_stable_id"]:
            result.append(lookup[gene_stable_id] if gene_stable_id in lookup else 0)
        result = np.array(result, dtype=np.float)
        return pd.Series(result)

    def deps(self, genes):
        return [self.aligned_lane.load()] + self.interval_strategy.deps(genes.genome)


# ## Raw tag count annos for analysis usage


class ExonSmartStranded(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyStranded(),
            IntervalStrategyExonSmart(),
            "Exon, protein coding, stranded smart tag count %s",
            "Tag count inside exons of protein coding transcripts (all if no protein coding transcripts) exons, correct strand only",
        )


class ExonSmartUnstranded(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyUnstranded(),
            IntervalStrategyExonSmart(),
            "Exon, protein coding, unstranded smart tag count %s",
            "Tag count inside exons of protein coding transcripts (all if no protein coding transcripts)  both strands",
        )


class ExonStranded(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyStranded(),
            IntervalStrategyExon(),
            "Exon, protein coding, stranded tag count %s",
            "Tag count inside exons of protein coding transcripts (all if no protein coding transcripts) exons, correct strand only",
        )


class ExonUnstranded(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyUnstranded(),
            IntervalStrategyExon(),
            "Exon, protein coding, unstranded tag count %s",
            "Tag count inside exons of protein coding transcripts (all if no protein coding transcripts)  both strands",
        )


class ExonSmartWeightedStranded(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyWeightedStranded(),
            IntervalStrategyExonSmart(),
            "Exon, protein coding, weighted stranded tag count %s",
            "Tag count inside exons of protein coding transcripts (all if no protein coding transcripts) exons, correct strand only. Multi-matching reads proportionally allocated",
        )


class GeneStranded(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyStranded(),
            IntervalStrategyGene(),
            "Gene, stranded tag count %s",
            "Tag count inside gene body (tss..tes), correct strand only",
        )


class GeneUnstranded(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyUnstranded(),
            IntervalStrategyGene(),
            "Gene unstranded tag count %s",
            "Tag count inside gene body (tss..tes), both strands",
        )


# ## Normalizing annotators - convert raw tag counts into something normalized


class NormalizationCPM(Annotator):
    """Normalize to 1e6 by taking the sum of all genes"""

    def __init__(self, raw_anno):
        self.genome = raw_anno.genome
        self.raw_anno = raw_anno
        self.vid = raw_anno.vid
        self.normalize_to = 1e6
        self.aligned_lane = raw_anno.aligned_lane
        self.columns = [self.raw_anno.columns[0] + " CPM"]
        self.cache_name = hashlib.md5(self.columns[0].encode("utf-8")).hexdigest()
        self.column_properties = {
            self.columns[0]: {
                "description": "Tag count inside protein coding (all if no protein coding transcripts) exons, normalized to 1e6 across all genes"
            }
        }

    def dep_annos(self):
        return [self.raw_anno]

    def calc(self, df):
        raw_counts = df[self.raw_anno.columns[0]]
        total = float(raw_counts.sum())
        # print raw_counts[:100]
        # print total, self.normalize_to
        result = raw_counts * (self.normalize_to / total)
        return pd.Series(result)


class NormalizationTPM(Annotator):
    """Normalize to transcripts per million, ie.
        count / length * (1e6 / (sum_i(count_/length_i)))

    """

    def __init__(self, raw_anno):
        self.genome = raw_anno.genome
        self.raw_anno = raw_anno
        self.vid = raw_anno.vid
        self.normalize_to = 1e6
        self.aligned_lane = raw_anno.aligned_lane
        self.columns = [self.raw_anno.columns[0] + " TPM"]
        self.cache_name = hashlib.md5(self.columns[0].encode("utf-8")).hexdigest()
        self.column_properties = {
            self.columns[0]: {"description": "transcripts per million"}
        }

    def dep_annos(self):
        return [self.raw_anno]

    def calc(self, df):
        raw_counts = df[self.raw_anno.columns[0]]
        length_by_gene = self.raw_anno.interval_strategy.get_interval_lengths_by_gene(
            self.genome
        )
        result = np.zeros(raw_counts.shape, float)
        for ii, gene_stable_id in enumerate(df["gene_stable_id"]):
            result[ii] = raw_counts[ii] / float(length_by_gene[gene_stable_id])
        total = float(result.sum())
        factor = 1e6 / total
        result = result * factor
        return pd.DataFrame({self.columns[0]: result})


class NormalizationCPMBiotypes(Annotator):
    """Tormalize to 1e6 by taking the sum of all [biotype, biotype2] genes.
    All other genes receive nan as their normalized value"""

    def __init__(self, raw_anno, biotypes):
        self.genome = raw_anno.genome
        if not isinstance(biotypes, tuple):
            raise ValueError("biotypes must be a tuple")
        self.biotypes = biotypes
        self.raw_anno = raw_anno
        self.vid = raw_anno.vid
        self.normalize_to = 1e6
        self.aligned_lane = raw_anno.aligned_lane
        self.columns = [
            self.raw_anno.columns[0] + " CPM(%s)" % (", ".join(sorted(biotypes)))
        ]
        self.cache_name = hashlib.md5(self.columns[0].encode("utf-8")).hexdigest()
        self.column_properties = {
            self.columns[0]: {
                "description": "Tag count inside protein coding (all if no protein coding transcripts) exons, normalized to 1e6 across genes in biotypes %s"
                % (biotypes,)
            }
        }

    def dep_annos(self):
        return [self.raw_anno]

    def calc(self, df):
        raw_counts = df[self.raw_anno.columns[0]].copy()
        ok = np.zeros(len(df), np.bool)
        for biotype in self.biotypes:
            ok |= df["biotype"] == biotype
        raw_counts[~ok] = np.nan
        total = float(raw_counts[ok].sum())
        result = raw_counts * (self.normalize_to / total)
        return pd.DataFrame({self.columns[0]: result})


class NormalizationTPMBiotypes(Annotator):
    """TPM, but only consider genes matching one of the biotypes
    All other genes receive nan as their normalized value"""

    def __init__(self, raw_anno, biotypes):
        self.genome = raw_anno.genome
        if not isinstance(biotypes, tuple):
            raise ValueError("biotypes must be a tuple")
        self.biotypes = biotypes
        self.raw_anno = raw_anno
        self.vid = raw_anno.vid
        self.normalize_to = 1e6
        self.aligned_lane = raw_anno.aligned_lane
        self.columns = [
            self.raw_anno.columns[0] + " tpm(%s)" % (", ".join(sorted(biotypes)))
        ]
        self.cache_name = hashlib.md5(self.columns[0].encode("utf-8")).hexdigest()
        self.column_properties = {
            self.columns[0]: {
                "description": "transcripts per million, considering only biotypes %s"
                % (biotypes,)
            }
        }

    def dep_annos(self):
        return [self.raw_anno]

    def calc(self, df):
        raw_counts = df[self.raw_anno.columns[0]].copy()
        ok = np.zeros(len(df), np.bool)
        for biotype in self.biotypes:
            ok |= df["biotype"] == biotype
        raw_counts[~ok] = np.nan
        print("raw", raw_counts)

        length_by_gene = self.raw_anno.interval_strategy.get_interval_lengths_by_gene(
            self.genome
        )
        result = np.zeros(raw_counts.shape, float)
        for ii, gene_stable_id in enumerate(df["gene_stable_id"]):
            result[ii] = raw_counts[ii] / float(length_by_gene[gene_stable_id])
        total = float(result[ok].sum())  # result.sum would be nan!
        factor = self.normalize_to / total
        result = result * factor
        return pd.DataFrame({self.columns[0]: result})


class NormalizationFPKM(Annotator):
    def __init__(self, raw_anno):
        self.genome = raw_anno.genome
        self.raw_anno = raw_anno
        self.vid = raw_anno.vid
        self.aligned_lane = raw_anno.aligned_lane
        self.columns = [self.raw_anno.columns[0] + " FPKM"]
        self.cache_name = hashlib.md5(self.columns[0].encode("utf-8")).hexdigest()
        self.column_properties = {
            self.columns[0]: {
                "description": "Tag count inside protein coding (all if no protein coding transcripts) exons, normalized to FPKM"
            }
        }

    def deps(self, genes):
        res = [self.raw_anno.interval_strategy.load_intervals(genes.genome)]
        return res

    def dep_annos(self):
        return [self.raw_anno]

    def calc(self, df):
        raw_counts = df[self.raw_anno.columns[0]]
        # RPKM = (CDS read count * 10^9) / (CDS length * total mapped read
        # count)
        total = float(raw_counts.sum())
        result = np.zeros(raw_counts.shape, float)
        length_by_gene = self.raw_anno.interval_strategy.get_interval_lengths_by_gene(
            self.genome
        )
        for ii, gene_stable_id in enumerate(df["gene_stable_id"]):
            result[ii] = raw_counts[ii] * 1e9 / (length_by_gene[gene_stable_id] * total)
        return pd.DataFrame({self.columns[0]: result})


class NormalizationFPKMBiotypes(Annotator):
    def __init__(self, raw_anno, biotypes):
        self.genome = raw_anno.genome
        self.raw_anno = raw_anno
        self.vid = raw_anno.vid
        if not isinstance(biotypes, tuple):
            raise ValueError("biotypes must be a tuple")
        self.biotypes = biotypes
        self.aligned_lane = raw_anno.aligned_lane
        self.columns = [self.raw_anno.columns[0] + " FPKM"]
        self.cache_name = hashlib.md5(self.columns[0].encode("utf-8")).hexdigest()
        self.column_properties = {
            self.columns[0]: {
                "description": "Fragments per kilobase per million, considering only %s"
                % (biotypes,)
            }
        }

    def deps(self, genes):
        res = [
            self.raw_anno.interval_strategy.load_intervals(genes.genome),
            ppg.ParameterInvariant(self.columns[0] + "_biotypes", list(self.biotypes)),
        ]
        return res

    def dep_annos(self):
        return [self.raw_anno]

    def calc(self, df):
        raw_counts = df[self.raw_anno.columns[0]]
        # RPKM = (CDS read count * 10^9) / (CDS length * total mapped read
        # count)
        ok = np.zeros(len(df), np.bool)
        for biotype in self.biotypes:
            ok |= df["biotype"] == biotype
        raw_counts[~ok] = np.nan
        total = float(raw_counts.sum())
        result = np.zeros(raw_counts.shape, float)
        length_by_gene = self.raw_anno.interval_strategy.get_interval_lengths_by_gene(
            self.genome
        )
        for ii, gene_stable_id in enumerate(df["gene_stable_id"]):
            result[ii] = raw_counts[ii] * 1e9 / (length_by_gene[gene_stable_id] * total)
        return pd.DataFrame({self.columns[0]: result})
