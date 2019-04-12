"""New style (fast) tag count annos

Use these for new projects.

"""
from mbf_genomics.annotator import Annotator
import collections
import numpy as np
import pypipegraph as ppg
import hashlib
import pandas as pd
from pathlib import Path
from dppd import dppd

dp, X = dppd()


# ## Base classes and strategies - skip these if you just care about using TagCount annotators


class _CounterStrategyBase:
    cores_needed = 1
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

    def extract_lookup(self, data):
        """Adapter for count strategies that have different outputs
        (e.g. one-hashmap-unstranded or two-hashmaps-one-forward-one-reversed)
        """
        return data


class CounterStrategyStranded(_CounterStrategyBase):
    """This counter fetches() all reads on one chromosome at once, then matches them to the respective intervals
    defined by self.strategy.get_interval_trees"""

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
        tree_forward, tree_reverse, gene_to_no = interval_strategy.get_interval_trees(
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
            for sub_start, sub_stop in read.get_blocks():
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


class CounterStrategyStrandedRust:
    cores_needed = -1
    def __init__(self):
        self.disable_sanity_check = False

    def count_reads(self, interval_strategy, genome, bamfile, reverse=False):
        # bam_filename = bamfil
        bam_filename = bamfile.filename.decode("utf-8")
        bam_index_name = bamfile.index_filename
        if bam_index_name is None:
            bam_index_name = bam_filename + ".bai"
        else:
            bam_index_name = str(bam_index_name)

        intervals = interval_strategy._get_interval_tuples_by_chr(genome)
        gene_intervals = IntervalStrategyGene()._get_interval_tuples_by_chr(genome)
        from mbf_bam import count_reads_stranded

        res = count_reads_stranded(
            bam_filename, bam_index_name, intervals, gene_intervals
        )
        self.sanity_check(res)
        return res

    def sanity_check(self, foward_and_reverse):
        if self.disable_sanity_check:
            return
        error_count = 0
        forward, reverse = foward_and_reverse
        for gene_stable_id, forward_count in forward.items():
            reverse_count = reverse.get(gene_stable_id, 0)
            if (reverse_count > 100) and (reverse_count > forward_count * 1.1):
                error_count += 1
        if error_count > 0.1 * len(forward):
            raise ValueError(
                "Found at least %.2f%% of genes to have a reverse read count (%s) "
                "above 110%% of the exon read count (and at least 100 tags). "
                "This indicates that this lane should have been reversed before alignment. "
                "Set reverse_reads=True on your Lane object"
                % (100.0 * error_count / len(forward), self.__class__.__name__)
            )

    def extract_lookup(self, data):
        """Adapter for count strategies that have different outputs
        (e.g. one-hashmap-unstranded or two-hashmaps-one-forward-one-reversed)
        """
        return data[0]


class CounterStrategyUnstranded(_CounterStrategyBase):
    """This counter fetches() all reads on one chromosome at once, 
    then matches them to the respective intervals
    defined by self.get_interval_trees"""

    def count_gene_reads_on_chromosome(
        self, interval_strategy, genome, samfile, chr, reverse=False
    ):
        """Return a dict of gene_stable_id -> spliced exon matching read counts"""
        # basic algorithm: for each aligned region, check whether it overlaps a (merged) exon
        # if so, consider this read a hit for the gene with that exon
        # reads may hit multiple genes (think overlapping genes, where each could have generated the read)
        # multi aligned reads may count for multiple genes
        # but not for the same gene multiple times
        tree_forward, tree_reverse, gene_to_no = interval_strategy.get_interval_trees(
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
            for sub_start, sub_stop in read.get_blocks():
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
        # if not real_counts and gene_to_no and len(chr) == 1:
        # print(counts)
        # print(real_counts)
        # print(not real_counts)
        #            raise ValueError("Nothing counted %s" % chr)
        return real_counts

    def sanity_check(self, genome, lookup, bam_file, interval_strategy):
        pass  # no op


class CounterStrategyUnstrandedRust(_CounterStrategyBase):
    cores_needed = -1
    def count_reads(self, interval_strategy, genome, bamfile, reverse=False):
        # bam_filename = bamfil
        bam_filename = bamfile.filename.decode("utf-8")
        bam_index_name = bamfile.index_filename
        if bam_index_name is None:
            bam_index_name = bam_filename + ".bai"
        else:
            bam_index_name = str(bam_index_name)

        intervals = interval_strategy._get_interval_tuples_by_chr(genome)
        gene_intervals = IntervalStrategyGene()._get_interval_tuples_by_chr(genome)
        # chr -> [gene_id, strand, [start], [stops]
        from mbf_bam import count_reads_unstranded

        res = count_reads_unstranded(
            bam_filename, bam_index_name, intervals, gene_intervals
        )
        return res


class CounterStrategyWeightedStranded(_CounterStrategyBase):
    """Counts reads matching multiple genes as 1/hit_count
    for each gene"""

    def count_reads(self, interval_strategy, genome, bamfile, reverse=False):
        lookup = collections.defaultdict(int)
        read_storage = {}
        for chr, length in genome.get_chromosome_lengths().items():
            tree_forward, tree_reverse, gene_to_no = interval_strategy.get_interval_trees(
                genome, chr
            )
            no_to_gene = dict((v, k) for (k, v) in gene_to_no.items())

            for read in bamfile.fetch(chr, 0, length):
                if read.is_reverse:
                    tree = tree_reverse
                else:
                    tree = tree_forward
                genes_hit = set()
                for sub_start, sub_stop in read.get_blocks():
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
                        if (
                            read_storage[read.qname] is False
                        ):  # pragma: no cover defensive
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
    def get_interval_trees(self, genome, chr):
        import bx.intervals

        by_chr = self._get_interval_tuples_by_chr(genome)
        tree_forward = bx.intervals.IntervalTree()
        tree_reverse = bx.intervals.IntervalTree()
        gene_to_no = {}
        ii = 0
        for tup in by_chr[chr]:  # stable_id, strand, [starts], [stops]
            length = 0
            for start, stop in zip(tup[2], tup[3]):
                if tup[1] == 1:
                    tree_forward.insert_interval(bx.intervals.Interval(start, stop, ii))
                else:
                    tree_reverse.insert_interval(bx.intervals.Interval(start, stop, ii))
                length += stop - start
            gene_stable_id = tup[0]
            gene_to_no[gene_stable_id] = ii
            ii += 1
        return tree_forward, tree_reverse, gene_to_no

    def get_interval_lengths_by_gene(self, genome):
        by_chr = self._get_interval_tuples_by_chr(genome)
        length_by_gene = {}
        for chr, tups in by_chr.items():
            for tup in tups:  # stable_id, strand, [starts], [stops]
                gene_stable_id = tup[0]
                length = 0
                for start, stop in zip(tup[2], tup[3]):
                    length += stop - start
                length_by_gene[gene_stable_id] = length
        return length_by_gene

    def _get_interval_tuples_by_chr(self, genome):  # pragma: no cover
        raise NotImplementedError()


class IntervalStrategyGene(_IntervalStrategy):
    """Count from TSS to TES"""

    def _get_interval_tuples_by_chr(self, genome):
        result = {chr: [] for chr in genome.get_chromosome_lengths()}
        gene_info = genome.df_genes
        for tup in gene_info[["chr", "start", "stop", "strand"]].itertuples():
            result[tup.chr].append((tup[0], tup.strand, [tup.start], [tup.stop]))
        return result


class IntervalStrategyExon(_IntervalStrategy):
    """count all exons"""

    def _get_interval_tuples_by_chr(self, genome):
        result = {chr: [] for chr in genome.get_chromosome_lengths()}
        for gene in genome.genes.values():
            exons = gene.exons_merged
            result[gene.chr].append(
                (gene.gene_stable_id, gene.strand, list(exons[0]), list(exons[1]))
            )
        return result


class IntervalStrategyIntron(_IntervalStrategy):
    """count all introns"""

    def _get_interval_tuples_by_chr(self, genome):
        result = {chr: [] for chr in genome.get_chromosome_lengths()}
        for gene in genome.genes.values():
            exons = gene.introns
            result[gene.chr].append(
                (gene.gene_stable_id, gene.strand, list(exons[0]), list(exons[1]))
            )
        return result


class IntervalStrategyExonSmart(_IntervalStrategy):
    """For protein coding genes: count only in exons of protein-coding transcripts.
    For other genes: count all exons"""

    def _get_interval_tuples_by_chr(self, genome):
        result = {chr: [] for chr in genome.get_chromosome_lengths()}
        for g in genome.genes.values():
            e = g.exons_protein_coding_merged
            if len(e[0]) == 0:
                e = g.exons_merged
            result[g.chr].append((g.gene_stable_id, g.strand, list(e[0]), list(e[1])))
        return result


class IntervalStrategyExonIntronClassification(_IntervalStrategy):
    """For QC purposes, defines all intron/exon intervals tagged
    with nothing but intron/exon

    See mbf_align.lanes.AlignedLane.register_qc_gene_strandedness
    
    """

    def _get_interval_tuples_by_chr(self, genome):
        from mbf_nested_intervals import IntervalSet

        coll = {chr: [] for chr in genome.get_chromosome_lengths()}
        ii = 0
        for g in genome.genes.values():
            exons = g.exons_overlapping
            if len(exons[0]) == 0:
                exons = g.exons_merged
            for start, stop in zip(*exons):
                coll[g.chr].append((start, stop, 0b0101 if g.strand == 1 else 0b0110))
            for start, stop in zip(*g.introns):
                coll[g.chr].append((start, stop, 0b1001 if g.strand == 1 else 0b1010))
        result = {}
        for chr, tups in coll.items():
            iset = IntervalSet.from_tuples_with_id(tups)
            # iset = iset.merge_split()
            iset = iset.merge_hull()
            if iset.any_overlapping():
                raise NotImplementedError("Should not be reached")
            result[chr] = []
            for start, stop, ids in iset.to_tuples_with_id():
                ids = set(ids)
                if len(ids) == 1:
                    id = list(ids)[0]
                    if id == 0b0101:
                        tag = "exon"
                        strand = +1
                    elif id == 0b0110:
                        tag = "exon"
                        strand = -1
                    elif id == 0b1001:
                        tag = "intron"
                        strand = +1
                    elif id == 0b1010:
                        tag = "intron"
                        strand = -1
                    else:  # pragma: no cover
                        raise NotImplementedError("Should not be reached")
                else:
                    down = 0
                    for i in ids:
                        down |= i
                    if down & 0b1100 == 0b1100:
                        tag = "both"
                    elif down & 0b0100 == 0b0100:
                        tag = "exon"
                    else:
                        tag = "intron"
                    if down & 0b11 == 0b11:
                        tag += "_undecidable"
                        strand = 1  # doesn't matter, but must be one or the other
                    elif down & 0b01:
                        strand = 1
                    else:
                        strand -= 1

                result[chr].append((tag, strand, [start], [stop]))
        return result


class IntervalStrategyWindows(_IntervalStrategy):
    """For QC purposes, spawn all chromosomes with 
    windows of the definied size

    See mbf_align.lanes.AlignedLane.register_qc_subchromosomal
    
    """

    def __init__(self, window_size):
        self.window_size = window_size

    def _get_interval_tuples_by_chr(self, genome):
        result = {}
        for chr, length in genome.get_chromosome_lengths().items():
            result[chr] = []
            for ii in range(0, length, self.window_size):
                result[chr].append(("%s_%i" % (chr, ii), 0, [ii], [ii + self.window_size]))
        return result


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
        self.columns = [(column_name % (self.aligned_lane.name,)).strip()]
        self.cache_name = hashlib.md5(self.columns[0].encode("utf-8")).hexdigest()
        self.column_properties = {self.columns[0]: {"description": column_desc}}
        self.vid = aligned_lane.vid
        self.cores_needed = count_strategy.cores_needed

    def calc(self, df):
        if ppg.inside_ppg():
            data = self._data
        else:
            data = self.calc_data()
        lookup = self.count_strategy.extract_lookup(data)
        result = []
        for gene_stable_id in df["gene_stable_id"]:
            result.append(lookup.get(gene_stable_id, 0))
        result = np.array(result, dtype=np.float)
        return pd.Series(result)

    def deps(self, _genes):
        return [self.load_data()]

    def calc_data(self):
        bam_file = self.aligned_lane.get_bam()
        return self.count_strategy.count_reads(
            self.interval_strategy, self.genome, bam_file
        )

    def load_data(self):
        cf = Path(ppg.util.global_pipegraph.cache_folder) / "FastTagCounters"
        cf.mkdir(exist_ok=True)
        return ppg.CachedAttributeLoadingJob(
            cf / self.cache_name, self, "_data", self.calc_data
        ).depends_on(self.aligned_lane.load())


# ## Raw tag count annos for analysis usage


class ExonSmartStrandedPython(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyStranded(),
            IntervalStrategyExonSmart(),
            "Exon, protein coding, stranded smart tag count %s",
            "Tag count inside exons of protein coding transcripts (all if no protein coding transcripts) exons, correct strand only",
        )


class ExonSmartStrandedRust(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyStrandedRust(),
            IntervalStrategyExonSmart(),
            "Exon, protein coding, stranded smart tag count %s",
            "Tag count inside exons of protein coding transcripts (all if no protein coding transcripts) exons, correct strand only",
        )


class ExonSmartUnstrandedPython(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyUnstranded(),
            IntervalStrategyExonSmart(),
            "Exon, protein coding, unstranded smart tag count %s",
            "Tag count inside exons of protein coding transcripts (all if no protein coding transcripts)  both strands",
        )


class ExonSmartUnstrandedRust(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyUnstrandedRust(),
            IntervalStrategyExonSmart(),
            "Exon, protein coding, unstranded smart tag count %s",
            "Tag count inside exons of protein coding transcripts (all if no protein coding transcripts)  both strands",
        )


class ExonStrandedPython(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyStranded(),
            IntervalStrategyExon(),
            "Exon, protein coding, stranded tag count %s",
            "Tag count inside exons of protein coding transcripts (all if no protein coding transcripts) exons, correct strand only",
        )


class ExonStrandedRust(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyStrandedRust(),
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


class GeneStrandedPython(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyStranded(),
            IntervalStrategyGene(),
            "Gene, stranded tag count %s",
            "Tag count inside gene body (tss..tes), correct strand only",
        )


class GeneStrandedRust(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyStrandedRust(),
            IntervalStrategyGene(),
            "Gene, stranded tag count %s",
            "Tag count inside gene body (tss..tes), correct strand only",
        )


class GeneUnstrandedPython(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyUnstranded(),
            IntervalStrategyGene(),
            "Gene unstranded tag count %s",
            "Tag count inside gene body (tss..tes), both strands",
        )


class GeneUnstrandedRust(_FastTagCounter):
    def __init__(self, aligned_lane):
        _FastTagCounter.__init__(
            self,
            aligned_lane,
            CounterStrategyUnstrandedRust(),
            IntervalStrategyGene(),
            "Gene unstranded tag count %s",
            "Tag count inside gene body (tss..tes), both strands",
        )


# we are keeping the python ones for now as reference implementations
GeneUnstranded = GeneUnstrandedRust
GeneStranded = GeneStrandedRust
ExonStranded = ExonStrandedRust
ExonSmartStranded = ExonSmartStrandedRust
ExonSmartUnstranded = ExonSmartUnstrandedRust

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
        return []

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
        return ppg.ParameterInvariant(
            self.columns[0] + "_biotypes", list(self.biotypes)
        )

    def dep_annos(self):
        return [self.raw_anno]

    def calc(self, df):
        raw_counts = df[self.raw_anno.columns[0]].copy()
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
