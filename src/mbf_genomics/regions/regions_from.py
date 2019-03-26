import pypipegraph as ppg
import pandas as pd
import numpy as np
import math

from .regions import GenomicRegions
from mbf_genomes.intervals import merge_intervals
from mbf_externals.util import to_string
from ..util import read_pandas


def verify_same_genome(list_of_grs):
    if len(set([x.genome for x in list_of_grs])) > 1:
        raise ValueError("Mixing GenomicRegions from different genomes not supported")


def GenomicRegions_FromGFF(
    name,
    filename,
    genome,
    filter_function=None,
    comment_char=None,
    on_overlap="raise",
    chromosome_mangler=None,
    fix_negative_coordinates=False,
    alternative_class=None,
    summit_annotator=None,
    vid=None,
):
    """Create a GenomicRegions from a gff file.
    You can filter entries with @filter_function(gff_entry_dict) -> Bool,
    remove comment lines starting with a specific character with @comment_char,
    mangle the chromosomes with @chromosome_mangler(str) -> str,
    replace negative coordinates with 0 (@fix_negative_coordinates),
    or provide an alternative constructor to call with @alternative_class
    """

    def load():
        from mbf_fileformats.gff import gffToDict

        entries = gffToDict(filename, comment_char=comment_char)
        data = {
            "chr": [],
            "start": [],
            "stop": [],
            "score": [],
            "strand": [],
            "name": [],
        }
        name_found = False
        for entry in entries:
            if filter_function and not filter_function(entry):
                continue
            if chromosome_mangler:
                chr = chromosome_mangler(entry["seqname"])
            else:
                chr = entry["seqname"]
            data["chr"].append(to_string(chr))
            start = entry["start"]
            if fix_negative_coordinates and start < 0:
                start = 0
            data["start"].append(start)
            data["stop"].append(entry["end"])
            data["score"].append(entry["score"])
            data["strand"].append(entry["strand"])
            name = entry["attributes"]["Name"] if "Name" in entry["attributes"] else ""
            data["name"].append(name)
            if name:
                name_found = True
        if not name_found:
            del data["name"]
        return pd.DataFrame(data)

    if alternative_class is None:
        alternative_class = GenomicRegions
    if ppg.inside_ppg():
        deps = [
            ppg.FileTimeInvariant(filename),
            ppg.ParameterInvariant(
                name + "_params_GenomicRegions_FromGFF",
                (comment_char, fix_negative_coordinates),
            ),
            ppg.FunctionInvariant(
                name + "_filter_func_GenomicRegions_FromGFF", filter_function
            ),
            ppg.FunctionInvariant(
                name + "_chromosome_manlger_GenomicRegions_FromGFF", chromosome_mangler
            ),
        ]
    else:
        deps = []
    return alternative_class(
        name, load, deps, genome, on_overlap, summit_annotator=summit_annotator, vid=vid
    )


def GenomicRegions_FromWig(
    name,
    filename,
    genome,
    enlarge_5prime=0,
    enlarge_3prime=0,
    on_overlap="raise",
    comment_char=None,
    summit_annotator=None,
    vid=None,
):
    """Create GenomicRegions from a Wiggle file.

    @enlarge_5prime and @enlarge_3prime increase the size of the fragments described in the wig in
    the respective direction (for example if a chip-chip array did not cover every base).
    @comment_char defines which lines to ignore in the wiggle (see {mbf_fileformats.wiggle_to_intervals})

    The resulting GenomicRegions has a column 'Score' that contains the wiggle score"""
    from mbf_fileformats.wiggle import wiggle_to_intervals

    def load():
        df = wiggle_to_intervals(filename, comment_char=comment_char)
        df["chr"] = [to_string(x) for x in df["chr"]]
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()
        if enlarge_5prime:
            df["start"] -= enlarge_5prime
        if enlarge_3prime:
            df["stop"] += enlarge_3prime
        return df

    if ppg.inside_ppg():
        deps = [ppg.FileTimeInvariant(filename)]
    else:
        deps = []

    return GenomicRegions(
        name, load, deps, genome, on_overlap, summit_annotator=summit_annotator, vid=vid
    )


def GenomicRegions_FromBed(
    name,
    filename,
    genome,
    chromosome_mangler=None,
    on_overlap="raise",
    filter_invalid_chromosomes=False,
    summit_annotator=None,
    sheet_name=None,
    vid=None,
):
    """Create GenomicRegions from a Bed file.

    The resulting GenomicRegions has a column 'Score' that contains the wiggle score"""
    from mbf_fileformats.bed import read_bed

    if chromosome_mangler is None:
        chromosome_mangler = lambda x: x  # noqa:E731

    def load():
        valid_chromosomes = set(genome.get_chromosome_lengths())
        data = {}
        entries = read_bed(filename)
        data["chr"] = np.array(
            [chromosome_mangler(to_string(e.refseq)) for e in entries], dtype=np.object
        )
        data["start"] = np.array([e.position for e in entries], dtype=np.int32)
        data["stop"] = np.array(
            [e.position + e.length for e in entries], dtype=np.int32
        )
        data["score"] = np.array([e.score for e in entries], dtype=np.float)
        data["strand"] = np.array([e.strand for e in entries], dtype=np.int8)
        data["name"] = np.array([to_string(e.name) for e in entries], dtype=np.object)
        data = pd.DataFrame(data)
        if filter_invalid_chromosomes:
            keep = [x in valid_chromosomes for x in data["chr"]]
            data = data[keep]
        res = data
        if len(res) == 0:
            raise ValueError("Emtpty Bed file - %s" % filename)
        if (np.isnan(res["score"])).all():
            res = res.drop(["score"], axis=1)
        if len(res["name"].unique()) == 1:
            res = res.drop(["name"], axis=1)
        return res

    if ppg.inside_ppg():
        deps = [
            ppg.FileTimeInvariant(filename),
            ppg.FunctionInvariant(name + "_chrmangler", chromosome_mangler),
        ]
    else:
        deps = []

    return GenomicRegions(
        name,
        load,
        deps,
        genome,
        on_overlap=on_overlap,
        summit_annotator=summit_annotator,
        sheet_name=sheet_name,
        vid=vid,
    )


def GenomicRegions_FromBigBed(
    name,
    filename,
    genome,
    chromosome_mangler=None,
    on_overlap="raise",
    summit_annotator=None,
    sheet_name=None,
    vid=None,
):
    """Create GenomicRegions from a BigBed file.
    @chromosome_mangler translates genome chromosomes into the bigbed's chromosomes!

    """
    from mbf_fileformats.bed import read_bigbed

    if chromosome_mangler is None:
        chromosome_mangler = lambda x: x  # noqa:E731

    def load():
        res = read_bigbed(filename, genome.get_chromosome_lengths(), chromosome_mangler)
        if (res["strand"] == 1).all():
            res = res.drop("strand", axis=1)
        if len(res) == 0:
            raise ValueError(
                "Emtpty BigBed file (or wrong chromosome names)- %s" % filename
            )
        return res

    if ppg.inside_ppg():
        deps = [
            ppg.FileTimeInvariant(filename),
            ppg.FunctionInvariant(name + "_chrmangler", chromosome_mangler),
        ]
    else:
        deps = []

    return GenomicRegions(
        name,
        load,
        deps,
        genome,
        on_overlap=on_overlap,
        summit_annotator=summit_annotator,
        sheet_name=sheet_name,
        vid=vid,
    )


def GenomicRegions_BinnedGenome(
    genome, bin_size, limit_to_chromosomes=None, new_name=None, vid=None
):
    """Create GenomicRegions than partition a given chromosome in bins
    of @bin_size bp
    """
    if limit_to_chromosomes and not hasattr(limit_to_chromosomes, "__iter__"):
        raise ValueError("Limit_to_chromosomes must be a list")

    def load():
        data = {"chr": [], "start": [], "stop": []}
        if limit_to_chromosomes:
            chr_plus_length = [
                (chr, chr_len)
                for (chr, chr_len) in genome.get_chromosome_lengths().items()
                if chr in limit_to_chromosomes
            ]
        else:
            chr_plus_length = genome.get_chromosome_lengths().items()
        if not chr_plus_length:
            raise ValueError("No chromosomes generated")
        for chr, chr_len in chr_plus_length:
            no_of_bins = int(math.ceil(float(chr_len) / bin_size))
            data["chr"].extend([chr] * no_of_bins)
            starts = np.array(range(0, chr_len, bin_size))
            stops = starts + bin_size - 1
            data["start"].extend(starts)
            data["stop"].extend(stops)
        res = pd.DataFrame(data)
        return res

    if new_name is None:
        name = genome.name + " binned (%i bp)" % bin_size
    else:
        name = new_name
    return GenomicRegions(name, load, [], genome, vid=vid)


def GenomicRegions_Union(
    name,
    list_of_grs,
    on_overlap="merge",
    summit_annotator=None,
    expand_by_x_bp=0,
    on_below_zero_by_expansion="raise",
    sheet_name="Overlaps",
):
    """Combine serveral GRs into one, similar to GR.union(), which handles only two.
    You can use @expand_by_x_bp = 1 to merge regions right next to each other (think binned genome)
    """
    allowed_on_below_zero_by_expansion = ("raise", "ignore", " drop", "truncate")
    if on_below_zero_by_expansion not in allowed_on_below_zero_by_expansion:
        raise ValueError(
            "on_below_zero_by_expansion  not in allowed values (%s)"
            % allowed_on_below_zero_by_expansion
        )
    verify_same_genome(list_of_grs)

    def load():
        if not expand_by_x_bp:
            dfs = [x.df[["chr", "start", "stop"]] for x in list_of_grs]
            return pd.concat(dfs, axis=0)
        else:
            dfs = [
                pd.DataFrame(
                    {
                        "chr": x.df["chr"],
                        "start": x.df["start"] - expand_by_x_bp,
                        "stop": x.df["stop"] + expand_by_x_bp,
                    }
                )
                for x in list_of_grs
            ]
            ret = pd.concat(dfs, axis=0)
            if (ret["start"] <= 0).any():
                if allowed_on_below_zero_by_expansion == "raise":
                    raise ValueError(
                        "expand_by_x_bp created regions with start < 0. Choose one of the options for on_below_zero_by_expansion (%s)"
                        % allowed_on_below_zero_by_expansion
                    )
                elif allowed_on_below_zero_by_expansion == "drop":
                    ret = ret[ret["start"] >= 0]
                elif allowed_on_below_zero_by_expansion == "truncate":
                    ret["start"] = ret["start"].clip(lower=0)
                elif allowed_on_below_zero_by_expansion == "ignore":
                    pass
                else:
                    # should not occur
                    raise NotImplementedError(
                        "Do not know how to handle on_below_zero_by_expansion of %s"
                        % on_below_zero_by_expansion
                    )
            return ret

    if ppg.inside_ppg():
        deps = [x.load() for x in list_of_grs]
        deps.append(
            ppg.ParameterInvariant(
                name + "_input_grs",
                list(sorted([x.name for x in list_of_grs])) + [expand_by_x_bp],
            )
        )
    else:
        deps = []
    vid = ("union", [x.vid for x in list_of_grs])
    return GenomicRegions(
        name,
        load,
        deps,
        list_of_grs[0].genome,
        on_overlap=on_overlap,
        summit_annotator=summit_annotator,
        sheet_name=sheet_name,
        vid=vid,
    )


def GenomicRegions_Overlapping(
    new_name, gr_a, gr_b, summit_annotator=None, sheet_name="Overlaps"
):
    return GenomicRegions_Common(new_name, [gr_a, gr_b], summit_annotator, sheet_name)


def GenomicRegions_Common(
    name, list_of_grs, summit_annotator=None, sheet_name="Overlaps"
):
    """Combine serveral GRs into one. Keep only those (union) regions occuring in all."""

    def load():
        union = merge_intervals(
            pd.concat([x.df[["chr", "start", "stop"]] for x in list_of_grs])
        )
        keep = np.ones((len(union),), dtype=np.bool)
        for gr in list_of_grs:
            for ii, row in union.iterrows():
                if keep[
                    ii
                ]:  # no point in checking if we already falsified - short circuit...
                    if not gr.has_overlapping(row["chr"], row["start"], row["stop"]):
                        keep[ii] = False
        return union[keep]

    verify_same_genome(list_of_grs)
    if ppg.inside_ppg():
        deps = [x.build_intervals() for x in list_of_grs]
        deps.append(
            ppg.ParameterInvariant(
                name + "_input_grs", sorted([x.name for x in list_of_grs])
            )
        )
    else:
        for x in list_of_grs:
            x.build_intervals()
        deps = []
    vid = ("common", [x.vid for x in list_of_grs])
    return GenomicRegions(
        name,
        load,
        deps,
        list_of_grs[0].genome,
        on_overlap="raise",
        summit_annotator=summit_annotator,
        sheet_name=sheet_name,
        vid=vid,
    )


def GenomicRegions_Invert(new_name, gr, summit_annotator=None, sheet_name="Inverted"):
    """Invert a GenomicRegions. What was covered becomes uncovered, what was uncovered becomes covered.
        [(10, 100), (400, 450)], in a chromosome of size 1000
        becomes
        [(0, 10), (450, 1000)]

        Note that all interval-set based operations (L{union}, L{intersection}, L{difference})
        drop all columns but chr, start, stop (annotators are merged and readded from all sets involved)
        """

    def do_load():
        new_rows = []
        chr_lens = gr.genome.get_chromosome_lengths()
        chrs_covered = set()
        # for chr, rows in itertools.groupby(self.df[['chr', 'start','stop']].iterrows(), lambda row: row['chr']):
        for chr, rows in gr.df[["chr", "start", "stop"]].groupby("chr"):
            chrs_covered.add(chr)
            start = 0
            for dummy_idx, row in rows.iterrows():
                if start != row["start"]:
                    new_rows.append({"chr": chr, "start": start, "stop": row["start"]})
                start = row["stop"]
            if start < chr_lens[chr]:
                new_rows.append({"chr": chr, "start": start, "stop": chr_lens[chr]})
        for (
            chr
        ) in (
            chr_lens
        ):  # we need to cover chromosomes that did not have a single entry so far.
            if not chr in chrs_covered:
                new_rows.append({"chr": chr, "start": 0, "stop": chr_lens[chr]})
        return pd.DataFrame(new_rows)

    if gr.load_strategy.build_deps:
        deps = [
            gr.load(),
            ppg.ParameterInvariant(
                "GenomicRegions_%s_parents" % new_name, (gr.name)
            ),  # so if you swap out the gr, it's detected...
        ]
    else:
        deps = []

    result = GenomicRegions(
        new_name,
        do_load,
        deps,
        gr.genome,
        on_overlap="merge",
        summit_annotator=summit_annotator,
        vid=["invert"] + gr.vid,
        sheet_name=sheet_name,
    )
    return result


def GenomicRegions_Difference(
    new_name, gr_a, gr_b, summit_annotator=None, sheet_name="difference"
):
    """Create a difference of these intervals wth other_gr's intervalls
        [(10, 100), (400, 450)],
        [(80, 120), (600, 700)]
        becomes
        [(10, 80), (400, 450)]
        (intervals may be split up!)

        Note that all interval-set based operations (L{union}, L{intersection}, L{difference})
        drop all columns but chr, start, stop (annotators are merged and readded from all sets involved)
        """
    verify_same_genome([gr_a, gr_b])

    def do_load():
        new_rows = []
        for idx, row in gr_a.df[["chr", "start", "stop"]].iterrows():
            overlaps = gr_b.get_overlapping(row["chr"], row["start"], row["stop"])
            if not len(overlaps):  # the easy case...
                new_rows.append(row)
            else:
                overlaps = gr_a.merge_intervals(
                    overlaps
                )  # they are now also sorted, so all we need to do is walk them, keep the regions between them (and within the original interval)
                start = row["start"]
                if overlaps.at[0, "start"] <= start and overlaps.at[0, "stop"] > start:
                    start_i = 1
                    start = overlaps.at[0, "stop"]
                else:
                    start_i = 0
                for ii in range(start_i, len(overlaps)):
                    stop = min(overlaps.at[ii, "start"], row["stop"])
                    new_rows.append({"chr": row["chr"], "start": start, "stop": stop})
                    start = overlaps.at[ii, "stop"]
                if start < row["stop"]:
                    new_rows.append(
                        {"chr": row["chr"], "start": start, "stop": row["stop"]}
                    )

                # todo: work the cutting up magic!
                pass
        if new_rows:
            return pd.DataFrame(new_rows)
        else:
            return pd.DataFrame({"chr": [], "start": [], "stop": []})

    if gr_a.load_strategy.build_deps:
        deps = [
            gr_b.load(),
            gr_a.load(),
            gr_b.build_intervals(),
            ppg.ParameterInvariant(
                "GenomicRegions_%s_parents" % new_name, (gr_a.name, gr_b.name)
            ),  # so if you swap out the gr, it's detected...
        ]
    else:
        gr_b.build_intervals()
        deps = []

    result = GenomicRegions(
        new_name,
        do_load,
        deps,
        gr_a.genome,
        on_overlap="merge",
        summit_annotator=summit_annotator,
        vid=["difference"] + list(gr_a.vid) + list(gr_b.vid),
        sheet_name=sheet_name,
    )
    return result


def GenomicRegions_Intersection(
    new_name, gr_a, gr_b, summit_annotator=None, sheet_name="intersection"
):
    """Create an intersection of all intervals...
        [(10, 100), (400, 450)],
        [(80, 120), (600, 700)]
        becomes
        [(80, 100),]

        Note that all interval-set based operations (L{union}, L{intersection}, L{difference})
        drop all columns but chr, start, stop (annotators are merged and readded from all sets involved)
        """
    verify_same_genome([gr_a, gr_b])

    def do_load():
        new_rows = []
        for chr, start, stop in gr_a._iter_intersections(gr_b):
            new_rows.append({"chr": chr, "start": start, "stop": stop})
        if new_rows:
            return pd.DataFrame(new_rows)
        else:
            return pd.DataFrame({"chr": [], "start": [], "stop": []})

    if gr_a.load_strategy.build_deps:
        deps = [
            gr_b.load(),
            gr_a.load(),
            gr_b.build_intervals(),
            ppg.ParameterInvariant(
                "GenomicRegions_%s_parents" % new_name, (gr_a.name, gr_b.name)
            ),  # so if you swap out the gr, it's detected...
        ]
    else:
        deps = []
        gr_b.build_intervals()

    result = GenomicRegions(
        new_name,
        do_load,
        deps,
        gr_a.genome,
        on_overlap="merge",
        summit_annotator=summit_annotator,
        vid=["intersection"] + gr_a.vid + gr_b.vid,
        sheet_name=sheet_name,
    )
    return result


def GenomicRegions_FromPartec(
    name, filename, genome, on_overlap="raise", summit_annotator=None, vid=None
):
    """create GenomicRegions from Partec's output"""
    import xlrd

    def load():
        print("loading for", name)

        try:
            df = pd.read_excel(filename)
        except xlrd.XLRDError:
            df = pd.read_csv(filename, sep="\t")

        renames = {}
        for column in df.columns:
            if "start" == column.lower():
                renames[column] = "start"
            if "stop" == column.lower() or "end" == column.lower():
                renames[column] = "stop"
            if "chromosome" == column.lower():
                renames[column] = "chr"
        if renames:
            df = df.rename(columns=renames)

        df["chr"] = df["chr"].astype(str)
        df["start"] = df["start"].astype(int)
        df["stop"] = df["stop"].astype(int)
        return df

    if ppg.util.inside_ppg():
        deps = [ppg.FileTimeInvariant(filename)]
    else:
        deps = []

    return GenomicRegions(
        name, load, deps, genome, on_overlap, summit_annotator=summit_annotator, vid=vid
    )


def GenomicRegions_FromTable(
    name,
    filename,
    genome,
    on_overlap="raise",
    filter_func=None,
    vid=None,
    sheet_name="FromTable",
    drop_further_columns=True,
):
    """Read a table file (csv/tsv/xls) with the correct chr/start/stop columns, drop all further columns"""

    def load():

        df = read_pandas(filename)
        df["chr"] = df["chr"].astype(str)
        df["start"] = df["start"].astype(int)
        df["stop"] = df["stop"].astype(int)
        if drop_further_columns:
            df = df[["chr", "start", "stop"]]
        if filter_func:
            df = filter_func(df)
        return df

    if ppg.inside_ppg():
        deps = [
            ppg.FileTimeInvariant(filename),
            ppg.FunctionInvariant(name + "_filter_func", filter_func),
        ]
    else:
        deps = []
    return GenomicRegions(
        name, load, deps, genome, on_overlap, sheet_name=sheet_name, vid=vid
    )


def GenomicRegions_FromGenome(genome):
    """For creating a gr from a FileBasedGenome"""

    def __loading_function():
        data = {"chr": [], "start": [], "stop": []}
        for key, value in genome.fasta_iterator(genome.sequence_fasta_filename):
            data["chr"].append(key.split(" ")[0])
            data["start"].append(0)
            data["stop"].append(len(value))
        return pd.DataFrame(data)

    if ppg.inside_ppg():
        deps = [genome.get_dependencies()]
    else:
        deps = []
    return GenomicRegions("spike_genomic_region", __loading_function, deps, genome)


def GenomicRegions_CommonInAtLeastX(
    name, list_of_grs, X, summit_annotator=None, sheet_name="Overlaps"
):
    """Combine serveral GRs into one. Keep only those (union) regions occuring in at least x."""

    def load():
        union = merge_intervals(
            pd.contact([x.df[["chr", "start", "stop"]] for x in list_of_grs])
        )
        keep = np.zeros((len(union),), dtype=np.bool)
        for ii, row in union.iterrows():
            count = 0
            for gr in list_of_grs:
                if gr.has_overlapping(row["chr"], row["start"], row["stop"]):
                    count += 1
            keep[ii] = count >= X
        if not keep.any():
            raise ValueError("Filtered all of them")
        return union[keep]

    if len(set([x.genome for x in list_of_grs])) > 1:
        raise ValueError("Can only merge GenomicRegions that have the same genome")
    if ppg.inside_ppg():
        deps = [x.build_intervals() for x in list_of_grs]
        deps.append(
            ppg.ParameterInvariant(
                name + "_input_grs", sorted([x.name for x in list_of_grs])
            )
        )
    else:
        deps = []
    vid = ("common at least %i" % X, [x.vid for x in list_of_grs])
    return GenomicRegions(
        name,
        load,
        deps,
        list_of_grs[0].genome,
        on_overlap="raise",
        summit_annotator=summit_annotator,
        sheet_name=sheet_name,
        vid=vid,
    )


def GenomicRegions_FromMotifHits(name, motif, threshold, genome):
    def load():
        chrs = []
        starts = []
        stops = []
        scores = []
        strands = []
        for chr, length in genome.get_chromosome_lengths().items():
            seq = genome.get_sequence(chr, 0, length)
            hits = motif.scan(seq, threshold)
            for dummy_idx, row in hits[0].iterrows():
                chrs.append(chr)
                start = min(row["start"], row["stop"])
                stop = max(row["start"], row["stop"])
                starts.append(start)
                stops.append(stop)
                strands.append("+" if row["start"] > row["stop"] else "-")
                scores.append(row["score"])
        return pd.DataFrame(
            {"chr": chrs, "start": starts, "stop": stops, "scores": scores}
        )

    if ppg.inside_ppg():
        deps = [
            motif.load(),
            ppg.ParameterInvariant("GR_%s" % name, (threshold,)),
            genome.get_dependencies(),
        ]
    else:
        deps = []
    return GenomicRegions(
        name, load, deps, genome, on_overlap="ignore", sheet_name="Motif"
    )


def GenomicRegions_Windows(
    genome,
    name,
    window_size,
    window_spacing,
    subset_of_chromosomes=None,
    sheet_name="Windowed",
):
    """Create a GenomicRegions that has a window of size @window_size (0 for next to each other), windows are spaced @window_spacing
    across all or a @subset_of_chromosomes"""

    def load():
        chrs_to_include = list(
            subset_of_chromosomes
            if subset_of_chromosomes
            else genome.get_chromosome_lengths().keys()
        )
        chrs = []
        starts = []
        stops = []
        chr_lengths = genome.get_chromosome_lengths()
        for c in chrs_to_include:
            ll = chr_lengths[c]
            for ii in range(0, ll, window_size + window_spacing):
                chrs.append(c)
                starts.append(ii)
                stops.append(ii + window_size)
        return pd.DataFrame({"chr": chrs, "start": starts, "stop": stops})

    return GenomicRegions(
        name, load, [], genome, on_overlap="raise", sheet_name=sheet_name
    )


def GenomicRegions_FilterRemoveOverlapping(
    new_name, gr_a, other_grs, summit_annotator=None, sheet_name="Overlaps"
):
    """Filter all from this GenomicRegions that have an overlapping region in other_gr
    Note that filtering does not change the coordinates, it only filters,
    non annotator additional rows are kept, annotators are recalculated.
    """
    if isinstance(other_grs, GenomicRegions):
        other_grs = [other_grs]
    verify_same_genome([gr_a] + other_grs)

    def filter_func(df):
        keep = np.zeros((len(df)), dtype=np.bool)
        for ii, row in df[["chr", "start", "stop"]].iterrows():
            for other_gr in other_grs:
                keep[ii] = keep[ii] | other_gr.has_overlapping(
                    row["chr"], row["start"], row["stop"]
                )
        return ~keep

    if not summit_annotator:
        summit_annotator = gr_a.summit_annotator

    if gr_a.load_strategy.build_deps:
        deps = [
            [x.build_intervals() for x in other_grs],
            ppg.ParameterInvariant(
                "GenomicRegions_%s_parents" % new_name,
                (gr_a.name, [x.name for x in other_grs]),
            ),  # so if you swap out the gr, it's detected...
        ]
    else:
        for other_gr in other_grs:
            other_gr.build_intervals()
        deps = []

    return gr_a.filter(
        new_name,
        df_filter_function=filter_func,
        dependencies=deps,
        summit_annotator=summit_annotator,
        sheet_name=sheet_name,
    )


def GenomicRegions_FilterToOverlapping(
    new_name, gr_a, other_grs, summit_annotator=None, sheet_name="Overlaps"
):
    """Filter to just those that overlap one in *all* other_grs.
    Note that filtering does not change the coordinates, it only filters,
    non annotator additional rows are kept, annotators are recalculated.
    """
    if isinstance(other_grs, GenomicRegions):
        other_grs = [other_grs]
    verify_same_genome([gr_a] + other_grs)

    def filter_func(df):
        keep = np.ones((len(df)), dtype=np.bool)
        for ii, row in df[["chr", "start", "stop"]].iterrows():
            for gr in other_grs:
                keep[ii] &= gr.has_overlapping(row["chr"], row["start"], row["stop"])
        return keep

    if gr_a.load_strategy.build_deps:
        deps = [other_gr.build_intervals() for other_gr in other_grs] + [
            ppg.ParameterInvariant(
                "GenomicRegions_%s_parents" % new_name,
                (gr_a.name, [other_gr.name for other_gr in other_grs]),
            )  # so if you swap out the gr, it's detected...
        ]
    else:
        for other_gr in other_grs:
            other_gr.build_intervals()
        deps = []

    return gr_a.filter(
        new_name,
        df_filter_function=filter_func,
        dependencies=deps,
        summit_annotator=summit_annotator,
        sheet_name=sheet_name,
    )
