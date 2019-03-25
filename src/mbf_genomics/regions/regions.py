import pypipegraph as ppg
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Iterator


from mbf_genomics.delayeddataframe import DelayedDataFrame
from mbf_genomes import GenomeBase
from mbf_externals.util import lazy_property
from .annotators import SummitMiddle
from mbf_genomes.intervals import merge_intervals, merge_intervals_with_callback


def merge_identical(sub_df):
    if (len(sub_df["start"].unique()) != 1) or (len(sub_df["stop"].unique()) != 1):
        raise ValueError(
            "Overlapping intervals: %s, merge_identical was set." % (sub_df,)
        )
    return sub_df.iloc[0].to_dict()


def get_overlapping_interval_indices(
    start, stop, start_array, stop_array, handle_overlapping
):
    """Return the indices of all intervals stored in start_array, stop_array that overlap start, stop"""
    # TODO: replace with better algorithm - e.g. rustbio intervl trees
    if not handle_overlapping:
        first_start_smaller = np.searchsorted(start_array, start) - 1
        first_end_larger = np.searchsorted(stop_array, stop, "right") + 1
    else:
        first_start_smaller = 0
        first_end_larger = len(stop_array)
    result = []
    for possible_match in range(
        max(0, first_start_smaller), min(first_end_larger, len(stop_array))
    ):
        s = max(start, start_array[possible_match])
        e = min(stop, stop_array[possible_match])
        if s < e:
            result.append(possible_match)
    return result


region_registry = {}


class GenomicRegions(DelayedDataFrame):

    """A genomic regions object encapsulates intervals across the chromosomes
    of a genome. They model everything from restriction sites over binding sites
    to gene locations.

    Internally a genomic region consists of a dataframe {chr, start, stop} (.df),
    and GR-annotators can be used to add further columns.
    Please note that the intervals are 'standard python' ie. left inclusive, right exclusive.
    An interval 10:20, is 10 bp long, and covers 10, 11, ..., 18, 19.
    That also means a 'point' interval is actually x, x+1, not x,x (that's an empty interval)

    There is a variety of set operations you can perform on them that are done efficiently
    via interval trees.

    GRs do not require disjoint regions. More specifically, they can be configured
    to do a variety of things if they're passed overlapping regions - ignore them,
    raise an exception, merge them...

    You can query them for all of their internal regions overlapping an interval

    Annotators are inherited by descendands, their values are recalculated for
    the new intervals.

    """

    # basic loading
    def __init__(
        self,
        name,
        loading_function,
        dependencies,
        genome,
        on_overlap="raise",
        result_dir=None,
        summit_annotator=None,
        sheet_name=None,
        vid=None,
    ):
        """Create a lazy loaded GenomicRegion, available once it's load() job has been completed.
        @dependencies allow you to inject dependencies for the load() job.
        @on_overlap may be one of (raise, ignore, merge, drop, see below), which decides how overlapping regions are handled.

        @loading_function is a parameterless function that returns dataframe of {chr, start, stop}.
        The dataframe does not need to sort the data in any way. chr must be a string column, start and stop integers
        (this is checked).

        The internal df is always sorted by chr, start...


        @on_overlap, allowed values:
            -raise: - raise a ValueError if any overlapping regions are detected (default)
            -merge: - combine overlapping regions into larger ones
            -merge_identical: combine identical regions, but raise if there are non identical overlapping ones
            -(merge, merge_function) - combine regions - call merge_function(sub_df) which must return a row (dict) to keep and is only called when there are multiple rows to pick! Its row['start'], row['stop'] are ignored though
            -drop: - drop all regions that are overlapping one another
            -ignore: - ignore overlapping regions (this implies a nested list search for overlap queries
            and will lead to certain functions that assume non-overlapping regions (such as covered_bases)
            to raise ValueErrors)

        """
        if (
            isinstance(dependencies, str)
            or not hasattr(dependencies, "__iter__")
            or (
                hasattr(dependencies, "__iter__") and isinstance(dependencies, Iterator)
            )
        ):
            raise ValueError(
                "dependencies must be iterable (use [] for no external dependencies"
            )
        if not isinstance(dependencies, list):
            dependencies = list(dependencies)

        allowed_overlap_modes = ("raise", "merge", "ignore", "drop", "merge_identical")
        if not on_overlap in allowed_overlap_modes and not (
            isinstance(on_overlap, tuple)
            and on_overlap[0] == "merge"
            and hasattr(on_overlap[1], "__call__")
        ):
            raise ValueError(
                "Invalid on_overlap mode %s. Allowed: %s, or a tuple of ('merge', function)"
                % (on_overlap, allowed_overlap_modes)
            )
        if on_overlap == "merge_identical":
            on_overlap = ("merge", merge_identical)

        self.name = name
        self.gr_loading_function = loading_function
        self.genome = genome
        self.on_overlap = on_overlap
        self._default_mangler = True
        if self.on_overlap == "ignore":
            self.need_to_handle_overlapping_regions = True
        elif (
            self.on_overlap == "raise"
            or self.on_overlap == "merge"
            or self.on_overlap == "drop"
            or (isinstance(self.on_overlap, tuple) and self.on_overlap[0] == "merge")
        ):
            self.need_to_handle_overlapping_regions = False
        else:  # pragma: no cover - defensive
            raise ValueError(
                "Don't know how to decide on has_overlapping from %s" % self.on_overlap
            )

        if result_dir:
            result_dir = Path(result_dir)
        else:
            if sheet_name:
                result_dir = Path("results") / "GenomicRegions" / sheet_name / name
            else:
                result_dir = Path("results") / "GenomicRegions" / name
        self.sheet_name = sheet_name

        super().__init__(name, self._load, dependencies, result_dir)
        if self.load_strategy.build_deps:
            dependencies = self.load_strategy.deps
            if isinstance(on_overlap, tuple):
                dependencies.append(
                    ppg.ParameterInvariant("grload_params" + name, ("function",))
                )
                dependencies.append(
                    ppg.FunctionInvariant("grload_overlap_func" + name, on_overlap[1])
                )
            else:
                dependencies.append(
                    ppg.ParameterInvariant("grload_params" + name, (on_overlap,))
                )
            dependencies.extend(genome.download_genome())
            dependencies.append(
                ppg.FunctionInvariant("grload_func" + name, self.gr_loading_function)
            )

        if not hasattr(self, "column_properties"):
            self.column_properties = {
                "chr": {
                    "user_visible": False,
                    "priority": -997,
                    "description": "On which chromosome (or contig) the region is loacted",
                },
                "start": {
                    "user_visible": False,
                    "priority": -997,
                    "description": "Left most position of this region",
                },
                "stop": {
                    "user_visible": False,
                    "priority": -997,
                    "description": "Right most position of this region",
                },
            }
        # self.genome = genome

        self.random_count = 0
        if summit_annotator:
            self.summit_annotator = summit_annotator
        elif summit_annotator is False:
            self.summit_annotator = None
        else:
            self.summit_annotator = SummitMiddle()
        if self.summit_annotator is not None:
            self.add_annotator(self.summit_annotator)
        self.register()
        if vid is None:
            vid = []
        self.vid = vid

    def get_default_columns(self):
        return ("chr", "start", "stop")

    def _load(self):

        df = self.gr_loading_function()
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "GenomicRegion(%s).loading_function must return a pandas.DataFrame, was: %s\n%s"
                % (self.name, type(df), self.gr_loading_function)
            )
        for col in self.get_default_columns():
            if not col in df.columns:
                raise ValueError(
                    "%s not in dataframe returned by GenomicRegion.loading_function"
                    % col
                )
        allowed_chromosomes = set(self.genome.get_chromosome_lengths().keys())
        if len(df):
            # if isinstance(df.iloc[0]["chr"], six.string_types):
            # df["chr"] = [str(x) for x in df["chr"].values]
            if not (isinstance(df.iloc[0]["chr"], str)):
                raise ValueError(
                    "Chromosomes must be a string, was: %s, first instance %s of type %s"
                    % (
                        df["chr"].dtype,
                        repr(df.iloc[0]["chr"]),
                        type(df.iloc[0]["chr"]),
                    )
                )
            chrs = set([str(x) for x in df["chr"].unique()])
            if chrs.difference(allowed_chromosomes):
                raise ValueError(
                    "Invalid chromosomes found: %s, expected one of: %s"
                    % (
                        chrs.difference(allowed_chromosomes),
                        sorted(allowed_chromosomes),
                    )
                )
            if not np.issubdtype(df["start"].dtype, np.integer):
                raise ValueError(
                    "start needs to be an integer, was: %s" % df["start"].dtype
                )
            if not np.issubdtype(df["stop"].dtype, np.integer):
                raise ValueError(
                    "stop needs to be an integer, was: %s" % df["stop"].dtype
                )
            if (df["start"] < 0).any():
                print(df[df["start"] < 0])
                raise ValueError("All starts need to be positive!")
            if (df["start"] > df["stop"]).any():
                error = df[(df["start"] > df["stop"])][["chr", "start", "stop"]]
                error = error[:100]  # not too many...
                error = str(error)
                raise ValueError(
                    "%s.loading_function returned a negative interval:\n%s"
                    % (self, error)
                )
            df = self.handle_overlap(df)
        else:
            df = df.assign(
                start=np.array([], dtype=np.int32),
                stop=np.array([], dtype=np.int32),
                chr=np.array([], dtype=np.object),
            )
        # enforce column order
        cols = ["chr", "start", "stop"]
        for x in df.columns:
            if not x in cols:
                cols.append(x)
        df = df[cols].reset_index(drop=True)

        return df

    def handle_overlap(self, df):
        """depending on L{GenomicRegion.on_overlap}, check
        for overlapping regions and handle accordingly"""
        if self.on_overlap == "raise":
            df = df.sort_values(["chr", "start"], ascending=[True, True]).reset_index(
                drop=True
            )
            last_chr = None
            last_stop = 0
            last_row = None
            for idx, row in df.iterrows():
                if row["chr"] != last_chr:
                    last_chr = row["chr"]
                    last_stop = 0
                if row["start"] < last_stop:
                    raise ValueError(
                        "%s: Overlapping intervals: %s %i..%i vs %i..%i"
                        % (
                            self.name,
                            row["chr"],
                            row["start"],
                            row["stop"],
                            last_row["start"],
                            last_row["stop"],
                        )
                    )
                last_stop = row["stop"]
                last_row = row
            return df
        elif self.on_overlap == "merge":
            return self.merge_intervals(df)
        elif isinstance(self.on_overlap, tuple) and self.on_overlap[0] == "merge":
            return self.merge_intervals_with_callback(df, self.on_overlap[1])
        elif self.on_overlap == "ignore":
            df = df.sort_values(["chr", "start"], ascending=[True, True]).reset_index(
                drop=True
            )  # still need to sort at least by chr...
            if "is_overlapping" in df.columns:
                del df["is_overlapping"]
            is_overlapping = np.zeros((len(df),), dtype=np.bool)
            last_chr = None
            last_stop = 0
            last_row = None
            last_row_index = 0
            for ii, row in df.iterrows():
                if row["chr"] != last_chr:
                    last_chr = row["chr"]
                    last_stop = 0
                if row["start"] < last_stop:
                    is_overlapping[last_row_index] = True
                    is_overlapping[ii] = True
                last_stop = row["stop"]
                last_row = row
                last_row_index = ii
            df = df.assign(is_overlapping=is_overlapping)
            return df

        elif self.on_overlap == "drop":
            return self.drop_intervals(df)

    def merge_intervals(self, df):
        return merge_intervals(df)

    def merge_intervals_with_callback(self, df, callback):
        """take a {chr, start, end, *} dataframe and merge overlapping intervals, calling callback for group larger than one.."""
        return merge_intervals_with_callback(df, callback)

    def drop_intervals(self, df):
        """take a {chr, start, end, *} dataframe, and drop all intervals that overlap one or more others"""
        df = df.sort_values(
            ["chr", "start"], ascending=[True, True]
        )  # you need to do this here so it's true later...
        last_chr = None
        last_stop = 0
        last_row = None

        chrs = np.array(df["chr"])
        starts = np.array(df["start"])
        stops = np.array(df["stop"])

        ii = 0
        lendf = len(df)
        keep = np.ones((len(df),), dtype=np.bool)
        while ii < lendf:
            if chrs[ii] != last_chr:
                last_chr = chrs[ii]
                last_stop = 0
            if starts[ii] < last_stop:
                keep[ii] = False
                keep[ii - 1] = False
            else:
                pass
            if stops[ii] > last_stop:
                last_stop = stops[ii]
            last_row = ii
            ii += 1
        if last_row is not None:
            keep[last_row] = True
        # new_rows.append(df.get_row(last_row))
        # return pydataframe.DataFrame(new_rows)
        return df.iloc[keep]

    def do_build_intervals(self):
        """"Build the interval trees right now, ignoring all dependencies"""
        # find out where the start and stop's of for each chromosome are
        chr_diffs = {}
        if len(self.df):  # no dataframes, no chromosome_intervals...
            chrs_per_region = self.df["chr"]
            shifted = np.roll(chrs_per_region, 1)
            changed = chrs_per_region != shifted
            if not changed.any():  # only a single chromosome...
                chr_diffs[chrs_per_region[0]] = (0, len(self.df))

            else:
                breaks = list(np.where(changed)[0]) + [len(self.df)]
                for ii in range(0, len(breaks) - 1):
                    chr = chrs_per_region[breaks[ii]]
                    start = breaks[ii]
                    stop = breaks[ii + 1]
                    chr_diffs[chr] = (start, stop)

        self.chromosome_intervals = chr_diffs

    def has_overlapping(self, chr, start, stop):
        """is there an interval overlapping the region passed"""
        try:
            chr_start, chr_stop = self.chromosome_intervals[chr]
        except KeyError:
            # print 'False1'
            return False
        # if chr_stop == chr_start: # no intrvals on this chromosome
        # print 'False2'
        # return None
        start_array = self.df["start"][chr_start:chr_stop]
        stop_array = self.df["stop"][chr_start:chr_stop]
        first_start_smaller = np.searchsorted(start_array, start) - 1
        first_end_larger = np.searchsorted(stop_array, stop, "right") + 1
        # when was this necessary?
        # if hasattr(first_start_smaller, "__iter__"):
        # first_start_smaller = first_start_smaller[0]
        # if hasattr(first_end_larger, "__iter__"):
        # first_end_larger = first_end_larger[0]

        for possible_match in range(
            max(0, first_start_smaller), min(first_end_larger, len(stop_array))
        ):
            s = max(start, start_array.iloc[possible_match])
            e = min(stop, stop_array.iloc[possible_match])
            # print s, e
            if s < e:
                # print 'True'
                return True
        # print 'False3'
        return False

    def get_overlapping(self, chr, start, stop):
        """Retrieve the rows for the region passed in.
        Returns an empty dataframe if there is no overlap

        Please note that the interval is end excluding - ie start == stop
        means an empty interval and nothing ever overlapping!
        """
        # print 'testing overlap for', chr, start, stop
        try:
            chr_start, chr_stop = self.chromosome_intervals[chr]
        except KeyError:
            return self.df[0:0]
        # if chr_stop == chr_start: # no intervals on this chromosome
        # return self.df[0:0]
        start_array = np.array(self.df["start"][chr_start:chr_stop])
        stop_array = np.array(self.df["stop"][chr_start:chr_stop])
        selected_entries = get_overlapping_interval_indices(
            start,
            stop,
            start_array,
            stop_array,
            self.need_to_handle_overlapping_regions,
        )
        return self.df.iloc[[x + chr_start for x in selected_entries]]

    def get_closest(self, chr, point):
        """Find the interval that is closest to the passed point.
        Returns a df with that interval, or an empty df!
        """
        # first we check whether we're on top of a region...
        # overlapping = self.get_overlapping(chr, point, point+1)
        # if len(overlapping):
        # return overlapping # could be more than one?!
        try:
            chr_start, chr_stop = self.chromosome_intervals[chr]
        except KeyError:
            return self.df[0:0]  # ie an empty df
        if chr_stop == chr_start:  # no intervals on this chromosome
            return self.df[0:0]  # ie an empty df
        start_array = np.array(self.df["start"][chr_start:chr_stop])
        stop_array = np.array(self.df["stop"][chr_start:chr_stop])
        if self.need_to_handle_overlapping_regions:
            # TODO this could use some optimization with a more appropriate data structure
            distance_to_start = np.abs(start_array - point)
            distance_to_stop = np.abs(stop_array - point)
            argmin_start = distance_to_start.argmin()
            argmin_stop = distance_to_stop.argmin()
            if distance_to_start[argmin_start] < distance_to_stop[argmin_stop]:
                return self.df.iloc[
                    chr_start + argmin_start : chr_start + argmin_start + 1
                ]
            else:
                return self.df.iloc[
                    chr_start + argmin_stop : chr_start + argmin_stop + 1
                ]
        else:

            # now, we already know that there is no overlapping interval... so...
            first_end_smaller = max(
                0, min(len(stop_array) - 1, np.searchsorted(stop_array, point) - 1)
            )
            first_start_larger = max(
                0,
                min(len(stop_array) - 1, np.searchsorted(start_array, point, "right")),
            )
            distance_left = abs(point - stop_array[first_end_smaller])
            distance_right = abs(start_array[first_start_larger] - point)
            # print start_array, stop_array, first_end_smaller, first_start_larger, distance_left, distance_right
            if distance_left < distance_right:
                return self.df.iloc[
                    chr_start + first_end_smaller : chr_start + first_end_smaller + 1
                ]
            else:
                return self.df.iloc[
                    chr_start + first_start_larger : chr_start + first_start_larger + 1
                ]

    # various statistics
    def get_no_of_entries(self):
        """How many intervals are there"""
        return len(self.df)

    @lazy_property
    def covered_bases(self):
        """How many base pairs are covered by these intervals"""
        if self.on_overlap == "ignore":
            raise ValueError(
                "covered_bases is currently only implemented for not overlapping GenomicRegions (on_overlap != 'ignore')"
            )
        return (self.df["stop"] - self.df["start"]).sum()

    @lazy_property
    def mean_size(self):
        """Get the mean size of the intervals defined in this GR"""
        return np.mean(self.df["stop"] - self.df["start"])

    def register(self):
        region_registry[self.name] = self

    # magic
    def __hash__(self):
        return hash("GR" + self.name)

    def __str__(self):
        return "GenomicRegion(%s)" % self.name

    def __repr__(self):
        return "GenomicRegion(%s)" % self.name

    # interval querying
    def build_intervals(self):
        """Prepare the internal datastructure for all overlap/closest/set based operations"""
        if self.load_strategy.build_deps:
            return (
                ppg.DataLoadingJob(self.name + "_build", self.do_build_intervals)
                .depends_on(self.load())
                .depends_on(self.genome.download_genome())
            )
        else:
            self.do_build_intervals()

    # output functions

    def write_bed(self, output_filename=None, region_name=None, include_header=False):
        """Store the intervals of the GenomicRegion in a BED file
        @region_name: Insert the column of the GenomicRegions-Object e.g. 'repeat_name'"""
        from mbf_fileformats.bed import BedEntry, write_bed

        output_filename = self.pathify(output_filename, self.name + ".bed")

        def write(output_filename=output_filename):
            bed_entries = []
            for ii, row in self.df.iterrows():
                if region_name is None:

                    def get_name(row):
                        return None

                else:

                    def get_name(row):
                        if region_name in row:
                            return row[region_name]
                        else:
                            raise ValueError(
                                "key: %s not in genomic regions object. These objects are available: %s"
                                % (region_name, self.df.columns)
                            )

                entry = BedEntry(
                    row["chr"],
                    row["start"],
                    row["stop"],
                    name=get_name(row),
                    strand=row["strand"] if "strand" in row else None,
                    score=row["score"] if "score" in row else None,
                )
                bed_entries.append(entry)
            write_bed(
                output_filename,
                bed_entries,
                {},
                self.name,
                include_header=include_header,
            )

        if self.load_strategy.build_deps:
            deps = [
                self.load(),
                ppg.ParameterInvariant(output_filename, (include_header,)),
            ]
        else:
            deps = []
        return self.load_strategy.generate_file(output_filename, write, deps)

    def write_bigbed(self, output_filename=None):
        """Store the intervals of the GenomicRegion in a big bed file"""
        from mbf_fileformats.bed import BedEntry, write_big_bed

        output_filename = self.pathify(output_filename, self.name + ".bigbed")

        def write(output_filename=output_filename):

            bed_entries = []
            for idx, row in self.df.iterrows():
                if "repeat_name" in row:
                    entry = BedEntry(
                        row["chr"],
                        row["start"],
                        row["stop"],
                        name=row["repeat_name"],
                        strand=row["strand"] if "strand" in row else 0,
                    )
                elif "name" in row:
                    entry = BedEntry(
                        row["chr"],
                        row["start"],
                        row["stop"],
                        name=row["name"],
                        strand=row["strand"] if "strand" in row else 0,
                    )
                else:
                    entry = BedEntry(
                        row["chr"],
                        row["start"],
                        row["stop"],
                        strand=row["strand"] if "strand" in row else 0,
                    )
                bed_entries.append(entry)
            if len(self.df):
                write_big_bed(
                    bed_entries, output_filename, self.genome.get_chromosome_lengths()
                )
            else:
                with open(output_filename, "wb"):
                    pass

        if self.load_strategy.build_deps:
            deps = [self.load()]
        else:
            deps = []
        return self.load_strategy.generate_file(
            output_filename, write, deps, empty_ok=True
        )

    # filtering
    def _new_for_filtering(self, new_name, load_func, deps, **kwargs):
        """When filtering, a new object of this class is created.
        To pass it the right options from the parent, overwrite this
        """
        kwargs["sheet_name"] = kwargs.get("sheet_name", "Filtered")
        for k in "on_overlap", "result_dir", "summit_annotator", "vid":
            if not k in kwargs:
                kwargs[k] = getattr(self, k)
        return GenomicRegions(new_name, load_func, deps, genome=self.genome, **kwargs)

    def __remove_me_fil_er_to_overlapping_count(self, other_gr):
        """Return the length a GenomicRegions that was created via filter_to_overlapping would have,
        without actually creating it.... Must have called other_gr.build_intervals() first (and this one must have been loaded)"""
        if other_gr.genome != self.genome:
            raise ValueError(
                "Unequal genomes betwen %s %s in filter_remove_overlapping"
                % (self.name, other_gr.name)
            )
        count = 0
        for ii, row in self.df[["chr", "start", "stop"]].iterrows():
            if other_gr.has_overlapping(row["chr"], row["start"], row["stop"]):
                count += 1
        return count

    def _iter_intersections(self, other_gr):
        """Iterate over (chr, start, stop) tuples of the intersections between this GenomicRegions
        and other_gr.

        Refactored from intersection and overlap_basepairs"""
        if self.genome != other_gr.genome:
            raise ValueError(
                "GenomicRegions set-operations only work if both have the same genome. You had %s and %s"
                % (self.genome, other_gr.genome)
            )
        if self.need_to_handle_overlapping_regions:
            raise ValueError(
                "_iter_intersections currently only works on non_overlapping intervalsets - %s was %s"
                % (self.name, self.on_overlap)
            )
        if other_gr.need_to_handle_overlapping_regions:
            other_df = self.merge_intervals(other_gr.df)
        else:
            other_df = other_gr.df
        for chr in self.df["chr"].unique():
            df_here = self.df[self.df["chr"] == chr]
            starts_here = df_here["start"]
            stops_here = df_here["stop"]
            print(other_df)
            print(other_df.dtypes)
            df_there = other_df[other_df["chr"] == chr]
            starts_there = df_there["start"]
            stops_there = df_there["stop"]
            ii = 0
            jj = 0
            while ii < len(df_here) and jj < len(df_there):
                if starts_there.iloc[jj] <= starts_here.iloc[ii] < stops_there.iloc[jj]:
                    yield (
                        chr,
                        max(starts_here.iloc[ii], starts_there.iloc[jj]),
                        min(stops_here.iloc[ii], stops_there.iloc[jj]),
                    )
                    ii += 1
                elif (
                    starts_here.iloc[ii] <= starts_there.iloc[jj] <= stops_here.iloc[ii]
                ):
                    yield (
                        chr,
                        max(starts_here.iloc[ii], starts_there.iloc[jj]),
                        min(stops_here.iloc[ii], stops_there.iloc[jj]),
                    )
                    jj += 1
                elif starts_here.iloc[ii] < starts_there.iloc[jj]:
                    ii += 1
                else:
                    jj += 1
        return

    def copy_annotators(self, *other_grs):
        """Copy annotators from other GRs"""
        for p in other_grs:
            for anno in p.annotators.values():
                self += anno

    # overlap calculations
    def overlap_basepair(
        self, other_gr
    ):  # todo: given to GRs with on_overlap != ignore, we could do a sorted search like we did for the GIS, that should be faster...
        """calculate the overlap between to GenomicRegions, on a base pair level - ie. the size of the intersection set
        (this converts other_gr into a disjoint set, so each base is counted only once. self must be disjoint)"""
        if self.genome != other_gr.genome:
            raise ValueError(
                "GenomicRegions set-operations only work if both have the same genome. You had %s and %s"
                % (self.genome, other_gr.genome)
            )
        overlap = 0
        for chr, start, stop in self._iter_intersections(other_gr):
            overlap += stop - start
        return overlap

    def overlap_percentage(self, other_gr):
        """calculate the percentage of overlapping base pairs
        Basically overlap in bp / min(bp self, bp other_gr)
        if either is empty, percentage = 0
        """
        if self.genome != other_gr.genome:
            raise ValueError(
                "GenomicRegions set-operations only work if both have the same genome. You had %s and %s"
                % (self.genome, other_gr.genome)
            )
        bp_overlap = self.overlap_basepair(other_gr)
        divisor = min(self.covered_bases, other_gr.covered_bases)
        if divisor == 0:
            return 0
        else:
            return float(bp_overlap) / divisor

    def overlap_count(self, other_gr):
        """Count the size of the union of all intersecting intervals. Basically, len(overlapping(..., other_gr))

        See L{overlapping}.

        Requires self.load() to have been done.

        This is (much) faster than building overlap()  and calling len,.

        The invariant len(a) + len(b) == len(a.filter_remove_overlapping(b))
            + len(b.filter_remove_overlapping(a)) + a.overlap_count(b) is not true,
             because the overlapping count only counts each region once,
             even if it was compromised out of multiple regions in the input. Say you have
        ---  ----  -----
          ----  ----
        that count's as 3 (or two) regions in len(a) (len(b)),
         but just as one in overlapping/overlap_count (and the filtered ones are of len(0)). Still, I argue that this is the right one to use in venn diagrams.

        """
        if self.genome != other_gr.genome:
            raise ValueError(
                "GenomicRegions set-operations only work if both have the same genome. You had %s and %s"
                % (self.genome, other_gr.genome)
            )
        overlap = 0
        for chrom in self.genome.get_chromosome_lengths():
            a = self.df[self.df.chr == chrom][["start", "stop"]].assign(color=1)
            b = other_gr.df[other_gr.df.chr == chrom][["start", "stop"]].assign(color=2)
            comb = pd.concat([a, b]).sort_values("start")
            last_stop = 0
            run_start = -1
            run_colors = 0
            for dummy_idx, row in comb.iterrows():
                start = row["start"]
                stop = row["stop"]
                color = row["color"]
                if start < last_stop:  # a continuation
                    run_colors |= color
                    last_stop = max(last_stop, stop)
                else:  # end of this run
                    if run_start > 0:
                        if run_colors == 3:
                            overlap += 1
                    run_start = start
                    last_stop = stop
                    run_colors = color
            if (
                len(comb) and start != run_start and run_colors == 3
            ):  # capture the last one that will not have triggered an 'end-of-run'
                overlap += 1
        return overlap

    def intersection_count(self, other_gr):
        """calculate the number of intersecting regions between two GenomicRegions
        Note that
        len(here) - intersection_count(here, other_gr) != len(here.filter_remove_overlapping(...,other_gr)
        since they handle this situation differently:
        here : ---    ----
        there:   ------
        (which here & in intersection gives you two intervals, while in overlapping (and overlap_count) it
        gives just one)

        """
        if self.genome != other_gr.genome:
            raise ValueError(
                "GenomicRegions set-operations only work if both have the same genome. You had %s and %s"
                % (self.genome, other_gr.genome)
            )
        if not self.load_strategy.build_deps:
            other_gr.build_intervals()
        overlap = 0
        for chr, start, stop in self._iter_intersections(other_gr):
            overlap += 1
        return overlap

    def convert(
        self,
        new_name,
        conversion_function,
        new_genome=None,
        dependencies=None,
        on_overlap="raise",
        sheet_name="Converted",
        anno_columns_to_keep=[],
        summit_annotator=None,
    ):
        """Convert the intervals into some other coordinate system,
        possibly changing the genome as well.
        Does not conserve annotators, non canonical rows depend on the conversion fucntion

        @conversion_function is passed the dataframe, and must return one containing
        at least (chr, start, stop).
        @conversion_function may be a tuple (function, [annotators]), 
            in which case the function is treated as conversion_function 
            and the annotators are added as dependencies
        @conversion_function may be a tuple (function, [annotators], parameters), 
            in which case the function is treated as conversion_function and the 
            annotators are added as dependencies, 
            and an additional parameter_dependency is added
        @dependencies must be a list of jobs
        """
        if not isinstance(new_name, str):
            raise ValueError("Name must be a string")
        if isinstance(conversion_function, tuple):
            if len(conversion_function) > 2:
                convert_parameters = conversion_function[2]
            else:
                convert_parameters = None
            annotators_required = conversion_function[1]
            conversion_function = conversion_function[0]
        else:
            annotators_required = []
            convert_parameters = None

        def do_load():
            df = conversion_function(self.df)
            if not isinstance(df, pd.DataFrame):
                raise ValueError(
                    "GenomicRegions.convert conversion_function must return a pandas.DataFrame."
                )
            for col in df.columns[:]:
                if not (
                    col in self.non_annotator_columns or col in anno_columns_to_keep
                ):
                    df = df.drop(col, axis=1)
            return df

        if new_genome is None:
            new_genome = self.genome
        else:
            if not isinstance(new_genome, GenomeBase):
                raise ValueError(
                    "new_genome %s was not a genome. Did you mean to pass in dependencies, that's the next parameter?"
                    % new_genome
                )
        if self.load_strategy.build_deps:
            deps = [self.load()]
            if dependencies:
                deps.extend(dependencies)
            for anno in annotators_required:
                deps.append(self.add_annotator(anno))
            deps.append(
                ppg.ParameterInvariant(
                    new_name + "_conversion_paramaters", convert_parameters
                )
            )
            deps.append(
                ppg.FunctionInvariant(
                    new_name + "_conversion_function", conversion_function
                )
            )
        else:
            deps = []
        if hasattr(conversion_function, "dependencies"):
            deps.extend(conversion_function.dependencies)
        if summit_annotator is None:  # pragma: no cover
            summit_annotator = self.summit_annotator
        return GenomicRegions(
            new_name,
            do_load,
            deps,
            new_genome,
            on_overlap=on_overlap,
            summit_annotator=summit_annotator,
            sheet_name=sheet_name,
            vid=self.vid,
        )
