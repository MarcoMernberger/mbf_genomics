import tempfile
import pypipegraph as ppg
import os
import numpy as np
import pandas as pd
import itertools
import math
import random
import six
from pathlib import Path


from mbf_genomics.delayeddataframe import DelayedDataFrame
from mbf_genomics.annotator import Annotator
from mbf_genomes.common import reverse_complement
from mbf_genomes import GenomeBase
from mbf_externals.util import lazy_property
from mbf_fileformats.util import pathify
from .annotators import SummitMiddle


def get_overlapping_interval_indices(start, stop, start_array, stop_array):
    """Return the indices of all intervals stored in start_array, stop_array that overlap start, stop"""
    # TODO: replace with better algorithm - e.g. rustbio intervl trees
    first_start_smaller = np.searchsorted(start_array, start) - 1
    first_end_larger = np.searchsorted(stop_array, stop, "right") + 1
    result = []
    for possible_match in range(
        max(0, first_start_smaller), min(first_end_larger, len(stop_array))
    ):
        s = max(start, start_array[possible_match])
        e = min(stop, stop_array[possible_match])
        if s < e:
            result.append(possible_match)
    return result


def merge_intervals(df):
    """take a {chr, start, end, *} dataframe and merge overlapping intervals"""
    if hasattr(df, "to_pandas"):
        raise ValueError("pydataframe passed")
    df = df.sort_values(["chr", "start"], ascending=[True, True]).reset_index(
        drop=True
    )  # you need to do this here so it's true later...
    keep = _merge_choose_rows_to_keep_and_update_positon(df)
    return df.loc[keep].reset_index(drop=True)


def merge_intervals_with_callback(df, callback):
    """take a {chr, start, end, *} dataframe and merge overlapping intervals, calling callback for group larger than one.."""
    df = df.sort_values(["chr", "start"], ascending=[True, True]).reset_index(
        drop=True
    )  # you need to do this here so it's true later...
    keep = _merge_choose_rows_to_keep_and_update_positon(df)
    # now, I have each run identified by ending in keep=True, so it's easy to just walk them and call the merge function appropriatly
    # this saves me from having to figure out where the run started, which the normal merge doesn't need...
    last_keep = -1
    res = []
    for ii, do_keep in enumerate(keep):
        if do_keep:  # the end of a run...
            if last_keep < ii - 1:  # more than one...
                subset = df.iloc[last_keep + 1 : ii + 1]
                row_data = callback(subset)
                if not isinstance(
                    row_data, dict
                ):  # and not (isinstance(row_data, pd.core.series.Series) and len(row_data.shape) == 1):
                    print("type", type(row_data))
                    # print 'len(shape)', len(row_data.shape)
                    print(callback)
                    raise ValueError(
                        "Merge_function returned something other than dict (writing to the pandas series directly is very slow, call to_dict() on it, then modify it.)"
                    )
                if set(row_data.keys()) != set(df.columns):
                    raise ValueError(
                        "Merge_function return wrong columns. Expected %s, was %s"
                        % (df.columns, list(row_data.keys()))
                    )
                row_data["start"] = df.at[ii, "start"]
                row_data["stop"] = df.at[ii, "stop"]
                res.append(row_data)
            else:
                res.append(df.iloc[last_keep + 1].to_dict())
            last_keep = ii
    print(res)
    res = pd.DataFrame(res)[df.columns].reset_index(drop=True)
    return res


def _merge_choose_rows_to_keep_and_update_positon(df):
    """A helper for the merge_intervals and merge_intervals_with_callback functions that refactors some common code.
    Basically, updates continuous 'runs' of overlapping intervals so that the last one has start = start_of_first, end = end of longest,
    and then returns a boolean array with True if it's the last one"""
    if hasattr(df, "to_pandas"):
        raise ValueError("pydataframe passed")
    chrs = np.array(df["chr"])
    starts = np.array(df["start"])
    stops = np.array(df["stop"])
    ii = 0
    lendf = len(df)
    keep = np.zeros((lendf,), dtype=np.bool)
    last_chr = None
    last_stop = 0
    last_row = None
    while ii < lendf:
        if chrs[ii] != last_chr:
            last_chr = chrs[ii]
            last_stop = 0
        if starts[ii] < last_stop:
            starts[ii] = starts[last_row]
            stops[ii] = max(stops[ii], last_stop)
        else:
            if last_row is not None:
                keep[last_row] = True
                # new_rows.append(df.get_row(last_row))
        if stops[ii] > last_stop:
            last_stop = stops[ii]
        last_row = ii
        ii += 1
    if last_row is not None:
        keep[last_row] = True
    df["start"] = starts
    df["stop"] = stops
    return keep


def _merge_identical(sub_df):
    if (len(sub_df["start"].unique()) != 1) or (len(sub_df["stop"].unique()) != 1):
        raise ValueError(
            "Overlapping intervals: %s, merge_identical was set." % (sub_df,)
        )
    return sub_df.iloc[0].to_dict()


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
        skip_adding_summit_anno_for_filtering=False,
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
        if not hasattr(dependencies, "__iter__"):
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
            on_overlap = ("merge", _merge_identical)

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
        else:
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
        else:

            self.summit_annotator = SummitMiddle()

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
        if (df.index != np.array(range(len(df)))).any():  # TODO: check for RangeIndex?
            print(df.index)
            raise ValueError("df.index was not 0..n")

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
        result = []
        if hasattr(first_start_smaller, "__iter__"):
            first_start_smaller = first_start_smaller[0]
        if hasattr(first_end_larger, "__iter__"):
            first_end_larger = first_end_larger[0]

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

    def has_overlapping_generator(self):
        """Returns a function that takes (chr, start, stop) in sorted order, and returns self.has_overlapping(...).
        This is an optimization, in that it exploits that both the queries and the internal storage are sorted,
        thereby avoiding having to (log) search all intervals on each query.

        No optimizating for genomic regions than need_to_handle_overlapping_regions
        """
        gen = self.get_overlapping_generator()

        def call(chr, start, stop, gen=gen):
            g = gen(chr, start, stop)
            if (
                g is True
            ):  # this is a fairly stupid workaround for the dataframe not being 'false' if it's empty
                return g
            elif g is False:
                return g
            else:
                return len(g) != 0

        return call

    def get_overlapping_generator(self):
        """Returns a function that takes (chr, start, stop) in sorted order, and returns self.has_overlapping(...).
        This is an optimization, in that it exploits that both the queries and the internal storage are sorted,
        thereby avoiding having to (log) search all intervals on each query.

        No optimizating for genomic regions than need_to_handle_overlapping_regions
        """
        if self.need_to_handle_overlapping_regions:

            class GetOverlappingGeneratorFake(object):
                """A tiny wrapper so that the overlapping regions in gr case throws the same
                exception if the order is not kept"""

                def __init__(self, genomic_regions):
                    self.gr = genomic_regions
                    self.last_start = 0
                    self.last_chr = None

                def __call__(self, chr, start, stop):
                    if chr != self.last_chr:
                        self.last_chr = chr
                        self.last_start = 0
                    if start < self.last_start:
                        raise ValueError(
                            "Query sequence was not orderd by ascending start"
                        )
                    self.last_start = start
                    return self.gr.has_overlapping(chr, start, stop)

            return GetOverlappingGeneratorFake(self)

        class GetOverlappingGenerator(object):
            def __init__(self, genomic_regions):
                self.gr = genomic_regions
                self.df = genomic_regions.df
                self.last_chr = None
                self.last_start = 0

            def __call__(self, chr, start, stop):
                if chr != self.last_chr:
                    self.last_chr = chr
                    if chr in self.gr.chromosome_intervals:
                        chr_start, chr_stop = self.gr.chromosome_intervals[chr]
                        self.start_array = np.array(
                            self.df["start"][chr_start:chr_stop]
                        )
                        self.stop_array = np.array(self.df["stop"][chr_start:chr_stop])
                        self.current_index = 0
                        self.stop_index = self.stop_array.shape[0]
                        self.last_start = 0
                    elif (
                        chr in self.gr.genome.get_chromosome_lengths()
                    ):  # gene less chromosome
                        self.start_array = np.array((0,), dtype=np.uint32)
                        self.stop_array = np.array((0,), dtype=np.uint32)
                        self.current_index = 0
                        self.stop_index = 0
                        self.last_start = 0
                    else:
                        raise ValueError("Unknown chromosome: %s" % chr)
                if start < self.last_start:
                    raise ValueError("Query sequence was not orderd by ascending start")
                self.last_start = start

                while (
                    self.current_index < self.stop_index - 1
                    and self.stop_array[self.current_index] < start
                ):  # these are not overlapping since their end is to the left of our start...
                    self.current_index += 1
                # so, the one at current_index might be the first one overlapping
                s = max(start, self.start_array[self.current_index])
                e = min(stop, self.stop_array[self.current_index])
                if s < e:
                    chr_start, chr_stop = self.gr.chromosome_intervals[chr]
                    selected_entries = get_overlapping_interval_indices(
                        start,
                        stop,
                        self.start_array[self.current_index :],
                        self.stop_array[self.current_index :],
                    )
                    return self.df.iloc[[x + chr_start for x in selected_entries]]
                else:
                    return self.df.iloc[1:1]

        return GetOverlappingGenerator(self)

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
            start, stop, start_array, stop_array
        )
        return self.df.iloc[[x + chr_start for x in selected_entries]]

    def get_closest(self, chr, point):
        """Find the interval that is closest to the passed point.
        Returns a df with that interval, or an empty df!
        """
        # first we check whether we're on top of a region...
        overlapping = self.get_overlapping(chr, point, point)
        if len(overlapping):
            return overlapping
        if self.need_to_handle_overlapping_regions:
            raise ValueError(
                "BX python is currently broken - get_closest only works for non overlapping regions right now"
            )
            # previous = self.interval_trees[chr].before(point + 1)
            # after = self.interval_trees[chr].after(point - 1)
            # if previous and not after:
            # return self.df[previous[0].value, :]
            # elif after and not previous:
            # return self.df[after[0].value, :]
            # elif after and previous:
            # dist_previous = min( abs( point - previous[0].start), abs(point - previous[0].end))
            # dist_after = min( abs( point - after[0].start), abs(point - after[0].end))
            # if dist_previous < dist_after:
            # return self.df[previous[0].value, :]
            # else:
            # return self.df[after[0].value, :]
            # else:
            # print 'returning empty df for', chr, point, after, previous
            # return self.df[0:0, :] #ie an empty df
        else:
            try:
                chr_start, chr_stop = self.chromosome_intervals[chr]
            except KeyError:
                return self.df[0:0]  # ie an empty df
            if chr_stop == chr_start:  # no intrvals on this chromosome
                return self.df[0:0]  # ie an empty df
            start_array = np.array(self.df["start"][chr_start:chr_stop])
            stop_array = np.array(self.df["stop"][chr_start:chr_stop])
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

    def write_bed(
        self,
        output_filename=None,
        region_name=None,
        additional_info=False,
        include_header=False,
    ):
        """Store the intervals of the GenomicRegion in a BED file
        @region_name: Insert the column of the GenomicRegions-Object e.g. 'repeat_name'"""
        from mbf_fileformats.bed import BedEntry, write_bed

        output_filename = pathify(
            output_filename, Path(self.result_dir) / (self.name + ".bed")
        )

        def write(output_filename=output_filename):
            bed_entries = []
            for ii, row in self.df.iterrows():
                if region_name is None:
                    if additional_info:
                        entry = BedEntry(
                            row["chr"],
                            row["start"],
                            row["stop"],
                            strand=row["strand"] if "strand" in row else None,
                            score=row["score"] if "score" in row else None,
                            name=row["name"] if "name" in row else None,
                        )
                    else:
                        entry = BedEntry(
                            row["chr"],
                            row["start"],
                            row["stop"],
                            strand=row["strand"] if "strand" in row else None,
                        )
                else:
                    entry = BedEntry(
                        row["chr"],
                        row["start"],
                        row["stop"],
                        name=row[region_name]
                        if region_name in row
                        else ValueError(
                            "key: %s not in genomic regions object. These objects are available: %s"
                            % (region_name, self.df.columns)
                        ),
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

        output_filename = pathify(
            output_filename, Path(self.result_dir) / (self.name + ".bigbed")
        )

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
            deps = [
                self.load(),
            ]
        else:
            deps = []
        return self.load_strategy.generate_file(output_filename, write,
                                                deps, empty_ok=True)
        
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

    def filter_remove_overlapping(
        self, new_name, other_gr, summit_annotator=None, sheet_name="Overlaps"
    ):
        """Filter all from this GenomicRegions that have an overlapping region in other_gr
        Note that filtering does not change the coordinates, it only filters,
        non annotator additional columns are kept, annotators are recalculated.
        """
        if other_gr.genome != self.genome:
            raise ValueError(
                "Unequal genomes betwen %s %s in filter_remove_overlapping"
                % (self.name, other_gr.name)
            )

        def filter_func(df):
            keep = np.zeros((len(df)), dtype=np.bool)
            for ii, row in df[["chr", "start", "stop"]].iterrows():
                keep[ii] = other_gr.has_overlapping(
                    row["chr"], row["start"], row["stop"]
                )
            return ~keep

        if not summit_annotator:
            summit_annotator = self.summit_annotator
        if self.load_strategy.build_deps:
            deps = [
                other_gr.build_intervals(),
                ppg.ParameterInvariant(
                    "GenomicRegions_%s_parents" % new_name, (self.name, other_gr.name)
                ),  # so if you swap out the gr, it's detected...
            ]
        else:
            other_gr.build_intervals()
            deps = []
        return self.filter(
            new_name,
            df_filter_function=filter_func,
            dependencies=deps,
            summit_annotator=summit_annotator,
            sheet_name=sheet_name,
        )

    def filter_remove_overlapping_multiple(
        self, new_name, other_grs, summit_annotator=None, sheet_name="Overlaps"
    ):
        """Filter all from this GenomicRegions that have an overlapping region in other_gr
        Note that filtering does not change the coordinates, it only filters,
        non annotator additional rows are kept, annotators are recalculated.
        """
        for other_gr in other_grs:
            if other_gr.genome != self.genome:
                raise ValueError(
                    "Unequal genomes betwen %s %s in filter_remove_overlapping"
                    % (self.name, other_gr.name)
                )

        def filter_func(df):
            keep = np.zeros((len(df)), dtype=np.bool)
            for ii, row in df[["chr", "start", "stop"]].iterrows():
                for other_gr in other_grs:
                    keep[ii] = keep[ii] | other_gr.has_overlapping(
                        row["chr"], row["start"], row["stop"]
                    )
            return ~keep

        if not summit_annotator:
            summit_annotator = self.summit_annotator

        return self.filter(
            new_name,
            df_filter_function=filter_func,
            dependencies=[
                [x.build_intervals() for x in other_grs],
                ppg.ParameterInvariant(
                    "GenomicRegions_%s_parents" % new_name,
                    (self.name, [x.name for x in other_grs]),
                ),  # so if you swap out the gr, it's detected...
            ],
            summit_annotator=summit_annotator,
            sheet_name=sheet_name,
        )

    def filter_remove_overlapping_count(self, other_gr):
        """Return the length a GenomicRegions that was created via filter_remove_overlapping would have,
        without actually creating it.... Must have called other_gr.build_intervals() first (and this one must have been loaded)"""
        if other_gr.genome != self.genome:
            raise ValueError(
                "Unequal genomes betwen %s %s in filter_remove_overlapping"
                % (self.name, other_gr.name)
            )
        count = len(self)
        for ii, row in self.df[["chr", "start", "stop"]].iterrows():
            if other_gr.has_overlapping(row["chr"], row["start"], row["stop"]):
                count -= 1
        return count

    def filter_remove_overlapping_count_multiple(
        self, other_grs, summit_annotator=None
    ):
        """Return the length a GenomicRegions that was created via filter_remove_overlapping_multiple would have,
        without actually creating it.... Must have called other_gr.build_intervals() first (and this one must have been loaded)"""
        for other_gr in other_grs:
            if other_gr.genome != self.genome:
                raise ValueError(
                    "Unequal genomes betwen %s %s in filter_remove_overlapping"
                    % (self.name, other_gr.name)
                )
        count = len(self)
        for ii, row in self.df[["chr", "start", "stop"]].iterrows():
            for other_gr in other_grs:
                if other_gr.has_overlapping(row["chr"], row["start"], row["stop"]):
                    count -= 1
                    break
        return count

    def filter_to_overlapping(
        self, new_name, other_gr, summit_annotator=None, sheet_name=None
    ):
        """Filter to just those that overlap one in other_gr.
        Note that filtering does not change the coordinates, it only filters,
        non annotator additional rows are kept, annotators are recalculated.
        """
        if other_gr.genome != self.genome:
            raise ValueError(
                "Unequal genomes betwen %s %s in filter_remove_overlapping"
                % (self.name, other_gr.name)
            )

        def filter_func(df):
            keep = np.zeros((len(df)), dtype=np.bool)
            for ii, row in df[["chr", "start", "stop"]].iterrows():
                keep[ii] = other_gr.has_overlapping(
                    row["chr"], row["start"], row["stop"]
                )
            return keep

        if self.load_strategy.build_deps:
            deps = [
                other_gr.build_intervals(),
                ppg.ParameterInvariant(
                    "GenomicRegions_%s_parents" % new_name, (self.name, other_gr.name)
                ),  # so if you swap out the gr, it's detected...
            ]
        else:
            other_gr.build_intervals()
            deps = []

        return self.filter(
            new_name,
            df_filter_function=filter_func,
            dependencies=deps,
            summit_annotator=summit_annotator,
            sheet_name=sheet_name,
        )

    def filter_to_overlapping_multiple(
        self, new_name, other_grs, summit_annotator=None, sheet_name="Overlaps"
    ):
        """Filter to just those that overlap one in *all* other_grs.
        Note that filtering does not change the coordinates, it only filters,
        non annotator additional rows are kept, annotators are recalculated.
        """
        for other_gr in other_grs:
            if other_gr.genome != self.genome:
                raise ValueError(
                    "Unequal genomes betwen %s %s in filter_remove_overlapping"
                    % (self.name, other_gr.name)
                )

        def filter_func(df):
            keep = np.ones((len(df)), dtype=np.bool)
            for ii, row in df[["chr", "start", "stop"]].iterrows():
                for gr in other_grs:
                    keep[ii] &= gr.has_overlapping(
                        row["chr"], row["start"], row["stop"]
                    )
            return keep

        return self.filter(
            new_name,
            df_filter_function=filter_func,
            dependencies=[other_gr.build_intervals() for other_gr in other_grs]
            + [
                ppg.ParameterInvariant(
                    "GenomicRegions_%s_parents" % new_name,
                    (self.name, [other_gr.name for other_gr in other_grs]),
                )  # so if you swap out the gr, it's detected...
            ],
            summit_annotator=summit_annotator,
            sheet_name=sheet_name,
        )

    def filter_to_overlapping_count(self, other_gr):
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

    # set operations
    def union(self, new_name, other_gr, summit_annotator=None, add_annotators=True):
        """Create a union of all intervals in two GR, merging overlapping ones
        [(10, 100), (400, 450)],
        [(80, 120), (600, 700)]
        becomes
        [(10, 120), (400, 450), (600, 700)]

        Note that all interval-set based operations (L{union}, L{intersection}, L{difference})
        drop all columns but chr, start, stop (annotators are merged and readded from all sets involved)

        """
        return self.unions(new_name, [other_gr], summit_annotator, add_annotators)

    def unions(self, new_name, other_grs, summit_annotator=None, add_annotators=True):
        """Create a union of all intervals in multiple GRs, merging overlapping ones
        [(10, 100), (400, 450)],
        [(80, 120), (600, 700)]
        becomes
        [(10, 120), (400, 450), (600, 700)]

        Note that all interval-set based operations (L{union}, L{intersection}, L{difference})
        drop all columns but chr, start, stop (annotators are merged and readded from all sets involved)

        """
        from .regions_from import GenomicRegions_Union

        parts = [self] + other_grs
        result = GenomicRegions_Union(
            new_name, parts, summit_annotator=summit_annotator
        )
        if add_annotators:
            for x in parts:
                for anno in x.annotators.values():
                    result += anno
        return result

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

    def intersection(
        self, new_name, other_gr, summit_annotator=None, add_annotators=True
    ):
        """Create an intersection of all intervals...
        [(10, 100), (400, 450)],
        [(80, 120), (600, 700)]
        becomes
        [(80, 100),]

        Note that all interval-set based operations (L{union}, L{intersection}, L{difference})
        drop all columns but chr, start, stop (annotators are merged and readded from all sets involved)
        """
        if self.genome != other_gr.genome:
            raise ValueError(
                "GenomicRegions set-operations only work if both have the same genome. You had %s and %s"
                % (self.genome, other_gr.genome)
            )

        def do_load():
            new_rows = []
            for chr, start, stop in self._iter_intersections(other_gr):
                new_rows.append({"chr": chr, "start": start, "stop": stop})
            if new_rows:
                return pd.DataFrame(new_rows)
            else:
                return pd.DataFrame({"chr": [], "start": [], "stop": []})

        if self.load_strategy.build_deps:
            deps = [
                other_gr.load(),
                self.load(),
                other_gr.build_intervals(),
                ppg.ParameterInvariant(
                    "GenomicRegions_%s_parents" % new_name, (self.name, other_gr.name)
                ),  # so if you swap out the gr, it's detected...
            ]
        else:
            deps = []
            other_gr.build_intervals()

        result = GenomicRegions(
            new_name,
            do_load,
            deps,
            self.genome,
            on_overlap="merge",
            summit_annotator=summit_annotator,
            vid=["intersection"] + self.vid + other_gr.vid,
        )
        for p in [self, other_gr]:
            for anno in p.annotators.values():
                result += anno
        return result

    def difference(
        self, new_name, other_gr, summit_annotator=None, add_annotators=True
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
        if self.genome != other_gr.genome:
            raise ValueError(
                "GenomicRegions set-operations only work if both have the same genome. You had %s and %s"
                % (self.genome, other_gr.genome)
            )

        def do_load():
            new_rows = []
            for idx, row in self.df[["chr", "start", "stop"]].iterrows():
                overlaps = other_gr.get_overlapping(
                    row["chr"], row["start"], row["stop"]
                )
                if not len(overlaps):  # the easy case...
                    new_rows.append(row)
                else:
                    overlaps = self.merge_intervals(
                        overlaps
                    )  # they are now also sorted, so all we need to do is walk them, keep the regions between them (and within the original interval)
                    start = row["start"]
                    if (
                        overlaps.at[0, "start"] <= start
                        and overlaps.at[0, "stop"] > start
                    ):
                        start_i = 1
                        start = overlaps.at[0, "stop"]
                    else:
                        start_i = 0
                    for ii in range(start_i, len(overlaps)):
                        stop = min(overlaps.at[ii, "start"], row["stop"])
                        new_rows.append(
                            {"chr": row["chr"], "start": start, "stop": stop}
                        )
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

        if self.load_strategy.build_deps:
            deps = [
                other_gr.load(),
                self.load(),
                other_gr.build_intervals(),
                ppg.ParameterInvariant(
                    "GenomicRegions_%s_parents" % new_name, (self.name, other_gr.name)
                ),  # so if you swap out the gr, it's detected...
            ]
        else:
            other_gr.build_intervals()
            deps = []

        result = GenomicRegions(
            new_name,
            do_load,
            deps,
            self.genome,
            on_overlap="merge",
            summit_annotator=summit_annotator,
            vid=["difference"] + list(self.vid) + list(other_gr.vid),
        )
        if add_annotators:
            for p in self, other_gr:
                for anno in p.annotators.values():
                    result += anno
        return result

    def overlapping(
        self, new_name, other_gr, summit_annotator=None, add_annotators=True
    ):
        """Create an union of all intersecting intervals...
        Basically, if intervals overlap, keep their union, if they don't have a partner, drop them
        [(10, 100), (400, 450)],
        [(80, 120), (600, 700)]
        becomes
        [(10, 120),]
        """
        if self.genome != other_gr.genome:
            raise ValueError(
                "GenomicRegions set-operations only work if both have the same genome. You had %s and %s"
                % (self.genome, other_gr.genome)
            )

        def do_load():
            new_rows = []
            for dummy_idx, row in self.df[["chr", "start", "stop"]].iterrows():
                overlaps = other_gr.get_overlapping(
                    row["chr"], row["start"], row["stop"]
                )
                if not len(overlaps):  # the easy case...
                    pass  # no overlap, no intersection, no new region!
                else:
                    overlaps = self.merge_intervals(overlaps)
                    start = min(row["start"], (np.min(overlaps["start"])))
                    stop = max(row["stop"], (np.max(overlaps["stop"])))
                    new_rows.append({"chr": row["chr"], "start": start, "stop": stop})
            if new_rows:
                return pd.DataFrame(new_rows)
            else:
                return pd.DataFrame({"chr": [], "start": [], "stop": []})

        if self.load_strategy.build_deps:
            deps = [
                other_gr.load(),
                self.load(),
                other_gr.build_intervals(),
                ppg.ParameterInvariant(
                    "GenomicRegions_%s_parents" % new_name, (self.name, other_gr.name)
                ),  # so if you swap out the gr, it's detected...
            ]
        else:
            deps = []
            other_gr.build_intervals()

        result = GenomicRegions(
            new_name,
            do_load,
            deps,
            self.genome,
            on_overlap="merge",
            summit_annotator=summit_annotator,
            vid=["overlapping"] + self.vid + other_gr.vid,
        )
        if add_annotators:
            for p in [self, other_gr]:
                for anno in p.annotators.values():
                    result += anno
        return result

    def invert(self, new_name, summit_annotator=None, add_annotators=True):
        """Invert a GenomicRegions. What was covered becomes uncovered, what was uncovered becomes covered.
        [(10, 100), (400, 450)], in a chromosome of size 1000
        becomes
        [(0, 10), (450, 1000)]

        Note that all interval-set based operations (L{union}, L{intersection}, L{difference})
        drop all columns but chr, start, stop (annotators are merged and readded from all sets involved)
        """

        def do_load():
            new_rows = []
            chr_lens = self.genome.get_chromosome_lengths()
            chrs_covered = set()
            # for chr, rows in itertools.groupby(self.df[['chr', 'start','stop']].iterrows(), lambda row: row['chr']):
            for chr, rows in self.df[["chr", "start", "stop"]].groupby("chr"):
                chrs_covered.add(chr)
                start = 0
                for dummy_idx, row in rows.iterrows():
                    if start != row["start"]:
                        new_rows.append(
                            {"chr": chr, "start": start, "stop": row["start"]}
                        )
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

        if self.load_strategy.build_deps:
            deps = [
                self.load(),
                ppg.ParameterInvariant(
                    "GenomicRegions_%s_parents" % new_name, (self.name)
                ),  # so if you swap out the gr, it's detected...
            ]
        else:
            deps = []

        result = GenomicRegions(
            new_name,
            do_load,
            deps,
            self.genome,
            on_overlap="merge",
            summit_annotator=summit_annotator,
            vid=["invert"] + self.vid,
        )
        if add_annotators:
            for p in [self]:
                for anno in p.annotators.values():
                    result += anno
        return result

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
            if len(comb):
                print(chrom, comb)
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
        @conversion_function may be a tuple (function, [annotators]), in which case the function is treated as conversion_function and the annotators are added as dependencies
        @conversion_function may be a tuple (function, [annotators], parameters), in which case the function is treated as conversion_function and the annotators are added as dependencies, and an additional parameter_dependency is added
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
        if summit_annotator is None:
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
