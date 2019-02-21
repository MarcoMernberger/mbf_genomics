"""Utility functions to pass to GenomicRegions.convert(..., convert_func)

"""

# TODO: Liftover utility -> mbf_externals
# TODO: chains not included?

import tempfile
import pandas as pd
import pypipegraph as ppg
import numpy as np
import subprocess
from mbf_externals.util import chmod, to_string, to_bytes
from pathlib import Path

file_path = Path(__file__).parent


def grow(basepairs):
    """A function for GenomicRegions.convert that enlarges the regions
    bei doing start = start - basepairs, stop = stop + basepairs"""

    def do_grow(df):
        starts = df["start"] - basepairs
        starts[starts < 0] = 0
        stops = df["stop"] + basepairs
        new_df = df.copy()
        new_df = new_df.assign(start=starts, stop=stops)
        return new_df

    return do_grow, [], basepairs


def promotorize(basepairs=1250):
    """Genes.convert - returns [-basepairs...tss] regions"""

    def do_promotorize(df):
        res = {"chr": df["chr"]}
        res["start"] = np.zeros((len(df),), dtype=np.int32)
        res["stop"] = np.zeros((len(df),), dtype=np.int32)
        forward = df["strand"] == 1
        res["start"][:] = df["tss"]  # Assign within array.
        res["stop"][:] = df["tss"]  # Assign within array.
        res["start"][forward] -= basepairs
        res["start"][res["start"] < 0] = 0
        res["stop"][~forward] += basepairs
        return pd.DataFrame(res)

    return do_promotorize, [], basepairs


def promotorize_trna(basepairs=1250):
    """Genes.convert - returns [-basepairs...start] regions
    it is the promotorize function from above but works with 'start' instead of 'tss',
    especially good to promotorize rRNA and tRNA regions imported via rnaseq/datasets.py"""

    def do_promotorize(df):
        res = {"chr": df["chr"]}
        res["start"] = np.zeros((len(df),), dtype=np.int32)
        res["stop"] = np.zeros((len(df),), dtype=np.int32)
        forward = df["strand"] == 1
        res["start"][:] = df["start"]  # Assign within array.
        res["stop"][:] = df["start"]  # Assign within array.
        res["start"][forward] -= basepairs
        res["start"][res["start"] < 0] = 0
        res["stop"][~forward] += basepairs
        return pd.DataFrame(res)

    return do_promotorize, [], basepairs


def shift(basepairs):
    def do_shift(df):
        res = {
            "chr": df["chr"],
            "start": df["start"] + basepairs,
            "stop": df["stop"] + basepairs,
        }
        return pd.DataFrame(res)

    return do_shift


def summit(summit_annotator):
    def do_summits(df):
        summit_col = summit_annotator.column_name
        res = {
            "chr": df["chr"],
            "start": df["start"] + df[summit_col],
            "stop": df["start"] + df[summit_col] + 1,
        }
        return pd.DataFrame(res)

    return do_summits, [summit_annotator]


def merge_connected():
    """Merge regions that are next to each other.
    100..200, 200..300 becomes 100..300
    """

    def do_merge(df):
        res = {"chr": [], "start": [], "stop": []}
        df = df.sort_values(
            ["chr", "start"], ascending=[True, True]
        )  # you need to do this here so it's true later. Also it makes a copy...
        new_rows = []
        last_chr = None
        last_stop = -1
        last_row = None

        chrs = df["chr"]
        starts = df["start"]
        stops = df["stop"]

        ii = 0
        lendf = len(df)
        keep = np.zeros((lendf,), dtype=np.bool)
        while ii < lendf:
            if chrs[ii] != last_chr:
                last_chr = chrs[ii]
                last_stop = -1
                if last_row is not None:
                    keep[last_row] = True
            else:
                if (
                    starts[ii] <= last_stop + 1
                ):  # +1 so that being next to each other is enough
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
            # new_rows.append(df.get_row(last_row))
        # return pd.DataFrame(new_rows)
        return df[keep]

    return do_merge


class LiftOver(object):
    data_path = file_path / "liftOvers"

    replacements = {"hg19to38": {"11_gl000202_random": "GL000202.1"}}

    @staticmethod
    def do_liftover(listOfChromosomeIntervals, chain_file):
        """perform a lift over. Error messages are silently swallowed!"""
        tmp_input = tempfile.NamedTemporaryFile(mode="wb")
        tmp_output = tempfile.NamedTemporaryFile(mode="wb")
        tmp_error = tempfile.NamedTemporaryFile(mode="wb")
        max_len = 0
        strip_chr = False
        listOfChromosomeIntervals = [list(row) for row in listOfChromosomeIntervals]
        for row in listOfChromosomeIntervals:
            if not row[0].startswith("chr"):
                row[0] = "chr" + row[0]
                strip_chr = True
            tmp_input.write(b" ".join(to_bytes(str(x)) for x in row))
            tmp_input.write(b"\n")
            max_len = max(len(row), max_len)
        tmp_input.write(b"\n")
        tmp_input.flush()  # it's magic ;)
        chmod(file_path / "liftOver", 0o777)
        cmd = [
            file_path / "liftOver",
            tmp_input.name,
            chain_file,
            tmp_output.name,
            tmp_error.name,
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        dummy_stdout, stderr = p.communicate()
        if p.returncode != 0:
            raise ValueError(
                "do_liftover failed. Returncode: %s, stderr: %s"
                % (p.returncode, stderr)
            )
        tmp_output_in = open(tmp_output.name, "rb")
        res = []
        for row in tmp_output_in:
            row = row.strip().split(b"\t")
            if strip_chr:
                row[0] = row[0][3:]
            row[0] = to_string(row[0])
            row[1] = int(row[1])
            row[2] = int(row[2])
            res.append(tuple(row))
        tmp_error_in = open(tmp_error.name, "rb")
        tmp_error_in.read()
        tmp_input.close()
        tmp_output.close()
        tmp_error.close()
        return res

    def get_convert_func(self, key, keep_name=False, filter_to_these_chromosomes=None):
        chain_file = self.data_path / (key + ".over.chain")
        if not chain_file.exists():
            raise ValueError("invalid liftover key, file not found: %s" % chain_file)
        if filter_to_these_chromosomes:
            filter_to_these_chromosomes = set(filter_to_these_chromosomes)

        def do_convert(df):
            if keep_name:
                input_tuples = [
                    ("chr" + row["chr"], row["start"], row["stop"], row["name"])
                    for dummy_idx, row in df.iterrows()
                ]
            else:
                input_tuples = [
                    ("chr" + row["chr"], row["start"], row["stop"])
                    for dummy_idx, row in df.iterrows()
                ]
            output_tuples = self.do_liftover(input_tuples, chain_file)
            output_lists = list(zip(*output_tuples))
            res = pd.DataFrame(
                {
                    "chr": output_lists[0],
                    "start": output_lists[1],
                    "stop": output_lists[2],
                }
            )
            if keep_name:
                res = res.assign(name=output_lists[3])
            new_chr = []
            for x in res["chr"]:
                x = x[3:]
                if x == "m":
                    x = "MT"
                elif key in self.replacements and x in self.replacements[key]:
                    x = self.replacements[key][x]
                new_chr.append(x)
            res["chr"] = new_chr
            for col in df.columns:
                if col not in res.columns:
                    res = res.assign(**{col: df[col]})
            if filter_to_these_chromosomes:
                res = res[res["chr"].isin(filter_to_these_chromosomes)]
            return res

        do_convert.dependencies = [
            ppg.FileTimeInvariant(chain_file),
            ppg.FunctionInvariant(
                "genomics.regions.convert.LiftOver.do_liftover", LiftOver.do_liftover
            ),
        ]
        return do_convert


def hg19_to_hg38(filter_to_these_chromosomes=None):
    """Map a human genome 19 genomic regions into hg 38(=grch38)"""
    return LiftOver().get_convert_func(
        "hg19ToHg38", filter_to_these_chromosomes=filter_to_these_chromosomes
    )


def hg18_to_hg19():
    """Map a human genome 18 (genome == chipseq.genomes.NBCI36()) genomic regions into hg 19(=grch37
    or ensembl.EnsemblGenome('Homo_sapiens',58+)"""
    return LiftOver().get_convert_func("hg18ToHg19")


def hg18_to_hg19_keep_name():
    """Map a human genome 18 (genome == chipseq.genomes.NBCI36()) genomic regions into hg 19(=grch37
    or ensembl.EnsemblGenome('Homo_sapiens',58+)"""
    return LiftOver().get_convert_func("hg18ToHg19", True)


def hg17_to_hg19():
    """Map a human genome 18 (genome == chipseq.genomes.HG17()) genomic regions into hg 19(=grch37
    or ensembl.EnsemblGenome('Homo_sapiens',58+)"""
    return LiftOver().get_convert_func("hg17ToHg19")


def mm8_to_mm9():
    """Map a mouse rev. 8 genome to rev 9
    """
    return LiftOver().get_convert_func("mm8ToMm9")


def mm8_to_mm10(filter_to_these_chromosomes=None):
    """Map a mouse rev. 8 genome to rev 10
    """
    return LiftOver().get_convert_func(
        "mm8ToMm10", filter_to_these_chromosomes=filter_to_these_chromosomes
    )


def mm9_to_mm10():
    """Map a mouse rev. 9 genome to rev 10
    """
    return LiftOver().get_convert_func("mm9ToMm10")


def cookie_cutter(bp):
    """ transform all their binding regions to -1/2 * bp ... 1/2 * bp centered
    around the old midpoint... (so pass in the final size of the region)
    inspired by Lupien et al (doi 10.1016/j.cell.2008.01.018")
    """

    def convert(df):
        peak_lengths = df["stop"] - df["start"]
        centers = np.array(df["start"] + peak_lengths / 2, dtype=np.int32)
        new_starts = centers - bp / 2
        new_stops = new_starts + bp
        new_starts[new_starts < 0] = 0
        res = pd.DataFrame({"chr": df["chr"], "start": new_starts, "stop": new_stops})
        if "strand" in df.columns:
            res["strand"] = df["strand"]
        return res

    return convert, [], bp


def cookie_cutter_asym(
    summit_annotator,
    bp_minus,
    bp_plus,
    retain_additional_columns=None,
    retain_alternate=None,
):
    """ transform all their binding regions to - bp_minus ... + bp_plus
    around the position given in column_name.
    """

    def convert(df):
        centers = df["start"] + df[summit_annotator.column_name]
        new_starts = centers - bp_minus
        new_stops = centers + bp_plus
        new_starts[new_starts < 0] = 0
        ret = {"chr": df["chr"], "start": new_starts, "stop": new_stops}
        if retain_additional_columns is not None:
            for col_name in retain_additional_columns:
                ret[col_name] = df[col_name]
        if retain_alternate is not None:
            for col_name in retain_alternate:
                ret[retain_alternate[col_name]] = df[col_name]
        df = pd.DataFrame(ret)
        return df

    return convert, [], (bp_plus + bp_minus)


def cookie_summit(summit_annotator, bp, drop_those_outside_chromosomes=False):
    """ transform all their binding regions to -1/2 * bp ... 1/2 * bp centered
    around the summit (so pass in the final size of the region)

    if @drop_those_outside_chromosomes is set, regions < 0 are dropped
    """

    def do_summits(df):
        summit_col = summit_annotator.column_name
        res = {
            "chr": df["chr"],
            "start": df["start"] + df[summit_col].astype(int) - bp // 2,
            "stop": df["start"] + df[summit_col].astype(int) + bp // 2,
        }
        res = pd.DataFrame(res)
        if drop_those_outside_chromosomes:
            res = res[res["start"] >= 0]
        return res

    return do_summits, [summit_annotator], (bp, drop_those_outside_chromosomes)


def cookie_summit_alternative_column_name(
    summit_annotator,
    bp,
    drop_those_outside_chromosomes=False,
    alternative_column_name=None,
):
    """ transform all their binding regions to -1/2 * bp ... 1/2 * bp centered
    around the summit (so pass in the final size of the region)

    if @drop_those_outside_chromosomes is set, regions < 0 are dropped
    """

    def do_summits(df):
        summit_col = summit_annotator.column_name
        if alternative_column_name is not None:
            summit_col = alternative_column_name
        res = {
            "chr": df["chr"],
            "start": df["start"] + df[summit_col] - bp / 2,
            "stop": df["start"] + df[summit_col] + 1 + bp / 2,
        }
        res = pd.DataFrame(res)
        if drop_those_outside_chromosomes:
            res = res[res["start"] >= 0]
        return res

    return do_summits, [summit_annotator], (bp, drop_those_outside_chromosomes)


def windows(window_size, drop_smaller_windows=False):
    """Chuck the region into window_size sized windows.
    if @drop_smaller_windows is True, the right most windows get chopped"""

    def create_windows(df):
        res = {"chr": [], "start": [], "stop": []}
        for dummy_idx, row in df.iterrows():
            for start in range(row["start"], row["stop"], window_size):
                stop = min(start + window_size, row["stop"])
                if drop_smaller_windows and stop - start < window_size:
                    continue
                res["chr"].append(row["chr"])
                res["start"].append(start)
                res["stop"].append(stop)
        return pd.DataFrame(res)

    return create_windows, [], (window_size, drop_smaller_windows)
