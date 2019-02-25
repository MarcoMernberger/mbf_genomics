import logging

logging.basicConfig(level=logging.WARNING)
import pytest
import numpy
import os
import pypipegraph as ppg
import pandas as pd
from pathlib import Path
import dppd
import dppd_plotnine
import inspect
import sys


import mbf_genomics.regions as regions
from mbf_genomics.annotator import Constant, Annotator
from mbf_genomes.filebased import InteractiveFileBasedGenome

from .shared import (
    get_genome,
    get_genome_chr_length,
    force_load,
    assert_image_equal,
    inside_ppg,
    run_pipegraph,
    RaisesDirectOrInsidePipegraph
)

dp, X = dppd.dppd()
data_path = Path(__file__).parent / "sample_data"


@pytest.mark.usefixtures("new_pipegraph")
class TestGenomicRegionsLoadingPPGOnly:
    def test_dependency_passing(self):
        job = ppg.ParameterInvariant("sha", (None,))
        a = regions.GenomicRegions("shu", lambda: None, [job], get_genome())
        load_job = a.load()
        assert job in load_job.lfg.prerequisites

    def test_dependency_may_be_iterable_instead_of_list(self):
        job = ppg.ParameterInvariant("shu", (None,))
        a = regions.GenomicRegions("shu", lambda: None, (job,), get_genome())
        load_job = a.load()
        assert job in load_job.lfg.prerequisites

    def test_depenencies_must_be_jobs(self):
        job = ppg.ParameterInvariant("shu", (None,))
        with pytest.raises(ValueError):
            a = regions.GenomicRegions("shu", lambda: None, ["shu"], get_genome())


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestGenomicRegionsLoading:
    def tearDown(self):
        import shutil

        try:
            shutil.rmtree("cache/")
        except OSError:
            pass

    def test_raises_on_duplicate_name(self, both_ppg_and_no_ppg):
        def sample_data():
            return pd.DataFrame(
                {"chr": ["Chromosome"], "start": [1000], "stop": [1100]}
            )

        a = regions.GenomicRegions("shu", sample_data, [], get_genome())

        if inside_ppg():
            with pytest.raises(ValueError):
                b = regions.GenomicRegions("shu", sample_data, [], get_genome())
            both_ppg_and_no_ppg.new_pipegraph()
            c = regions.GenomicRegions(
                "shu", sample_data, [], get_genome()
            )  # should not raise

    def test_loading(self):
        def sample_data():
            return pd.DataFrame(
                {"chr": ["Chromosome"], "start": [1000], "stop": [1100]}
            )

        a = regions.GenomicRegions("sha", sample_data, [], get_genome())
        if inside_ppg():
            assert not hasattr(a, "df")
            force_load(a.load())
        else:
            assert hasattr(a, "df")
        run_pipegraph()
        assert hasattr(a, "df")
        assert len(a.df) == 1
        assert "chr" in a.df.columns
        assert "start" in a.df.columns
        assert "stop" in a.df.columns

    def test_raises_on_invalid_on_overlap(self):
        def inner():
            a = regions.GenomicRegions(
                "shu",
                lambda: None,
                [],
                get_genome(),
                on_overlap="run in circles all about",
            )

        with pytest.raises(ValueError):
            inner()

    def test_magic(self):
        def sample_data():
            return pd.DataFrame(
                {"chr": ["Chromosome"], "start": [1000], "stop": [1100]}
            )

        a = regions.GenomicRegions("shu", sample_data, [], get_genome())
        hash(a)
        str(a)
        repr(a)
        bool(a)
        a.load()
        run_pipegraph()
        with pytest.raises(TypeError):
            iter(a)

    def test_loading_missing_start(self):
        def sample_data():
            return pd.DataFrame({"chr": "1", "stop": [1100]})

        with RaisesDirectOrInsidePipegraph(ValueError):
            a = regions.GenomicRegions("sha", sample_data, [], get_genome())
            force_load(a.load)

    def test_loading_missing_chr(self):
        def sample_data():
            return pd.DataFrame({"start": [1000], "stop": [1100]})

        with RaisesDirectOrInsidePipegraph(ValueError):
            a = regions.GenomicRegions("sha", sample_data, [], get_genome())
            force_load(a.load)

    def test_loading_missing_stop(self):
        def sample_data():
            return pd.DataFrame({"chr": "Chromosome", "start": [1200]})

        with RaisesDirectOrInsidePipegraph(ValueError):
            a = regions.GenomicRegions("sha", sample_data, [], get_genome())
            force_load(a.load)

    def test_loading_raises_on_invalid_chromosome(self):
        def sample_data():
            return pd.DataFrame({"chr": ["1b"], "start": [1200], "stop": [1232]})

        with RaisesDirectOrInsidePipegraph(ValueError):
            a = regions.GenomicRegions("sharum", sample_data, [], get_genome())
            force_load(a.load)

    def test_loading_raises_on_no_int_start(self):
        def sample_data():
            return pd.DataFrame(
                {"chr": ["Chromosome"], "start": ["shu"], "stop": [1232]}
            )

        with RaisesDirectOrInsidePipegraph(ValueError):
            a = regions.GenomicRegions("sharum", sample_data, [], get_genome())
            force_load(a.load)

    def test_loading_raises_on_no_int_stop(self):
        def sample_data():
            return pd.DataFrame({"chr": ["Chromosome"], "start": [2], "stop": [20.0]})

        with RaisesDirectOrInsidePipegraph(ValueError):
            a = regions.GenomicRegions("sharum", sample_data, [], get_genome())
            force_load(a.load)

    def test_loading_raises_on_no_str_chr(self):
        def sample_data():
            return pd.DataFrame({"chr": [1], "start": [2], "stop": [20]})

        with RaisesDirectOrInsidePipegraph(ValueError):
            a = regions.GenomicRegions("sharum", sample_data, [], get_genome())
            force_load(a.load)

    def test_loading_raises_on_not_dataframe(self):
        def sample_data():
            return None

        with RaisesDirectOrInsidePipegraph(ValueError):
            a = regions.GenomicRegions("sharum", sample_data, [], get_genome())
            force_load(a.load)

    def test_loading_raises_on_overlapping(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["Chromosome", "Chromosome"],
                    "start": [1000, 1010],
                    "stop": [1100, 1020],
                }
            )

        with RaisesDirectOrInsidePipegraph(ValueError):

            a = regions.GenomicRegions(
                "sha", sample_data, [], get_genome(), on_overlap="raise"
            )
            force_load(a.load)

    def test_raises_on_negative_intervals(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [8, 110, 1110, 11110, 111_110],
                }
            )

        with RaisesDirectOrInsidePipegraph(ValueError):
            a = regions.GenomicRegions("sharum", sample_data, [], get_genome())
            force_load(a.load)

    def test_raises_on_overlapping_intervals(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [1010, 110, 1110, 11110, 111_110],
                }
            )

        with RaisesDirectOrInsidePipegraph(ValueError):
            a = regions.GenomicRegions("sharum", sample_data, [], get_genome())
            force_load(a.load)

    def test_merges_overlapping_intervals(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [1010, 110, 1110, 11110, 111_110],
                }
            )

        a = regions.GenomicRegions(
            "shu", sample_data, [], get_genome_chr_length(), on_overlap="merge"
        )
        force_load(a.load)
        run_pipegraph()
        assert len(a.df) == 4
        assert a.df.iloc[0]["start"] == 10
        assert a.df.iloc[0]["stop"] == 1110

    def test_merge_overlapping_with_function(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [1010, 110, 1110, 11110, 111_110],
                    "pick_me": [1, 2, 3, 4, 5],
                }
            )

        def merge_function(subset_df):
            row = subset_df.iloc[0].to_dict()
            row["pick_me"] = numpy.max(subset_df["pick_me"])
            return row

        a = regions.GenomicRegions(
            "shu",
            sample_data,
            [],
            get_genome_chr_length(),
            on_overlap=("merge", merge_function),
        )
        force_load(a.load)
        run_pipegraph()
        assert len(a.df) == 4
        assert a.df.iloc[0]["start"] == 10
        assert a.df.iloc[0]["stop"] == 1110
        assert a.df.iloc[0]["pick_me"] == 3

    def test_merge_overlapping_with_function2(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [1010, 110, 1110, 11110, 111_110],
                    "pick_me": [10, 2, 3, 4, 5],
                }
            )

        def merge_function(subset_df):
            row = subset_df.iloc[0].to_dict()
            row["pick_me"] = numpy.max(subset_df["pick_me"])
            return row

        a = regions.GenomicRegions(
            "shu",
            sample_data,
            [],
            get_genome_chr_length(),
            on_overlap=("merge", merge_function),
        )
        force_load(a.load)
        run_pipegraph()
        assert len(a.df) == 4
        assert a.df.iloc[0]["start"] == 10
        assert a.df.iloc[0]["stop"] == 1110
        assert a.df.iloc[0]["pick_me"] == 10

    def test_merge_overlapping_with_function_ignores_returned_intervals(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [1010, 110, 1110, 11110, 111_110],
                    "pick_me": [1, 2, 3, 4, 5],
                }
            )

        def merge_function(subset_df):
            row = subset_df.iloc[0].to_dict()
            row["start"] = 9000
            row["pick_me"] = numpy.max(subset_df["pick_me"])
            return row

        a = regions.GenomicRegions(
            "shu",
            sample_data,
            [],
            get_genome_chr_length(),
            on_overlap=("merge", merge_function),
        )
        force_load(a.load)
        run_pipegraph()
        assert len(a.df) == 4
        assert a.df.iloc[0]["start"] == 10
        assert a.df.iloc[0]["stop"] == 1110
        assert a.df.iloc[0]["pick_me"] == 3

    def test_merge_overlapping_with_function_raises_on_non_dict(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [1010, 110, 1110, 11110, 111_110],
                    "pick_me": [1, 2, 3, 4, 5],
                }
            )

        def merge_function(subset_df):
            row = subset_df.iloc[0].copy()
            row["pick_me"] = numpy.max(subset_df["pick_me"])
            return row

        with RaisesDirectOrInsidePipegraph(ValueError):
            a = regions.GenomicRegions(
                "shu",
                sample_data,
                [],
                get_genome_chr_length(),
                on_overlap=("merge", merge_function),
            )
            force_load(a.load)

    def test_merge_overlapping_with_function_raises_on_unknown_column(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [1010, 110, 1110, 11110, 111_110],
                    "pick_me": [1, 2, 3, 4, 5],
                }
            )

        def merge_function(subset_df):
            row = subset_df.iloc[0]
            row["does not exist"] = row["pick_me"]
            return row

        with RaisesDirectOrInsidePipegraph(ValueError):
            a = regions.GenomicRegions(
                "shu",
                sample_data,
                [],
                get_genome_chr_length(),
                on_overlap=("merge", merge_function),
            )
            force_load(a.load)

    def test_merge_overlapping_with_function_raises_on_missing(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [1010, 110, 1110, 11110, 111_110],
                    "pick_me": [1, 2, 3, 4, 5],
                }
            )

        def merge_function(subset_df):
            row = subset_df.iloc[0]
            del row["pick_me"]
            return None

        with RaisesDirectOrInsidePipegraph(ValueError):
            a = regions.GenomicRegions(
                "shu",
                sample_data,
                [],
                get_genome_chr_length(),
                on_overlap=("merge", merge_function),
            )
            force_load(a.load)

    def test_merges_overlapping_intervals_next_to_each_other(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["Chromosome", "Chromosome"],
                    "start": [10, 21],
                    "stop": [20, 100],
                }
            )

        a = regions.GenomicRegions(
            "shu", sample_data, [], get_genome(), on_overlap="merge"
        )
        force_load(a.load)
        run_pipegraph()
        assert (a.df["start"] == [10, 21]).all()
        assert (a.df["stop"] == [20, 100]).all()

    def test_merges_overlapping_earlier_overlaps_later_with_one_in_between(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["Chromosome", "Chromosome", "Chromosome", "Chromosome"],
                    "start": [3100, 3000, 3750, 4910],
                    "stop": [4900, 3500, 4000, 5000],
                }
            )

        a = regions.GenomicRegions(
            "shu", sample_data, [], get_genome(), on_overlap="merge"
        )
        force_load(a.load)

        run_pipegraph()
        assert (a.df["start"] == [3000, 4910]).all()
        assert (a.df["stop"] == [4900, 5000]).all()

    def test_merges_overlapping_intervals_multiple(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5", "1"],
                    "start": [10, 100, 1000, 10000, 100_000, 1005],
                    "stop": [1010, 110, 1110, 11110, 111_110, 2000],
                }
            )

        a = regions.GenomicRegions(
            "shu", sample_data, [], get_genome_chr_length(), on_overlap="merge"
        )
        force_load(a.load)

        run_pipegraph()
        assert len(a.df) == 4
        assert a.df.iloc[0]["start"] == 10
        assert a.df.iloc[0]["stop"] == 2000

    def test_merges_overlapping_intervals_multiple_with_function(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5", "1"],
                    "start": [10, 100, 1000, 10000, 100_000, 1005],
                    "stop": [1010, 110, 1110, 11110, 111_110, 2000],
                    "pick_me": [23, 1234, 2, 4, 5, 50],
                }
            )

        def merge_function(subset_df):
            row = subset_df.iloc[0].to_dict()
            row["pick_me"] = numpy.max(subset_df["pick_me"])
            return row

        a = regions.GenomicRegions(
            "shu",
            sample_data,
            [],
            get_genome_chr_length(),
            on_overlap=("merge", merge_function),
        )
        force_load(a.load)

        run_pipegraph()
        assert len(a.df) == 4
        assert a.df.iloc[0]["start"] == 10
        assert a.df.iloc[0]["stop"] == 2000
        assert a.df.iloc[0]["pick_me"] == 50

    def test_on_overlap_drop(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5", "1", "5"],
                    "start": [10, 100, 1000, 10000, 100_000, 1005, 5000],
                    "stop": [1010, 110, 1110, 11110, 111_110, 2000, 5050],
                }
            )

        a = regions.GenomicRegions(
            "shu", sample_data, [], get_genome_chr_length(), on_overlap="drop"
        )
        force_load(a.load)
        run_pipegraph()
        assert len(a.df) == 4
        assert a.df.iloc[0]["start"] == 100
        assert a.df.iloc[0]["stop"] == 110
        assert a.df.iloc[1]["start"] == 10000
        assert a.df.iloc[1]["stop"] == 11110
        assert a.df.iloc[2]["start"] == 5000
        assert a.df.iloc[2]["stop"] == 5050
        assert a.df.iloc[3]["start"] == 100_000
        assert a.df.iloc[3]["stop"] == 111_110

    def test_on_overlap_ignore(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5", "1", "5"],
                    "start": [10, 100, 1000, 10000, 100_000, 1005, 5000],
                    "stop": [1010, 110, 1110, 11110, 111_110, 2000, 5050],
                    "index": ["a", "b", "c", "d", "e", "f", "g"],
                }
            ).set_index("index")

        a = regions.GenomicRegions(
            "shu", sample_data, [], get_genome_chr_length(), on_overlap="ignore"
        )
        force_load(a.load)

        run_pipegraph()
        assert len(a.df) == 7
        assert (a.df.index == [0, 1, 2, 3, 4, 5, 6]).all()

    def test_merging_apperantly_creating_negative_intervals(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "start": [
                        140_688_139,
                        140_733_871,
                        140_773_241,
                        140_792_022,
                        141_032_547,
                        141_045_565,
                        141_069_938,
                        141_075_938,
                        141_098_775,
                        141_108_518,
                        141_131_159,
                        -4423,
                        -4352,
                        -3398,
                        -3329,
                        -1770,
                        -1693,
                        -737,
                        3400,
                        -598,
                    ],
                    "chr": [
                        str(x)
                        for x in [
                            "1",
                            "1",
                            "1",
                            "1",
                            "1",
                            "1",
                            "1",
                            "1",
                            "1",
                            "1",
                            "1",
                            "2",
                            "2",
                            "2",
                            "2",
                            "2",
                            "2",
                            "2",
                            "2",
                            "2",
                        ]
                    ],
                    "stop": [
                        140_767_241,
                        140_786_022,
                        141_026_547,
                        141_039_565,
                        141_064_437,
                        141_070_437,
                        141_092_775,
                        141_102_518,
                        141_125_159,
                        141_145_045,
                        141_213_431,
                        1577,
                        1648,
                        2602,
                        2671,
                        4230,
                        4307,
                        5263,
                        9400,
                        5402,
                    ],
                }
            )

        with RaisesDirectOrInsidePipegraph(ValueError, "All starts need to be positive"):
            a = regions.GenomicRegions(
                "shu", sample_data, [], get_genome_chr_length(), on_overlap="merge"
            )
            force_load(a.load)

    def test_merge_test(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "1", "1", "3", "5"],
                    "start": [10, 10, 1000, 10000, 100_000],
                    "stop": [100, 100, 1110, 11110, 111_110],
                }
            )

        a = regions.GenomicRegions(
            "shu", sample_data, [], get_genome_chr_length(), on_overlap="merge"
        )
        force_load(a.load)

        run_pipegraph()
        assert len(a.df) == 4
        assert a.df.iloc[0]["start"] == 10
        assert a.df.iloc[0]["stop"] == 100

    def test_merge_identical_ok(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "1", "1", "3", "5"],
                    "start": [10, 10, 1000, 10000, 100_000],
                    "stop": [100, 100, 1110, 11110, 111_110],
                }
            )

        a = regions.GenomicRegions(
            "shu",
            sample_data,
            [],
            get_genome_chr_length(),
            on_overlap="merge_identical",
        )
        force_load(a.load)

        run_pipegraph()
        assert len(a.df) == 4
        assert a.df.iloc[0]["start"] == 10
        assert a.df.iloc[0]["stop"] == 100

    def test_merge_identical_raises(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "1", "1", "3", "5"],
                    "start": [10, 10, 1000, 10000, 100_000],
                    "stop": [100, 120, 1110, 11110, 111_110],
                }
            )

        with RaisesDirectOrInsidePipegraph(ValueError):
            a = regions.GenomicRegions(
                "shu",
                sample_data,
                [],
                get_genome_chr_length(),
                on_overlap="merge_identical",
            )
            force_load(a.load)

    def test_plot_job(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [1010, 110, 1110, 11110, 111_110],
                }
            )

        x = regions.GenomicRegions(
            "shu", sample_data, [], get_genome_chr_length(), on_overlap="merge"
        )
        pj = x.plot(
            "shu.png",
            lambda df: dp(df)
            .p9()
            .add_scatter("chr", "start")
            .pd,  # also tests the write-to-result_dir_part
        )
        fn = "results/GenomicRegions/shu/shu.png"
        if inside_ppg():
            assert isinstance(pj, ppg.FileGeneratingJob)
            assert pj.filenames[0] == fn
        else:
            assert str(pj) == fn
        run_pipegraph()
        assert_image_equal(fn)

    def test_plot_job_with_custom_calc_function(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [1010, 110, 1110, 11110, 111_110],
                }
            )

        x = regions.GenomicRegions(
            "shu", sample_data, [], get_genome_chr_length(), on_overlap="merge"
        )

        def calc(df):
            df = df.copy()
            df = df.assign(shu=["shu"] * len(df))
            return df

        pj = x.plot(
            Path("shu.png").absolute(),
            lambda df: dp(df).p9().add_scatter("shu", "start"),
            calc,
        )
        fn = str(Path("shu.png").absolute())
        if inside_ppg():
            assert isinstance(pj, ppg.FileGeneratingJob)
            assert pj.filenames[0] == fn
        else:
            assert str(pj) == fn
        run_pipegraph()
        assert_image_equal(fn)


@pytest.mark.usefixtures("new_pipegraph")
class TestGenomicRegionsAnnotationDependencyies:
    def setUp(self):
        def sample_data():
            df = pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [12, 110, 1110, 11110, 111_110],
                }
            )
            df = df.assign(summit=df["start"] + (df["stop"] - df["start"]) / 2)
            return df

        self.a = regions.GenomicRegions("shu", sample_data, [], get_genome_chr_length())

    def test_anno_job_depends_on_load(self):
        self.setUp()
        if inside_ppg():
            assert not hasattr(self.a, "df")
        else:
            assert hasattr(self.a, "df")
        ca = Constant("Constant", 5)
        anno_job = self.a.add_annotator(ca)
        assert isinstance(anno_job(), ppg.Job)


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestGenomicRegionsAnnotation:
    def setUp(self):
        def sample_data():
            df = pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [12, 110, 1110, 11110, 111_110],
                }
            )
            df = df.assign(summit=df["start"] + (df["stop"] - df["start"]) / 2)
            return df

        self.a = regions.GenomicRegions("shu", sample_data, [], get_genome_chr_length())

    def test_anno_jobs_are_singletonic(self):
        self.setUp()
        ca = Constant("Constant", 5)
        assert len(self.a.annotators) == 1
        anno_job = self.a.add_annotator(ca)
        anno_job2 = self.a.add_annotator(ca)
        assert anno_job is anno_job2

    def test_anno_jobs_are_singletonic_across_names(self):
        self.setUp()
        ca = Constant("Constant", 5)
        assert len(self.a.annotators) == 1
        anno_job = self.a.add_annotator(ca)
        anno_job2 = self.a.add_annotator(Constant("Constant", 5))
        assert anno_job is anno_job2

    def test_has_annotator(self):
        self.setUp()
        ca = Constant("Constant", 5)
        assert not self.a.has_annotator(ca)
        anno_job = self.a.add_annotator(ca)
        assert self.a.has_annotator(ca)

    def test_annotator_by_name(self):
        self.setUp()
        ca = Constant("Constant", 5)
        assert not self.a.has_annotator(ca)
        anno_job = self.a.add_annotator(ca)
        assert ca == self.a.get_annotator(ca.columns[0])

    def test_anno_jobs_add_columns(self):
        self.setUp()
        ca = Constant("Constant", 5)
        assert len(self.a.annotators) == 1
        anno_job = self.a.add_annotator(ca)
        force_load(self.a.annotate(), "test_anno_jobs_add_columns")
        run_pipegraph()
        assert ca.columns[0] in self.a.df.columns


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestGenomicRegionsWriting:
    def setUp(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [11, 110, 1110, 11110, 111_110],
                }
            )

        self.a = regions.GenomicRegions("shu", sample_data, [], get_genome_chr_length())
        self.sample_filename = "sample.dat"
        try:
            os.unlink(self.sample_filename)
        except OSError:
            pass

    def test_write_bed(self):
        # TODO: rewrite with non-fileformats?
        self.setUp()
        from mbf_fileformats.bed import read_bed

        self.a.write_bed(self.sample_filename)
        run_pipegraph()
        assert len(self.a.df) > 0
        read = read_bed(self.sample_filename)
        assert len(read) == len(self.a.df)
        assert read[0].refseq == b"1"
        assert read[1].refseq == b"1"
        assert read[2].refseq == b"2"
        assert read[3].refseq == b"3"
        assert read[4].refseq == b"5"
        assert read[0].position == 10
        assert read[1].position == 1000
        assert read[2].position == 100
        assert read[3].position == 10000
        assert read[4].position == 100_000
        assert read[0].length == 1
        assert read[1].length == 110
        assert read[2].length == 10
        assert read[3].length == 1110
        assert read[4].length == 11110

    def test_write(self):
        self.setUp()
        self.a.write(self.sample_filename)
        run_pipegraph()
        assert len(self.a.df) > 0
        df = pd.read_csv(self.sample_filename, sep="\t")
        df["chr"] = df["chr"].astype(str)
        for col in self.a.df.columns:
            assert (self.a.df[col] == df[col]).all()

    def test_write_without_filename(self):
        self.setUp()
        self.a.result_dir = ""
        self.a.write()
        run_pipegraph()
        assert os.path.exists("shu.tsv")
        os.unlink("shu.tsv")

    def test_write_sorted(self):
        self.setUp()
        # sorting by chromosome means they would've been equal anyhow, since we
        # internally sort by chr
        self.a.write(self.sample_filename, lambda df: df.sort_values("start"))
        run_pipegraph()
        assert len(self.a.df) > 0
        df = pd.read_csv(self.sample_filename, sep="\t")
        df["chr"] = df["chr"].astype(str)
        assert set(df.columns) == set(self.a.df.columns)
        df = df[self.a.df.columns]  # make them have the same order
        assert not (df == self.a.df).all().all()
        assert (df == self.a.df.sort_values("start").reset_index(drop=True)).all().all()

    def test_plot_plots(self):
        self.setUp()

        pj = self.a.plot("shu.png", lambda df: dp(df).p9().add_scatter("start", "stop"))
        fn = "results/GenomicRegions/shu/shu.png"
        if inside_ppg():
            assert isinstance(pj, ppg.FileGeneratingJob)
            assert pj.filenames[0] == fn
        else:
            assert str(pj) == fn
        run_pipegraph()
        assert_image_equal(fn)


class GRNameAnnotator(Annotator):
    def __init__(self, column_name="GR"):
        self.columns = [column_name]

    def calc_ddf(self, genomic_regions):
        """Return a dataframe with new columns, len(new_df) == len(genomic_regions.df).
        Merging is done by the genomic_region"""
        return pd.DataFrame(
            {self.columns[0]: [genomic_regions.name] * len(genomic_regions.df)}
        )


@pytest.mark.usefixtures("new_pipegraph")
class TestFilterTestDependencies:
    def setUp(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [11, 110, 1110, 11110, 111_110],
                }
            )

        self.genome = get_genome_chr_length()
        self.a = regions.GenomicRegions("shu", sample_data, [], self.genome)

    def test_dependecies(self):
        self.setUp()
        job = ppg.ParameterInvariant("shu_param", (None,))
        b = self.a.filter("sha", lambda df: df["chr"] == "1", dependencies=[job])
        force_load(b.load())
        assert job in b.load().lfg.prerequisites
        run_pipegraph()
        assert len(b.df) == 2

    def test_filter_can_depend_on_anno_jobs(self):
        self.setUp()
        anno = Constant("Constant", 5)
        anno_job = self.a.add_annotator(anno)

        def filter(df):
            return df[anno.columns[0]] == 5

        b = self.a.filter("sha", filter, dependencies=[anno_job])
        b.write("shu.tsv")
        run_pipegraph()
        # self.assertTrue(len(b.df) == 5)

    def test_filter_can_depend_on_single_job(self):
        self.setUp()
        anno = Constant("Constant", 5)
        anno_job = self.a.add_annotator(anno)

        def filter(df):
            return df[anno.columns[0]] == 5

        b = self.a.filter("sha", filter, dependencies=[anno_job])
        b.write("shu.tsv")
        run_pipegraph()
        assert len(b.df) == 5


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestFilter:
    def setUp(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [11, 110, 1110, 11110, 111_110],
                }
            )

        self.genome = get_genome_chr_length()
        self.a = regions.GenomicRegions("shu", sample_data, [], self.genome)

    def test_filtering(self):
        self.setUp()
        b = self.a.filter("sha", lambda df: df["chr"] == "1")
        force_load(b.load)
        run_pipegraph()
        assert len(b.df) == 2
        assert (b.df["start"] == [10, 1000]).all()

    def test_filter_raises_on_no_params(self):
        self.setUp()
        with pytest.raises(TypeError):
            self.a.filter("no_parms")

    def test_sorting(self):
        self.setUp()
        b = self.a.filter("sha", lambda df: df.sort_values("chr").index)
        force_load(b.load)
        run_pipegraph()
        assert len(b.df) == len(self.a.df)
        assert (b.df["chr"] == ["1", "1", "2", "3", "5"]).all()
        assert (b.df["start"] == [10, 1000, 100, 10000, 100_000]).all()

    def test_select_top_k(self):
        self.setUp()
        b = self.a.filter(
            "sha", lambda df: df.sort_values("start", ascending=False)[:2].index
        )
        force_load(b.load)
        run_pipegraph()
        assert len(b.df) == 2
        assert b.df.iloc[1]["chr"] == "5"
        assert b.df.iloc[0]["chr"] == "3"
        assert b.df.iloc[1]["start"] == 100_000
        assert b.df.iloc[0]["start"] == 10000
        assert b.df.iloc[1]["stop"] == 111_110
        assert b.df.iloc[0]["stop"] == 11110

    def test_select_top_k_raises_on_non_int(self):
        self.setUp()
        with RaisesDirectOrInsidePipegraph(TypeError):
            b = self.a.filter(
                "sha", lambda df: df.sort_values("start", ascending=False)[:2.0]
            )
            force_load(b.load)

    def test_annotator_inheritance(self):
        self.setUp()
        anno = GRNameAnnotator()
        self.a.add_annotator(anno)
        b = self.a.filter("sha", lambda df: df["chr"] == "1")
        assert b.has_annotator(anno)
        force_load(self.a.annotate, "test_annotator_inheritance")
        force_load(b.annotate, "test_annotator_inheritanceb")
        run_pipegraph()
        assert len(self.a.df) == 5
        assert len(b.df) == 2
        assert self.a.df.iloc[0][anno.columns[0]] == self.a.name
        assert b.df.iloc[0][anno.columns[0]] == self.a.name

    def test_filter_remove_overlapping(self):
        self.setUp()

        def sample_data():
            return pd.DataFrame({"chr": ["1"], "start": [1001], "stop": [1002]})

        b = regions.GenomicRegions("b", sample_data, [], self.genome)
        c = self.a.filter_remove_overlapping("c", b)
        c.write("shu.tsv")
        run_pipegraph()
        assert len(c.df) == 4
        assert (c.df["start"] == [10, 100, 10000, 100_000]).all()

    def test_filter_to_overlapping(self):
        self.setUp()

        def sample_data():
            return pd.DataFrame({"chr": ["1"], "start": [1001], "stop": [1002]})

        b = regions.GenomicRegions("b", sample_data, [], self.genome)
        c = self.a.filter_to_overlapping("c", b)
        c.write("shu.tsv")
        run_pipegraph()
        assert len(c.df) == 1
        assert (c.df["chr"] == ["1"]).all()
        assert (c.df["start"] == [1000]).all()
        assert (c.df["stop"] == [1110]).all()

    def test_filtering_keeps_non_annotated_non_canonical_columns(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [11, 110, 1110, 11110, 111_110],
                    "hello": [1, 2, 3, 4, 5],
                }
            )

        b = regions.GenomicRegions("b", sample_data, [], get_genome_chr_length())
        c = b.filter("c", df_filter_function=lambda df: df["chr"] == "1")
        c.write("shu.tsv")
        run_pipegraph()
        assert len(c.df) == 2
        assert (c.df["chr"] == ["1", "1"]).all()
        assert (c.df["start"] == [10, 1000]).all()
        assert (c.df["hello"] == [1, 3]).all()


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestInterval:
    def setUp(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [11, 110, 1110, 11110, 111_110],
                }
            )

        self.genome = get_genome_chr_length()
        self.a = regions.GenomicRegions("shu", sample_data, [], self.genome)

    def test_overlapping(self):
        self.setUp()
        force_load(self.a.build_intervals())
        run_pipegraph()
        found = self.a.get_overlapping("1", 800, 1001)
        assert len(found) == 1

    def test_overlapping_excludes_right_border(self):
        self.setUp()

        def sample_data():
            return pd.DataFrame({"chr": ["1"], "start": [100], "stop": [200]})

        a = regions.GenomicRegions("a", sample_data, [], self.genome)
        force_load(a.build_intervals())
        run_pipegraph()
        print(dir(a))
        assert not len(a.get_overlapping("1", 5, 99))
        assert not len(a.get_overlapping("1", 5, 100))
        assert len(a.get_overlapping("1", 5, 101))

    def test_overlapping_with_generator_excludes_right_border(self):
        self.setUp()

        def sample_data():
            return pd.DataFrame({"chr": ["1"], "start": [100], "stop": [200]})

        a = regions.GenomicRegions("a", sample_data, [], self.genome)
        force_load(a.build_intervals())
        run_pipegraph()
        g = a.get_overlapping_generator()
        assert not len(g("1", 5, 99))
        assert not len(g("1", 5, 100))
        assert len(g("1", 5, 101))

    def test_end_exclusion(self):
        self.setUp()

        def sample_data():
            return pd.DataFrame(
                [
                    {"chr": "1", "start": 7_724_885, "stop": 7_841_673},
                    {"chr": "1", "start": 8_236_026, "stop": 8_392_008},
                ]
            )

        a = regions.GenomicRegions("sha", sample_data, [], self.genome)
        force_load(a.build_intervals())
        run_pipegraph()
        # emtpy query
        assert not a.has_overlapping("1", 7_778_885, 7_778_885)
        assert a.has_overlapping("1", 7_778_885, 7_778_886)
        ov = a.get_overlapping("1", 7_778_885, 7_778_885)
        assert len(ov) == 0
        ov = a.get_overlapping("1", 7_778_885, 7_778_886)
        assert len(ov) == 1
        assert ov.iloc[0]["start"] == 7_724_885

    def test_end_exclusion_generator(self):
        self.setUp()

        def sample_data():
            return pd.DataFrame(
                [
                    {"chr": "1", "start": 7_724_885, "stop": 7_841_673},
                    {"chr": "1", "start": 8_236_026, "stop": 8_392_008},
                ]
            )

        a = regions.GenomicRegions("sha", sample_data, [], self.genome)
        force_load(a.build_intervals())
        run_pipegraph()
        h = a.has_overlapping_generator()
        g = a.get_overlapping_generator()
        assert not h("1", 7_778_885, 7_778_885)  # emtpy query
        assert h("1", 7_778_885, 7_778_886)
        ov = g("1", 7_778_885, 7_778_885)
        assert len(ov) == 0
        ov = g("1", 7_778_885, 7_778_886)
        assert len(ov) == 1
        assert ov.iloc[0]["start"] == 7_724_885

    def test_has_overlapping(self):
        self.setUp()
        force_load(self.a.build_intervals())
        run_pipegraph()
        assert self.a.has_overlapping("1", 800, 1001)

    def test_has_overlapping_generator_simple(self):
        self.setUp()
        force_load(self.a.build_intervals())
        run_pipegraph()
        g = self.a.has_overlapping_generator()
        res = []
        for chr, start, stop in [("1", 800, 1001)]:
            res.append(g(chr, start, stop))
        assert (numpy.array(res, dtype=numpy.bool) == [True]).all()

    def test_overlapping_point(self):
        self.setUp()
        force_load(self.a.build_intervals())
        run_pipegraph()
        found = self.a.get_overlapping("2", 105, 106)
        assert len(found) == 1
        found = self.a.get_overlapping("2", 105, 105)
        assert len(found) == 0

    def test_has_overlapping_point(self):
        self.setUp()
        force_load(self.a.build_intervals())
        run_pipegraph()
        assert not self.a.has_overlapping("2", 105, 105)
        assert self.a.has_overlapping("2", 105, 106)

    def test_has_overlapping_generator_point(self):
        self.setUp()
        force_load(self.a.build_intervals())
        run_pipegraph()
        g = self.a.has_overlapping_generator()
        res = []
        for chr, start, stop in [("2", 105, 105), ("2", 105, 106)]:
            res.append(g(chr, start, stop))
        assert (numpy.array(res, dtype=numpy.bool) == [False, True]).all()

    def test_has_overlapping_generator_point_raises_on_going_back(self):
        self.setUp()
        force_load(self.a.build_intervals())
        run_pipegraph()
        g = self.a.has_overlapping_generator()
        res = []
        assert g("2", 105, 106)
        # querying the same start again is ok
        assert g("2", 105, 108)

        with pytest.raises(ValueError):
            g("2", 102, 106)

    def test_has_overlapping_generator_chromosome_switch(self):
        self.setUp()
        # note that the generator restarts when you switch a chromosome.
        force_load(self.a.build_intervals())
        run_pipegraph()
        g = self.a.has_overlapping_generator()
        res = []
        for chr, start, stop in [
            ("2", 105, 105),
            ("2", 105, 106),
            ("1", 9, 20),
            ("1", 11, 12),
            ("2", 5, 50),
            ("2", 100, 120),
            ("2", 120, 150),
        ]:
            res.append(g(chr, start, stop))
        assert (
            numpy.array(res, dtype=numpy.bool)
            == [False, True, True, False, False, True, False]
        ).all()

    def test_has_overlapping_on_filtered(self):
        self.setUp()
        # note that the generator restarts when you switch a chromosome.
        b = self.a.filter("b", lambda df: df["chr"] == "1")
        force_load(b.build_intervals())
        run_pipegraph()
        g = b.has_overlapping_generator()
        res = []
        for chr, start, stop in [
            ("2", 105, 105),
            ("2", 105, 106),
            ("1", 9, 20),
            ("1", 11, 12),
            ("2", 5, 50),
            ("2", 100, 120),
            ("2", 120, 150),
        ]:
            res.append(g(chr, start, stop))
        assert (
            numpy.array(res, dtype=numpy.bool)
            == [False, False, True, False, False, False, False]
        ).all()

    def test_non_overlapping(self):
        self.setUp()
        force_load(self.a.build_intervals())
        run_pipegraph()
        found = self.a.get_overlapping("1", 1200, 1300)
        assert len(found) == 0
        assert isinstance(found, pd.DataFrame)

        found = self.a.get_overlapping("4", 1200, 1300)
        assert len(found) == 0
        assert isinstance(found, pd.DataFrame)

    def test_has_overlapping_not(self):
        self.setUp()
        force_load(self.a.build_intervals())
        run_pipegraph()
        assert not self.a.has_overlapping("1", 1200, 1300)

    def test_build_intervals_works_with_empty_df(self):
        self.setUp()

        def sample_data():
            return pd.DataFrame({"chr": [], "start": [], "stop": []})

        b = regions.GenomicRegions("shub", sample_data, [], get_genome())
        force_load(b.build_intervals())
        run_pipegraph()
        assert len(b.df) == 0
        assert not len(b.get_overlapping("1", 0, 10000))
        assert not b.has_overlapping("1", 0, 10000)
        assert not len(b.get_closest("1", 500))

    def test_closest(self):
        self.setUp()
        force_load(self.a.build_intervals())
        run_pipegraph()
        found = self.a.get_closest("1", 5)
        assert len(found) == 1
        assert found.iloc[0]["chr"] == "1"
        assert found.iloc[0]["start"] == 10

        found = self.a.get_closest("2", 150)  # before, but not after...
        assert len(found) == 1
        assert found.iloc[0]["chr"] == "2"
        assert found.iloc[0]["start"] == 100

        # that's still closer to the left one..
        found = self.a.get_closest("1", 501)
        assert len(found) == 1
        assert found.iloc[0]["chr"] == "1"
        assert found.iloc[0]["start"] == 10

        # definatly closer to the second one...
        found = self.a.get_closest("1", 701)
        assert len(found) == 1
        assert found.iloc[0]["chr"] == "1"
        assert found.iloc[0]["start"] == 1000

        found = self.a.get_closest("1", 1050)  # within an interval...
        assert len(found) == 1
        assert found.iloc[0]["chr"] == "1"
        assert found.iloc[0]["start"] == 1000
        found = self.a.get_closest("1", 505)  # test the border
        assert len(found) == 1
        assert found.iloc[0]["chr"] == "1"
        assert found.iloc[0]["start"] == 10
        found = self.a.get_closest("1", 506)  # test the border
        assert len(found) == 1
        assert found.iloc[0]["chr"] == "1"
        assert found.iloc[0]["start"] == 1000

        found = self.a.get_closest("4", 701)  # empty chromosome...
        assert len(found) == 0

    def test_closest_broken_under_bx(self):
        self.setUp()

        def sample_data():
            return pd.DataFrame(
                {
                    "start": [
                        566_564,
                        569_592,
                        713_866,
                        935_162,
                        1_051_311,
                        1_279_151,
                        1_282_803,
                        1_310_387,
                        1_337_193,
                        1_447_089,
                    ],
                    "chr": [
                        "Chromosome",
                        "Chromosome",
                        "Chromosome",
                        "Chromosome",
                        "Chromosome",
                        "Chromosome",
                        "Chromosome",
                        "Chromosome",
                        "Chromosome",
                        "Chromosome",
                    ],
                    "stop": [
                        567_063,
                        570_304,
                        714_288,
                        937_142,
                        1_052_403,
                        1_281_233,
                        1_283_631,
                        1_311_060,
                        1_337_881,
                        1_447_626,
                    ],
                }
            )

        b = regions.GenomicRegions("shbu", sample_data, [], get_genome())
        force_load(b.build_intervals())
        run_pipegraph()
        closest = b.get_closest("Chromosome", 1_015_327)
        assert len(closest)
        assert closest.iloc[0]["chr"] == "Chromosome"
        assert closest.iloc[0]["start"] == 1_051_311


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestIntervalTestsNeedingOverlapHandling(TestInterval):
    def test_nested(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "1", "1", "1", "3"],
                    "start": [10, 25, 30, 100, 0],
                    "stop": [25, 35, 35, 110, 100],
                }
            )

        genome = get_genome_chr_length()
        a = regions.GenomicRegions("shu", sample_data, [], genome, on_overlap="ignore")
        force_load(a.build_intervals())
        run_pipegraph()
        assert (a.get_overlapping("1", 25, 26).start == [25]).all()
        assert (a.get_overlapping("1", 15, 26).start == [10, 25]).all()
        assert (a.get_overlapping("1", 30, 40).start == [25, 30]).all()


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestAssortedGenomicRegion:
    def test_get_no_of_entries(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [11, 110, 1110, 11110, 111_110],
                }
            )

        a = regions.GenomicRegions("shu", sample_data, [], get_genome_chr_length())
        force_load(a.load)
        run_pipegraph()
        assert a.get_no_of_entries() == 5

    def test_get_covered_bases(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [11, 110, 1110, 11110, 111_110],
                }
            )

        a = regions.GenomicRegions("shu", sample_data, [], get_genome_chr_length())
        force_load(a.load)
        run_pipegraph()
        assert a.covered_bases == 1 + 10 + 110 + 1110 + 11110

    def test_get_covered_bases_raises_on_possible_overlaps(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [11, 110, 1110, 11110, 111_110],
                }
            )

        a = regions.GenomicRegions(
            "shu", sample_data, [], get_genome_chr_length(), on_overlap="ignore"
        )
        force_load(a.load())
        run_pipegraph()

        with pytest.raises(ValueError):
            a.covered_bases

    def test_get_mean_size(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [11, 110, 1110, 11110, 111_110],
                }
            )

        a = regions.GenomicRegions("shu", sample_data, [], get_genome_chr_length())
        force_load(a.load)
        run_pipegraph()
        assert a.mean_size == (1.0 + 10 + 110 + 1110 + 11110) / 5


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestSetOperationsOnGenomicRegions:
    def sample_to_gr(self, a, name, on_overlap="merge"):
        if not hasattr(self, "genome"):
            self.genome = get_genome()

        def sample_a(a=a):
            data = {"chr": [], "start": [], "stop": []}
            for start, stop in a:
                data["chr"].append("Chromosome")
                data["start"].append(start)
                data["stop"].append(stop)
            return pd.DataFrame(data)

        a = regions.GenomicRegions(
            name, sample_a, [], self.genome, on_overlap=on_overlap
        )
        return a

    def handle(self, a, b, should, operation):
        a = self.sample_to_gr(a, "a")
        b = self.sample_to_gr(b, "b")
        c = operation(a, "c", b)
        c.write("shu.tsv")
        run_pipegraph()
        assert len(should) == len(c.df)
        starts = []
        stops = []
        for start, stop in should:
            starts.append(start)
            stops.append(stop)
        if len(starts) == 0:
            assert len(c.df) == 0
        else:
            assert (c.df["start"].values == starts).all()
            assert (c.df["stop"].values == stops).all()
        return a, b, c

    def test_union(self):
        a = [(10, 100), (400, 450)]
        b = [(80, 120), (600, 700)]
        should = [(10, 120), (400, 450), (600, 700)]
        self.handle(a, b, should, regions.GenomicRegions.union)

    def test_union_preserves_annos(self):
        a = [(10, 100), (400, 450)]
        ca1 = Constant("one", 1)
        b = [(80, 120), (600, 700)]
        ca2 = Constant("two", 2)
        should = [(10, 120), (400, 450), (600, 700)]
        a = self.sample_to_gr(a, "a")
        a.add_annotator(ca1)
        b = self.sample_to_gr(b, "b")
        b.add_annotator(ca2)
        c = a.union("c", b)
        assert c.has_annotator(ca1)
        assert c.has_annotator(ca2)

    def test_difference(self):
        a = [(10, 100), (400, 450)]
        b = [(80, 120), (600, 700)]
        should = [(10, 80), (400, 450)]
        self.handle(a, b, should, regions.GenomicRegions.difference)

    def test_difference_preserves_annos(self):
        a = [(10, 100), (400, 450)]
        ca1 = Constant("one", 1)
        b = [(80, 120), (600, 700)]
        ca2 = Constant("two", 2)
        should = [(10, 120), (400, 450), (600, 700)]
        a = self.sample_to_gr(a, "a")
        a.add_annotator(ca1)
        b = self.sample_to_gr(b, "b")
        b.add_annotator(ca2)
        c = a.difference("c", b)
        assert c.has_annotator(ca1)
        assert c.has_annotator(ca2)

    def test_difference_start(self):
        a = [(10, 100), (400, 450)]
        b = [(10, 20), (600, 700)]
        should = [(20, 100), (400, 450)]
        self.handle(a, b, should, regions.GenomicRegions.difference)

    def test_difference_start_2(self):
        a = [(10, 100), (400, 450)]
        b = [(5, 20), (600, 700)]
        should = [(20, 100), (400, 450)]
        self.handle(a, b, should, regions.GenomicRegions.difference)

    def test_difference_split(self):
        a = [(100, 1000)]
        b = [(80, 120), (500, 600), (800, 1200)]
        should = [(120, 500), (600, 800)]
        self.handle(a, b, should, regions.GenomicRegions.difference)

    def test_difference_split_2(self):
        a = [(10, 100), (400, 450)]
        b = [(11, 20), (600, 700)]
        should = [(10, 11), (20, 100), (400, 450)]
        self.handle(a, b, should, regions.GenomicRegions.difference)

    def test_difference_empty(self):
        a = [(10, 100), (400, 450)]
        b = [(5, 2000)]
        should = []
        self.handle(a, b, should, regions.GenomicRegions.difference)

    def test_difference_adjoining_regions(self):
        # in the end, this is a fence-post error.
        a = [(495, 687)]
        b = [(124, 495)]
        should = [(495, 687)]
        self.handle(a, b, should, regions.GenomicRegions.difference)

    def test_intersection(self):
        a = [(10, 100), (400, 450)]
        b = [(80, 120), (600, 700)]
        should = [(80, 100)]
        self.handle(a, b, should, regions.GenomicRegions.intersection)

    def test_from_common(self):
        a = [(10, 100), (400, 450)]
        b = [(80, 120), (600, 700)]
        a = self.sample_to_gr(a, "a")
        b = self.sample_to_gr(b, "b")
        c = regions.GenomicRegions_Common("c", [a, b])
        c.write()
        run_pipegraph()
        assert (c.df["start"] == [10]).all()
        assert (c.df["stop"] == [120]).all()

    def test_intersection_preserves_annos(self):
        a = [(10, 100), (400, 450)]
        ca1 = Constant("one", 1)
        b = [(80, 120), (600, 700)]
        ca2 = Constant("two", 2)
        should = [(10, 120), (400, 450), (600, 700)]
        a = self.sample_to_gr(a, "a")
        a.add_annotator(ca1)
        b = self.sample_to_gr(b, "b")
        b.add_annotator(ca2)
        c = a.intersection("c", b)
        assert c.has_annotator(ca1)
        assert c.has_annotator(ca2)

    def test_intersection_empty(self):
        a = [(10, 100), (400, 450)]
        b = [(600, 700)]
        should = []
        self.handle(a, b, should, regions.GenomicRegions.intersection)

    def test_overlapping(self):
        a = [(10, 100), (400, 450)]
        b = [(80, 120), (600, 700)]
        should = [(10, 120)]
        self.handle(a, b, should, regions.GenomicRegions.overlapping)

    def test_overlapping_empty(self):
        a = [(10, 100), (400, 450)]
        b = [(110, 120), (600, 700)]
        should = []
        self.handle(a, b, should, regions.GenomicRegions.overlapping)

    def test_overlapping_preserves_annos(self):
        a = [(10, 100), (400, 450)]
        ca1 = Constant("one", 1)
        b = [(80, 120), (600, 700)]
        ca2 = Constant("two", 2)
        should = [(10, 120), (400, 450), (600, 700)]
        a = self.sample_to_gr(a, "a")
        a.add_annotator(ca1)
        b = self.sample_to_gr(b, "b")
        b.add_annotator(ca2)
        c = a.overlapping("c", b)
        assert c.has_annotator(ca1)
        assert c.has_annotator(ca2)

    def test_invert(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1"],
                    "start": [10, 100, 1000],
                    "stop": [12, 110, 1110],
                }
            )

        a = regions.GenomicRegions("sharum", sample_data, [], get_genome_chr_length())
        b = a.invert("b")

        def should_data():
            return pd.DataFrame(
                {
                    "chr": [
                        "1",
                        "1",
                        "1",
                        "2",
                        "2",
                        "3",
                        "4",
                        "5",
                    ],  # interval-less chromosomes become totally covered...
                    "start": [0, 12, 1110, 0, 110, 0, 0, 0],
                    "stop": [
                        10,
                        1000,
                        100_000,
                        100,
                        200_000,
                        300_000,
                        400_000,
                        500_000,
                    ],
                }
            )

        should = regions.GenomicRegions(
            "should", should_data, [], get_genome_chr_length()
        )
        force_load(b.load())
        force_load(should.load())
        run_pipegraph()
        assert (b.df == should.df).all().all()

    def test_invert_twice(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1"],
                    "start": [10, 100, 1000],
                    "stop": [12, 110, 1110],
                }
            )

        a = regions.GenomicRegions("sharum", sample_data, [], get_genome_chr_length())
        b = a.invert("b")
        c = b.invert("c")
        force_load(c.load())
        run_pipegraph()
        assert (c.df == a.df).all().all()

    def test_invert_preserves_annos(self):
        a = [(10, 100), (400, 450)]
        ca1 = Constant("one", 1)
        a = self.sample_to_gr(a, "a")
        a.add_annotator(ca1)
        c = a.invert("c")
        assert c.has_annotator(ca1)

    def test_overlap_basepairs(self):
        a = [(10, 100), (400, 450)]
        b = [(80, 120), (600, 700)]
        a = self.sample_to_gr(a, "a")
        b = self.sample_to_gr(b, "b")
        a.load()
        force_load(a.build_intervals())
        b.load()
        force_load(b.build_intervals())
        run_pipegraph()
        assert a.overlap_basepair(b) == 20
        assert b.overlap_basepair(a) == 20

    def test_overlap_basepairs_identical(self):
        a = [(10, 100), (400, 450)]
        b = a
        a = self.sample_to_gr(a, "a")
        b = self.sample_to_gr(b, "b")
        a.load()
        force_load(a.build_intervals())
        b.load()
        force_load(b.build_intervals())
        run_pipegraph()
        assert a.overlap_basepair(b) == a.covered_bases

    def test_overlap_basepairs_identical_plus_overlapping(self):
        a = [(10, 100), (400, 450), (498, 499)]
        b = [(10, 100), (400, 425), (412, 499)]
        a = self.sample_to_gr(a, "a")
        b = self.sample_to_gr(b, "b", on_overlap="ignore")
        a.load()
        force_load(a.build_intervals())
        b.load()
        force_load(b.build_intervals())
        run_pipegraph()
        assert a.overlap_basepair(b) == a.covered_bases

    def test_overlap_percentage(self):
        a = [(10, 100), (400, 450)]
        b = [(80, 120), (600, 700)]
        a = self.sample_to_gr(a, "a")
        b = self.sample_to_gr(b, "b")
        a.load()
        force_load(a.build_intervals())
        b.load()
        force_load(b.build_intervals())
        run_pipegraph()
        assert a.overlap_percentage(b) == 20 / (40 + 100.0)
        assert b.overlap_percentage(a) == 20 / (40 + 100.0)

    def test_intersection_count(self):
        a = [(10, 100), (400, 450)]
        b = [(80, 120), (600, 700)]
        a = self.sample_to_gr(a, "a")
        b = self.sample_to_gr(b, "b")
        a.load()
        force_load(a.build_intervals())
        b.load()
        force_load(b.build_intervals())
        run_pipegraph()
        assert a.intersection_count(b) == 1
        assert b.intersection_count(a) == 1

    def test_intersection_count_unequal_filter_count(self):

        a = [(500, 600)]
        b = [(400, 510), (590, 650)]
        a = self.sample_to_gr(a, "a")
        b = self.sample_to_gr(b, "b")
        just_a = a.filter_remove_overlapping("a-b", b)
        just_b = b.filter_remove_overlapping("b-a", a)
        force_load(a.build_intervals())
        force_load(b.build_intervals())
        force_load(just_a.load())
        force_load(just_b.load())
        run_pipegraph()
        # coming this way, we only have one intersection
        assert len(a.df) - len(just_a.df) == a.intersection_count(b)
        # coming this way, it's two of them
        assert len(b.df) - len(just_b.df) != b.intersection_count(a)

    def test_overlap_count(self):
        a = [(10, 100), (400, 450)]
        b = [(80, 120), (600, 700)]
        a = self.sample_to_gr(a, "a")
        b = self.sample_to_gr(b, "b")
        a.load()
        force_load(a.build_intervals())
        b.load()
        force_load(b.build_intervals())
        run_pipegraph()
        assert a.overlap_count(b) == 1
        assert b.overlap_count(a) == 1

    def test_overlap_count_slightly_more_complex(self):

        a = [(500, 600), (1000, 1100), (3000, 3001)]
        b = [(400, 510), (590, 650), (1050, 1055), (2000, 2002)]
        a = self.sample_to_gr(a, "a")
        b = self.sample_to_gr(b, "b")
        force_load(a.build_intervals())
        force_load(b.build_intervals())
        just_a = a.filter_remove_overlapping("just_a", b)
        just_b = b.filter_remove_overlapping("just_b", a)
        force_load(just_a.load())
        force_load(just_b.load())
        both = a.overlapping("ab", b)
        force_load(both.load())
        run_pipegraph()
        assert a.overlap_count(b) == 2
        assert b.overlap_count(a) == 2
        assert len(just_a.df) == 1
        assert len(just_b.df) == 1
        assert a.overlap_count(b) == len(both.df)
        assert b.overlap_count(a) == len(both.df)
        # self.assertEqual(len(a) + len(b), len(just_a) + len(just_b) + 2 * a.overlap_count(b))
        # self.assertEqual(len(a) + len(b), len(just_a) + len(just_b) + 2 * a.overlap_count(a))

    def test_overlap_count_extending(self):

        a = [(500, 600), (650, 1000)]
        b = [(400, 510), (590, 700), (10000, 10001)]
        a = self.sample_to_gr(a, "a")
        b = self.sample_to_gr(b, "b")
        a.build_intervals()
        b.build_intervals()
        m1 = a.overlapping("ab", b)
        m2 = b.overlapping("ba", a)
        force_load(m1.load())
        force_load(m2.load())
        run_pipegraph()
        assert len(m1.df) == 1
        assert len(m2.df) == 1
        assert a.overlap_count(b) == 1
        assert b.overlap_count(a) == 1

    def test_overlap_count_extending_does_not_miss_last(self):

        a = [(500, 600), (650, 1000)]
        b = [(400, 510), (590, 700)]
        a = self.sample_to_gr(a, "a")
        b = self.sample_to_gr(b, "b")
        a.build_intervals()
        b.build_intervals()
        m1 = a.overlapping("ab", b)
        m2 = b.overlapping("ba", a)

        force_load(m1.load())
        force_load(m2.load())
        run_pipegraph()
        assert len(m1.df) == 1
        assert len(m2.df) == 1
        assert a.overlap_count(b) == 1
        assert b.overlap_count(a) == 1

    def test_raises_on_unequal_genome(self):
        def sample_data():
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [11, 110, 1110, 11110, 111_110],
                }
            )

        a = regions.GenomicRegions("shu", sample_data, [], get_genome_chr_length())
        genome2 = get_genome_chr_length(name="arosebyanyothername")
        b = regions.GenomicRegions("sha", sample_data, [], genome2)

        def inner_union():
            a.union("shu", b)

        def inner_intersection():
            c = a.intersection("sha", b)

        def inner_difference():
            a.difference("shi", b)

        def inner_overlapping():
            a.overlapping("sho", b)

        def inner_overlap_basepair():
            a.overlap_basepair(b)

        def inner_overlap_percentag():
            a.overlap_percentage(b)

        def inner_intersection_count():
            a.intersection_count(b)

        with pytest.raises(ValueError):
            inner_union()
        with pytest.raises(ValueError):
            inner_intersection()
        with pytest.raises(ValueError):
            inner_difference()
        with pytest.raises(ValueError):
            inner_overlapping()
        with pytest.raises(ValueError):
            inner_overlap_basepair()
        with pytest.raises(ValueError):
            inner_overlap_percentag()
        with pytest.raises(ValueError):
            inner_intersection_count()
        force_load(a.load())
        force_load(b.load())
        run_pipegraph()

        def inner_iter_intersections():
            next(a._iter_intersections(b))

        with pytest.raises(ValueError):
            inner_iter_intersections()


@pytest.mark.usefixtures("both_ppg_and_no_ppg")
class TestFromXYZ:
    def test_gff(self):
        a = regions.GenomicRegions_FromGFF(
            "shu",
            data_path / "test.gff3",
            get_genome_chr_length(),
            filter_function=lambda entry: entry["source"]
            == b"Regions_of_sig_enrichment",
            comment_char="#",
        )
        force_load(a.load())
        run_pipegraph()
        assert len(a.df) == 4
        assert (a.df["chr"] == ["2", "3", "3", "3"]).all()
        assert (a.df["start"] == [662_743, 53968, 58681, 68187]).all()
        print(a.df["score"])
        assert (
            a.df["score"]
            == [
                b"-0.583140831299777",
                b"0.41667200444801",
                b"1.20483507668127",
                b"0.742024408736128",
            ]
        ).all()

    def test_gff_chromosome_mangler(self):
        a = regions.GenomicRegions_FromGFF(
            "shu",
            data_path / "test.gff3",
            get_genome_chr_length(),
            filter_function=lambda entry: entry["source"]
            == b"Regions_of_sig_enrichment",
            comment_char="#",
            chromosome_mangler=lambda chr: str(int(chr) + 1),
        )
        force_load(a.load())
        run_pipegraph()
        assert len(a.df) == 4
        assert (a.df["chr"] == ["3", "4", "4", "4"]).all()
        assert (a.df["start"] == [662_743, 53968, 58681, 68187]).all()
        assert (
            a.df["score"]
            == [
                b"-0.583140831299777",
                b"0.41667200444801",
                b"1.20483507668127",
                b"0.742024408736128",
            ]
        ).all()

    def test_bed(self):
        a = regions.GenomicRegions_FromBed(
            "shu", data_path / "test.bed", get_genome_chr_length()
        )
        force_load(a.load())
        run_pipegraph()
        assert len(a.df) == 6
        assert (a.df["chr"] == ["2", "2", "2", "3", "3", "3"]).all()
        assert (
            a.df["start"] == [356_591, 662_743, 1_842_875, 53968, 58681, 68187]
        ).all()
        should = [
            -0.610_793_694_004_74,
            -0.583_140_831_299_777,
            numpy.nan,
            0.416_672_004_448_01,
            1.204_835_076_681_27,
            0.742_024_408_736_128,
        ]
        assert (a.df["name"] == ["four", "five", "six", "one", "two", "three"]).all()
        for ii, val in enumerate(a.df["score"]):
            if numpy.isnan(should[ii]):
                assert numpy.isnan(val)
            else:
                assert round(abs(val - should[ii]), 7) == 0

    def test_bed_without_score(self):
        a = regions.GenomicRegions_FromBed(
            "shu", data_path / "test_without_score.bed", get_genome_chr_length()
        )
        force_load(a.load())
        run_pipegraph()
        assert len(a.df) == 6
        assert (a.df["chr"] == ["2", "2", "2", "3", "3", "3"]).all()
        assert (
            a.df["start"] == [356_591, 662_743, 1_842_875, 53968, 58681, 68187]
        ).all()
        assert not ("score" in a.df.columns)

    def test_wig(self):
        a = regions.GenomicRegions_FromWig(
            "shu",
            data_path / "test.wig",
            get_genome_chr_length(),
            enlarge_5prime=2,
            enlarge_3prime=1,
        )
        force_load(a.load())
        run_pipegraph()
        assert len(a.df) == 5
        assert (a.df["chr"] == ["3", "3", "3", "3", "3"]).all()
        assert (a.df["start"] + 2 == [2, 41, 81, 120, 158]).all()
        assert (a.df["stop"] - 1 == [37, 76, 116, 155, 193]).all()
        assert (
            numpy.abs(
                a.df["score"]
                - [
                    0.150_787_654_060_043,
                    0.158_118_185_847_438,
                    0.165_947_064_796_052,
                    0.175_360_778_350_916,
                    0.191_375_614_608_046,
                ]
            )
            <= 0.0001
        ).any()

    def test_partec(self):
        a = regions.GenomicRegions_FromPartec(
            "shu", data_path / "test_partec.txt", get_genome_chr_length()
        )
        force_load(a.load())
        run_pipegraph()
        assert len(a.df) == 5
        assert (a.df["chr"] == ["1", "2", "2", "4", "5"]).all()
        assert (
            a.df["start"]
            == [119_019_045, 33_327_521, 216_002_580, 45_933_913, 138_423_628]
        ).all()
        assert (
            a.df["stop"]
            == [119_019_245, 33_327_690, 216_002_792, 45_934_151, 138_423_806]
        ).all()


@pytest.mark.usefixtures("no_pipegraph")
class TestOutsideOfPipegraph:
    def test_ignores_second_loading(self):
        genome = get_genome_chr_length()
        counter = [0]

        def sample_data():
            counter[0] += 1
            return pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 1000, 10000, 100_000],
                    "stop": [1010, 110, 1110, 11110, 111_110],
                }
            )

        a = regions.GenomicRegions("shu", sample_data, [], genome, on_overlap="merge")
        assert counter[0] == 1  # load is immediate
        a.load()
        assert counter[0] == 1
        a.load()
        assert counter[0] == 1
