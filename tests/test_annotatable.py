import pytest
import collections
from pathlib import Path
import pandas as pd
from mbf_genomics import DelayedDataFrame
from mbf_genomics.annotator import Constant, Annotator
import pypipegraph as ppg
from pandas.testing import assert_frame_equal


class LenAnno(Annotator):
    def __init__(self, name):
        self.columns = [name]

    def calc(self, df):
        return pd.DataFrame(
            {self.columns[0]: ["%s%i" % (self.columns[0], len(df))] * len(df)}
        )


@pytest.mark.usefixtures("no_pipegraph")
@pytest.mark.usefixtures("clear_annotators")
class Test_DelayedDataFrameDirect:
    def test_create(self):
        test_df = pd.DataFrame({"A": [1, 2]})

        def load():
            return test_df

        a = DelayedDataFrame("shu", load)
        assert_frame_equal(a.df, test_df)
        assert a.non_annotator_columns == "A"

    def test_write(self):
        test_df = pd.DataFrame({"A": [1, 2]})

        def load():
            return test_df

        a = DelayedDataFrame("shu", load, result_dir="sha")
        assert Path("sha").exists()
        assert_frame_equal(a.df, test_df)
        assert a.non_annotator_columns == "A"
        fn = a.write()
        assert Path("sha") in fn.parents
        assert Path(fn).exists()
        assert_frame_equal(pd.read_csv(fn, sep="\t"), test_df)

    def test_write_excel(self):
        test_df = pd.DataFrame({"A": [1, 2]})

        def load():
            return test_df

        a = DelayedDataFrame("shu", load, result_dir="sha")
        assert Path("sha").exists()
        assert_frame_equal(a.df, test_df)
        assert a.non_annotator_columns == "A"
        fn = a.write("sha.xls")
        assert Path(fn).exists()
        assert_frame_equal(pd.read_excel(fn), test_df)

    def test_write_excel2(self):
        data = {}
        for i in range(0, 257):
            c = "A%i" % i
            d = [1, 1]
            data[c] = d
        test_df = pd.DataFrame(data)

        def load():
            return test_df

        a = DelayedDataFrame("shu", load, result_dir="sha")
        fn = a.write("sha.xls")
        assert Path(fn).exists()
        assert_frame_equal(pd.read_csv(fn, sep="\t"), test_df)

    def test_write_mangle(self):
        test_df = pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})

        def load():
            return test_df

        a = DelayedDataFrame("shu", load)
        assert_frame_equal(a.df, test_df)
        assert (a.non_annotator_columns == ["A", "B"]).all()

        def mangle(df):
            df = df.drop("A", axis=1)
            df = df[df.B == "c"]
            return df

        fn = a.write("test.csv", mangle)
        assert Path(fn).exists()
        assert_frame_equal(pd.read_csv(fn, sep="\t"), mangle(test_df))

    def test_magic(self):
        test_df = pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        a = DelayedDataFrame("shu", lambda: test_df)
        assert hash(a)
        assert a.name in str(a)
        assert a.name in repr(a)
        assert len(a) == 2
        assert bool(a)
        b = DelayedDataFrame("shb", lambda: pd.DataFrame({}))
        assert not bool(b)
        assert len(b) == 0

    def test_annotator(self):
        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        a += Constant("column", "value")
        a.annotate()
        assert "column" in a.df.columns
        assert (a.df["column"] == "value").all()

    def test_add_non_anno(self):
        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        with pytest.raises(TypeError):
            a += 5

    def test_annotator_wrong_columns(self):
        class WrongConstant(Annotator):
            def __init__(self, column_name, value):
                self.columns = [column_name]
                self.value = value

            def calc(self, df):
                return pd.DataFrame({"shu": self.value}, index=df.index)

            def deps(self):
                return []

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        with pytest.raises(ValueError):
            a += WrongConstant("column", "value")

    def test_annotator_minimum_columns(self):
        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        assert "Direct" in str(a.load_strategy)

        class MissingCalc(Annotator):
            column_names = ["shu"]

            def deps(self):
                return []

        with pytest.raises(AttributeError):
            a += MissingCalc()

        class MissingDeps(Annotator):
            column_names = ["shu"]

            def calc(self, df):
                pass

        with pytest.raises(AttributeError):
            a += MissingDeps()

        class EmptyColumnNames(Annotator):
            columns = []

            def calc(self, df):
                return pd.DataFrame({})

            def deps(self):
                return []

        with pytest.raises(IndexError):
            a += EmptyColumnNames()

        class EmptyColumnNamesButCacheName(Annotator):
            cache_name = "shu"
            columns = []

            def calc(self, df):
                return pd.DataFrame({})

            def deps(self):
                return []

        with pytest.raises(IndexError):
            a += EmptyColumnNamesButCacheName()

        class MissingColumnNames(Annotator):
            def calc(self, df):
                pass

            def deps(self):
                return []

        with pytest.raises(AttributeError):
            a += MissingColumnNames()

    def test_DynamicColumNames(self):
        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )

        class Dynamic(Annotator):
            @property
            def columns(self):
                return ["a"]

            def calc(self, df):
                return pd.DataFrame({"a": ["x", "y"]})

        a += Dynamic()
        a.annotate()
        assert_frame_equal(
            a.df, pd.DataFrame({"A": [1, 2], "B": ["c", "d"], "a": ["x", "y"]})
        )

    def test_annos_added_only_once(self):
        count = [0]

        class CountingConstant(Annotator):
            def __init__(self, column_name, value):
                count[0] += 1
                self.columns = [column_name]
                self.value = value

            def calc(self, df):
                return pd.DataFrame({self.columns[0]: self.value}, index=df.index)

            def deps(self):
                return []

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        c = CountingConstant("hello", "c")
        a += c
        a.annotate()
        assert "hello" in a.df.columns
        assert count[0] == 1
        a += c  # this get's ignored

    def test_annos_same_column_different_anno(self):
        count = [0]

        class CountingConstant(Annotator):
            def __init__(self, column_name, value):
                count[0] += 1
                self.columns = [column_name]
                self.value = value

            def calc(self, df):
                return pd.DataFrame({self.columns[0]: self.value}, index=df.index)

            def deps(self):
                return []

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        c = CountingConstant("hello", "c")
        a += c
        a.annotate()
        assert "hello" in a.df.columns
        assert count[0] == 1
        c = CountingConstant("hello2", "c")
        a += c
        a.annotate()
        assert "hello2" in a.df.columns
        assert count[0] == 2
        d = CountingConstant("hello2", "d")
        assert c is not d
        with pytest.raises(ValueError):
            a += d

    def test_annos_same_column_different_anno2(self):
        class A(Annotator):
            cache_name = "hello"
            columns = ["aa"]

            def calc(self, df):
                return pd.DataFrame({self.columns[0]: "a"}, index=df.index)

        class B(Annotator):
            cache_name = "hello2"
            columns = ["aa"]

            def calc(self, df):
                return pd.DataFrame({self.columns[0]: "a"}, index=df.index)

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        a += A()
        with pytest.raises(ValueError):
            a += B()

    def test_annos_dependening(self):
        class A(Annotator):
            cache_name = "hello"
            columns = ["aa"]

            def calc(self, df):
                return pd.DataFrame({self.columns[0]: "a"}, index=df.index)

        class B(Annotator):
            cache_name = "hello2"
            columns = ["ab"]

            def calc(self, df):
                return df["aa"] + "b"

            def dep_annos(self):
                return [A()]

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        a += B()
        a.annotate()
        assert "ab" in a.df.columns
        assert "aa" in a.df.columns
        assert (a.df["ab"] == (a.df["aa"] + "b")).all()

    def test_filtering(self):
        class A(Annotator):
            cache_name = "A"
            columns = ["aa"]

            def calc(self, df):
                return pd.DataFrame({self.columns[0]: "a"}, index=df.index)

        class B(Annotator):
            cache_name = "B"
            columns = ["ab"]

            def calc(self, df):
                return df["aa"] + "b"

            def dep_annos(self):
                return [A()]

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        a += Constant("C", "c")
        assert "C" in a.df.columns
        b = a.filter("sha", lambda df: df["A"] == 1)
        assert "C" in b.df.columns
        a += A()
        assert "aa" in a.df.columns
        assert "aa" in b.df.columns
        b += B()
        assert "ab" in b.df.columns
        assert not "ab" in a.df.columns

    def test_filtering2(self):
        counts = collections.Counter()

        class A(Annotator):
            cache_name = "A"
            columns = ["aa"]

            def calc(self, df):
                counts["A"] += 1
                return pd.DataFrame({self.columns[0]: "a"}, index=df.index)

        class B(Annotator):
            cache_name = "B"
            columns = ["ab"]

            def calc(self, df):
                counts["B"] += 1
                return df["aa"] + "b"

            def dep_annos(self):
                return [A()]

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        b = a.filter("sha", lambda df: df["A"] == 1)
        b += B()
        assert "aa" in b.df.columns
        assert "ab" in b.df.columns
        assert not "aa" in a.df.columns
        assert not "ab" in a.df.columns
        assert counts["A"] == 1
        a += A()
        assert "aa" in a.df.columns
        assert counts["A"] == 2  # no two recalcs
        assert not "ab" in a.df.columns
        a += B()
        assert "ab" in a.df.columns
        assert counts["A"] == 2  # no two recalcs
        assert counts["B"] == 2  # no two recalcs

    def test_filtering_on_annotator(self):
        class A(Annotator):
            cache_name = "A"
            columns = ["aa"]

            def calc(self, df):
                return pd.DataFrame(
                    {self.columns[0]: (["a", "b"] * int(len(df) / 2 + 1))[: len(df)]},
                    index=df.index,
                )

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        with pytest.raises(KeyError):
            b = a.filter("sha", lambda df: df["aa"] == "a")
        b = a.filter("sha", lambda df: df["aa"] == "a", [A()])
        canno = Constant("C", "c")
        a += canno
        b += canno
        assert (b.df["A"] == [1]).all()

    def test_multi_level(self):
        a = DelayedDataFrame(
            "shu",
            lambda: pd.DataFrame(
                {"A": [1, 2, 3], "B": ["a", "b", "c"], "idx": ["x", "y", "z"]}
            ).set_index("idx"),
        )
        b = a.filter("sha", lambda df: df["C"] == 4, Constant("C", 4))
        a1 = LenAnno("count")
        b += a1
        c = b.filter("shc", lambda df: df["A"] >= 2)
        a2 = LenAnno("count2")
        c += a2
        c.annotate()
        print(c.df)
        assert len(c.df) == 2
        assert (c.df["A"] == [2, 3]).all()
        assert (c.df["count"] == "count3").all()
        assert (c.df["count2"] == "count22").all()

    def test_anno_not_returning_enough_rows_and_no_index_range_index_on_df(self):
        class BrokenAnno(Annotator):
            columns = ["X"]

            def calc(self, df):
                return pd.DataFrame({"X": [1]})

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
        )
        with pytest.raises(ValueError) as excinfo:
            a += BrokenAnno()
            print(str(excinfo))
            assert "Length and index mismatch " in str(excinfo.value)

    def test_anno_returning_series(self):
        a = DelayedDataFrame(
            "shu",
            lambda: pd.DataFrame(
                {"A": [1, 2, 3], "B": ["a", "b", "c"], "idx": ["x", "y", "z"]}
            ).set_index("idx"),
        )

        class SeriesAnno(Annotator):
            columns = ["C"]

            def calc(self, df):
                return pd.Series(list(range(len(df))))

        a += SeriesAnno()
        assert (a.df["C"] == [0, 1, 2]).all()

    def test_anno_returning_series_but_defined_two_columns(self):
        a = DelayedDataFrame(
            "shu",
            lambda: pd.DataFrame(
                {"A": [1, 2, 3], "B": ["a", "b", "c"], "idx": ["x", "y", "z"]}
            ).set_index("idx"),
        )

        class SeriesAnno(Annotator):
            columns = ["C", "D"]

            def calc(self, df):
                return pd.Series(list(range(len(df))))

        with pytest.raises(ValueError) as excinfo:
            a += SeriesAnno()
            assert "result was no dataframe" in str(excinfo)

    def test_anno_returning_string(self):
        a = DelayedDataFrame(
            "shu",
            lambda: pd.DataFrame(
                {"A": [1, 2, 3], "B": ["a", "b", "c"], "idx": ["x", "y", "z"]}
            ).set_index("idx"),
        )

        class SeriesAnno(Annotator):
            columns = ["C", "D"]

            def calc(self, df):
                return "abc"

        with pytest.raises(ValueError) as excinfo:
            a += SeriesAnno()
            assert "result was no dataframe" in str(excinfo)

    def test_lying_about_columns(self):
        a = DelayedDataFrame(
            "shu",
            lambda: pd.DataFrame(
                {"A": [1, 2, 3], "B": ["a", "b", "c"], "idx": ["x", "y", "z"]}
            ).set_index("idx"),
        )

        class SeriesAnno(Annotator):
            columns = ["C"]

            def calc(self, df):
                return pd.DataFrame({"D": [0, 1, 2]})

        with pytest.raises(ValueError) as excinfo:
            a += SeriesAnno()
            assert "declared different" in str(excinfo)


@pytest.mark.usefixtures("new_pipegraph")
class Test_DelayedDataFramePPG:
    def test_create(self):
        test_df = pd.DataFrame({"A": [1, 2]})

        def load():
            return test_df

        a = DelayedDataFrame("shu", load)
        assert not hasattr(a, "df")
        ppg.JobGeneratingJob("shu", lambda: 55).depends_on(a.load())
        ppg.run_pipegraph()
        assert_frame_equal(a.df, test_df)
        assert a.non_annotator_columns == "A"

    def test_write(self):
        test_df = pd.DataFrame({"A": [1, 2]})

        def load():
            return test_df

        a = DelayedDataFrame("shu", load)
        fn = a.write()
        ppg.run_pipegraph()
        assert Path(fn.filenames[0]).exists()
        assert_frame_equal(pd.read_csv(fn.filenames[0], sep="\t"), test_df)

    def test_annotator_basic(self):
        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        a += Constant("aa", "aa")
        ppg.JobGeneratingJob("shu", lambda: 55).depends_on(a.annotate())
        ppg.run_pipegraph()
        assert (a.df["aa"] == "aa").all()

    def test_annotator_raising(self):
        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )

        class RaiseAnno(Annotator):
            columns = ["aa"]
            cache_name = "empty"

            def calc(self, df):
                raise ValueError("hello")

            def deps(self):
                return []

        anno1 = RaiseAnno()
        a += anno1
        ppg.JobGeneratingJob("shu", lambda: 55).depends_on(a.annotate())
        with pytest.raises(ppg.RuntimeError):
            ppg.run_pipegraph()
        anno_job = a.anno_jobs[RaiseAnno().get_cache_name()]
        assert "hello" in str(anno_job.lfg.exception)

    def test_annotator_columns_not_list(self):
        class BrokenAnno(Annotator):
            def __init__(self,):
                self.columns = "shu"

            def calc(self, df):
                return pd.DataFrame(
                    {self.columns[0]: ["%s%i" % (self.columns[0], len(df))] * len(df)}
                )

        with pytest.raises(ValueError):
            BrokenAnno().get_cache_name()

    def test_annotator_empty_columns(self):
        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )

        class EmptyColumnNames(Annotator):
            columns = []
            cache_name = "empty"

            def calc(self, df):
                return pd.DataFrame({"shu": [1, 2]})

            def deps(self):
                return []

        a += EmptyColumnNames()
        ppg.JobGeneratingJob("shu", lambda: 55).depends_on(a.annotate())
        anno_job_cb = a.anno_jobs[EmptyColumnNames().get_cache_name()]
        with pytest.raises(ppg.RuntimeError):
            ppg.run_pipegraph()
        assert anno_job_cb() is anno_job_cb()
        assert "anno.columns was empty" in repr(anno_job_cb().exception)

    def test_annotator_missing_columns(self):
        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )

        class MissingColumnNames(Annotator):
            cache_name = "MissingColumnNames"

            def calc(self, df):
                return pd.DataFrame({})

            def deps(self):
                return []

        with pytest.raises(AttributeError):
            a += MissingColumnNames()

    def test_DynamicColumNames(self):
        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )

        class Dynamic(Annotator):
            @property
            def columns(self):
                return ["a"]

            def calc(self, df):
                return pd.DataFrame({"a": ["x", "y"]})

            def deps(self):
                return []

        a += Dynamic()
        a.anno_jobs[Dynamic().get_cache_name()]
        ppg.JobGeneratingJob("shu", lambda: 55).depends_on(a.annotate())
        ppg.run_pipegraph()
        assert_frame_equal(
            a.df, pd.DataFrame({"A": [1, 2], "B": ["c", "d"], "a": ["x", "y"]})
        )

    def test_annos_same_column_different_anno(self):
        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        c = Constant("hello", "c")
        a += c
        c = Constant("hello2", "c")
        a += c
        c = Constant("hello2", "d")
        with pytest.raises(ValueError):
            a += c

    def test_annos_dependening(self):
        class A(Annotator):
            cache_name = "hello"
            columns = ["aa"]

            def calc(self, df):
                return pd.DataFrame({self.columns[0]: "a"}, index=df.index)

        class B(Annotator):
            cache_name = "hello2"
            columns = ["ab"]

            def calc(self, df):
                return df["aa"] + "b"

            def dep_annos(self):
                return [A()]

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        a += B()
        ppg.JobGeneratingJob("shu", lambda: 55).depends_on(a.annotate())
        ppg.run_pipegraph()
        assert "ab" in a.df.columns
        assert "aa" in a.df.columns
        assert (a.df["ab"] == (a.df["aa"] + "b")).all()

    def test_filteringA(self):
        ppg.util.global_pipegraph.quiet = False

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        b = a.filter("sha", lambda df: df["A"] == 1)
        a += LenAnno("C")
        b.write()
        ppg.run_pipegraph()
        assert "C" in b.df.columns
        assert "C" in a.df.columns
        assert (b.df["C"] == "C2").all()
        assert (a.df["C"] == "C2").all()

    def test_filteringB(self):
        ppg.util.global_pipegraph.quiet = False

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        b = a.filter("sha", lambda df: df["A"] == 1)
        a += LenAnno("C")
        b += LenAnno("D")
        assert not LenAnno("D").get_cache_name() in a.anno_jobs
        b.write()
        ppg.run_pipegraph()
        assert not LenAnno("D").get_cache_name() in a.anno_jobs
        assert "C" in b.df.columns
        assert "C" in a.df.columns
        assert not "D" in a.df.columns
        assert len(a.df) == 2
        assert len(b.df) == 1
        assert (b.df["C"] == "C2").all()
        assert (b.df["D"] == "D1").all()
        assert (a.df["C"] == "C2").all()
        assert not "D" in a.df.columns

    def test_filteringC(self):
        ppg.util.global_pipegraph.quiet = False

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        # a += LenAnno("C")
        b = a.filter("sha", lambda df: df["C"] == 2, LenAnno("C"), set())
        b.write()
        ppg.run_pipegraph()
        assert "C" in a.df
        assert "C" in b.df

    def test_filter_and_clone_without_annos(self):
        ppg.util.global_pipegraph.quiet = False

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        # a += LenAnno("C")
        b = a.filter("sha", lambda df: df["C"] == 2, LenAnno("C"), set())
        b.write()
        with pytest.raises(ValueError):
            b.clone_without_annotators("shc", "hello")
        c = b.clone_without_annotators("shc", result_dir="dir_c")
        fn = c.write().job_id
        ppg.run_pipegraph()
        assert "C" in a.df
        assert "C" in b.df
        assert "C" not in c.df
        written = pd.read_csv(fn, sep="\t")
        assert set(c.df.columns) == set(written.columns)
        for col in c.df.columns:
            assert (c.df[col] == written[col]).all()

    def test_filtering_on_annotator_missing(self):
        class A(Annotator):
            cache_name = "A"
            columns = ["aa"]

            def calc(self, df):
                return pd.DataFrame(
                    {self.columns[0]: (["a", "b"] * int(len(df) / 2 + 1))[: len(df)]},
                    index=df.index,
                )

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        b = a.filter("sha", lambda df: df["aaA"] == "a")
        load_job = b.load()
        a.write()
        print("run now")
        with pytest.raises(ppg.RuntimeError):
            ppg.run_pipegraph()
        assert "KeyError" in repr(load_job.lfg.exception)

    def test_forbidden_cache_names(self):
        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2], "B": ["c", "d"]})
        )
        c1 = Constant("c1*", "*")
        c2 = Constant("c2/", "*")
        c3 = Constant("c3?", "*")
        c4 = Constant("c4" * 100, "*")
        with pytest.raises(ValueError):
            a += c1
        with pytest.raises(ValueError):
            a += c2
        with pytest.raises(ValueError):
            a += c3
        with pytest.raises(ValueError):
            a += c4

    def test_multi_level(self):
        a = DelayedDataFrame(
            "shu",
            lambda: pd.DataFrame(
                {"A": [1, 2, 3], "B": ["a", "b", "c"], "idx": ["x", "y", "z"]}
            ).set_index("idx"),
        )
        b = a.filter("sha", lambda df: df["C"] == 4, Constant("C", 4))
        a1 = LenAnno("count")
        b += a1
        c = b.filter("shc", lambda df: df["A"] >= 2)
        a2 = LenAnno("count2")
        c += a2
        c.write()
        ppg.run_pipegraph()
        assert len(c.df) == 2
        assert (c.df["A"] == [2, 3]).all()
        assert (c.df["count"] == "count3").all()
        assert (c.df["count2"] == "count22").all()

    def test_anno_not_returning_enough_rows_and_no_index(self):
        class BrokenAnno(Annotator):
            columns = ["X"]

            def calc(self, df):
                return pd.DataFrame({"X": [1]})

        a = DelayedDataFrame(
            "shu",
            lambda: pd.DataFrame(
                {"A": [1, 2, 3], "B": ["a", "b", "c"], "idx": ["x", "y", "z"]}
            ).set_index("idx"),
        )
        a += BrokenAnno()
        lj = a.anno_jobs["X"]
        ppg.JobGeneratingJob("shu", lambda: 55).depends_on(a.annotate())
        with pytest.raises(ppg.RuntimeError):
            ppg.run_pipegraph()
        assert "Length and index mismatch " in str(lj().exception)

    def test_anno_not_returning_enough_rows_and_no_index_range_index_on_df(self):
        class BrokenAnno(Annotator):
            columns = ["X"]

            def calc(self, df):
                return pd.DataFrame({"X": [1]})

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
        )
        a += BrokenAnno()
        lj = a.anno_jobs["X"]
        ppg.JobGeneratingJob("shu", lambda: 55).depends_on(a.annotate())
        with pytest.raises(ppg.RuntimeError):
            ppg.run_pipegraph()
        assert "Length and index mismatch " in str(lj().exception)

    def test_annotator_coliding_with_non_anno_column(self):
        a = DelayedDataFrame(
            "shu",
            lambda: pd.DataFrame(
                {"A": [1, 2, 3], "B": ["a", "b", "c"], "idx": ["x", "y", "z"]}
            ).set_index("idx"),
        )
        a += Constant("A", "aa")
        lj = a.anno_jobs["A"]
        ppg.JobGeneratingJob("shu", lambda: 55).depends_on(a.annotate())
        with pytest.raises(ppg.RuntimeError):
            ppg.run_pipegraph()
        assert "were already present" in str(lj().exception)

    def test_anno_returning_series(self):
        a = DelayedDataFrame(
            "shu",
            lambda: pd.DataFrame(
                {"A": [1, 2, 3], "B": ["a", "b", "c"], "idx": ["x", "y", "z"]}
            ).set_index("idx"),
        )

        class SeriesAnno(Annotator):
            columns = ["C"]

            def calc(self, df):
                return pd.Series(list(range(len(df))))

        a += SeriesAnno()
        ppg.JobGeneratingJob("shu", lambda: 55).depends_on(a.annotate())
        ppg.run_pipegraph()
        assert (a.df["C"] == [0, 1, 2]).all()

    def test_anno_returning_series_but_defined_two_columns(self):
        a = DelayedDataFrame(
            "shu",
            lambda: pd.DataFrame(
                {"A": [1, 2, 3], "B": ["a", "b", "c"], "idx": ["x", "y", "z"]}
            ).set_index("idx"),
        )

        class SeriesAnno(Annotator):
            columns = ["C", "D"]

            def calc(self, df):
                return pd.Series(list(range(len(df))))

        a += SeriesAnno()
        lj = a.anno_jobs["C"]
        ppg.JobGeneratingJob("shu", lambda: 55).depends_on(a.annotate())
        with pytest.raises(ppg.RuntimeError):
            ppg.run_pipegraph()
        assert "result was no dataframe" in str(lj().lfg.exception)

    def test_anno_returning_string(self):
        a = DelayedDataFrame(
            "shu",
            lambda: pd.DataFrame(
                {"A": [1, 2, 3], "B": ["a", "b", "c"], "idx": ["x", "y", "z"]}
            ).set_index("idx"),
        )

        class SeriesAnno(Annotator):
            columns = ["C", "D"]

            def calc(self, df):
                return "abc"

        a += SeriesAnno()
        lj = a.anno_jobs["C"]
        ppg.JobGeneratingJob("shu", lambda: 55).depends_on(a.annotate())
        with pytest.raises(ppg.RuntimeError):
            ppg.run_pipegraph()
        assert "result was no dataframe" in str(lj().lfg.exception)

    def test_lying_about_columns(self):
        a = DelayedDataFrame(
            "shu",
            lambda: pd.DataFrame(
                {"A": [1, 2, 3], "B": ["a", "b", "c"], "idx": ["x", "y", "z"]}
            ).set_index("idx"),
        )

        class SeriesAnno(Annotator):
            columns = ["C"]

            def calc(self, df):
                return pd.DataFrame({"D": [0, 1, 2]})

        a += SeriesAnno()
        lj = a.anno_jobs["C"]
        ppg.JobGeneratingJob("shu", lambda: 55).depends_on(a.annotate())
        with pytest.raises(ppg.RuntimeError):
            ppg.run_pipegraph()
        assert "declared different " in str(lj().exception)

    def test_annotator_depending_on_actual_jobs(self):
        def wf():
            Path("fileA").write_text("hello")

        class TestAnno(Annotator):
            columns = ["C"]

            def calc(self, df):
                prefix = Path("fileA").read_text()
                return pd.Series([prefix] * len(df))

            def deps(self):
                return [ppg.FileGeneratingJob("fileA", wf)]

        a = DelayedDataFrame(
            "shu", lambda: pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
        )
        a += TestAnno()
        a.write()
        ppg.run_pipegraph()
        assert (a.df["C"] == "hello").all()
