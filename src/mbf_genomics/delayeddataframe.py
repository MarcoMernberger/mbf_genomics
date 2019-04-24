from pathlib import Path
import pandas as pd
import pypipegraph as ppg

from .annotator import Annotator
from mbf_externals.util import lazy_method


class DelayedDataFrame(object):
    """Base class for  DataFrame + annotators style of classes.

    An annotator is an object that can calculate additional columns for a DataFrame.

    This is a dual object - it can be used in a pipegraph - in which
    case it .annotate() returns jobs, and calculatin is 'smart & lazy'
    or without a global pipegraph in which case loading and += annotator evaluation
    happens immediatly (though it still chases the dep_anno chain defined by the annotators)

    Annotators may be annotated to any child (created by .filter),
    and in the ppg case, they will be evaluated on the top most parent that
    actually needs them.

    """

    def __init__(self, name, loading_function, dependencies=[], result_dir=None):
        # assert_uniqueness_of_object is taking core of by the load_strategy
        self.name = name
        if result_dir:
            self.result_dir = Path(result_dir)
        else:
            self.result_dir = Path("results") / self.__class__.__name__ / self.name
        self.result_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(loading_function, pd.DataFrame):
            # don't you just love lambda variable binding?
            loading_function = (
                lambda loading_function=loading_function: loading_function
            )

        if not ppg.inside_ppg():
            self.load_strategy = Load_Direct(self, loading_function)
        else:
            self.load_strategy = Load_PPG(self, loading_function, dependencies)
        self.column_to_annotators = {}
        self.annotators = {}
        self.parent = None
        self.children = []
        # this prevents writing the same file with two different mangler functions
        # but still allows you to call write() in ppg settings multiple times
        # if different parts need to ensure it's being written out
        self.mangler_dict = {self.get_table_filename(): None}
        self.load()

    # magic
    def __hash__(self):
        return hash("DelayedDataFrame" + self.name)

    def __str__(self):
        return "DelayedDataFrame(%s)" % self.name

    def __repr__(self):
        return "DelayedDataFrame(%s)" % self.name

    def load(self):
        return self.load_strategy.load()

    def __iadd__(self, other):
        """Add and return self"""
        if isinstance(other, Annotator):
            if ppg.inside_ppg():
                if not self.has_annotator(other):
                    self.load_strategy.add_annotator(other)
                elif self.get_annotator(other.get_cache_name()) is not other:
                    raise ValueError(
                        "trying to add different annotators with identical cache_names\n%s\n%s"
                        % (other, self.get_annotator(other.get_cache_name()))
                    )
            else:
                self.load_strategy.add_annotator(other)

            return self
        else:
            return NotImplemented

    def add_annotator(self, anno):
        """Sugar for
        self += anno
        return self.anno_jobs[anno.get_cache_name()]
        """
        self += anno
        if self.load_strategy.build_deps:  # pragma: no branch
            return self.anno_jobs[anno.get_cache_name()]

    def has_annotator(self, anno):
        return anno.get_cache_name() in self.annotators

    def get_annotator(self, cache_name):
        return self.annotators[cache_name]

    @property
    def root(self):
        x = self
        while x.parent is not None:
            x = x.parent
        return x

    def annotate(self):
        """Job: create plotjobs and dummy annotation job.

        This is all fairly convoluted.
        If you require a particular column, don't depend on this, but on the result of add_annotator().
        This is only the master job that jobs that require *all* columns (table dump...) depend on.
        """
        return self.load_strategy.annotate()

    def filter(
        self, new_name, df_filter_function, annotators=[], dependencies=None, **kwargs
    ):
        """Filter an ddf to a new one called new_name.

        Paramaters
        -----------
            df_filter_function: function
              take a df, return a valid index
            dependencies:
                list of ppg.Jobs
        """

        def load():
            idx_or_bools = df_filter_function(self.df)
            if isinstance(idx_or_bools, pd.Index):
                res = self.df.loc[idx_or_bools][self.non_annotator_columns]
                res = res.assign(parent_row=idx_or_bools)
            else:
                res = self.df[idx_or_bools][self.non_annotator_columns]
                res = res.assign(parent_row=self.df.index[idx_or_bools])
            return res

        if dependencies is None:
            dependencies = []
        elif not isinstance(dependencies, list):  # pragma: no cover
            dependencies = list(dependencies)
        if isinstance(annotators, Annotator):
            annotators = [annotators]
        if self.load_strategy.build_deps:
            dependencies.append(
                ppg.ParameterInvariant(
                    self.__class__.__name__ + "_" + new_name + "_parent", self.name
                )
            )
            dependencies.append(
                ppg.FunctionInvariant(
                    self.__class__.__name__ + "_" + new_name + "_filter",
                    df_filter_function,
                )
            )

            dependencies.append(self.load())
            for anno in annotators:
                self += anno
                dependencies.append(self.anno_jobs[anno.get_cache_name()])

        else:
            for anno in annotators:
                self += anno

        result = self._new_for_filtering(new_name, load, dependencies, **kwargs)
        result.parent = self
        result.filter_annos = annotators
        for anno in self.annotators.values():
            result += anno

        self.children.append(result)
        return result

    def _new_for_filtering(self, new_name, load_func, deps, **kwargs):
        if not "result_dir" in kwargs:
            kwargs["result_dir"] = self.result_dir / new_name
        return self.__class__(new_name, load_func, deps, **kwargs)

    def clone_without_annotators(
        self, new_name, non_annotator_columns_to_copy=None, result_dir=None
    ):
        """Create a clone of this DelayedDataFrame without any of the old annotators attached"""
        if isinstance(non_annotator_columns_to_copy, str):
            raise ValueError(
                "non_annotator_columns_to_copy must be an iterable - maybe you were trying to set result_dir?"
            )

        def load():
            return self.df[self.non_annotator_columns]

        deps = [self.load()]
        result = self.__class__(new_name, load, deps, result_dir)
        return result

    def get_table_filename(self):
        return self.result_dir / (self.name + ".tsv")

    def mangle_df_for_write(self, df):
        return df

    def write(self, output_filename=None, mangler_function=None):
        """Job: Store the internal DataFrame (df) in a table.
        To sort, filter, remove columns, etc before output,
        pass in a mangler_function (takes df, returns df)
        """
        output_filename = self.pathify(
            output_filename, self.get_table_filename().absolute()
        )

        def write(output_filename):
            if mangler_function:
                df = mangler_function(self.df.copy())
            else:
                df = self.mangle_df_for_write(self.df)
            if str(output_filename).endswith(".xls"):
                try:
                    df.to_excel(output_filename, index=False)
                except (ValueError):
                    df.to_csv(output_filename, sep="\t", index=False)
            else:
                df.to_csv(output_filename, sep="\t", index=False, encoding="utf-8")

        if self.load_strategy.build_deps:
            deps = [
                self.annotate(),
                ppg.FunctionInvariant(
                    str(output_filename) + "_mangler", mangler_function
                ),
            ]
        else:
            deps = []
        return self.load_strategy.generate_file(output_filename, write, deps)

    def plot(self, output_filename, plot_func, calc_func=None):
        output_filename = self.pathify(output_filename)

        def do_plot(output_filename=output_filename):
            df = self.df
            if calc_func is not None:
                df = calc_func(df)
            p = plot_func(df)
            if hasattr(p, "pd"):
                p = p.pd
            p.save(output_filename, verbose=False)

        if self.load_strategy.build_deps:
            deps = [
                self.annotate(),
                ppg.FunctionInvariant(
                    output_filename.with_name(output_filename.name + "_plot_func"),
                    plot_func,
                ),
            ]
        else:
            deps = []
        return self.load_strategy.generate_file(output_filename, do_plot, deps)

    def pathify(self, output_filename, default=None):
        """Turn output_filename into a Path. If it's a relative path, treat
        it as relative to self.result_dir,
        if it's absolute, take it is at is
        """
        if output_filename is None:
            output_filename = default
        output_filename = Path(output_filename)
        if not output_filename.is_absolute():
            output_filename = self.result_dir / output_filename
        return output_filename.absolute()


class Load_Direct:
    def __init__(self, ddf, loading_function):
        self.ddf = ddf
        self.loading_function = loading_function
        self.build_deps = False

    def load(self):
        if not hasattr(self.ddf, "df"):
            self.ddf.df = self.loading_function()
            self.ddf.non_annotator_columns = self.ddf.df.columns

    def generate_file(self, filename, write_callback, dependencies, empty_ok=False):
        write_callback(filename)
        return Path(filename)

    def add_annotator(self, anno):
        if anno.get_cache_name() in self.ddf.annotators:
            if self.ddf.annotators[anno.get_cache_name()] != anno:
                raise ValueError(
                    "Trying to add two different annotators with same cache name: %s and %s"
                    % (anno, self.ddf.annotators[anno.get_cache_name()])
                )
            return
        self.ddf.annotators[anno.get_cache_name()] = anno
        for d in anno.dep_annos():
            self.ddf += d
        s_should = set(anno.columns)
        if not s_should:
            raise IndexError("Empty columns")
        if (
            self.ddf.parent is not None
            and anno.get_cache_name() in self.ddf.parent.annotators
        ):
            a_df = self.ddf.parent.df.loc[self.ddf.df.index]
            a_df = a_df[s_should]
        else:
            if not isinstance(anno.columns, list):
                raise ValueError("Columns was not a list")
            if hasattr(anno, "calc_ddf"):
                a_df = anno.calc_ddf(self.ddf)
            else:
                a_df = anno.calc(self.ddf.df)
        if isinstance(a_df, pd.Series) and len(s_should) == 1:
            a_df = pd.DataFrame({next(iter(s_should)): a_df})
        elif not isinstance(a_df, pd.DataFrame):
            raise ValueError(
                "Annotator return non DataFrame result (nor a Series and len(anno.columns) == 1)"
            )
        s_actual = set(a_df.columns)
        if s_should != s_actual:
            raise ValueError(
                "Annotator declared different columns from those actualy calculated: %s"
                % (s_should.symmetric_difference(s_actual))
            )
        for k in s_actual:
            if k in self.ddf.df.columns:
                raise ValueError("Same column form two annotators", k)
        if isinstance(a_df.index, pd.RangeIndex):
            if len(a_df) == len(self.ddf.df):  # assume it's simply ordered by the df
                a_df.index = self.ddf.df.index
            else:
                raise ValueError(
                    "Length and index mismatch between DataFrame and Annotator result - "
                    "Annotator must return either a DF with a compatible index "
                    "or at least one with the same length (and a RangeIndex)"
                )

        self.ddf.df = pd.concat([self.ddf.df, a_df], axis=1)
        for c in self.ddf.children:
            c += anno

    def annotate(self):  # a noop
        return None


class Load_PPG:
    def __init__(self, ddf, loading_function, deps):
        ppg.assert_uniqueness_of_object(ddf)
        ddf.cache_dir = (
            Path(ppg.util.global_pipegraph.cache_folder)
            / ddf.__class__.__name__
            / ddf.name
        )
        ddf.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ddf = ddf
        self.ddf.anno_jobs = {}

        self.loading_function = loading_function
        self.deps = deps
        self.build_deps = True
        self.tree_fixed = False

    def add_annotator(self, anno):
        if ppg.util.global_pipegraph.running:
            raise ppg.JobContractError(
                "Can not add_annotator in a running pipegraph"
                " - the annotator structure get's fixed when a "
                "pipegraph is run, you can't add to it in e.g. a "
                "JobGeneratingJob"
            )
        cache_name = anno.get_cache_name()
        forbidden_chars = "/", "?", "*"
        if any((x in cache_name for x in forbidden_chars)) or len(cache_name) > 60:
            raise ValueError(
                "annotator.column_names[0] not suitable as a cache_name (was %s), add cache_name property"
                % repr(cache_name)
            )
        if not cache_name in self.ddf.annotators:
            # if not hasattr(anno, "columns"): # handled by get_cache_name
            # raise AttributeError("no columns property on annotator %s" % repr(anno))
            self.ddf.annotators[cache_name] = anno
            self.ddf.anno_jobs[cache_name] = self.get_anno_dependency_callback(anno)
        for c in self.ddf.children:
            c += anno
        return self.ddf.anno_jobs[cache_name]

    @lazy_method
    def load(self):
        def load_func(df):
            self.ddf.df = df
            self.ddf.non_annotator_columns = self.ddf.df.columns

        job = ppg.CachedDataLoadingJob(
            self.ddf.cache_dir / "calc", self.loading_function, load_func
        )
        job.depends_on(self.deps).depends_on(
            ppg.FunctionInvariant(
                self.ddf.__class__.__name__ + "_" + self.ddf.name + "_load",
                self.loading_function,
            )
        )
        return job

    def generate_file(self, filename, write_callback, dependencies, empty_ok=False):
        return (
            ppg.FileGeneratingJob(filename, write_callback, empty_ok=empty_ok)
            .depends_on(dependencies)
            .depends_on(self.load())
        )

    def get_anno_dependency_callback(self, anno):
        if anno.get_cache_name() in self.ddf.anno_jobs:
            return self.ddf.anno_jobs[anno.get_cache_name()]
        else:

            def gen():
                self.ddf.root.load_strategy.fix_anno_tree()
                return self.ddf.anno_jobs[anno.get_cache_name()]

            return gen

    def fix_anno_tree(self):
        if self.ddf.parent is not None:
            raise NotImplementedError("Should only be called on the root")
        if self.tree_fixed:
            return
        self.tree_fixed = True

        def recursivly_add_annos(deps, a):
            new = a.dep_annos()
            if None in new:  # pragma: no cover
                raise ValueError("None returned in %s.dep_annos" % a)
            for n in new:
                if n not in deps:
                    recursivly_add_annos(deps, n)
                    deps.add(n)

        def descend_and_add_annos(node):
            annos_here = set(node.ddf.annotators.values())
            deps = set()
            for a in annos_here:
                recursivly_add_annos(deps, a)
            annos_here.update(deps)
            for anno in annos_here:
                node.add_annotator(anno)
                for c in node.ddf.children:
                    c.load_strategy.add_annotator(anno)
                    descend_and_add_annos(c.load_strategy)

        def descend_and_jobify(node):
            for anno in node.ddf.annotators.values():
                if (
                    node.ddf.parent is not None
                    and anno.get_cache_name() in node.ddf.parent.annotators
                ):
                    node.ddf.anno_jobs[anno.get_cache_name()] = node._anno_load(anno)
                else:
                    # this is the top most node with this annotator
                    node.ddf.anno_jobs[
                        anno.get_cache_name()
                    ] = node._anno_cache_and_calc(anno)
                if hasattr(anno, "register_qc"):
                    anno.register_qc(node.ddf)
            for c in node.ddf.children:
                descend_and_jobify(c.load_strategy)

        descend_and_add_annos(self)
        for anno in self.ddf.annotators.values():
            job = self._anno_cache_and_calc(anno)
            self.ddf.anno_jobs[anno.get_cache_name()] = job
        # for c in self.ddf.children:
        descend_and_jobify(self)

    def _anno_load(self, anno):
        def load():
            self.ddf.df = pd.concat(
                [self.ddf.df, self.ddf.parent.df[anno.columns].loc[self.ddf.df.index]],
                axis=1,
            )

        job = ppg.DataLoadingJob(self.ddf.cache_dir / anno.get_cache_name(), load)
        job.depends_on(
            ppg.FunctionInvariant(
                self.ddf.cache_dir / (anno.get_cache_name() + "_funcv"), anno.calc
            ),
            self.ddf.parent.anno_jobs[anno.get_cache_name()],
            self.ddf.load(),
        )
        return job

    def _anno_cache_and_calc(self, anno):
        def calc():
            if not isinstance(anno.columns, list):
                raise ValueError("Columns was not a list")

            if hasattr(anno, "calc_ddf"):
                df = anno.calc_ddf(self.ddf)
            else:
                df = anno.calc(self.ddf.df)
            if isinstance(df, pd.Series) and len(anno.columns) == 1:
                df = pd.DataFrame({anno.columns[0]: df})
            if not isinstance(df, pd.DataFrame):
                raise ValueError(
                    "result was no dataframe (or series and len(anno.columns) == 1)"
                )
            return df

        def load(df):
            s_should = set(anno.columns)
            if not len(s_should):
                raise ValueError("anno.columns was empty")
            s_actual = set(df.columns)
            if s_should != s_actual:
                raise ValueError(
                    "Annotator declared different columns from those actualy calculated: %s"
                    % (s_should.symmetric_difference(s_actual))
                )
            if set(df.columns).intersection(self.ddf.df.columns):
                raise ValueError(
                    "Annotator created columns that were already present.",
                    self.ddf.name,
                    anno.get_cache_name(),
                    set(df.columns).intersection(self.ddf.df.columns),
                )
            if isinstance(df.index, pd.RangeIndex):
                if len(df) == len(self.ddf.df):  # assume it's simply ordered by the df
                    df.index = self.ddf.df.index
                else:
                    raise ValueError(
                        "Length and index mismatch between DataFrame and Annotator result - "
                        "Annotator must return either a DF with a compatible index "
                        "or at least one with the same length (and a RangeIndex)"
                    )

            self.ddf.df = pd.concat([self.ddf.df, df], axis=1)

        (self.ddf.cache_dir / anno.__class__.__name__).mkdir(exist_ok=True)
        job = ppg.CachedDataLoadingJob(
            self.ddf.cache_dir / anno.__class__.__name__ / anno.get_cache_name(),
            calc,
            load,
        )
        ppg.Job.depends_on(
            job, self.load()
        )  # both the load and nthe calc needs our ddf.df
        job.depends_on(
            self.load(),
            ppg.FunctionInvariant(
                self.ddf.cache_dir / (anno.get_cache_name() + "_calc_func"),
                anno.calc if hasattr(anno, "calc") else anno.calc_ddf,
            ),
        )
        for d in anno.dep_annos():
            job.depends_on(self.ddf.anno_jobs[d.get_cache_name()])
        job.depends_on(anno.deps(self.ddf))
        job.lfg.cores_needed = getattr(anno, "cores_needed", 1)
        return job

    def annotate(self):
        res = lambda: list(self.ddf.anno_jobs.values())  # noqa: E731
        res.job_id = self.ddf.name + "_annotate_callback"
        return res
