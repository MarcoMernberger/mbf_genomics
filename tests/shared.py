import pytest
import pypipegraph as ppg
import inspect
import sys
from matplotlib.testing.compare import compare_images
from pathlib import Path
from mbf_genomes.example_genomes import get_Candidatus_carsonella_ruddii_pv
from mbf_genomes import InteractiveFileBasedGenome
from mbf_genomes import HardCodedGenome

ppg_genome = None


def get_genome(name=None):
    global ppg_genome
    cache_dir = Path(__file__).parent / "run" / "genome_cache"
    if ppg_genome is None:
        old_pipegraph = ppg.util.global_pipegraph
        ppg.new_pipegraph()
        g = get_Candidatus_carsonella_ruddii_pv(
            name, cache_dir=cache_dir, ignore_code_changes=True
        )
        g.download_genome()
        # g.job_genes()
        # g.job_transcripts()
        ppg_genome = g
        ppg.run_pipegraph()
        ppg.util.global_pipegraph = old_pipegraph
    return InteractiveFileBasedGenome(
        name,
        ppg_genome._filename_lookups["genome.fasta"],
        ppg_genome._filename_lookups["cdna.fasta"],
        ppg_genome._filename_lookups["proteins.fasta"],
        ppg_genome._filename_lookups["genes.gtf"],
        ppg_genome.cache_dir,
    )


def get_genome_chr_length(chr_lengths=None, name=None):
    if chr_lengths is None:
        chr_lengths = {
            "1": 100_000,
            "2": 200_000,
            "3": 300_000,
            "4": 400_000,
            "5": 500_000,
        }
    genome = get_genome(name + "_chr" if name else "dummy_genome_chr")
    genome.get_chromosome_lengths = lambda: chr_lengths
    return genome


def inside_ppg():
    return ppg.util.inside_ppg()


def force_load(job, prefix=None):
    """make sure a dataloadingjob has been loaded (if applicable)"""
    if inside_ppg():
        if prefix is None:
            if not isinstance(job, ppg.Job):
                job = job()
            prefix = job.job_id
        return ppg.JobGeneratingJob(prefix + "_force_load", lambda: None).depends_on(
            job
        )


def caller_name(skip=2):
    """Get a name of a caller in the format module.class.method

       `skip` specifies how many levels of stack to skip while getting caller
       name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.

       An empty string is returned if skipped levels exceed stack height
    """

    def stack_(frame):
        framelist = []
        while frame:
            framelist.append(frame)
            frame = frame.f_back
        return framelist

    stack = stack_(sys._getframe(1))
    start = 0 + skip
    if len(stack) < start + 1:
        return ""
    parentframe = stack[start]

    name = []
    module = inspect.getmodule(parentframe)
    # `modname` can be None when frame is executed directly in console
    # TODO(techtonik): consider using __main__
    if module:
        name.append(module.__name__)
    # detect classname
    if "self" in parentframe.f_locals:
        # I don't know any way to detect call from the object method
        # XXX: there seems to be no way to detect static method call - it will
        #      be just a function call
        name.append(parentframe.f_locals["self"].__class__.__name__)
    codename = parentframe.f_code.co_name
    if codename != "<module>":  # top level usually
        name.append(codename)  # function or a method
    del parentframe
    return ".".join(name)


def assert_image_equal(generated_image_path, tolerance=2):
    """assert that the generated image and the base_images/... (test case name) is identical"""
    generated_image_path = Path(generated_image_path).absolute()
    extension = generated_image_path.suffix
    caller = caller_name(1)
    parts = caller.split(".")
    func = parts[-1]
    cls = parts[-2]
    module = parts[-3]
    if cls.lower() == cls:  # not actually a class, a module instead
        module = cls
        cls = "_"
    should_path = (
        Path(__file__).parent / "base_images" / module / cls / (func + extension)
    )
    if not should_path.exists():
        should_path.parent.mkdir(exist_ok=True, parents=True)
        raise ValueError(
            f"Base_line image not found, perhaps: \ncp {generated_image_path} {should_path}"
        )
    err = compare_images(
        str(should_path), str(generated_image_path), tolerance, in_decorator=True
    )
    assert not err


def run_pipegraph():
    if inside_ppg():
        ppg.run_pipegraph()
    else:
        pass


class RaisesDirectOrInsidePipegraph(object):
    """Piece of black magic from the depths of _pytest
    that will check whether a piece of code will raise the
    expected expcition (if outside of ppg), or if it will
    rise the exception when the pipegraph is running

    Use as a context manager like pytest.raises"""

    def __init__(self, expected_exception, search_message=None):
        self.expected_exception = expected_exception
        self.message = "DID NOT RAISE {}".format(expected_exception)
        self.search_message = search_message
        self.excinfo = None

    def __enter__(self):
        import _pytest

        self.excinfo = object.__new__(_pytest._code.ExceptionInfo)
        return self.excinfo

    def __exit__(self, *tp):
        from _pytest.outcomes import fail

        if inside_ppg():
            with pytest.raises(ppg.RuntimeError) as e:
                run_pipegraph()
            assert isinstance(e.value.exceptions[0], self.expected_exception)
            if self.search_message:
                assert self.search_message in str(e.value.exceptions[0])
        else:
            __tracebackhide__ = True
            if tp[0] is None:
                fail(self.message)
            self.excinfo.__init__(tp)
            suppress_exception = issubclass(self.excinfo.type, self.expected_exception)
            if sys.version_info[0] == 2 and suppress_exception:
                sys.exc_clear()
            return suppress_exception


def MockGenome(df_genes, df_transcripts=None, chr_lengths=None):
    if chr_lengths is None:
        chr_lengths = {
            "1": 100_000,
            "2": 200_000,
            "3": 300_000,
            "4": 400_000,
            "5": 500_000,
        }

    df_genes = df_genes.rename(columns={"stable_id": "gene_stable_id"})
    if not "start" in df_genes.columns:
        starts = []
        stops = []
        if not "strand" in df_genes:
            df_genes = df_genes.assign(strand=1)
        for idx, row in df_genes.iterrows():
            starts.append(min(row["tss"], row["tes"]))
            stops.append(max(row["tss"], row["tes"]))
        df_genes = df_genes.assign(start=starts, stop=stops)
    if not "biotype" in df_genes.columns:
        df_genes = df_genes.assign(biotype="protein_coding")
    df_genes = df_genes.sort_values(["chr", "start"])
    df_genes = df_genes.set_index("gene_stable_id")
    if not df_genes.index.is_unique:
        raise ValueError("gene_stable_ids not unique")
    if df_transcripts is not None and len(df_transcripts):
        if not "transcript_stable_id" in df_transcripts.columns:
            print(df_transcripts.columns)
            df_transcripts = df_transcripts.assign(
                transcript_stable_id=df_transcripts["name"]
            )
        if not "biotype" in df_transcripts.columns:
            df_transcripts = df_transcripts.assign(biotype="protein_coding")
        if "exons" in df_transcripts.columns:
            if len(df_transcripts["exons"].iloc[0]) == 3:
                df_transcripts = df_transcripts.assign(
                    exons=[(x[0], x[1]) for x in df_transcripts["exons"]]
                )
        stops = []
        if not "strand" in df_transcripts:
            df_transcripts = df_transcripts.assign(strand=1)
        if not "tss" in df_transcripts:
            tss = []
            tes = []
            for tr, row in df_transcripts.iterrows():
                if row["strand"] == 1:
                    tss.append(min((x[0] for x in row["exons"])))
                    tes.append(max((x[1] for x in row["exons"])))
                else:
                    tss.append(max((x[1] for x in row["exons"])))
                    tes.append(min((x[0] for x in row["exons"])))
            df_transcripts = df_transcripts.assign(tss=tss, tes=tes)
        if not "start" in df_transcripts:
            starts = []
            stops = []
            for idx, row in df_transcripts.iterrows():
                starts.append(min(row["tss"], row["tes"]))
                stops.append(max(row["tss"], row["tes"]))
            df_transcripts = df_transcripts.assign(start=starts, stop=stops)
        df_transcripts = df_transcripts.set_index("transcript_stable_id")
        if not df_transcripts.index.is_unique:
            raise ValueError("transcript_stable_ids not unique")
    result = HardCodedGenome("dummy", chr_lengths, df_genes, df_transcripts, None)
    result.sanity_check_genes(df_genes)
    if df_transcripts is not None and len(df_transcripts):
        result.sanity_check_transcripts(df_transcripts)
    return result