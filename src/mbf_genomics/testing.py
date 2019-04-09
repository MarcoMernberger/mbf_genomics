import pytest
import pypipegraph as ppg
import sys
from mbf_genomes import HardCodedGenome


def run_pipegraph():
    if ppg.inside_ppg():
        ppg.run_pipegraph()
    else:
        pass



class RaisesDirectOrInsidePipegraph(object):
    """Piece of black magic from the depths of _pytest
    that will check whether a piece of code will raise the
    expected expcition (if outside of ppg), or if it will
    raise the exception when the pipegraph is running

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

        if ppg.inside_ppg():
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
            tes_larger = df_genes['tes'] > df_genes['tss']
            strand = tes_larger.replace({True: 1, False: -1})
            df_genes = df_genes.assign(strand=strand)
        for idx, row in df_genes.iterrows():
            starts.append(min(row["tss"], row["tes"]))
            stops.append(max(row["tss"], row["tes"]))
        df_genes = df_genes.assign(start=starts, stop=stops)
    if not "biotype" in df_genes.columns:
        df_genes = df_genes.assign(biotype="protein_coding")
    if not "name" in df_genes.columns:
            df_genes = df_genes.assign(
                name=df_genes.gene_stable_id
            )
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
        if not "name" in df_transcripts.columns:
            df_transcripts = df_transcripts.assign(
                name=df_transcripts.transcript_stable_id
            )
        if "exons" in df_transcripts.columns:
            if len(df_transcripts["exons"].iloc[0]) == 3:
                df_transcripts = df_transcripts.assign(
                    exons=[(x[0], x[1]) for x in df_transcripts["exons"]]
                )
            df_transcripts = df_transcripts.assign(
                exon_stable_ids=[
                    "exon_%s_%i" % (idx, ii)
                    for (ii, idx) in enumerate(df_transcripts["exons"])
                ]
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
