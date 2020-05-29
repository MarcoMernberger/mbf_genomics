from mbf_genomics.annotator import Annotator
from mbf_align.raw import Sample
from mbf_genomes import EnsemblGenome
from pypipegraph import Job
import pandas as pd


class Description(Annotator):
    """Add the description for the genes from genome.

    @genome may be None (default), then the ddf is queried for a '.genome'
    Requires a genome with df_genes_meta - e.g. EnsemblGenomes
    """

    columns = ["description"]

    def __init__(self, genome=None):
        self.genome = genome

    def calc_ddf(self, ddf):
        if self.genome is None:
            try:
                genome = ddf.genome
            except AttributeError:
                raise AttributeError(
                    "ddf had no .genome and no genome was passed to Description"
                )
        else:
            genome = self.genome
        lookup = dict(genome.df_genes_meta["description"].items())
        result = []
        for gene_stable_id in ddf.df["gene_stable_id"]:
            result.append(lookup.get(gene_stable_id, ""))
        return pd.Series(result, index=ddf.df.index)


class GeneStrandedSalmon(Annotator):
    """
    GeneStrandedSalmon is an Annotator to add Salmon counts to a ddf.

    Adds two columns to the ddf containing the raw counts and TPMs
    as privided by Salmon per lane.

    Parameters
    ----------
    lane : Sample
        Raw lane on which the Salmon call was performed.
    salmon_job : Job
        A job that generates the Salmon quant output from which the counts
        are derived.
    """
    def __init__(self, lane: Sample, salmon_job: Job):
        self.lane = lane
        self.name = f"Raw count estimate {lane.name} (Salmon)"
        self.salmon_output = f"results/Salmon/quant/{lane.name}/quant.genes.sf"
        self.columns = [f"{lane.name} TPM (Salmon)", f"{lane.name} counts (Salmon)"]
        self.cache_name = f"GeneStrandedSalmon_{lane.name}"
        self.dependencies = [salmon_job]

    def calc(self, df):
        """
        Calculates new columns for the DelayedDataFrame.

        Adds a TPM column and a raw count column as counted by Salmon and
        written to the the salmon output file.

        Parameters
        ----------
        ddf : mbf_genomics.delayeddataframe.DelayedDataFrame
            The ddf to add the columns to.

        Returns
        -------
        DataFrame : pandas.DataFrame
        Dataframe to add to the ddf.
        """
        df_salmon = pd.read_csv(self.salmon_output, sep="\t")
        lookup = dict(zip(df["gene_stable_id"].values, df.index.values))
        salmon_index = []
        for stable_id in df_salmon["Name"]:
            salmon_index.append(lookup[stable_id])
        df_salmon.index = salmon_index
        del df_salmon["Name"]
        del df_salmon["Length"]
        del df_salmon["EffectiveLength"]
        df_salmon = df_salmon.rename(
            columns={"TPM": self.columns[0], "NumReads": self.columns[1]}
        )
        df_salmon = df_salmon.reindex(labels=df.index).fillna(0)
        return df_salmon

    def deps(self, ddf):
        """Return ppg.jobs"""
        return self.dependencies
