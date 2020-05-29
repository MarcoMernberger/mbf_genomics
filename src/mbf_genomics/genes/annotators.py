from mbf_genomics.annotator import Annotator
from mbf_align.raw import Sample
from mbf_genomes import EnsemblGenome
from pathlib import Path
from typing import List
from pypipegraph import Job
import pandas as pd
import numpy as np


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
class FromFile(Annotator):

    def __init__(self, tablepath: Path, columns_to_add: List[str], index_column_table: str = "gene_stable_id", index_column_genes: str = "gene_stable_id", fill_value: float = None):
        """
        Adds arbitrary columns from a table.

        This requires that both the table and the ddf have a common column on
        which we can index.

        Parameters
        ----------
        tablepath : Path
            Path to table with additional columns.
        columns_to_add : List[str]
            List of columns to append.
        index_column_table : str, optional
            Index column in table, by default "gene_stable_id".
        index_column_genes : str, optional
            Index column in ddf to append to, by default "gene_stable_id".
        fill_value : float, optonal
            Value to fill for missing rows, defaults to np.NaN.
        """
        self.tablepath = tablepath
        self.columns = columns_to_add
        self.index_column_table = index_column_table
        self.index_column_genes = index_column_genes
        self.fill = fill_value if fill_value is not None else np.NaN

    def parse(self):
        if (self.tablepath.suffix == ".xls") or (self.tablepath.suffix == ".xlsx"):
            return pd.read_excel(self.tablepath)
        else:
            return pd.read_csv(self.tablepath, sep="\t")

    def get_cache_name(self):
        return f"FromFile_{self.tablepath.name}"

    def calc_ddf(self, ddf):
        """Calculates the ddf to append."""
        df_copy = ddf.df.copy()
        if self.index_column_genes not in df_copy.columns:
            raise ValueError(f"Column {self.index_column_genes} not found in ddf index, found was:\n{[str(x) for x in df_copy.columns]}.")
        df_in = self.parse()
        if self.index_column_table not in df_in.columns:
            raise ValueError(f"Column {self.index_column_table} not found in table, found was:\n{[str(x) for x in df_in.columns]}.")
        for column in self.columns:
            if column not in df_in.columns:
                raise ValueError(f"Column {column} not found in table, found was:\n{[str(x) for x in df_in.columns]}.")
        df_copy.index = df_copy[self.index_column_genes]
        df_in.index = df_in[self.index_column_table]
        df_in = df_in.reindex(df_copy.index, fill_value=self.fill)
        df_in = df_in[self.columns]
        df_in.index = ddf.df.index
        return df_in

    def deps(self, ddf):
        """Return ppg.jobs"""
        return self.dependencies
