import pytest
import pandas as pd
import pypipegraph as ppg
from mbf_genomics import genes, DelayedDataFrame
from mbf_genomics.testing import MockGenome
from pypipegraph.testing import force_load


@pytest.mark.usefixtures("new_pipegraph")
class TestDescription:
    def test_simple(self):
        genome = MockGenome(
            pd.DataFrame(
                {
                    "stable_id": ["a", "b", "c"],
                    "chr": "1",
                    "tss": [0, 100, 1000],
                    "tes": [10, 101, 1010],
                }
            ),
            df_genes_meta=pd.DataFrame(
                {
                    "gene_stable_id": ["a", "b", "c"],
                    "description": ["hello", "world", "!"],
                }
            ).set_index("gene_stable_id"),
        )
        g = genes.Genes(genome)
        anno = genes.annotators.Description()
        g += anno
        force_load(g.annotate())
        ppg.run_pipegraph()
        assert "description" in g.df.columns
        assert (
            g.df.sort_values("gene_stable_id")["description"] == ["hello", "world", "!"]
        ).all()

    def test_external_genome(self):
        genome = MockGenome(
            pd.DataFrame(
                {
                    "stable_id": ["a", "b", "c"],
                    "chr": "1",
                    "tss": [0, 100, 1000],
                    "tes": [10, 101, 1010],
                }
            ),
            df_genes_meta=pd.DataFrame(
                {
                    "gene_stable_id": ["a", "b", "c"],
                    "description": ["hello", "world", "!"],
                }
            ).set_index("gene_stable_id"),
        )
        g = DelayedDataFrame("ex", pd.DataFrame({"gene_stable_id": ["a", "c", "b"]}))
        anno = genes.annotators.Description(genome)
        g += anno
        force_load(g.annotate())
        ppg.run_pipegraph()
        assert "description" in g.df.columns
        assert (
            g.df.sort_values("gene_stable_id")["description"] == ["hello", "world", "!"]
        ).all()

    def test_missing_external_genome(self):
        g = DelayedDataFrame("ex", pd.DataFrame({"gene_stable_id": ["a", "c", "b"]}))
        anno = genes.annotators.Description()
        g += anno
        force_load(g.annotate())
        with pytest.raises(ppg.RuntimeError):
            ppg.run_pipegraph()
        assert "ddf had no .genome and no genome was passed to Description" in str(
            g.anno_jobs[anno.get_cache_name()].lfg.exception
        )
