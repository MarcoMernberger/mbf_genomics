import pytest
import pypipegraph as ppg
from pathlib import Path
from mbf_genomes.testing import get_Candidatus_carsonella_ruddii_pv
from mbf_genomes import InteractiveFileBasedGenome
from mbf_genomes import HardCodedGenome
from mbf_genomics.testing import RaisesDirectOrInsidePipegraph, MockGenome, run_pipegraph  # noqa: F401

ppg_genome = None


def get_genome(name=None):
    global ppg_genome
    cache_dir = Path(__file__).parent / "run" / "genome_cache"
    if ppg_genome is None:
        old_pipegraph = ppg.util.global_pipegraph
        ppg.new_pipegraph()
        g = get_Candidatus_carsonella_ruddii_pv(
            name, cache_dir=cache_dir  # , ignore_code_changes=True
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


