from genomics.sources import Source, CombinedSource
import exptools
from . import motifs
import pypipegraph as ppg
import itertools
import os


class ManualMotifSource(Source):

    def __init__(self, name, list):
        self.name = name
        self.list = list

    def __iter__(self):
        """Iterate over the available GenomicRegions.
        This is called after all jobs in get_dependencies() have been satisfied"""
        return iter(self.list)

    def get_dependencies(self):
        """Jobs returned here must be completed
        before Source.__iter__ is accessed"""
        return []


__global_matbase = {}
def Matbase(genome):
    if not genome in __global_matbase:
        __global_matbase[genome] = _Matbase(genome)
    return __global_matbase[genome]


class _Matbase(Source):
    """Interface to our MatBase dataset"""

    def __init__(self, genome):
        exptools.load_software('motif_databases')
        import motif_databases
        self.matbase = motif_databases.Matbase()
        self.name = 'Matbase'
        self.genome = genome
        self._pwms = None

    @exptools.common.lazy_member('_len')
    def __len__(self):
        return self.matbase.get_num_factors()

    def __iter__(self):
        self.build_pwms()
        l = self._pwms.values()
        if isinstance(l, list):  # python 2
            return iter(l)
        else:
            return iter(l)

    def build_pwms(self):
        if self._pwms is None:
            self._pwms = {}
            for mb_factor in self.matbase.iter_factors():
                name = 'MB' + mb_factor.get_name()
                counts = mb_factor.get_pwm() #actually, should be called get_counts
                self._pwms[name] = motifs.PWMMotif(name, counts, self.genome, re_normalize = True)

    def __getitem__(self, name):
        self.build_pwms()
        return self._pwms[name]

    def get_dependencies(self):
        return []


class _JasparSource(Source):
    """Interface to the Jaspar dataset.
    This is the super class for sereveral sub-datasets.
    Use the JasparXY classes for your use case.
    """

    def __init__(self, genome):
        self.motifs = None
        self.genome = genome

    def get_dependencies(self):
        return [ppg.FileChecksumInvariant(self.data_file)]

    def __iter__(self):
        """Iterate over the available GenomicRegions.
        This is called after all jobs in get_dependencies() have been satisfied"""
        self.parse()
        return iter(self.motifs.values())

    def __getitem__(self, name):
        self.parse()
        return self.motifs[name]

    def parse(self):
        if self.motifs is None:
            op = open(self.data_file, 'rb')
            data = op.read().decode('utf-8').split(">")[:]
            op.close()
            self.motifs = {}
            for chunk in data:
                if chunk.strip():
                    chunk = chunk.split("\n")
                    name = chunk[0]
                    if '[' in chunk[1]:
                        row_a = [int(x) for x in chunk[1][chunk[1].find('[') + 1:chunk[1].find(']')].strip().split()]
                        row_c = [int(x) for x in chunk[2][chunk[2].find('[') + 1:chunk[2].find(']')].strip().split()]
                        row_g = [int(x) for x in chunk[3][chunk[3].find('[') + 1:chunk[3].find(']')].strip().split()]
                        row_t = [int(x) for x in chunk[4][chunk[4].find('[') + 1:chunk[4].find(']')].strip().split()]
                    else:
                        row_a = [int(x) for x in chunk[1].strip().split()]
                        row_c = [int(x) for x in chunk[2].strip().split()]
                        row_g = [int(x) for x in chunk[3].strip().split()]
                        row_t = [int(x) for x in chunk[4].strip().split()]
                    pwm = []
                    for ii in range(0, len(row_a)):
                        pwm.append({
                            'A': row_a[ii],
                            'C': row_c[ii],
                            'G': row_g[ii],
                            'T': row_t[ii],
                        })
                    self.motifs[name] = motifs.PWMMotif(name, pwm, self.genome, re_normalize=True)


class JasparVertebrates(_JasparSource):
    def __init__(self, genome):
        exptools.load_software('motif_databases')
        self.name = 'JASPAR'
        self.data_file = "code3/motif_databases/jaspar/incoming/20180827_jaspar_core_version_5_pfm_vertebrates.txt"
        _JasparSource.__init__(self, genome)


class JasparInsects(_JasparSource):
    def __init__(self, genome):
        exptools.load_software('motif_databases')
        self.name = 'JASPAR_insects'
        self.data_file = 'code3/motif_databases/jaspar/incoming/20180827_jaspar_core_version_5_pfm_vertebrates.txt'
        _JasparSource.__init__(self, genome)


class UniProbe(Source):
    """interface to the UniProbe dataset"""
    def __init__(self, genome):
        exptools.load_software('motif_databases')
        self.name = 'UniProbe'
        self.genome = genome
        self.data_file = "code3/motif_databases/UniProbe/incoming/20110822_all_pwms.zip"
        self.motifs = None

    def get_dependencies(self):
        return [ppg.FileTimeInvariant(self.data_file)]

    def __iter__(self):
        """Iterate over the available GenomicRegions.
        This is called after all jobs in get_dependencies() have been satisfied"""
        self.parse()
        return iter(self.motifs.values())

    def __getitem__(self, name):
        self.parse()
        return self.motifs[name]

    def parse(self):
        if self.motifs is None:
            self.motifs = {}
            seen = {}
            import zipfile
            zf = zipfile.ZipFile(self.data_file)
            for fn in zf.namelist():
                if fn.find('readme.txt') != -1 or fn.find("_RC.") != -1:
                    continue
                x = zf.open(fn)
                raw = x.read().decode('utf-8')
                x.close()
                if not raw:
                    continue
                lines = raw.strip().split("\n")
                if 'Protein' in lines[0] or 'Gene' in lines[0]:
                    name = lines[0][lines[0].find(":") + 1:]
                    if 'Seed' in name:
                        name = name[:name.find('Seed')].strip()
                    elif 'Enrichment' in name:
                        name = name[:name.find('Enrichment')].strip()
                    else:
                        raise ValueError("Can't parse" % lines[0])
                else:
                    name = fn[fn.rfind('/') + 1:]
                if 'primary' in fn:
                    name += '_primary'
                if 'secondary' in fn:
                    name += '_secondary'
                if name in seen:
                    name = name + '_%i' % seen[name]
                    seen[name] += 1
                else:
                    seen[name] = 1
                if raw.find('Probability matrix') == -1:
                    lines = lines[1:]
                else:
                    block = raw[raw.find('Probability matrix'):]
                    block = block.strip()
                    lines = block.split("\n")[1:]

                by_letter = {}
                for line in lines:
                    if not line:
                        continue
                    letter = line[0].upper()
                    by_letter[letter] = []
                    for number in line.split()[1:]:
                        by_letter[letter].append(float(number))
                ref_len = None
                for col in by_letter.values():
                    if ref_len is None:
                        ref_len = len(col)
                    elif ref_len != len(col):
                        raise ValueError("Error parsing %s" % name)
                        continue
                counts = []
                for a, c, g, t in zip(by_letter['A'], by_letter['C'], by_letter['G'], by_letter['T']):
                    counts.append({"A": a * 10000, 'C': c * 10000, 'G': g * 10000, 'T': t * 10000})  #act as if we had seen 40k sites - the data should be pretty good coming from these protein binding microarrays
                self.motifs[name] = motifs.PWMMotif(name, counts, self.genome, re_normalize = True)



            pass

remove_spaces = lambda row: row['name'].replace(' ', '_')


class DeNovoMotifSource:
    """A MotifSource created from the output of a de-novo motif searcher"""

    def __init__(self, motif_searcher, foreground_sequences, background_sequences):
        """@MotifSearcher is a motifs.denovo.Algorithm instance
        @foreground_sequences a genomics.Sequences object
        @background_sequences is a genomics.Sequences object or None,
        """
        self.motif_searcher = motif_searcher
        self.foreground_sequences = foreground_sequences
        self.background_sequences = background_sequences
        self.genome = foreground_sequences.genome
        self.name = 'Denovo' + motif_searcher.name + '_in_' + foreground_sequences.name
        if self.background_sequences:
            self.name += '_vs_%s' % self.background_sequences.name
        ppg.assert_uniqueness_of_object(self)
        self.cache_dir = os.path.join('cache', 'Denovo_motifsearch', self.name.replace(' ', '_'))
        exptools.common.ensure_path(self.cache_dir)
        self.foreground_fasta_filename = os.path.join(self.cache_dir, 'foreground.fasta')
        if self.background_sequences:
            self.background_fasta_filename = os.path.join(self.cache_dir, 'background.fasta')
        else:
            self.background_fasta_filename = None

    def get_dependencies(self):
        algorithm = self.motif_searcher
        dependcies = algorithm.get_dependencies()
        parameter_dependency = ppg.ParameterInvariant(self.name + algorithm.name + '_params', algorithm.get_parameters())

        def run_algo():
            if len(self.foreground_sequences.df) > 1:
                return algorithm.run(self.foreground_fasta_filename, self.background_fasta_filename, self.cache_dir)
            else:
                return "NO INPUT"

        def store_algo_output(output):
            self.algo_output = output

        def parse_algo_results():
            if self.algo_output == 'NO INPUT':
                return []
            found_motifs = []
            if hasattr(algorithm, 'parse_to_counts'):
                for motif_no, tup in enumerate(algorithm.parse_to_counts(self.algo_output)):
                    count_matrix, re_normalize = tup
                    found_motifs.append(
                            motifs.PWMMotif(algorithm.name + '_' + self.name + '_' + str(motif_no), count_matrix, self.genome, re_normalize = re_normalize))
            else:
                value = algorithm.parse(self.algo_output)
                for motif_no, rows in value.groupby('motif_no'):
                    seqs = []
                    for dummy_idx, row in rows.iterrows():
                        if not ('N' in row['sequence'] or 'X' in row['sequence']):  # filter weird meme output, hits at the beginning etc
                            seqs.append(row['sequence'])
                    found_motifs.append(motifs.PWMMotif_FromSeqs(algorithm.name + '_' + self.name + '_' + str(motif_no), seqs, self.genome))
            return found_motifs

        def store_algo_parsed(found_motifs):
            self.motifs = found_motifs
        run_job = ppg.CachedDataLoadingJob(os.path.join(self.cache_dir, 'alggo_output_' + algorithm.name),
                run_algo, store_algo_output).depends_on(dependcies).depends_on(self.foreground_sequences.load())
        run_job.depends_on(ppg.FunctionInvariant(algorithm.__class__.__name__ + '.run', algorithm.__class__.run))
        run_job.depends_on(self.foreground_sequences.write_fasta(self.foreground_fasta_filename, remove_spaces))  # no spaces in sequence names...
        if self.background_sequences:
            run_job.depends_on(self.background_sequences.write_fasta(self.background_fasta_filename, remove_spaces))  # no spaces in sequence names...
        run_job.depends_on(parameter_dependency)
        if hasattr(algorithm, 'needs_all_cores') and algorithm.needs_all_cores:
            run_algo.needs_all_cores = True
        parse_job = ppg.CachedDataLoadingJob(os.path.join(self.cache_dir, 'algo_results_' + algorithm.name),
                parse_algo_results, store_algo_parsed)
        ppg.Job.depends_on(parse_job.lfg, run_job)
        return [parse_job]

    def __len__(self):
        return len(self.motifs)

    def __iter__(self):
        for motif in self.motifs:
            yield motif

    def __getitem__(self, key):
        if not hasattr(self, 'motifs'):
            raise ValueError("MotifSearcher_Source[] called before it actually was run. Make sure this only happens in jobs that depend on MotifSearcher_Source.get_dependencies()")
        return self.motifs[key]

class MemeFile:
    def __init__(self, name, meme_file, genome):
        self.name = name
        self.meme_file = meme_file
        self.genome = genome
    def do_load(self):
        from . import denovo
        found_motifs = []
        with open(self.meme_file) as op:
            xml = op.read()
            value = denovo.Meme().parse(xml)
            for motif_no, rows in value.groupby('motif_no'):
                seqs = []
                for row in rows:
                    if not ('N' in row['sequence'] or 'X' in row['sequence']):  # filter weird meme output, hits at the beginning etc
                        seqs.append(row['sequence'])
                found_motifs.append(motifs.PWMMotif_FromSeqs(self.name + '_' + str(motif_no), seqs, self.genome))
        self.motifs = found_motifs

    def get_dependencies(self):
        return ppg.DataLoadingJob('MemeFile_%s' % self.name, self.do_load).depends_on(ppg.FileChecksumInvariant(self.meme_file))

    def __len__(self):
        return len(self.motifs)

    def __iter__(self):
        for motif in self.motifs:
            yield motif

    def __getitem__(self, key):
        if not hasattr(self, 'motifs'):
            raise ValueError("MotifSearcher_Source[] called before it actually was run. Make sure this only happens in jobs that depend on MotifSearcher_Source.get_dependencies()")
        return self.motifs[key]

_default_source_cache = {}

def default_sources(genome):
    if not genome in _default_source_cache:
        _default_source_cache[genome] = CombinedSource([
            Matbase(genome),
            JasparVertebrates(genome),
            UniProbe(genome),
            ],
            'Default sources %s' % genome.short_name)
    return _default_source_cache[genome]


class MemeFromMemeFile(MemeFile):
    """
    This allows to create a Meme-Motif-object from a .meme text file, supplied by http://meme-suite.org/meme-software/Databases/motifs/motif_databases.12.17.tgz
    """
    def do_load(self):
        self.motifs = []
        for motif_name, count_matrix in self.parse():
            self.motifs.append(motifs.PWMMotif(motif_name, count_matrix, self.genome, re_normalize=True))

    def parse(self):
        with open(self.meme_file) as op:
            d = op.read()
            blocks = d.split("MOTIF")
            blocks = blocks[1:]  # first one is no motif...
            for b in blocks:
                motif_name = b[:b.find("\n")]                
                p = b[b.find('letter-'):b.find("URL")]
                p = p.split("\n")[1:-1]  # letter und url zeile abschneiden
                counts = []
                for line in p:
                    if line.strip():
                        values = [float(x.strip()) for x in line.strip().split("\t")]
                        counts.append({"A": values[0], 'C': values[1], 'G' : values[2], 'T' : values[3]})
                yield motif_name, counts
