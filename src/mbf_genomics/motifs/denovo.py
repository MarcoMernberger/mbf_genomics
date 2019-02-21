"""Interfaces for denovo motif search algorithms,
to use with the DeNovoMotifSource"""

import exptools
import subprocess
import time
import shutil
import os
import pypipegraph as ppg
import math
import threading
import pandas as pd

class Command(object):
    '''
    Enables to run subprocess commands in a different thread
    with TIMEOUT option!

    Based on jcollado's solution:
    http://stackoverflow.com/questions/1191374/subprocess-with-timeout/4825933#4825933
    '''
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None
        self.stdout = None
        self.stderr = None

    def run(self, timeout=None, **kwargs):
        def target(**kwargs):
            self.process = subprocess.Popen(self.cmd, **kwargs)
            self.stdout, self.stderr = self.process.communicate()

        thread = threading.Thread(target=target, kwargs=kwargs)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            #first, ask nicely...
            p = subprocess.Popen(['pkill', '-TERM', '-P', str(self.process.pid)])
            p.communicate()
            time.sleep(10)
            #and now I expect you to die!
            try:
                self.process.kill()
            except OSError: #though if you were already dead ;)
                pass
            thread.join()
            self.returncode = self.process.returncode
        else:
            self.returncode = 143
        return self.stdout, self.stderr



class Meme:
    """De-Novo motif discovery with Meme (see L{MotifSearcher_Source})"""
    name = 'Meme'

    def __init__(self, nmotifs = 5, minw=6, maxw=24, maxsize=None):
        #self.meme_path = meme_path
        self.nmotifs = nmotifs
        self.meme_cmd = 'bin/meme'
        self.needs_all_cores = False
        self.minw = minw #minimum motif length
        self.maxw = maxw #maximum motif length
        self.maxsize = maxsize
        self.additional_enviroment = {}
        exptools.load_software('meme')

    def get_dependencies(self):
        import meme
        return ppg.ParameterInvariant('meme_version',meme.get_version())

    def get_parameters(self):
        return self.minw, self.maxw, self.nmotifs

    def run(self, foreground_fasta, background_fasta, cache_dir):
        import fileformats
        import meme
        cache_dir = os.path.join(cache_dir)
        try:
            exptools.common.ensure_path(cache_dir) #after all another meme run might have removed this before hand...
            try:
                os.unlink(os.path.join(cache_dir, 'meme.xml'))# make sure we have no old output lying around
            except OSError:
                pass
            if background_fasta:
                background_markov_filename = os.path.abspath(os.path.join(cache_dir, 'background.meme'))
                self.prepareMemeBackground(background_fasta, background_markov_filename)
            self.input_size = sum(len(x) for x in fileformats.fastaToDict(foreground_fasta).values())
            if self.maxsize is None:
                max_size = math.ceil(self.input_size / 100000.0) * 100000
                max_size = max(max_size, 100000)
            else:
                max_size = self.maxsize
            meme_cmd = os.path.join(meme.meme_path, 'meme')
            cmd = [
                    meme_cmd,
                        os.path.abspath(foreground_fasta),
                        '-oc', os.path.abspath(cache_dir),
                        '-dna',
                        #'-text',
                        '-mod', 'zoops', #one motif per sequence
                        '-nmotifs', str(self.nmotifs),
                        '-minw', '%i' % self.minw,
                        '-maxw', '%i' % self.maxw,
                           '-maxsize', '%i' % max_size,
                        '-revcomp',
                        #'-p', '%i' % ppg.util.CPUs() #can't get this to compile right now...
                        #'-time', "%s" % (20000 * self.nmotifs) # 20k secones per motif - if more, abort the run.
                        ]
            if background_fasta:
                cmd.extend([
                        '-bfile', background_markov_filename,
                        ])

            print(" ".join(cmd))
            print('cwd', meme.meme_path)
            environ = dict(os.environ)
            environ.update(self.additional_enviroment)
            print(self.additional_enviroment)
            print(environ)
            p = Command(cmd)
            stdout, stderr = p.run(timeout=20000 * self.nmotifs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=meme.meme_path, env=environ)
            print(stdout)
            print(stderr)
            #stdout, stderr = p.communicate()
            if p.returncode != 0 or 'FATAL' in stderr: #log output
                op = open(os.path.join(cache_dir, 'stdout'), 'wb')
                op.write(str(stdout))
                op.close()
                op = open(os.path.join(cache_dir, 'stderr'), 'wb')
                op.write(str(stderr))
                op.close()
                if 'FATAL' in stderr:
                    fatal = stderr[stderr.find("FATAL"):]
                    fatal = fatal[:fatal.find("\n")]
                    raise ValueError("Meme printed FATAL on stderr: %s" % fatal)
                else:
                    if p.returncode != 143: #which seems what timelimit returns...
                        print(stdout, stderr)
                        raise ValueError("Meme did not return 0: %i - stderr in %s" % (p.returncode,os.path.join(cache_dir, 'stderr')))
            if os.path.exists(os.path.join(cache_dir, 'meme.xml')): #meme output something
                op = open(os.path.join(cache_dir, 'meme.xml'),'rb')
                xml = op.read()
                op.close()
            else:
                xml = "No result"
            return xml
        finally:
            #try:
                ##shutil.rmtree(cache_dir)
                #pass
            #except OSError:
                #pass
            pass

    def parse(self, xmlstring):
        """Parse to a list of [ [binding_site1_sequence, binding_site_2_sequence, ...], ... ]"""
        import xml.etree.ElementTree
        if xmlstring == 'No result' or xmlstring == self.get_empty_xml():
            results = {'sequence': [], 'strand': [], 'start': [], 'end': [], 'motif_no': []}
        else:
            # print(xmlstring)
            tree = xml.etree.ElementTree.XML(xmlstring)
            meme = tree
            motifs = meme.findall('motifs')[0]
            results = []
            for motif_no, aMotif in enumerate(motifs.findall('motif')):
                sites = aMotif.findall('contributing_sites')[0]
                for anInstance in sites.findall('contributing_site'):
                    seq = ""
                    for letter in anInstance.findall('site')[0].findall('letter_ref'):
                        seq += letter.get('letter_id')[-1]
                    geneName = anInstance.get('sequence_id')
                    #chr = geneName[:geneName.find(':')]
                    begin = int(anInstance.get("position"))
                    end = len(seq) + begin
                    #begin = int(geneName[geneName.find(':') + 1: geneName.find('.')]) + begin
                    #end = int(geneName[geneName.rfind('.') + 1:] ) + end
                    strand = anInstance.get('strand')
                    if strand == 'plus':
                        strand = 1
                    else:
                        strand = -1
                    results.append({
                        'sequence': seq,
                        'strand': strand,
                    #    'chr': chr,
                        'start': begin,
                        'end': end,
                        'motif_no': motif_no
                    })
        return pd.DataFrame(results)

    def prepareMemeBackground(self, backgroundFasta,memeFrequencies):
        import meme
        meme_path = meme.meme_path
        op_out = open(memeFrequencies, 'wb')
        op_out.close()
        for ii in range(2, 3): #  newer fasta-get-markov output the lower models each time as well, so we only need to do the upper ones
            cmd = [os.path.join(meme_path, 'fasta-get-markov'), '-m', str(ii)]
            op_in = open(backgroundFasta, 'rb')
            op_out = open(memeFrequencies, 'ab')
            p = subprocess.Popen(cmd, stdin=op_in, stdout=op_out, stderr=subprocess.PIPE, cwd=meme_path)
            stdout, stderr = p.communicate()
            if not p.returncode == 0:
                raise ValueError("wrong return code from fasta-get-markov: %i %s, input file: %s" % (p.returncode, cmd, backgroundFasta))
            op_in.close()
            op_out.close()

    def get_empty_xml(self):
        return """
<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<!-- Document definition -->
<!DOCTYPE MEME[
<!ELEMENT MEME (
  training_set,
  model,
  motifs,
  scanned_sites_summary?
)>
<!ATTLIST MEME
  version CDATA #REQUIRED
  release CDATA #REQUIRED
>
<!-- Training-set elements -->
<!ELEMENT training_set (alphabet, ambigs, sequence+, letter_frequencies)>
<!ATTLIST training_set datafile CDATA #REQUIRED length CDATA #REQUIRED>
<!ELEMENT alphabet (letter+)>
<!ATTLIST alphabet id (amino-acid|nucleotide) #REQUIRED
                   length CDATA #REQUIRED>
<!ELEMENT ambigs (letter+)>
<!ELEMENT letter EMPTY>
<!ATTLIST letter id ID #REQUIRED>
<!ATTLIST letter symbol CDATA #REQUIRED>
<!ELEMENT sequence EMPTY>
<!ATTLIST sequence id ID #REQUIRED
                   name CDATA #REQUIRED
                   length CDATA #REQUIRED
                   weight CDATA #REQUIRED
>
<!ELEMENT letter_frequencies (alphabet_array)>

<!-- Model elements -->
<!ELEMENT model (
  command_line,
  host,
  type,
  nmotifs,
  evalue_threshold,
  object_function,
  min_width,
  max_width,
  minic,
  wg,
  ws,
  endgaps,
  minsites,
  maxsites,
  wnsites,
  prob,
  spmap,
  spfuzz,
  prior,
  beta,
  maxiter,
  distance,
  num_sequences,
  num_positions,
  seed,
  seqfrac,
  strands,
  priors_file,
  reason_for_stopping,
  background_frequencies
)>
<!ELEMENT command_line (#PCDATA)*>
<!ELEMENT host (#PCDATA)*>
<!ELEMENT type (#PCDATA)*>
<!ELEMENT nmotifs (#PCDATA)*>
<!ELEMENT evalue_threshold (#PCDATA)*>
<!ELEMENT object_function (#PCDATA)*>
<!ELEMENT min_width (#PCDATA)*>
<!ELEMENT max_width (#PCDATA)*>
<!ELEMENT minic (#PCDATA)*>
<!ELEMENT wg (#PCDATA)*>
<!ELEMENT ws (#PCDATA)*>
<!ELEMENT endgaps (#PCDATA)*>
<!ELEMENT minsites (#PCDATA)*>
<!ELEMENT maxsites (#PCDATA)*>
<!ELEMENT wnsites (#PCDATA)*>
<!ELEMENT prob (#PCDATA)*>
<!ELEMENT spmap (#PCDATA)*>
<!ELEMENT spfuzz (#PCDATA)*>
<!ELEMENT prior (#PCDATA)*>
<!ELEMENT beta (#PCDATA)*>
<!ELEMENT maxiter (#PCDATA)*>
<!ELEMENT distance (#PCDATA)*>
<!ELEMENT num_sequences (#PCDATA)*>
<!ELEMENT num_positions (#PCDATA)*>
<!ELEMENT seed (#PCDATA)*>
<!ELEMENT seqfrac (#PCDATA)*>
<!ELEMENT strands (#PCDATA)*>
<!ELEMENT priors_file (#PCDATA)*>
<!ELEMENT reason_for_stopping (#PCDATA)*>
<!ELEMENT background_frequencies (alphabet_array)>
<!ATTLIST background_frequencies source CDATA #REQUIRED>

<!-- Motif elements -->
<!ELEMENT motifs (motif+)>
<!ELEMENT motif (scores, probabilities, regular_expression?, contributing_sites)>
<!ATTLIST motif id ID #REQUIRED
                name CDATA #REQUIRED
                width CDATA #REQUIRED
                sites CDATA #REQUIRED
                llr CDATA #REQUIRED
                ic CDATA #REQUIRED
                re CDATA #REQUIRED
                bayes_threshold CDATA #REQUIRED
                e_value CDATA #REQUIRED
                elapsed_time CDATA #REQUIRED
                url CDATA ""
>
<!ELEMENT scores (alphabet_matrix)>
<!ELEMENT probabilities (alphabet_matrix)>
<!ELEMENT regular_expression (#PCDATA)*>

<!-- Contributing site elements -->
<!-- Contributing sites are motif occurences found during the motif discovery phase -->
<!ELEMENT contributing_sites (contributing_site+)>
<!ELEMENT contributing_site (left_flank, site, right_flank)>
<!ATTLIST contributing_site sequence_id IDREF #REQUIRED
                          position CDATA #REQUIRED
                          strand (plus|minus|none) 'none'
                          pvalue CDATA #REQUIRED
>
<!-- The left_flank contains the sequence for 10 bases to the left of the motif start -->
<!ELEMENT left_flank (#PCDATA)>
<!-- The site contains the sequence for the motif instance -->
<!ELEMENT site (letter_ref+)>
<!-- The right_flank contains the sequence for 10 bases to the right of the motif end -->
<!ELEMENT right_flank (#PCDATA)>

<!-- Scanned site elements -->
<!-- Scanned sites are motif occurences found during the sequence scan phase -->
<!ELEMENT scanned_sites_summary (scanned_sites+)>
<!ATTLIST scanned_sites_summary p_thresh CDATA #REQUIRED>
<!ELEMENT scanned_sites (scanned_site*)>
<!ATTLIST scanned_sites sequence_id IDREF #REQUIRED
                        pvalue CDATA #REQUIRED
                        num_sites CDATA #REQUIRED>
<!ELEMENT scanned_site EMPTY>
<!ATTLIST scanned_site  motif_id IDREF #REQUIRED
                        strand (plus|minus|none) 'none'
                        position CDATA #REQUIRED
                        pvalue CDATA #REQUIRED>

<!-- Utility elements -->
<!-- A reference to a letter in the alphabet -->
<!ELEMENT letter_ref EMPTY>
<!ATTLIST letter_ref letter_id IDREF #REQUIRED>
<!-- A alphabet-array contains one floating point value for each letter in an alphabet -->
<!ELEMENT alphabet_array (value+)>
<!ELEMENT value (#PCDATA)>
<!ATTLIST value letter_id IDREF #REQUIRED>

<!-- A alphabet_matrix contains one alphabet_array for each position in a motif -->
<!ELEMENT alphabet_matrix (alphabet_array+)>

]>
<!-- Begin document body -->
<MEME version="4.8.1" release="Tue Feb  7 14:03:40 EST 2012">
<training_set datafile="empty_sample" length="135">
<alphabet id="nucleotide" length="4">
<letter id="letter_A" symbol="A"/>
<letter id="letter_C" symbol="C"/>
<letter id="letter_G" symbol="G"/>
<letter id="letter_T" symbol="T"/>
</alphabet>
<ambigs>
<letter id="letter_B" symbol="B"/>
<letter id="letter_D" symbol="D"/>
<letter id="letter_H" symbol="H"/>
<letter id="letter_K" symbol="K"/>
<letter id="letter_M" symbol="M"/>
<letter id="letter_N" symbol="N"/>
<letter id="letter_R" symbol="R"/>
<letter id="letter_S" symbol="S"/>
<letter id="letter_U" symbol="U"/>
<letter id="letter_V" symbol="V"/>
<letter id="letter_W" symbol="W"/>
<letter id="letter_Y" symbol="Y"/>
<letter id="letter_star" symbol="*"/>
<letter id="letter_-" symbol="-"/>
<letter id="letter_X" symbol="X"/>
</ambigs>
<letter_frequencies>
<alphabet_array>
<value letter_id="letter_A">0.287</value>
<value letter_id="letter_C">0.213</value>
<value letter_id="letter_G">0.213</value>
<value letter_id="letter_T">0.287</value>
</alphabet_array>
</letter_frequencies>
</training_set>
<model>
<command_line>meme foreground.fasta -oc /tmp/meme -dna -mod zoops -nmotifs 1 -minw 6 -maxw 24 -maxsize 100000 -revcomp </command_line>
<host>pcmt230</host>
<type>zoops</type>
<nmotifs>1</nmotifs>
<evalue_threshold>inf</evalue_threshold>
<object_function>E-value of product of p-values</object_function>
<min_width>6</min_width>
<max_width>24</max_width>
<minic>    0.00</minic>
<wg>11</wg>
<ws>1</ws>
<endgaps>yes</endgaps>
<minsites>2</minsites>
<maxsites>135</maxsites>
<wnsites>0.8</wnsites>
<prob>1</prob>
<spmap>uni</spmap>
<spfuzz>0.5</spfuzz>
<prior>dirichlet</prior>
<beta>0.01</beta>
<maxiter>50</maxiter>
<distance>1e-05</distance>
<num_sequences>135</num_sequences>
<num_positions>40635</num_positions>
<seed>0</seed>
<seqfrac>       1</seqfrac>
<strands>both</strands>
<priors_file></priors_file>
<reason_for_stopping>Stopped because nmotifs = 1 reached.</reason_for_stopping>
<background_frequencies source="dataset with add-one prior applied">
<alphabet_array>
<value letter_id="letter_A">0.287</value>
<value letter_id="letter_C">0.213</value>
<value letter_id="letter_G">0.213</value>
<value letter_id="letter_T">0.287</value>
</alphabet_array>
</background_frequencies>
</model>
<motifs>
</motifs>
<scanned_sites_summary p_thresh="0.0001">
</scanned_sites_summary>
</MEME>
"""

class DREME:
    """De-novo motif search for chipseq data using DREME (see doi:10.1093/bioinformatics/btr261)"""
    name = "DREME"

    def __init__(self, e_threshold = None, m = None, g = None, s = None):
        """See DREME documentation for description, meme/__init__.run_dreme for defaults"""
        self.params = {'e_threshold': e_threshold, 'm': m, 'g': g, 's': s}
        exptools.load_software('meme')

    def get_dependencies(self):
        import meme
        return ppg.ParameterInvariant('meme_version',meme.get_version())

    def get_parameters(self): #get's turned into a ParameterInvariant
        return self.params

    def run(self, foreground_fasta, background_fasta, cache_dir):
        foreground_fasta = os.path.abspath(foreground_fasta)
        if background_fasta:
            background_fasta = os.path.abspath(background_fasta)
        exptools.common.ensure_path(cache_dir)
        current_dir = os.getcwd()
        os.chdir(cache_dir)
        import meme
        try:
            stdout, stderr = meme.run_dreme(foreground_fasta, background_fasta)
        except ValueError as e:
            if 'No sequences on FASTA format found in this file' in str(e): # ignore if we're empty...
                stderr = ''
                stdout = ''
                pass
            else:
                raise
        os.chdir(current_dir)
        op = open(os.path.join(cache_dir, 'stderr.txt'), 'wb')
        op.write(stderr)
        op.close()
        print(os.listdir(cache_dir))
        op = open(os.path.join(cache_dir, 'dreme_out', 'dreme.txt'))
        dreme_output = op.read()
        op.close()
        return dreme_output

    def parse_to_counts(self, dreme_stdout):
        """Parse to [ [ {A: 5, C: 3...}, {}, ... ], [...] ]"""
        return parse_meme_output_file(dreme_stdout)

def parse_meme_output_file(text):
    motif_blocks = text.strip().split("MOTIF")[1:] #throw away initial nonsense ;)
    for block in motif_blocks:
        block = block[block.find('letter-probability'):]
        lines = block.split("\n")
        print('"' + block + '"')
            #if lines[0].strip():
                #iupac = lines[0].splineslit()[1]
        nsites = lines[0]
        nsites = nsites[nsites.find('nsites=') + len('nsites='):].strip()
        nsites = int(nsites[:nsites.find(' ')])
        if nsites == '0':
            raise ValueError("Nsites was 0")
        probabilities = []
        for line in lines[1:]: #skip 'letter probability matrix line' (& MOTIF iupac line, which is now ignored
            line = line.strip()
            if line == '':
                break
            values = [(float(x) * nsites) for x in line.split()]
            probabilities.append({"A": values[0],'C': values[1], 'G': values[2], 'T': values[3]})
        yield probabilities, True #yes, please renormalize



class MemeChip:
    def __init__(self):
        self.name ='Memechip'
        exptools.load_software('meme')
        pass

    def get_parameters(self): #get's turned into a ParameterInvariant
        return []

    def run(self, foreground_fasta, background_fasta, cache_dir):
        """Run memechip and create outpt files
        exptools.common.ensure_path(cache_dir) #after all another meme run might have removed this before hand...
        """
        import meme
        stdout, stderr = meme.run_meme_chip(foreground_fasta, background_fasta, cache_dir, max_time_in_minutes=120)
        try:
            op = open(os.path.join(cache_dir, 'combined.meme'))
            res = op.read()
            op.close()
        except:
            print('meme output', stdout, stderr)
            raise
        return res

    def get_dependencies(self):
        import meme
        return ppg.ParameterInvariant('meme_version',meme.get_version())


    def parse_to_counts(self, combined_meme):
        """Parse to [ [ {A: 5, C: 3...}, {}, ... ], [...] ]"""
        return parse_meme_output_file(combined_meme)
