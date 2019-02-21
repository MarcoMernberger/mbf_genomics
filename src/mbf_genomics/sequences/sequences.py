from collections import OrderedDict
from pathlib import Path

import pypipegraph as ppg
from ..delayeddataframe import DelayedDataFrame
import os
import common


class Sequences(DelayedDataFrame):
    def __init__(self, name, loading_function, dependencies, genome, sheet_name=None):
        """@loading_function must return a df with name and seq column"""
        self.sheet_name = sheet_name
        if sheet_name:
            result_dir = Path("results") / "Sequences" / sheet_name / name
        else:
            result_dir = Path("results") / "Sequences" / name
        super().__init__(name, loading_function, dependencies, result_dir)
        self.genome = genome

    def do_load(self):
        if not hasattr(self, "df"):
            self.df = self.loading_function()
            for col in self.get_default_columns():
                if not col in self.df.columns:
                    print(self.df)
                    raise ValueError(
                        "DataFrame return by %s for Sequences %s did not contain %s"
                        % (self.loading_function, self.name, col)
                    )
            # self.df.row_names = self.df['name']
            self.non_annotator_columns = self.df.columns

    def _new_for_filtering(
        self, new_name, load_func, dependencies, result_dir=None, sheet_name=None
    ):
        if result_dir:
            raise ValueError(
                "Sequences filtering doesn't take a result dir right now - you should go and fix that"
            )
        return Sequences(
            new_name, load_func, dependencies, self.genome, sheet_name=sheet_name
        )

    def get_default_columns(self):
        return ("name", "seq")

    def __hash__(self):
        return hash("Sequences" + self.name)

    def __str__(self):
        return "Sequences(%s)" % self.name

    def __repr__(self):
        return "Sequences(%s)" % self.name

    def write_fasta(
        self, output_filename=None, key_mangling_function=None, annotators=None
    ):
        """JOB: Store the sequences of the Sequence in a FASTA file"""
        import fileformats

        if output_filename is None:
            output_filename = self.result_dir / (self.name + ".fasta")
        output_filename.parent.mkdir(exist_ok=True)

        def write(
            self=self,
            output_filename=output_filename,
            key_mangling_function=key_mangling_function,
        ):
            fasta_dict = OrderedDict()
            if key_mangling_function is None:
                for dummy_idx, row in self.df.iterrows():
                    name = row["name"]
                    if "chr" in row:
                        name = name + " %s:%i..%i (%ibp)" % (
                            row["chr"],
                            row["start"],
                            row["stop"],
                            row["stop"] - row["start"],
                        )
                    fasta_dict[name] = row["seq"]
            else:
                for dummy_idx, row in self.df.iterrows():
                    fasta_dict[key_mangling_function(row)] = row["seq"]
            if fasta_dict:
                fileformats.dictToFasta(fasta_dict, output_filename)
            else:
                # write an empty file so the pipegraph does not complain
                op = open(output_filename, "wb")
                op.write("\n" * 5)
                op.close()

        job = (
            ppg.FileGeneratingJob(output_filename, write)
            .depends_on(self.load())
            .depends_on(
                ppg.FunctionInvariant(
                    output_filename + "_key_mangler", key_mangling_function
                )
            )
        )
        if annotators:
            for anno in annotators:
                job.depends_on(self.add_annotator(anno))
        return job

    def plot_motifs(self, motif_source, background_sequences):
        """For each motif in a motif source, plot it's logo and ROC curve"""
        output_directory = self.result_dir / motif_source.name
        if background_sequences is not None:
            output_directory.with_name(
                output_directory.name + "_vs_%s" % background_sequences.name
            )

        def generate_plotjobs():
            foreground_seqs = self.df["seq"]
            if background_sequences is not None:
                background_seqs = background_sequences.df["seq"]
            else:
                background_seqs = []
            for motif in motif_source:
                (output_directory, motif.name).mkdir(exist_ok=True)
                motif.plot_logo(output_directory / motif.name / "logo.pdf")
                motif.plot_logo(
                    output_directory / motif.name / "logo_reverse.pdf", reverse=True
                )
                if len(background_seqs):
                    motif.plot_roc(
                        foreground_seqs,
                        background_seqs,
                        output_directory / motif.name / "roc.png",
                    )
                else:
                    print("skipping roc for", motif.name)
                motif.plot_support(self, output_directory / motif.name / "support.png")

        job = ppg.JobGeneratingJob(output_directory, generate_plotjobs)
        job.depends_on(motif_source.get_dependencies())
        job.depends_on(self.load())
        if background_sequences:
            job.depends_on(background_sequences.load())
        return job

    def tomtom_motifs(self, query_motif_source, reference_motif_source):
        output_directory = (
            self.result_dir / query_motif_source.name / reference_motif_source.name
        )

        def write_summary(summary_filename, plot_files, query_motif_source):
            op = open(summary_filename, "wb")
            op.write("<html><head></head><body>")
            op.write("<h1>Tomtom output overview for %s</h1>" % query_motif_source.name)
            op.write("<table><tr><th>Motif no</th><th>Logo</th><th>Details</th></tr>")
            for ii, motif in enumerate(query_motif_source):
                op.write("<tr><td>%i</td>" % ii)
                op.write("<td ><img src='%s'></img></td>" % plot_files[motif.name])
                op.write(
                    "<td ><img src='%s' height='200'></img></td>"
                    % (Path(motif.name) / "support.tiny.png")
                )
                op.write(
                    "<td><a href='%s'>TomTom</a>"
                    % Path(motif.name)
                    / "tomtom_out"
                    / "index.html"
                )

                op.write("</td>")
                op.write("</tr>\n")
            op.write("</table>")
            op.write("</body></html>")

        def generate_tomtomjobs(output_directory=output_directory):
            tt_jobs = []
            plot_jobs = []
            plot_files = {}
            for motif in query_motif_source:
                target_dir = output_directory / motif.name
                target_dir.mkdir(exist_ok=True)
                tt_jobs.append(motif.tomtom(reference_motif_source, target_dir))
                plot_files[motif.name] = Path(motif.name) / "logo.pdf"
                plot_files[motif.name] = Path(motif.name) / "logo.png"
                plot_jobs.append(
                    motif.plot_logo(target_dir / "logo.pdf", width=2, height=0.5)
                )
                plot_jobs.append(
                    motif.plot_logo(target_dir / "logo.png", width=2, height=0.5)
                )
                plot_jobs.append(
                    motif.plot_support(self, target_dir / "support.png")
                )  # , width=2, height=0.5))
                motif.write_matrix(target_dir / "logo.txt")
            summary_filename = output_directory / "index.html"
            output_directory.mkdir(exist_ok=True)
            fg = (
                ppg.FileGeneratingJob(
                    summary_filename,
                    lambda: write_summary(
                        summary_filename, plot_files, query_motif_source
                    ),
                )
                .depends_on(tt_jobs)
                .depends_on(plot_jobs)
                .depends_on(
                    ppg.FunctionInvariant(
                        output_directory
                        + "genomics.Sequences.tomtom_motifs.write_summary",
                        write_summary,
                    )
                )
            )

        job = ppg.JobGeneratingJob(output_directory + "_tomtom", generate_tomtomjobs)
        job.depends_on(query_motif_source.get_dependencies())
        job.depends_on(reference_motif_source.get_dependencies())
        job.depends_on(self.load())
        return job

    def centrimo_motifs(self, reference_motif_source, local=False, ethresh=False):
        centrimo_input_filename = self.result_dir / "centrimo" / "input.fasta"
        output_directory = self.result_dir / "centrimo" / reference_motif_source.name
        import exptools

        exptools.load_software("meme")
        input_fasta = self.result_dir / "centrimo" / "input.fasta"

        def run():
            lengths = set()
            for seq in self.df["seq"]:
                lengths.add(len(seq))
            if len(lengths) != 1:
                raise ValueError(
                    "Trying to pass uneven sized sequences to centrimo - that's not going to work"
                )
            output_directory.mkdir(exist_ok=True)
            input_filename = centrimo_input_filename
            import meme

            meme.run_centrimo(
                centrimo_input_filename.absolute(),
                reference_motif_source,
                output_directory.absolute(),
                local,
                ethresh,
            )

        return (
            ppg.FileGeneratingJob(output_directory / "centrimo.html", run)
            .depends_on(self.load())
            .depends_on(self.write_fasta(centrimo_input_filename))
            .depends_on(reference_motif_source.load())
        )

    def fimo_motifs(self, reference_motif_source, alpha=1, qthresh=False):
        """
        Scans sequences for individual matches to each of the motifs you provide.
        """
        fimo_input_filename = self.result_dir / "fimo" / "input.fasta"
        output_directory = self.result_dir / "fimo" / reference_motif_source.name
        import exptools

        def run():
            lengths = set()
            for seq in self.df["seq"].values:
                lengths.add(len(seq))
            if len(lengths) != 1:
                raise ValueError(
                    "Trying to pass uneven sized sequences to FIMO - that's not going to work"
                )
            exptools.common.ensure_path(output_directory)
            import meme

            meme.run_FIMO(
                (fimo_input_filename).absolute(),
                reference_motif_source,
                (output_directory).absolute(),
                alpha,
                qthresh,
            )

        return (
            ppg.FileGeneratingJob(output_directory / "fimo.xml", run)
            .depends_on(self.load())
            .depends_on(self.write_fasta(fimo_input_filename))
            .depends_on(reference_motif_source.load())
        )

    def dump_sorted_highlighted_fasta(self, motif_source):
        def gen():
            jobs = []
            for motif in motif_source:
                output_filename = (
                    self.result_dir
                    / motif_source.name
                    / motif.name
                    / ("%s_sorted.fasta" % motif.name)
                )
                output_filename.parent.mkdir(exist_ok=True)

                def dump(motif=motif, output_filename=output_filename):
                    entries = []
                    for dummy_idx, row in self.df.iterrows():
                        name = row["name"]
                        if "chr" in row:
                            name = name + " %s:%i..%i (%ibp)" % (
                                row["chr"],
                                row["start"],
                                row["stop"],
                                row["stop"] - row["start"],
                            )
                        hits, cum_score, best_score = motif.scan(
                            row["seq"], motif.max_score * 0.5
                        )
                        seq = row["seq"].lower()
                        seq_len = len(seq)
                        if len(hits):
                            hits = hits.sort_values("score", ascending=False)
                            name += " best hit at %i bp, %.2f score" % (
                                hits.get_value(0, "start"),
                                hits.get_value(0, "score"),
                            )
                            for dummy_idx2, h in hits.iterrows():
                                start = int(min(h["start"], h["stop"]))
                                stop = int(max(h["start"], h["stop"]))
                                seq = seq[:start] + seq[start:stop].upper() + seq[stop:]
                        if len(seq) != seq_len:
                            raise ValueError("You messed up the casting to upper case")
                        entries.append((best_score, name, seq))
                    entries.sort(reverse=True)
                    op = open(output_filename, "wb")
                    for (best_score, name, seq) in entries:
                        op.write(
                            b">"
                            + common.to_bytes(name)
                            + b"\n"
                            + common.to_bytes(seq)
                            + b"\n"
                        )
                    op.close()

                jobs.append(ppg.FileGeneratingJob(output_filename, dump))
            return jobs

        return (
            ppg.JobGeneratingJob(
                self.name + motif_source.name + "_generated_sorted_fasta", gen
            )
            .depends_on(motif_source.get_dependencies())
            .depends_on(self.load())
        )
