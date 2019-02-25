import pytest
import mbf_genomics.genes as genes
if False:

    @pytest.mark.usefixtures("new_pipegraph")
    class TestRegionAnnotationWithGenes:
        def test_anno_next_genes(self):
            genome = DummyGenome(
                pd.DataFrame(
                    [
                        {
                            "stable_id": "fake1",
                            "chr": "1",
                            "strand": 1,
                            "tss": 5000,
                            "tes": 5500,
                            "description": "bla",
                        },
                        {
                            "stable_id": "fake2",
                            "chr": "1",
                            "strand": -1,
                            "tss": 5400,
                            "tes": 4900,
                            "description": "bla",
                        },
                        {
                            "stable_id": "fake3",
                            "chr": "2",
                            "strand": -1,
                            "tss": 5400,
                            "tes": 4900,
                            "description": "bla",
                        },
                    ]
                )
            )

            def sample_data():
                df = pd.DataFrame(
                    {
                        "chr": ["1", "2", "1", "3", "5"],
                        "start": [10, 100, 6000, 10000, 100000],
                        "stop": [12, 110, 6110, 11110, 111110],
                    }
                )
                df = df.assign(summit=(df["stop"] - df["start"]) / 2)
                return df

            a = regions.GenomicRegions("shu", sample_data, [], genome)
            anno = regions.annotators.NextGenes(still_ok=True)
            a.add_annotator(anno)
            a.load()
            ppg.run_pipegraph()

            assert (
                a.df["Primary gene stable_id"] == ["fake1", "fake2", "fake3", "", ""]
            ).all()
            should = [
                -1.0 * (5000 - (11)),
                -1.0 * (6055 - 5400),
                -1.0 * (105 - 5400),
                numpy.nan,
                numpy.nan,
            ]
            assert (
                (a.df["Primary gene distance"] == should)
                | numpy.isnan(a.df["Primary gene distance"])
            ).all()


    def test_anno_next_genes(self):
        genome = DummyGenome(
            pd.DataFrame(
                [
                    {
                        "stable_id": "fake1",
                        "chr": "1",
                        "strand": 1,
                        "tss": 5000,
                        "tes": 5500,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake2",
                        "chr": "1",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                    {
                        "stable_id": "fake3",
                        "chr": "2",
                        "strand": -1,
                        "tss": 5400,
                        "tes": 4900,
                        "description": "bla",
                    },
                ]
            )
        )

        def sample_data():
            df = pd.DataFrame(
                {
                    "chr": ["1", "2", "1", "3", "5"],
                    "start": [10, 100, 6000, 10000, 100000],
                    "stop": [12, 110, 6110, 11110, 111110],
                }
            )
            df = df.assign(summit=(df["stop"] - df["start"]) / 2)
            return df

        a = regions.GenomicRegions("shu", sample_data, [], genome)
        anno = regions.annotators.NextGenes(still_ok=True)
        a.add_annotator(anno)
        a.load()
        ppg.run_pipegraph()

        assert (a.df["Primary gene stable_id"] == ["fake1", "fake2", "fake3", "", ""]).all()
        should = [
            -1.0 * (5000 - (11)),
            -1.0 * (6055 - 5400),
            -1.0 * (105 - 5400),
            numpy.nan,
            numpy.nan,
        ]
        assert (
            (a.df["Primary gene distance"] == should)
            | numpy.isnan(a.df["Primary gene distance"])
        ).all()
