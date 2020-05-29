from ..annotator import Annotator
import pandas as pd


class SummitBase(Annotator):
    pass


class SummitMiddle(SummitBase):
    """Place a summit right in the center (ie. a fake summit"""

    columns = ["summit middle"]
    column_properties = {
        columns[0]: {
            "description": "Fake summit, just the center of the region (given relative to start)"
        }
    }

    def calc(self, df):
        res = []
        for dummy_idx, row in df.iterrows():
            res.append((row["stop"] + row["start"]) / 2 - row["start"])
        return pd.Series(res)


# from ..genes.anno_tag_counts import GRUnstrandedRust as TagCount
# from ..genes.anno_tag_counts import GRStrandedRust as TagCountStranded
from ..genes.anno_tag_counts import _NormalizationAnno


class NormalizationCPM(_NormalizationAnno):
    """Normalize to 1e6 by taking the sum of all genes"""

    def __init__(self, base_column_spec):
        self.name = "CPM(lane)"
        self.normalize_to = 1e6
        super().__init__(base_column_spec)
        self.column_properties = {
            self.columns[0]: {"description": "Tag count normalized to lane tag count"}
        }

    def calc(self, df):
        raw_counts = df[self.raw_column]
        total = max(
            1, sum((x.mapped for x in self.raw_anno.aligned_lane.get_bam().get_index_statistics()))
        )
        result = raw_counts * (self.normalize_to / total)
        return pd.Series(result)
