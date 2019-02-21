from ..annotator import Annotator
import pandas as pd

class SummitBase(Annotator):
    pass


class SummitMiddle(SummitBase):
    """Place a summit right in the center (ie. a fake summit"""
    columns = ['summit middle']
    column_properties = {
            columns[0]: {
                'description': "Fake summit, just the center of the region (given relative to start)"
            }
        }
    
    def calc(self, df):
        res = []
        for dummy_idx, row in df.iterrows():
            res.append((row['stop'] + row['start']) / 2 - row['start'])
        return pd.Series(res)


