from bkanalysis.transforms import master_transform
from bkanalysis.process import process
import tests.unit.config_helper as ch
import unittest
import os


class TestProcess(unittest.TestCase):

    def test_process(self):
        mt = master_transform.Loader(ch.get_config())
        df_raw = mt.load_all()

        pr = process.Process(ch.get_config())
        df = pr.extend(df_raw)
        self.assertTrue(len(df) > 0, 'empty DataFrame loaded')
        self.assertEqual(df.drop('Week', axis=1).isnull().sum().sum(), 0, 'Loaded DataFrame contains NaN.')


if __name__ == '__main__':
    unittest.main()
