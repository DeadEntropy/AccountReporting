from bkanalysis.transforms import master_transform
from bkanalysis.process import process
import tests.unit.config_helper as ch
import unittest
import os


class TestSum(unittest.TestCase):

    def test_process(self):
        mt = master_transform.Loader(ch.get_config())
        df_raw = mt.load_all()

        pr = process.Process(ch.get_config())
        df = pr.process(df_raw)
        self.assertTrue(len(df) > 0, 'empty DataFrame loaded')


if __name__ == '__main__':
    unittest.main()
