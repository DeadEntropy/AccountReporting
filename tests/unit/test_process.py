from bkanalysis.transforms import master_transform
import tests.unit.config_helper as ch
import unittest
import os


class TestSum(unittest.TestCase):

    def test_transform_load(self):
        mt = master_transform.Loader(ch.get_config())
        df = mt.load_all()
        self.assertTrue(len(df) > 0, 'empty DataFrame loaded')


if __name__ == '__main__':
    unittest.main()
