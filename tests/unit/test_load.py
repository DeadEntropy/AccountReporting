from bkanalysis.transforms import master_transform
import tests.unit.config_helper as ch
import unittest
import os


class TestLoad(unittest.TestCase):

    def test_exists_config(self):
        self.assertTrue(os.path.exists(ch.get_config_path()), f'Config path is not valid: {os.path.abspath(ch.get_config_path())}.')

    def test_get_config(self):
        self.assertIsNotNone(ch.get_config(), 'Did not successfully read the config.')

    def test_get_config_elt(self):
        config = ch.get_config()
        if 'IO' not in config:
            raise Exception('IO is not in config.')

    def test_transform_get_config_elt(self):
        mt = master_transform.Loader(ch.get_config())
        if 'IO' not in mt.config:
            raise Exception('IO is not in config.')
        if 'folder_lake' not in mt.config['IO']:
            raise Exception('IO.folder_lake is not in config.')

    def test_transform_get_files(self):
        mt = master_transform.Loader(ch.get_config())
        if 'folder_root' not in mt.config['IO']:
            raise Exception('IO.folder_root is not in config.')
        files = mt.get_files(mt.config['IO']['folder_lake'], mt.config['IO']['folder_root'])
        self.assertTrue(len(files) > 0, 'no files where found')

    def test_transform_load(self):
        mt = master_transform.Loader(ch.get_config())
        df = mt.load_all()
        self.assertTrue(len(df) > 0, 'empty DataFrame loaded')


if __name__ == '__main__':
    unittest.main()
