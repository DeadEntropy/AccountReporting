"""Tests for master_transform.Loader."""
import os
import pytest

from bkanalysis.transforms import master_transform


class TestLoad:
    """Tests for loading and validating data transforms."""

    def test_exists_config(self, config_path):
        """Verify configuration file exists at expected location."""
        assert os.path.exists(config_path), f"Config path is not valid: {os.path.abspath(config_path)}"

    def test_get_config(self, config):
        """Verify configuration was successfully loaded."""
        assert config is not None, "Did not successfully read the config."

    def test_get_config_elt(self, config):
        """Verify IO section exists in configuration."""
        assert "IO" in config, "IO is not in config."

    def test_transform_get_config_elt(self, master_transform_loader):
        """Verify master transform loader has correct config elements."""
        assert "IO" in master_transform_loader.config
        assert "folder_lake" in master_transform_loader.config["IO"]

    def test_transform_get_files(self, master_transform_loader):
        """Verify files are found in lake directory."""
        assert "folder_root" in master_transform_loader.config["IO"]
        files = master_transform_loader.get_files(
            master_transform_loader.config["IO"]["folder_lake"],
            master_transform_loader.config["IO"]["folder_root"]
        )
        assert len(files) > 0, "no files where found"

    def test_transform_load(self, master_transform_loader):
        """Verify data loads successfully."""
        df = master_transform_loader.load_all()
        assert len(df) > 0, "empty DataFrame loaded."

    def test_transform_load_check_nan(self, master_transform_loader):
        """Verify loaded data contains no NaN values."""
        df = master_transform_loader.load_all()
        assert df.isnull().sum().sum() == 0, "Loaded DataFrame contains NaN."
