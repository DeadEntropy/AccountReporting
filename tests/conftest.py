"""
Pytest configuration and shared fixtures for AccountReporting tests.
"""
import os
import pytest
import configparser
from pathlib import Path

from bkanalysis.transforms import master_transform
from bkanalysis.process import iat_identification


@pytest.fixture(scope="session")
def config_path():
    """Provide the path to the test config file."""
    config_path = os.path.join(Path(__file__).parent, "unit", "config.ini")
    if not os.path.exists(config_path):
        raise Exception(f"No Valid Config Path found: {os.path.abspath(config_path)}")
    return config_path


@pytest.fixture(scope="session")
def config(config_path):
    """Provide a fully configured ConfigParser object for tests."""
    config = configparser.ConfigParser()
    if len(config.read(os.path.abspath(config_path))) != 1:
        raise Exception(f"did not successfully load the config from {config_path}")

    # Set test data paths
    test_data_path = os.path.join(Path(__file__).parent, "test_data")
    config["IO"]["folder_root"] = test_data_path
    config["Mapping"]["folder_root"] = test_data_path
    return config


@pytest.fixture
def master_transform_loader(config):
    """Provide a master_transform.Loader instance for tests."""
    return master_transform.Loader(config)


@pytest.fixture
def iat_identifier(config):
    """Provide an IatIdentification instance for tests."""
    return iat_identification.IatIdentification(config)


@pytest.fixture
def test_data_path():
    """Provide the path to test data directory."""
    return os.path.join(Path(__file__).parent, "test_data")
