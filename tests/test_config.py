import os
from pathlib import Path

import pytest

from bombproofbasis.utils.config import parse_args, read_config

# def test_parse_args():
#     assert (parse_args(["--config", "config.ini"]).config) == "config.ini"


# def test_config():
#     # check if default config file exists
#     folder = os.path.dirname(__file__)
#     config_file = Path(os.path.join(folder, "config.ini"))
#     assert os.path.exists(config_file)
#     assert read_config(config_file) is not None
#     with pytest.raises(ValueError):
#         read_config("nonsense.xlsx")
#     with pytest.raises(ValueError):
#         read_config("nonsense")
