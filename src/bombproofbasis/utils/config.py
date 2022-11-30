"""
Various utilities
"""
import argparse
import configparser
import os
import platform
from pathlib import Path

import torch


def read_config(file: Path = None) -> configparser.ConfigParser:
    """
    Read config file, if None is provided uses the default config file.

    Args:
        file (Path, optional): The path to config file. Defaults to None.

    Raises:
        ValueError: If the config file is not in .ini format.
        OSError: _description_

    Returns:
        configparser.ConfigParser: _description_
    """
    if file is None:
        parse = argparse.ArgumentParser()
        parse.add_argument("-s")
        args = parse.parse_args()

        if args is None:
            print("Using default config")
            print(os.listdir())
            folder = os.path.dirname(__file__)
            config_file = Path(os.path.join(folder, "config.ini"))
        else:
            config_file = args.s
    else:
        config_file = file

    if not str(config_file).rsplit(".", maxsplit=1)[-1] == (".ini"):
        raise ValueError(
            f"Configuration file {config_file} is in the \
                {str(config_file).rsplit('.', maxsplit=1)[-1]} extension, \
                    should be .ini"
        )
    try:
        config = configparser.ConfigParser()
        config.read(config_file)
    except OSError as exc:
        raise OSError(f"Config file {config_file} is impossible to read") from exc
    if torch.cuda.is_available():
        GPU_NAME = torch.cuda.get_device_name(0)
    else:
        GPU_NAME = "No GPU"
    config["HARDWARE"]["GPU_name"] = GPU_NAME
    config["HARDWARE"]["CPU_name"] = platform.processor()
    return config


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-s")
    args = parse.parse_args()
    config = read_config(args.s)
    print(config.sections())
