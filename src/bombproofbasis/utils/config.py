"""
Various utilities
"""
import argparse
import configparser
import platform
from typing import List

import torch


def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Parse args from system input

    Args:
        args (List[str]): List of flags and values (e.g. \
            ["--config", "config.ini"])

    Returns:
        argparse.Namespace: The parsed arguments in a Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    return parser.parse_args(args)


def read_config(config_file) -> configparser.ConfigParser:
    """
    Read config file and output readable config.

    Args:
        config_file (Path): The path to config file.

    Raises:
        ValueError: If the config file is not in .ini format.

    Returns:
        configparser.ConfigParser: the config
    """
    # if file is None:
    #     arg_inputs = sys.argv[1:]
    #     print("arg_inputs", arg_inputs)
    #     try:
    #         args = parse_args(arg_inputs)
    #     except SystemExit:
    #         args = None
    #     if args is None:
    #         print("Using default config")

    #         base_folder = os.path.dirname(__file__)
    #         folder = os.path.join(base_folder, "../tests")
    #         print(os.path.join(folder, "config.ini"))
    #         config_file = Path(os.path.join(folder, "config.ini"))
    #     else:
    #         config_file = args.s
    # else:
    #     config_file = file

    if "." in str(config_file) and (
        not str(config_file).rsplit(".", maxsplit=1)[-1] == ("ini")
    ):
        raise ValueError(
            f"Configuration file {config_file} is in the \
                {str(config_file).rsplit('.', maxsplit=1)[-1]} extension, \
                    should be .ini"
        )
    if "." not in str(config_file):
        raise ValueError(
            f"File {config_file} impossible to read, it doesn't have any extension"
        )

    config = configparser.ConfigParser()
    config.read(config_file)
    GPU_NAME = get_GPU_name()
    config["HARDWARE"]["GPU_name"] = GPU_NAME
    config["HARDWARE"]["CPU_name"] = platform.processor()
    return config


def get_GPU_name() -> str:
    """
    Get GPU name from the system

    Returns:
        str: The GPU name or No GPU
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "No GPU"
