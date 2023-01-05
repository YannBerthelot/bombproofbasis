import shutil

from bombproofbasis.types import LoggingConfig
from bombproofbasis.utils.logging import Logger


def test_logging():
    log_config = LoggingConfig(
        logging_output="tensorboard", logging_frequency=1, log_path="./test_logging"
    )
    logger = Logger(log_config)
    for i in range(10):
        logger.log({"i": i, "ix2": i * 2}, i)
    shutil.rmtree("./test_logging", ignore_errors=True)
