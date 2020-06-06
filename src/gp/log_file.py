import logging
import os


class LogFile:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        logger = logging.getLogger()
        if len(logger.handlers) > 0:
            logger.handlers[0].stream.close()
            logger.removeHandler(logger.handlers[0])

        file_path = os.path.join(file_path, "log.out")
        logging.basicConfig(
            filename=file_path, level="DEBUG", format="%(asctime)s - %(message)s"
        )

    def record(self, val):
        logging.debug(val)
