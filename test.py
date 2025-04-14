import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.info("This is an INFO message.")
logging.debug("This is a DEBUG message.")