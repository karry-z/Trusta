from trusta.utils import set_seed
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.DEBUG)


def main():
    set_seed(42)
    logger.info("Hello from mmai26!")


if __name__ == "__main__":
    main()
