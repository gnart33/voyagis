import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from voyagis.core.rag.base import BaseRAG
from voyagis.core.rag.logging import setup_logging


def main():
    working_dir_path = Path(__file__).parent.parent
    setup_logging(log_file_path=working_dir_path / "scripts/logs/base_rag.log")

    rag = BaseRAG()
    rag.setup(resource_path=Path("examples/lilianweng-agent"))

    rag.invoke("What is Task Decomposition?")


if __name__ == "__main__":
    main()
