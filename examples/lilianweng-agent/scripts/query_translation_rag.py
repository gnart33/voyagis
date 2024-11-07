from pathlib import Path

from rags.core.query_translation import MultiQueryRAG
from rags.core.logging import setup_logging


def multi_query_rag():
    working_dir_path = Path(__file__).parent.parent
    setup_logging(
        log_file_path=working_dir_path / "scripts/logs/query_translation_rag.log"
    )

    rag = MultiQueryRAG()
    rag.setup(resource_path=Path("examples/lilianweng-agent"))

    rag.invoke("What is Task Decomposition?")


def main():
    multi_query_rag()


if __name__ == "__main__":
    main()
