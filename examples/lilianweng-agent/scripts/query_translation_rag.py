from pathlib import Path
from typing import Optional

from voyagis.core.query_translation import MultiQueryRAG, RAGFusion, DecompositionRAG
from voyagis.core.logging import setup_logging

working_dir_path = Path(__file__).parent.parent


def multi_query_rag():
    setup_logging(
        log_file_path=working_dir_path / "scripts/logs/query_translation_rag.log"
    )
    rag = MultiQueryRAG()
    rag.setup(resource_path=Path("examples/lilianweng-agent"))

    rag.invoke("What is Task Decomposition?")


def rag_fusion():
    log_file_path = working_dir_path / "scripts/logs/rag_fusion.log"
    setup_logging(log_file_path)

    resource_path = Path("examples/lilianweng-agent")
    rag = RAGFusion()
    rag.setup(resource_path)

    rag.invoke("What is Task Decomposition?")


def decomposition_rag():
    # log_file_path = working_dir_path / "scripts/logs/decomposition_rag.log"
    # setup_logging(log_file_path)
    resource_path = Path("examples/lilianweng-agent")

    rag = DecompositionRAG()
    rag.setup(resource_path)

    answer = rag.invoke("What is Task Decomposition?", method="recursively")
    print(answer)


def main():

    # multi_query_rag()
    # rag_fusion()
    decomposition_rag()


if __name__ == "__main__":
    main()
