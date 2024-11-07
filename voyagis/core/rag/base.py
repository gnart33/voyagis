"""
RAG (Retrieval Augmented Generation) Implementation
This module implements a basic RAG system.
"""

import logging
from typing import List, Optional, Tuple, Any
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from voyagis.core.rag.config import LLMConfig, RAGConfig
from voyagis.core.rag.prompts import base_rag_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseRAG:
    """The most basic RAG implementation."""

    def __init__(
        self,
        rag_config: Optional[RAGConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        """Initialize the RAG .

        Args:
            config: RAG configuration object. If None, uses default config.
        """
        self.rag_config = rag_config or RAGConfig()
        self.llm_config = llm_config or LLMConfig()

        self._llm = ChatOpenAI(
            model=self.llm_config.model_name,
            api_key=self.llm_config.api_key,
            temperature=self.llm_config.temperature,
        )

        self._vectorstore: Optional[Chroma] = None
        self._retriever: Optional[Any] = None
        self._chain: Optional[Any] = None

    def setup(self, resource_path: Path) -> None:
        """Setup the RAG with resources.

        Args:
            paths: List of Path objects to process

        Raises:
            Error: If document processing fails or retriever initialization fails
        """
        try:
            documents_path = resource_path / "documents"
            md_files = list(documents_path.glob("**/*.md"))
            logger.info(
                f"Found {len(md_files)} in {documents_path} markdown files, {md_files}"
            )
            documents = self._process_documents(
                md_files,
                self.rag_config.chunk_size,
                self.rag_config.chunk_overlap,
            )
            self._vectorstore, self._retriever = self._initialize_retrieval_system(
                documents, self.rag_config.retriever_k
            )
            self._setup_chain()
            logger.info("RAG  setup completed successfully")
        except Exception as e:
            logger.error(f"Error during setup: {str(e)}")
            raise e

    def _process_documents(
        self, paths: List[Path], chunk_size: int, chunk_overlap: int
    ) -> List[Document]:
        """Process documents and split them into chunks.

        Args:
            paths: List of file paths to process
            chunk_size: Size of each document chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of processed document chunks

        Raises:
            Error: If document processing fails
        """
        try:
            logger.info(f"Processing {len(paths)} files")

            documents = []
            for path in paths:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    documents.append(
                        Document(page_content=text, metadata={"source": str(path)})
                    )

            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            splits = splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} document splits")
            return splits
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise e

    def _initialize_retrieval_system(
        self, documents: List[Document], retriever_k: int = 1
    ) -> Tuple[Chroma, Any]:
        """Initialize the vector store and retriever.

        Args:
            documents: List of documents to index
            retriever_k: Number of documents to retrieve

        Returns:
            Tuple of (vectorstore, retriever)

        Raises:
            RetrieverInitializationError: If initialization fails
        """
        try:
            vectorstore = Chroma.from_documents(
                documents=documents, embedding=OpenAIEmbeddings()
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
            logger.info("Retrieval system initialized")
            return vectorstore, retriever
        except Exception as e:
            logger.error(f"Error initializing retrieval system: {str(e)}")
            raise e

    def _setup_chain(self, **kwargs) -> None:
        """Initialize the RAG chain with a prompt template and LLM.

        The chain combines context retrieval, prompt formatting, and LLM generation.
        """
        # Create prompt template for RAG
        prompt = ChatPromptTemplate.from_template(base_rag_prompt)

        # Initialize the chain components
        self._chain = (
            {"context": self._retriever, "question": RunnablePassthrough()}
            | prompt
            | self._llm
            | StrOutputParser()
        )

        logger.info("RAG chain initialized successfully")

    def invoke(self, question: str) -> str:
        answer = self._chain.invoke(question)
        logger.info(f"Answer: {answer}")
        return answer
