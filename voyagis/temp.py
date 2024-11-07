"""
RAG (Retrieval Augmented Generation) Implementation
This module implements a RAG system for processing web content and generating responses.
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

from voyagis.core.config import LLMConfig, RAGConfig
from voyagis.core.prompts import base_rag_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseRAG:
    """The most basic RAG implementation."""

    def __init__(
        self,
        resource_path: Path,
        rag_config: Optional[RAGConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        """Initialize the RAG .

        Args:
            config: RAG configuration object. If None, uses default config.
        """

        self.resource_path: Path = resource_path

        self.rag_config = rag_config or RAGConfig()
        self.llm_config = llm_config or LLMConfig()

        self.vectorstore: Optional[Chroma] = None
        self.retriever: Optional[Any] = None
        self.rag_chain: Optional[Any] = None

    def setup(self) -> None:
        """Setup the RAG with resources.

        Args:
            paths: List of Path objects to process

        Raises:
            Error: If document processing fails or retriever initialization fails
        """
        try:
            documents_path = self.resource_path / "documents"
            md_files = list(documents_path.glob("**/*.md"))
            documents = self._process_documents(
                md_files,
                self.rag_config.chunk_size,
                self.rag_config.chunk_overlap,
            )
            self.vectorstore, self.retriever = self._initialize_retrieval_system(
                documents, self.rag_config.retriever_k
            )
            self.initialize_chain()
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

    def initialize_chain(self) -> None:
        """Initialize the RAG chain with a prompt template and LLM.

        The chain combines context retrieval, prompt formatting, and LLM generation.
        """
        # Create prompt template for RAG
        prompt = ChatPromptTemplate.from_template(base_rag_prompt)

        # Initialize the chain components
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info("RAG chain initialized successfully")
