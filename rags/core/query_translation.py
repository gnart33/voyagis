"""
RAG (Retrieval Augmented Generation) Implementation
This module implements a basic RAG system.
"""

import logging
from typing import List, Optional, Tuple, Any
from pathlib import Path

from langchain.load import dumps, loads
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from rags.core.config import LLMConfig, RAGConfig
from rags.core.prompts import base_rag_prompt
from rags.core.base import BaseRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiQueryRAG(BaseRAG):
    """Advanced RAG implementation with multi-query retrieval."""

    def __init__(
        self,
        rag_config: Optional[RAGConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        num_perspectives: int = 5,
    ):
        super().__init__(rag_config=rag_config, llm_config=llm_config)
        self.num_perspectives = num_perspectives
        self._query_generator = None

    def _get_unique_union(self, documents: List[List[Document]]) -> List[Document]:
        """Get unique union of retrieved documents"""

        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        logger.info(f"Unique documents: {len(unique_docs)} from {len(flattened_docs)}")
        return [loads(doc) for doc in unique_docs]

    def _setup_chain(self) -> None:
        """Initialize the RAG chain with multi-query retrieval"""

        # Initialize query generator
        template = f"""You are an AI language model assistant. Your task is to generate {self.num_perspectives} 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {{question}}"""

        prompt_perspectives = ChatPromptTemplate.from_template(template)

        self._query_generator = (
            prompt_perspectives
            | ChatOpenAI(temperature=0)
            | StrOutputParser()
            | (lambda x: logger.info(f"Generated queries: {x}") or x.split("\n"))
        )
        try:
            # Create multi-query retrieval chain
            retrieval_chain = (
                self._query_generator | self._retriever.map() | self._get_unique_union
            )

            # Create prompt template for RAG
            prompt = ChatPromptTemplate.from_template(base_rag_prompt)

            # Initialize the chain components
            self._chain = (
                {"context": retrieval_chain, "question": RunnablePassthrough()}
                | prompt
                | self._llm
                | StrOutputParser()
            )

            logger.info("Multi-query RAG chain initialized")
        except Exception as e:
            logger.error(f"Error initializing chain: {str(e)}")
            raise e
