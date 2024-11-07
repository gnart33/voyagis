"""
RAG (Retrieval Augmented Generation) Implementation
This module implements a basic RAG system.
"""

import logging
from typing import List, Optional, Tuple
from operator import itemgetter


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
        # self._query_generator = None

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
                {
                    "context": retrieval_chain,
                    "question": RunnablePassthrough(),  # Changed from itemgetter
                }
                | prompt
                | self._llm
                | StrOutputParser()
            )

            logger.info("Multi-query RAG chain initialized")
        except Exception as e:
            logger.error(f"Error initializing chain: {str(e)}")
            raise e


class RAGFusion(BaseRAG):
    """Advanced RAG processor with RAG-Fusion implementation."""

    def __init__(
        self,
        rag_config: Optional[RAGConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        num_queries: int = 4,
        rrf_k: int = 60,
        **kwargs,
    ):
        super().__init__(rag_config=rag_config, llm_config=llm_config, **kwargs)

        self.num_queries = num_queries
        self.rrf_k = rrf_k  # Parameter for reciprocal rank fusion

        # self._query_generator = None

    def _reciprocal_rank_fusion(self, results: List[List[Document]]) -> List[Document]:
        """Implement reciprocal rank fusion scoring"""
        try:
            fused_scores = {}

            for docs in results:
                for rank, doc in enumerate(docs):
                    doc_str = dumps(doc)
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                    fused_scores[doc_str] += 1 / (rank + self.rrf_k)

            # Return just the documents in order of their scores
            reranked_results = [
                loads(doc)
                for doc, _ in sorted(
                    fused_scores.items(), key=lambda x: x[1], reverse=True
                )
            ]

            return reranked_results
        except Exception as e:
            logger.error(f"Error in reciprocal rank fusion: {str(e)}")
            raise

    def initialize_retrieval_system(self, documents: List[Document]) -> None:
        """Initialize retrieval system with RAG-Fusion capability"""
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents, embedding=OpenAIEmbeddings()
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.retriever_k}
            )

            logger.info("RAG-Fusion retrieval system initialized")
        except Exception as e:
            logger.error(f"Error initializing retrieval system: {str(e)}")
            raise

    def _setup_chain(self) -> None:
        """Initialize RAG chain with RAG-Fusion retrieval"""
        # Initialize query generator for RAG-Fusion
        template = f"""You are a helpful assistant that generates multiple search queries based on a single input query.

        Generate multiple search queries related to: {{question}}

        Output ({self.num_queries} queries):"""

        prompt_rag_fusion = ChatPromptTemplate.from_template(template)

        self._query_generator = (
            prompt_rag_fusion
            | ChatOpenAI(temperature=0)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )
        try:

            # Create RAG-Fusion retrieval chain
            retrieval_chain = (
                self._query_generator
                | self._retriever.map()
                | self._reciprocal_rank_fusion
            )

            # Create final RAG chain
            prompt = ChatPromptTemplate.from_template(base_rag_prompt)

            self._chain = (
                {"context": retrieval_chain, "question": RunnablePassthrough()}
                | prompt
                | self._llm
                | StrOutputParser()
            )

            logger.info("RAG-Fusion chain initialized")
        except Exception as e:
            logger.error(f"Error initializing chain: {str(e)}")
            raise
