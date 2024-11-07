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
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
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


class DecompositionRAG(BaseRAG):
    """Advanced RAG implementation with question decomposition."""

    def __init__(
        self,
        rag_config: Optional[RAGConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        num_subquestions: int = 3,
    ):
        super().__init__(rag_config=rag_config, llm_config=llm_config)
        self.num_subquestions = num_subquestions
        self._decomposition_chain = None  # Add this to store the chain components

    def _setup_chain(self, **kwargs) -> None:
        """Initialize the components needed for decomposition RAG"""
        # Initialize the template and other reusable components
        self._decomposition_template = """Here is the question you need to answer:

        \n --- \n {question} \n --- \n

        Here is any available background question + answer pairs:

        \n --- \n {q_a_pairs} \n --- \n

        Here is additional context relevant to the question: 

        \n --- \n {context} \n --- \n

        Use the above context and any background question + answer pairs to answer the question: \n {question}
        """
        self._decomposition_prompt = ChatPromptTemplate.from_template(
            self._decomposition_template
        )
        logger.info("Decomposition RAG chain components initialized")

    def invoke_recursively(self, question: str) -> str:
        """Process the question using decomposition approach"""
        # Generate sub-questions
        template = f"""You are a helpful assistant that generates multiple sub-questions related to an input question.
        
        The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.
        
        Generate multiple search queries related to: {{question}}
        
        Output ({self.num_subquestions} queries):"""

        prompt_decomposition = ChatPromptTemplate.from_template(template)

        query_generator = (
            prompt_decomposition
            | self._llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        questions = query_generator.invoke({"question": question})

        # Process each sub-question
        q_a_pairs = ""
        final_answer = None

        for q in questions:
            rag_chain = (
                {
                    "context": itemgetter("question") | self._retriever,
                    "question": itemgetter("question"),
                    "q_a_pairs": itemgetter("q_a_pairs"),
                }
                | self._decomposition_prompt
                | self._llm
                | StrOutputParser()
            )
            answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
            q_a_pairs += f"\n---\nQuestion: {q}\nAnswer: {answer}\n\n"
            final_answer = answer  # Keep the last answer as final

        return final_answer

    def invoke_individual(self, question: str) -> str:
        """Process the question using decomposition approach with individual answers"""
        # Generate sub-questions
        template = f"""You are a helpful assistant that generates multiple sub-questions related to an input question.
        
        The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.
        
        Generate multiple search queries related to: {{question}}
        
        Output ({self.num_subquestions} queries):"""

        prompt_decomposition = ChatPromptTemplate.from_template(template)

        query_generator = (
            prompt_decomposition
            | self._llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        sub_questions = query_generator.invoke({"question": question})

        # Process each sub-question individually
        answers = []
        for sub_q in sub_questions:
            # Retrieve documents for each sub-question
            retrieved_docs = self._retriever.invoke(sub_q)

            # Create RAG chain for individual sub-question
            rag_chain = (
                {
                    "context": RunnableLambda(lambda x: retrieved_docs),
                    "question": RunnableLambda(lambda x: sub_q),
                }
                | ChatPromptTemplate.from_template(base_rag_prompt)
                | self._llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke({})
            answers.append(answer)

        # Format Q&A pairs
        def format_qa_pairs(questions: List[str], answers: List[str]) -> str:
            """Format questions and answers into a readable string"""
            formatted_string = ""
            for i, (question, answer) in enumerate(zip(questions, answers), start=1):
                formatted_string += (
                    f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
                )
            return formatted_string.strip()

        qa_context = format_qa_pairs(sub_questions, answers)

        # Final synthesis prompt
        synthesis_template = """Here is a set of Q+A pairs that break down the original question:

        {context}

        Using these sub-answers, synthesize a complete response to the original question: {question}
        """

        synthesis_chain = (
            ChatPromptTemplate.from_template(synthesis_template)
            | self._llm
            | StrOutputParser()
        )

        return synthesis_chain.invoke({"context": qa_context, "question": question})

    def invoke(self, question: str, method: str = "recursively") -> str:
        """Invoke the RAG chain with decomposition"""
        if method == "recursively":
            return self.invoke_recursively(question)
        elif method == "individual":
            return self.invoke_individual(question)
        else:
            raise ValueError(
                f"Invalid method: {method}, must be 'recursively' or 'individual'"
            )


class StepBackRAG(BaseRAG):
    """Advanced RAG implementation with step-back"""

    def __init__(
        self,
        rag_config: Optional[RAGConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        super().__init__(rag_config=rag_config, llm_config=llm_config)


class HyDERAG(BaseRAG):
    """Advanced RAG implementation with HyDE"""

    def __init__(
        self,
        rag_config: Optional[RAGConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        super().__init__(rag_config=rag_config, llm_config=llm_config)
