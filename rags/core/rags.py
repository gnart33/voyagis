"""
RAG (Retrieval Augmented Generation) Implementation
This module implements a RAG system for processing web content and generating responses.
"""

import logging
from typing import List, Optional, Tuple

import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseRAGProcessor:
    """Base class for RAG (Retrieval Augmented Generation) processing."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        self.retriever = None
        self.llm = None

    def setup(self, urls: List[str]) -> None:
        """Common setup flow for both processors"""
        try:
            documents = self.process_documents(urls)
            self.initialize_retrieval_system(documents)
            self.initialize_chain()
            logger.info("RAG processor setup completed successfully")
        except Exception as e:
            logger.error(f"Error during setup: {str(e)}")
            raise

    def process_documents(self, urls: List[str]) -> List[Document]:
        """Process documents from URLs - to be implemented by subclasses"""
        raise NotImplementedError

    def initialize_retrieval_system(self, documents: List[Document]) -> None:
        """Initialize retrieval system - to be implemented by subclasses"""
        raise NotImplementedError

    def initialize_chain(self) -> None:
        """Initialize LLM chain - to be implemented by subclasses"""
        raise NotImplementedError


class RAGProcessor5(BaseRAGProcessor):
    """Advanced RAG processor with question decomposition."""

    def __init__(self, retriever_k: int = 1, num_subquestions: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.retriever_k = retriever_k
        self.num_subquestions = num_subquestions
        self.rag_chain = None
        self.query_generator = None
        self.parallel_rag_chain = None  # New chain for parallel processing

    def _format_qa_pair(self, question: str, answer: str) -> str:
        """Format question and answer pair"""
        return f"Question: {question}\nAnswer: {answer}\n\n".strip()

    def initialize_retrieval_system(self, documents: List[Document]) -> None:
        """Initialize retrieval system with decomposition capability"""
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents, embedding=OpenAIEmbeddings()
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.retriever_k}
            )

            # Initialize query generator for decomposition
            template = f"""You are a helpful assistant that generates multiple sub-questions related to an input question.
            
            The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.
            
            Generate multiple search queries related to: {{question}}
            
            Output ({self.num_subquestions} queries):"""

            prompt_decomposition = ChatPromptTemplate.from_template(template)

            self.query_generator = (
                prompt_decomposition
                | ChatOpenAI(temperature=0)
                | StrOutputParser()
                | (lambda x: x.split("\n"))
            )

            logger.info("Decomposition retrieval system initialized")
        except Exception as e:
            logger.error(f"Error initializing retrieval system: {str(e)}")
            raise

    def initialize_chain(self) -> None:
        """Initialize RAG chain with decomposition approach"""
        try:
            self.llm = ChatOpenAI(
                model_name=self.model_name, temperature=self.temperature
            )

            # Create decomposition prompt template
            template = """Here is the question you need to answer:

            \n --- \n {question} \n --- \n

            Here is any available background question + answer pairs:

            \n --- \n {q_a_pairs} \n --- \n

            Here is additional context relevant to the question: 

            \n --- \n {context} \n --- \n

            Use the above context and any background question + answer pairs to answer the question: \n {question}
            """

            decomposition_prompt = ChatPromptTemplate.from_template(template)

            # Create base RAG chain for sub-questions
            sub_rag_chain = (
                {
                    "context": itemgetter("question") | self.retriever,
                    "question": itemgetter("question"),
                    "q_a_pairs": itemgetter("q_a_pairs"),
                }
                | decomposition_prompt
                | self.llm
                | StrOutputParser()
            )

            # Create final chain that processes sub-questions and combines results
            def process_with_decomposition(input_dict):
                question = input_dict["question"]
                sub_questions = self.query_generator.invoke({"question": question})

                q_a_pairs = ""
                for sub_q in sub_questions:
                    answer = sub_rag_chain.invoke(
                        {"question": sub_q, "q_a_pairs": q_a_pairs}
                    )
                    q_a_pair = self._format_qa_pair(sub_q, answer)
                    q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair

                # Final answer using accumulated Q&A pairs
                final_answer = sub_rag_chain.invoke(
                    {"question": question, "q_a_pairs": q_a_pairs}
                )

                return final_answer

            self.rag_chain = process_with_decomposition

            logger.info("Decomposition RAG chain initialized")
        except Exception as e:
            logger.error(f"Error initializing chain: {str(e)}")
            raise

    def initialize_parallel_chain(self) -> None:
        """Initialize RAG chain with parallel sub-question processing"""
        try:
            self.llm = ChatOpenAI(
                model_name=self.model_name, temperature=self.temperature
            )

            # Create base RAG prompt for individual sub-questions
            rag_prompt = hub.pull("rlm/rag-prompt")

            def retrieve_and_rag(input_dict):
                """Process sub-questions in parallel"""
                question = input_dict["question"]
                sub_questions = self.query_generator.invoke({"question": question})

                # Initialize lists for results
                rag_results = []

                # Process each sub-question independently
                for sub_q in sub_questions:
                    # Retrieve documents
                    retrieved_docs = self.retriever.get_relevant_documents(sub_q)

                    # Get answer using RAG
                    answer = (rag_prompt | self.llm | StrOutputParser()).invoke(
                        {"context": retrieved_docs, "question": sub_q}
                    )

                    rag_results.append(answer)

                # Format Q&A pairs
                context = ""
                for i, (sub_q, answer) in enumerate(
                    zip(sub_questions, rag_results), start=1
                ):
                    context += f"Question {i}: {sub_q}\nAnswer {i}: {answer}\n\n"

                # Final synthesis prompt
                synthesis_template = """Here is a set of Q+A pairs:

                {context}

                Use these to synthesize an answer to the question: {question}
                """

                synthesis_prompt = ChatPromptTemplate.from_template(synthesis_template)

                # Get final synthesized answer
                final_answer = (synthesis_prompt | self.llm | StrOutputParser()).invoke(
                    {"context": context.strip(), "question": question}
                )

                return final_answer

            self.parallel_rag_chain = retrieve_and_rag

            logger.info("Parallel decomposition RAG chain initialized")
        except Exception as e:
            logger.error(f"Error initializing parallel chain: {str(e)}")
            raise

    def query(self, question: str, use_parallel: bool = False) -> str:
        """Query the RAG system with option for parallel processing"""
        try:
            if use_parallel:
                if self.parallel_rag_chain is None:
                    self.initialize_parallel_chain()
                return self.parallel_rag_chain({"question": question})
            else:
                return self.rag_chain({"question": question})
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            raise


class RAGProcessor6(BaseRAGProcessor):
    """Advanced RAG processor with step-back prompting."""

    def __init__(self, retriever_k: int = 1, examples: List[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.retriever_k = retriever_k
        self.examples = examples or [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "what can the members of The Police do?",
            },
            {
                "input": "Jan Sindel's was born in what country?",
                "output": "what is Jan Sindel's personal history?",
            },
        ]
        self.step_back_chain = None
        self.rag_chain = None

    def process_documents(self, urls: List[str]) -> List[Document]:
        """Process documents using tiktoken-based splitting"""
        try:
            loader = WebBaseLoader(
                web_paths=urls,
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                ),
            )
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} documents")

            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            splits = splitter.split_documents(docs)
            logger.info(f"Created {len(splits)} document splits")
            return splits
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            raise

    def initialize_retrieval_system(self, documents: List[Document]) -> None:
        """Initialize retrieval system with step-back capability"""
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents, embedding=OpenAIEmbeddings()
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.retriever_k}
            )

            # Initialize step-back query generator
            example_prompt = ChatPromptTemplate.from_messages(
                [
                    ("human", "{input}"),
                    ("ai", "{output}"),
                ]
            )

            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                examples=self.examples,
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
                    ),
                    few_shot_prompt,
                    ("user", "{question}"),
                ]
            )

            self.step_back_chain = (
                prompt | ChatOpenAI(temperature=0) | StrOutputParser()
            )

            logger.info("Step-back retrieval system initialized")
        except Exception as e:
            logger.error(f"Error initializing retrieval system: {str(e)}")
            raise

    def initialize_chain(self) -> None:
        """Initialize RAG chain with step-back prompting"""
        try:
            self.llm = ChatOpenAI(
                model_name=self.model_name, temperature=self.temperature
            )

            # Create response prompt
            response_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

            # {normal_context}
            # {step_back_context}

            # Original Question: {question}
            # Answer:"""

            response_prompt = ChatPromptTemplate.from_template(response_template)

            # Create final chain
            self.rag_chain = (
                {
                    # Retrieve context using the normal question
                    "normal_context": RunnableLambda(lambda x: x["question"])
                    | self.retriever,
                    # Retrieve context using the step-back question
                    "step_back_context": self.step_back_chain | self.retriever,
                    # Pass on the question
                    "question": lambda x: x["question"],
                }
                | response_prompt
                | self.llm
                | StrOutputParser()
            )

            logger.info("Step-back RAG chain initialized")
        except Exception as e:
            logger.error(f"Error initializing chain: {str(e)}")
            raise


class RAGProcessor7(BaseRAGProcessor):
    """HyDE"""

    pass
