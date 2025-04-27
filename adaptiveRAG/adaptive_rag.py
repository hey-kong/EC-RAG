from typing import Optional
import os
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    QueryBundle,
)
from llama_index.core.tools import FunctionTool, ToolMetadata
from llama_index.llms.deepseek import DeepSeek
from llama_index.core.selectors import LLMSingleSelector
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from reranker import local_reranker


def query_prompt(chunk_list, query):
    chunks = "\n\n".join(chunk_list)

    prompt_template = f"""\
{chunks}

Given the above information and not prior knowledge, answer the question: {query}

Respond with a concise answer only, do not output any other words.
"""

    return prompt_template


class AdaptiveRAG:
    def __init__(
        self, 
        index_dir: str,
        similarity_topk: int = 5,
        api_key: Optional[str] = None, 
        model_name: str = "deepseek-chat",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
        
        # Setup LLM and embedding model
        self.llm = DeepSeek(model=model_name, api_key=os.environ.get("DEEPSEEK_API_KEY"))
        Settings.llm = self.llm
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

        self.storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        self.index = load_index_from_storage(self.storage_context)
        self.retriever = self.index.as_retriever(
            similarity_top_k=similarity_topk,
        )
        
        # Setup query engines and router
        self._setup_query_router()
    
    def _setup_query_router(self):
        """Setup query engines and router"""
        
        # 1. Direct QA Tool - No retrieval
        direct_qa_tool = FunctionTool.from_defaults(
            name="direct_qa",
            description="For straightforward questions that don't require external knowledge",
            fn=self._direct_qa
        )
        
        # 2. Single-step retrieval QA Tool
        single_step_qa_tool = FunctionTool.from_defaults(
            name="single_step_qa",
            description="For simple questions that require one retrieval step",
            fn=self._single_step_qa
        )
        
        # 3. Multi-step iterative retrieval QA Tool
        multi_step_qa_tool = FunctionTool.from_defaults(
            name="multi_step_qa",
            description="For complex questions that require iterative retrieval and refinement",
            fn=self._multi_step_qa
        )
        
        # Create router selector        
        selector = LLMSingleSelector.from_defaults(
            llm=self.llm,     # use default prompt template
        )
        
        self.tools = [direct_qa_tool, single_step_qa_tool, multi_step_qa_tool]
        self.choices = [
            ToolMetadata(
                description="For straightforward questions that don't require external knowledge",
                name="direct_qa"
            ),
            ToolMetadata(
                description="For simple questions that require one retrieval step",
                name="single_step_qa"
            ),
            ToolMetadata(
                description="For complex questions that require iterative retrieval and refinement",
                name="multi_step_qa"
            ),
        ]

        self.selector = selector

    
    def _direct_qa(self, query: str) -> str:
        """Answer question directly with LLM without retrieval"""
        prompt = f"Answer the following question based on your knowledge, without external references:\n\nQuestion: {query}\n\nAnswer:"
        response = self.llm.complete(prompt)
        return response.text
    
    def _single_step_qa(self, query: str) -> str:
        """Answer question with single-step retrieval"""
        # retriever_engine = RetrieverQueryEngine.from_args(
        #     retriever=self.retriever,
        #     node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
        # )
        # response = retriever_engine.query(query)

        query_bundle = QueryBundle(query_str=query)
        nodes = self.retriever.retrieve(query_bundle)
        # rerank
        reranked_nodes = local_reranker.rerank_nodes(query, nodes)
        nodes = [node for node, _ in reranked_nodes]
        chunk_list = [node.text for node in nodes]
        prompt = query_prompt(chunk_list, query)
        response = self.llm.complete(prompt)

        return response.text
    
    def _multi_step_qa(self, query: str, max_iterations: int = 3) -> str:
        """Answer complex questions with multi-step iterative retrieval"""
        original_query = query
        current_answer = "I don't have enough information yet."
        retrieved_contexts = []
        retrieved_node_ids = []  # avoid duplicates
        current_query = query
        
        # Create retriever query engine
        # retriever_engine = RetrieverQueryEngine.from_args(
        #     retriever=self.retriever,
        #     node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
        # )
        
        for i in range(max_iterations):
            # Execute current query
            query_bundle = QueryBundle(query_str=current_query)
            nodes = self.retriever.retrieve(query_bundle)
            # rerank
            reranked_nodes = local_reranker.rerank_nodes(query, nodes)
            nodes = [node for node, _ in reranked_nodes]
            chunk_list = [node.text for node in nodes]
            prompt = query_prompt(chunk_list, current_query)
            response = self.llm.complete(prompt)

            # Extract retrieved context, notice avoid duplicates
            for node in nodes:
                if node.node_id not in retrieved_node_ids:
                    retrieved_node_ids.append(node.node_id)
                    retrieved_contexts.append(node.get_text())
            
            # Update current answer
            if i == 0:
                current_answer = response.text
            else:
                # Combine previous answer with new information
                combine_prompt = (
                    f"Original question: {original_query}\n"
                    f"Previous answer: {current_answer}\n"
                    f"New information: {response.text}\n"
                    
                    "Please update the answer to the original question incorporating this new information.\n"
                    "Focus on providing a complete answer to the original question. "
                    "Respond with a concise answer only, do not output any other words."
                )
                combine_response = self.llm.complete(combine_prompt)
                current_answer = combine_response.text

            # Check if answer is complete
            evaluation_prompt = (
                f"Original question: {original_query}\n"
                f"Current answer: {current_answer}\n"
                
                "Task: Determine if the current answer fully addresses all aspects of the original question."
                "Answer with only 'yes' or 'no'."
            )
            
            evaluation = self.llm.complete(evaluation_prompt)
            can_answer = "yes" in evaluation.text.lower()
            
            if can_answer or i == max_iterations - 1:
                break

                
            # Generate new query for next iteration
            all_contexts = "\n\n".join(retrieved_contexts)

            rewrite_prompt = (
                f"Original question: {original_query}\n"
                f"Current answer: {current_answer}\n"
                f"Information retrieved so far:\n{all_contexts}\n\n"
                
                "The current answer is incomplete. Please generate a new specific search query that will help find "
                "additional information needed to fully answer the original question. Focus on aspects not covered yet.\n\n"
                
                "New search query (be specific and concise):"
            )
            
            rewrite_response = self.llm.complete(rewrite_prompt)
            current_query = rewrite_response.text.strip()
        
        return current_answer
    
    def query(self, query_text: str):
        # Select appropriate tool based on query complexity
        selector_result = self.selector.select(
            choices=self.choices,
            query=query_text
        )
        
        select_idx = selector_result.selections[0].index
        selected_tool = self.tools[select_idx]
        
        response = selected_tool.fn(
            query_text
        )
        
        return response, select_idx