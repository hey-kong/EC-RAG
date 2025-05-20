import os
import re
import hashlib
import argparse
import torch
from typing import List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# LlamaIndex related
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
)
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from kvcache_io import save_kvcache
from slm_inference import chunk_with_prefix


def get_nodes_from_documents(
        documents: List[Document],
        splitter: SentenceSplitter,
) -> List[BaseNode]:
    nodes = []
    seen_hashes = set()

    for doc_id, document in tqdm(enumerate(documents)):
        doc_text = document.get_content()
        chunk_texts = splitter.split_text(doc_text)

        for chunk_id, chunk_text in enumerate(chunk_texts):
            chunk_text = re.sub(r'\s+', ' ', chunk_text.strip())
            chunk_hash = hashlib.md5(chunk_text.strip().encode('utf-8')).hexdigest()
            if chunk_hash in seen_hashes:
                continue
            seen_hashes.add(chunk_hash)

            node = TextNode(
                text=chunk_text,
                id_=f"{document.doc_id}_{chunk_id}",
            )
            nodes.append(node)

    return nodes


def main():
    parser = argparse.ArgumentParser(description='Run indexing for RAG')
    parser.add_argument('--embedding_model', type=str, default='BAAI/bge-small-en-v1.5',
                        help='Embedding model name or path')
    parser.add_argument('--slm_model_path', type=str, default='Qwen/Qwen3-4B', help='Path of local slm model')
    parser.add_argument('--chunk_size', type=int, default=512, help='chunk size for splitter')
    parser.add_argument('--chunk_overlap', type=int, default=20, help='chunk overlap for splitter')
    parser.add_argument('--dataset_name', type=str, default='dataset', help='dataset name')
    parser.add_argument('--docs_dir', type=str, default='../data/dataset/documents', help='directory of documents')
    parser.add_argument('--persist_dir', type=str, default='../docs_store', help='persist dir for docstore')
    parser.add_argument('--chunk_kvcache_dir', type=str, default='../chunk_kvcache',
                        help='persist dir for chunk kvcache')
    parser.add_argument('--save_kvcache', action='store_true', help='Whether to save chunk kvcache')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    splitter = SentenceSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding_model)

    # chunking
    print(f"Chunking documents with chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}")
    documents = SimpleDirectoryReader(args.docs_dir).load_data()
    nodes = get_nodes_from_documents(documents, splitter)

    if args.save_kvcache:
        tokenizer = AutoTokenizer.from_pretrained(args.slm_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.slm_model_path,
            torch_dtype=torch.float16
        ).to(device)
        model.eval()
        chunk_kvcache_dir = f'{args.chunk_kvcache_dir}/{args.dataset_name}'
        if not os.path.exists(chunk_kvcache_dir):
            os.makedirs(chunk_kvcache_dir)
        for idx, node in tqdm(enumerate(nodes, 1), total=len(nodes), desc="Processing chunks"):
            chunk = chunk_with_prefix(node.text, args.slm_model_path)
            inputs = tokenizer(chunk, return_tensors="pt").to(device)
            prefix_cache = DynamicCache()
            with torch.no_grad():
                prefix_cache = model(**inputs, past_key_values=prefix_cache, use_cache=True).past_key_values
            kvcache_file_path = f'{chunk_kvcache_dir}/kvcache_chunk_{node.node_id}.safetensors'
            save_kvcache(prefix_cache, kvcache_file_path)
            node.metadata["kvcache_file_path"] = kvcache_file_path

    # document store: for bm25 retrieval
    doc_store = SimpleDocumentStore()
    doc_store.add_documents(nodes)

    # vector index
    vector_store = SimpleVectorStore()
    index = VectorStoreIndex(
        nodes=nodes,
        vector_store=vector_store,
    )

    # persist
    print(f"Persisting docstore and vector index to {args.persist_dir}")
    if not os.path.exists(args.persist_dir):
        os.makedirs(args.persist_dir)
    persist_path = os.path.join(args.persist_dir, f"{args.dataset_name}_docstore.pkl")
    doc_store.persist(persist_path)

    index.storage_context.persist(persist_dir=args.persist_dir + f"/{args.dataset_name}_vec/")
    print("Done!")


if __name__ == '__main__':
    main()
