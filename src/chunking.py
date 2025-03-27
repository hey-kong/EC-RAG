import os
import hashlib
import argparse
import tiktoken
from typing import List
from tqdm import tqdm

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
    parser.add_argument('--chunk_size', type=int, default=512, help='chunk size for splitter')
    parser.add_argument('--chunk_overlap', type=int, default=10, help='chunk overlap for splitter')
    parser.add_argument('--dataset_name', type=str, default='hotpotqa', help='dataset name')
    parser.add_argument('--docs_dir', type=str, default='../data/hotpotqa/documents', help='directory of documents')
    parser.add_argument('--persist_dir', type=str, default='../docs_store', help='persist dir for docstore')
    args = parser.parse_args()

    tokenizer = tiktoken.get_encoding("o200k_base")
    splitter = SentenceSplitter(
        tokenizer=tokenizer.encode,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding_model)

    # chunking
    print(f"Chunking documents with chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}")
    documents = SimpleDirectoryReader(args.docs_dir).load_data()
    nodes = get_nodes_from_documents(documents, splitter)

    # document store: for bm25 retrieval
    doc_store = SimpleDocumentStore()
    doc_store.add_documents(nodes)

    # vector index: 向量索引构建需要较长时间，所以要在这里进行
    vector_store = SimpleVectorStore()
    index = VectorStoreIndex(
        nodes=nodes,
        vector_store=vector_store,
    )

    # persist
    print(f"Persisting docstore and vector index to {args.persist_dir}")
    if not os.path.exists(args.persist_dir):
        os.makedirs(args.persist_dir)
    persist_path = os.path.join(args.persist_dir, f"{args.dataset_name}_{args.chunk_size}_docstore.pkl")
    doc_store.persist(persist_path)

    index.storage_context.persist(persist_dir=args.persist_dir + f"/{args.dataset_name}_{args.chunk_size}_vec/")
    print("Done!")


if __name__ == '__main__':
    main()
