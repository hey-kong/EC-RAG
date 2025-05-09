import nltk
import numpy as np
from typing import List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import tiktoken
from scipy import spatial

class SemanticChunker:
    def __init__(
        self,
        embedding_model: HuggingFaceEmbedding,
        similarity_threshold: float = 0.8,
        max_tokens_per_chunk: int = 512,
    ):

        nltk.download('punkt')
        nltk.download('punkt_tab')
            
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        return 1 - spatial.distance.cosine(embedding1, embedding2)
    
    def _get_sentence_embeddings(self, sentences: List[str]) -> List[List[float]]:
        embeddings = []
        for sentence in sentences:
            if sentence.strip():  # ensure the sentence is not empty
                embedding = self.embedding_model.get_text_embedding(sentence)
                embeddings.append(embedding)
            else:
                embeddings.append(embeddings.append(np.zeros(self.embedding_model.embedding_dim).tolist()))
        return embeddings
    
    def split_text(self, doc: str) -> List[str]:
        sentences = nltk.sent_tokenize(doc)
        
        if not sentences:
            return []
        
        sentence_embeddings = self._get_sentence_embeddings(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_token_count = self._count_tokens(sentences[0])
        
        for i in range(1, len(sentences)):
            prev_embedding = sentence_embeddings[i-1]
            curr_embedding = sentence_embeddings[i]
            
            similarity = self._calculate_similarity(prev_embedding, curr_embedding)
            sentence_token_count = self._count_tokens(sentences[i])
            
            # if the similarity is below the threshold 
            # or adding the sentence exceeds the token limit
            # then save the current chunk and start a new one
            if similarity < self.similarity_threshold or current_token_count + sentence_token_count > self.max_tokens_per_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Start a new chunk
                current_chunk = [sentences[i]]
                current_token_count = sentence_token_count
            else:
                current_chunk.append(sentences[i])
                current_token_count += sentence_token_count

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks