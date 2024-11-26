from multiprocessing import Pool, cpu_count

import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix, vstack
from tqdm import tqdm

from .ngram import get_ngrams_parallel


class Bm25:
    def __init__(
        self,
        vocab,
        doc_freqs,
        tokenized_docs,
        ngram_range,
        max_features,
        k1: float = 1.1,
        b: float = 0.5,
    ):
        self.vocab = vocab
        self.doc_freqs = doc_freqs
        self.tokenized_docs = tokenized_docs
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.k1 = k1
        self.b = b
        self.vocab_terms = list(self.vocab.keys())
        self.vocab_size = len(self.vocab_terms)
        self.token_to_id = {term: idx for idx, term in enumerate(self.vocab_terms)}
        self.idf_array = None
        self.tokenized_docs_ids = self.tokens_to_ids(self.tokenized_docs)
        self.doc_lengths = np.array(
            [len(doc_ids) for doc_ids in self.tokenized_docs_ids]
        )
        self.average_document_length = np.mean(self.doc_lengths)

    def fit(self):
        num_docs = len(self.tokenized_docs_ids)
        doc_freqs_array = np.array(
            [self.doc_freqs[term] for term in self.vocab_terms], dtype=np.int32
        )
        self.idf_array = np.log(
            (num_docs - doc_freqs_array + 0.5) / (doc_freqs_array + 0.5) + 1
        )
        self.embedding_matrix = self._fit_transform()

    def get_embedding(self):
        if self.embedding_matrix is None:
            self.fit()
        return self.embedding_matrix

    def _fit_transform(self):
        num_docs = len(self.tokenized_docs_ids)
        batch_size = 1000  # 한 번에 처리할 문서 수
        embed_list = []
        for start in tqdm(range(0, num_docs, batch_size), desc="Calculating BM25"):
            end = min(start + batch_size, num_docs)
            batch_embed = self._process_batch(self.tokenized_docs_ids[start:end], start)
            embed_list.append(batch_embed)
        print("Finish BM25 Embedding")
        return vstack(embed_list)

    def _process_batch(self, batch_doc_ids, start_idx):
        batch_size = len(batch_doc_ids)
        doc_lengths_batch = self.doc_lengths[start_idx : start_idx + batch_size]
        batch_embed = _process_batch_numba(
            batch_doc_ids,
            doc_lengths_batch,
            self.average_document_length,
            self.idf_array,
            self.k1,
            self.b,
            self.vocab_size,
        )
        return csr_matrix(batch_embed)

    def transform(self, tokenized_query):
        query_ids = np.array(
            [
                self.token_to_id[token]
                for token in get_ngrams_parallel(tokenized_query, self.ngram_range)
                if token in self.token_to_id
            ],
            dtype=np.int64,
        )
        query_vector = _transform_query_numba(query_ids, self.vocab_size)
        return csr_matrix(query_vector)

    def tokens_to_ids(self, tokenized_docs):
        return [
            np.array(
                [
                    self.token_to_id[token]
                    for token in get_ngrams_parallel(doc, self.ngram_range)
                    if token in self.token_to_id
                ],
                dtype=np.int64,
            )
            for doc in tokenized_docs
        ]


@njit(parallel=True)
def _process_batch_numba(
    batch_doc_ids, doc_lengths_batch, average_doc_length, idf_array, k1, b, vocab_size
):
    batch_size = len(batch_doc_ids)
    batch_embed = np.zeros((batch_size, vocab_size), dtype=np.float32)
    for i in prange(batch_size):
        doc_ids = batch_doc_ids[i]
        doc_len = doc_lengths_batch[i]
        freq_dict = np.zeros(vocab_size, dtype=np.int32)

        for token_id in doc_ids:
            freq_dict[token_id] += 1

        for idx in range(vocab_size):
            freq = freq_dict[idx]
            if freq > 0:
                numerator = idf_array[idx] * freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * doc_len / average_doc_length)
                batch_embed[i, idx] = numerator / denominator
    return batch_embed


@njit
def _transform_query_numba(query_ids, vocab_size):
    query_vector = np.zeros(vocab_size, dtype=np.float32)
    for token_id in query_ids:
        query_vector[token_id] = 1.0
    return query_vector
