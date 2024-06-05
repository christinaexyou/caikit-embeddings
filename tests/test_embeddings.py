"""Test suite for Caikit Embeddings"""
from typing import List
import argparse
import requests

import numpy as np

from caikit.core.data_model.json_dict import JsonDict

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import (
        dot_score,
        normalize_embeddings as normalize,  # avoid parameter shadowing
        semantic_search
    )

def _parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "-m", "--model_id", required=True, help="Model ID"
    )
    parser.add_argument(
        "-n", "--hf_model_name", required=True, help="Source model name or path"
    )
    parser.add_argument(
        "-i", "--isvc_url", required=True, help="Inference service URL"
    )
    args = parser.parse_args()
    return args.model_id, args.hf_model_name, args.isvc_url
class testEmbeddings:
    """
    Tests to validate inference responses from Caikit Embedding endpoints
    """
    def __init__(
            self,
            model_id,
            model_name_or_path,
            isvc_url
        ):
        self.model_id = model_id
        self.model = SentenceTransformer(model_name_or_path)
        self.isvc_url = isvc_url

    def embeddingTask_test(self, text: str):
        """
        Validates embeddings for a single text input
        """
        response = requests.post(
            self.isvc_url + "/api/v1/task/embedding",
            json={
             "model_id": self.model_id,
             "inputs": text
            },
            verify=False
        )
        if response.status_code == 200:
            inference_response = response.json()['result']['data']['values']
            model_output = self.model.encode(text)

            np.testing.assert_almost_equal(inference_response, model_output, decimal=3)
        else:
            print(response.text)

    def embeddingTasks_test(self, texts: List[str]):
        """
        Validates embeddings for multiple text inputs
        """
        response = requests.post(
            self.isvc_url + "/api/v1/task/embedding-tasks",
            json={
                "model_id": self.model_id,
                "inputs":  texts
            },
            verify=False
        )
        if response.status_code == 200:
            inference_response = response.json()['results']['vectors'][0]['data']['values']
            model_output = self.model.encode(texts)[0]
            np.testing.assert_almost_equal(inference_response, model_output, decimal=3)
        else:
            print(response.text)

    def sentenceSimilarityTask_test(self, source_sentence: str, sentences: List[str]):
        """
        Validates sentence similarity  between a single source sentence and reference sentences
        """
        response = requests.post(
            self.isvc_url + "/api/v1/task/sentence-similarity",
            json={
             "model_id": self.model_id,
             "inputs": {
                 "source_sentence": source_sentence,
                 "sentences": sentences
                }
            },
            verify=False
        )
        if response.status_code == 200:
            inference_response = response.json()['result']['scores']
            model_output = self.model.similarity(
                self.model.encode(source_sentence),
                self.model.encode(sentences)
            ).flatten()
            np.testing.assert_almost_equal(inference_response, model_output, decimal=3)
        else:
            print(response.text)

    def sentenceSimilarityTasks_test(self, source_sentences: List[str], sentences: List[str]):
        """
        Validates sentence similarity between multiple source sentences and reference sentences
        """
        response = requests.post(
            self.isvc_url + "/api/v1/task/sentence-similarity-tasks",
            json={
                "model_id": self.model_id,
                "inputs": {
                    "source_sentences": source_sentences,
                    "sentences": sentences
                }
            },
            verify=False
        )
        if response.status_code == 200:
            inference_response = response.json()['results']
            scores = sum([result['scores'] for result in inference_response], [])
            model_output = self.model.similarity(
                self.model.encode(source_sentences), self.model.encode(sentences)
                ).flatten()

            np.testing.assert_almost_equal(scores, model_output, decimal=3)
        else:
            print(response.text)

    def rerankTask_test(self, query: str,  documents: List[JsonDict], top_n=None):
        """
        Validates reranking results for a single query and reference documents
        """
        if top_n is None:
            top_n = len(documents)
        response = requests.post(
            self.isvc_url + "/api/v1/task/rerank",
            json={
             "model_id": self.model_id,
             "inputs": {
                 "documents": documents,
                 "query": query
             },
            "parameters": {
                "top_n": top_n
                }
            },
            verify=False
        )
        if response.status_code == 200:
            inference_response = response.json()['result']
            corpus_ids = [
                result['index'] for result in inference_response['scores']
                ]
            scores = [
                result['score'] for result in inference_response['scores']
                ]
            query_embeddings = normalize(
                self.model.encode(
                    [query],
                    convert_to_tensor=True
                    )
                )
            document_embeddings = normalize(
                self.model.encode(
                    [list(doc.values())[0] for doc in documents],
                    convert_to_tensor=True
                    )
                )
            results = semantic_search(
                query_embeddings, document_embeddings, top_k=top_n, score_function=dot_score
                )[0]

            np.testing.assert_equal(corpus_ids, [res['corpus_id'] for res in results])
            np.testing.assert_almost_equal(scores, [res['score'] for res in results], decimal=3)
        else:
            print(response.text)

    def rerankTasks_test(self, queries: List[str], documents: List[JsonDict], top_n=None):
        """
        Validates reranking results for multiple queries and reference documents
        """
        if top_n is None:
            top_n = len(documents)
        response = requests.post(
            self.isvc_url + "/api/v1/task/rerank-tasks",
            json={
                "model_id": self.model_id,
                "inputs": {
                    "documents": documents,
                    "queries": queries
                },
            "parameters": {
                "top_n": top_n
                }
            },
            verify=False

        )
        if response.status_code == 200:
            inference_response = response.json()['results']
            corpus_ids = [
                score['index'] for result in inference_response for score in result['scores']
                ]
            scores = [
                score['score'] for result in inference_response for score in result['scores']
                ]
            query_embeddings = normalize(self.model.encode(queries, convert_to_tensor=True))
            document_embeddings = normalize(self.model.encode([list(doc.values())[0] for doc in documents], convert_to_tensor=True))
            results = semantic_search(query_embeddings, document_embeddings, top_k=top_n, score_function=dot_score)

            np.testing.assert_equal(corpus_ids, [doc['corpus_id'] for res in results for doc in res])
            np.testing.assert_almost_equal(scores, [doc['score'] for res in results for doc in res], decimal=3)
        else:
            print(response.text)

if __name__=="__main__":
    model_id, model_name_or_path, isvc_url = _parse_args()
    test_embeddings = testEmbeddings(
        model_id=model_id,
        model_name_or_path=model_name_or_path,
        isvc_url=isvc_url
    )
    text = "test first sentence"
    texts = ['test first sentence', 'another test sentence']
    documents = [
        {'text': 'first sentence', 'title': 'first title'},
        {'text': 'another sentence', 'more': 'more attributes here'},
        {'text': 'a doc with a nested metadata', 'meta': {'foo': 'bar', 'i': 999, 'f': 12.34}}
    ]
    test_embeddings.embeddingTask_test(text=text)
    test_embeddings.embeddingTasks_test(texts=texts)
    test_embeddings.sentenceSimilarityTask_test(source_sentence=text, sentences=texts)
    test_embeddings.sentenceSimilarityTasks_test(source_sentences=texts, sentences=texts)
    test_embeddings.rerankTask_test(query=text, documents=documents)
    test_embeddings.rerankTasks_test(queries=texts, documents=documents)
