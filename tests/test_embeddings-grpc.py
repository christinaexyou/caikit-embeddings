"""Test suite for Caikit Embeddings"""
from typing import List
import argparse
import subprocess
import json

import numpy as np

from caikit.core.data_model.json_dict import JsonDict

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import (
    dot_score,
    normalize_embeddings as normalize,  # avoid parameter shadowing
    semantic_search,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--model_id", required=True, help="Model ID")
    parser.add_argument(
        "-n", "--hf_model_name", required=True, help="Source model name or path"
    )
    parser.add_argument(
        "-i", "--isvc_hostname", required=True, help="Inference service hostname"
    )
    args = parser.parse_args()
    return args.model_id, args.hf_model_name, args.isvc_hostname


def grpc_request(command):
    try:
        response = subprocess.run(command, capture_output=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(e)
    if response.stdout:
        return json.loads(response.stdout.decode("utf-8"))
    return None


class testEmbeddingsGrpc:
    """
    Tests to validate inference responses from Caikit Embedding endpoints
    """

    def __init__(self, model_id, model_name_or_path, isvc_hostname):
        self.model_id = model_id
        self.model = SentenceTransformer(model_name_or_path)
        self.isvc_hostname = isvc_hostname

    def embeddingTask_test(self, text: str):
        payload = json.dumps({"text": text})
        command = f"grpcurl -insecure -d '{payload}' -H \"mm-model-id: {self.model_id}\" {self.isvc_hostname}:443 caikit.runtime.Nlp.NlpService/EmbeddingTaskPredict"
        response = grpc_request(command=command)
        if response:
            inference_response = response["result"]["data_npfloat32sequence"]["values"]
        model_output = self.model.encode(text)
        np.testing.assert_almost_equal(inference_response, model_output, decimal=3)

    def embeddingTasks_test(self, texts: List[str]):
        payload = json.dumps({"texts": texts})
        command = f"grpcurl -insecure -d '{payload}' -H \"mm-model-id: {self.model_id}\" {self.isvc_hostname}:443 caikit.runtime.Nlp.NlpService/EmbeddingTasksPredict"
        response = grpc_request(command=command)
        if response:
            inference_response = response["results"]["vectors"][0][
                "data_npfloat32sequence"
            ]["values"]
        model_output = self.model.encode(texts)[0]
        np.testing.assert_almost_equal(inference_response, model_output, decimal=3)

    def sentenceSimilarityTask_test(self, source_sentence: str, sentences: List[str]):
        payload = json.dumps(
            {"source_sentence": source_sentence, "sentences": sentences}
        )
        command = f"grpcurl -insecure -d '{payload}' -H \"mm-model-id: {self.model_id}\" {self.isvc_hostname}:443 caikit.runtime.Nlp.NlpService/SentenceSimilarityTaskPredict"
        response = grpc_request(command=command)
        if response:
            inference_response = response["result"]["scores"]
        model_output = self.model.similarity(
            self.model.encode(source_sentence), self.model.encode(sentences)
        ).flatten()
        np.testing.assert_almost_equal(inference_response, model_output, decimal=3)

    def sentenceSimilarityTasks_test(
        self, source_sentences: List[str], sentences: List[str]
    ):
        payload = json.dumps(
            {"source_sentences": source_sentences, "sentences": sentences}
        )
        command = f"grpcurl -insecure -d '{payload}' -H \"mm-model-id: {self.model_id}\" {self.isvc_hostname}:443 caikit.runtime.Nlp.NlpService/SentenceSimilarityTasksPredict"
        response = grpc_request(command=command)
        if response:
            inference_response = sum(
                [result["scores"] for result in response["results"]], []
            )
        model_output = self.model.similarity(
            self.model.encode(source_sentences), self.model.encode(sentences)
        ).flatten()

        np.testing.assert_almost_equal(inference_response, model_output, decimal=3)

    def rerankTask_test(self, query: str, documents: List[JsonDict], top_n=None):
        if top_n is None:
            top_n = len(documents)
        payload = json.dumps({"query": query, "documents": documents, "top_n": top_n})
        command = f"grpcurl -insecure -d '{payload}' -H \"mm-model-id: {self.model_id}\" {self.isvc_hostname}:443 caikit.runtime.Nlp.NlpService/RerankTaskPredict"
        response = grpc_request(command=command)
        if response:
            inference_response = [
                result["score"] for result in response["result"]["scores"]
            ]
        query_embeddings = normalize(self.model.encode([query], convert_to_tensor=True))
        document_embeddings = normalize(
            self.model.encode(
                [list(doc.values())[0] for doc in documents], convert_to_tensor=True
            )
        )
        results = semantic_search(
            query_embeddings, document_embeddings, top_k=top_n, score_function=dot_score
        )[0]

        np.testing.assert_almost_equal(
            inference_response, [res["score"] for res in results], decimal=3
        )

    def rerankTasks_test(
        self, queries: List[str], documents: List[JsonDict], top_n=None
    ):
        if top_n is None or top_n < 1:
            top_n = len(documents)
        payload = json.dumps(
            {"queries": queries, "documents": documents, "top_n": top_n}
        )
        command = f"grpcurl -insecure -d '{payload}' -H \"mm-model-id: {self.model_id}\" {self.isvc_hostname}:443 caikit.runtime.Nlp.NlpService/RerankTasksPredict"
        response = grpc_request(command=command)
        if response:
            scores = [
                score["score"]
                for result in response["results"]
                for score in result["scores"]
            ]
        query_embeddings = normalize(self.model.encode(queries, convert_to_tensor=True))
        document_embeddings = normalize(
            self.model.encode(
                [list(doc.values())[0] for doc in documents], convert_to_tensor=True
            )
        )
        results = semantic_search(
            query_embeddings, document_embeddings, top_k=top_n, score_function=dot_score
        )

        np.testing.assert_almost_equal(
            scores, [doc["score"] for res in results for doc in res], decimal=3
        )


if __name__ == "__main__":
    model_id, model_name_or_path, isvc_hostname = _parse_args()

    test_embeddings = testEmbeddingsGrpc(
        model_id=model_id,
        model_name_or_path=model_name_or_path,
        isvc_hostname=isvc_hostname,
    )
    text = "test first sentence"
    texts = ["test first sentence", "another test sentence"]
    documents = [
        {"text": "first sentence", "title": "first title"},
        {"text": "another sentence", "more": "more attributes here"},
        {
            "text": "a doc with a nested metadata",
            "meta": {"foo": "bar", "i": 999, "f": 12.34},
        },
    ]
    test_embeddings.embeddingTask_test(text=text)
    test_embeddings.embeddingTasks_test(texts=texts)
    test_embeddings.sentenceSimilarityTask_test(source_sentence=text, sentences=texts)
    test_embeddings.sentenceSimilarityTasks_test(
        source_sentences=texts, sentences=texts
    )
    test_embeddings.rerankTask_test(query=text, documents=documents)
    test_embeddings.rerankTasks_test(queries=texts, documents=documents)
