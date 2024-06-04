## Caikit Embeddings

### Convert model to Caikit format
```
python bootstrap_model.py -m=<model_name_or_path> -o=<output_path>
```

### Edit Dockerfile
Replace `<YOURMODEL>` with the name of your model in `Dockerfile`

```
COPY --chown=1000:0 <YOURMODEL> ${MODEL_DIR}/<YOURMODEL>/
.
.
.
```

### Build Dockerfile and push to Quay
Build the Dockerfile
```
QUAY_USERNAME=<quay_username>
MODEL_NAME=<model_name>
podman build --target minio-examples -t quay.io/${QUAY_USERNAME}/modelmesh-minio-examples:${MODEL_NAME} .
```
Push image to Podman
```
podman push quay.io/${QUAY_USERNAME}/modelmesh-minio-examples:<model_name>
```

### Edit MinIO manifest
Replace `<QUAY_USERNAME>` and `<MODEL_NAME>` with your Quay username and model name in `manifests/minio/minio.yaml`

```
    .
    .
    .
      image: quay.io/<QUAY_USERNAME>/modelmesh-minio-examples:<MODEL_NAME>
```

### Setup MinIO storage
```
ACCESS_KEY_ID=admin
SECRET_ACCESS_KEY=password
MINIO_NS=minio
oc new-project $MINIO_NS
```

```
oc apply -f ./manifests/minio/minio.yaml -n ${MINIO_NS}
sed "s/<minio_ns>/$MINIO_NS/g" manifests/minio/minio-secret.yaml | tee ./minio-secret-current.yaml | oc -n ${MINIO_NS} apply -f -
sed "s/<minio_ns>/$MINIO_NS/g" manifests/minio/serviceaccount-minio.yaml | tee ./serviceaccount-minio-current.yaml | oc -n ${MINIO_NS} apply -f -
```

### Deploy LLM with Caikit Standalone Serving Runtime
Create new test namespace
```
TEST_NS=<test_namespace>
oc new-project ${TEST_NS}
```

Deploy serving runtime
```
oc apply -f manifests/caikit/caikit-servingruntime.yaml -n ${TEST_NS}
```

Deploy the MinIO data connection and service account
```
oc apply -f ./minio-secret-current.yaml -n ${TEST_NS}
oc create -f ./serviceaccount-minio-current.yaml -n ${TEST_NS}
```

Replace the following sections in `manifests/caikit/caikit-isvc.yaml`
```
  .
  .
  .
  name: <isvc_name>
```

```
      .
      .
      .
      storageUri: <model_directory>
```
Deploy the inference service
```
oc apply -f manifests/caikit/caikit-isvc.yaml
```

Sanity check to make sure the inference service's `READY` state is `True`
```
ISVC_NAME=caikit-example-isvc
oc get isvc ${ISVC_NAME} -n ${TEST_NS}
```


### Make an inference request
```
ISVC_URL=$(oc get isvc ${ISVC_NAME} -n ${TEST_NS} -o jsonpath='{.status.components.predictor.url}')

curl -kL -H 'Content-Type: application/json' -d '{"model_id": <model_name>, "inputs": "At what temperature does Nitrogen boil?"}' ${ISVC_URL}/api/v1/task/embedding
```

### Clean up
Delete inference service and test namespace
```
oc delete isvc --all -n ${TEST_NS} --force --grace-period=0
oc delete ns ${TEST_NS}
```
Delete MinIO namespace
```
oc delete ns ${MINIO_NS}
```

## Testing
The following open source embeddings models have already been built as container images and pushed to Quay:
* all-minilm-l12-v2
* bge-large-en-v1.5
* multilingual-e5-large

To test them, follow the instructions starting from [Edit Minio manifest](#edit-minio-manifest) and replace the image URI with one of the following:
* quay.io/repository/christinaexyou/modelmesh-minio-examples/all-MiniLM-L12-v2-caikit
* quay.io/repository/christinaexyou/modelmesh-minio-examples/multilingual-e5-large-caikit
* quay.io/repository/christinaexyou/modelmesh-minio-examples/bge-large-en-v1.5-caikit

Follow the rest of the steps to create an inference service and stop at [Make an inference request](#make-an-inference-request) and continue the following steps.

### Install required libraries
```
pip install -r tests/requirements.txt
```

### Run test
Replace the `<model_name_or_path>` with its HuggingFace name
```
ISVC_URL=$(oc get ksvc $ISVC_NAME -n $TEST_NS -o jsonpath='{.status.url}' | cut -d'/' -f3)

python tests/test_embeddings.py -m=$MODEL_NAME - -n=<model_name_or_path> -i=$ISVC_URL
```

For example:
```
python tests/test_embeddings.py -m=$MODEL_NAME -n="sentence-transformers/all-MiniLM-L12-v2" -i=$ISVC_URL
```