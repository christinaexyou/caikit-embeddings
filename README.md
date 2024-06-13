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
  name: <ISVC_NAME>
```

```
      .
      .
      .
      storageUri: <MODEL_PATH>
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
The following open source embeddings models have already been containerized [here](quay.io/christinaexyou/modelmesh-minio-examples:embedding-models):
* all-minilm-l12-v2
* bge-large-en-v1.5
* multilingual-e5-large

### Prerequisites
* You have cloned this repository and set your working directory to the root
    ```
    https://github.com/christinaexyou/caikit-embeddings.git
    cd caikit-embeddings
    ```

### Procedure
1. Set the `TARGET_OPERATOR` to either `rhods` or `odh`. For example:
    ```
    export TARGET_OPERATOR=rhods
    ```

2. Deploy embeddings models

    For HTTP:
    ```
    tests/scripts/deploy-model.sh
    ```

    For gRPC
    ```
    tests/scripts/deploy-model.sh grpc
    ```

3. Validate inference responses

    For HTTP:
    ```
    tests/scripts/test-endpoints.sh
    ```

    For gRPC:
    ```
    tests/scripts/test-endpoints-grpc.sh
    ```

4. Delete model
    ```
    tests/scripts/delete-model.sh
    ```