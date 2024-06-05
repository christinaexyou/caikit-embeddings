#!/bin/bash
set -o pipefail
set -o nounset
set -o errtrace


# Deploys model for HTTP (default) or gRPC if "grpc" is passed as argument

# Check if at most one argument is passed
if [ "$#" -gt 1 ]; then
    echo "Error: at most a single argument ('http' or 'grpc') or no argument, default protocol being 'http'"
    exit 1
fi

# Default values that fit the default 'http' protocol:
INF_PROTO=""

# If we have an argument, check that it is either "http" or "grpc"
if [ "$#" -eq 1 ]; then
    if [ "$1" = "http" ]; then
	:  ### nothing to be done
    elif [ "$1" = "grpc" ]; then
	INF_PROTO="-grpc"
    else
	echo "Error: Argument must be either 'http' or 'grpc'."
	exit 1
    fi
fi

source "$(dirname "$(realpath "$0")")/env.sh"

# Deploy Minio
ACCESS_KEY_ID=THEACCESSKEY

## Check if ${MINIO_NS} exist
oc get ns ${MINIO_NS}
if [[ $? ==  1 ]]
then
  oc new-project ${MINIO_NS}
  SECRET_ACCESS_KEY=$(openssl rand -hex 32)
  oc apply -f ./tests/manifests/minio/minio.yaml -n ${MINIO_NS}
  sed "s/<accesskey>/$ACCESS_KEY_ID/g" ./manifests/minio/minio-secret.yaml | sed "s+<secretkey>+$SECRET_ACCESS_KEY+g" |sed "s/<minio_ns>/$MINIO_NS/g" | tee ./tests/manifests/minio/minio-secret-current.yaml | oc -n ${MINIO_NS} apply -f -
else
  SECRET_ACCESS_KEY=$(oc get pod minio  -n minio -o jsonpath='{.spec.containers[0].env[1].value}')
  sed "s/<accesskey>/$ACCESS_KEY_ID/g" ./manifests/minio/minio-secret.yaml | sed "s+<secretkey>+$SECRET_ACCESS_KEY+g" |sed "s/<minio_ns>/$MINIO_NS/g" | tee ./tests/manifests/minio/minio-secret-current.yaml
fi
sed "s/<minio_ns>/$MINIO_NS/g" ./manifests/minio/serviceaccount-minio.yaml | tee ./tests/manifests/minio/serviceaccount-minio-current.yaml

if ! oc get ns ${TEST_NS}
then
    oc new-project ${TEST_NS}
else
  echo "* ${TEST_NS} already exists."
fi

oc apply -f ./manifests/caikit/caikit-servingruntime"${INF_PROTO}".yaml -n ${TEST_NS}

BAAI_MODEL_NAME=bge-large-en
HF_MODEL_NAME=all-minilm
MICROSOFT_MODEL_NAME=multilingual-large

echo "Creating ISVC ${BAAI_MODEL_NAME}"
echo "Creating ISVC ${HF_MODEL_NAME}"
echo "Creating ISVC ${MICROSOFT_MODEL_NAME}"

oc apply -f ./tests/manifests/minio/minio-secret-current.yaml -n ${TEST_NS}
oc apply -f ./tests/manifests/minio/serviceaccount-minio-current.yaml -n ${TEST_NS}

###  create the isvc
oc apply -f ./tests/manifests/caikit/"$BAAI_MODEL_NAME"-isvc"${INF_PROTO}".yaml -n ${TEST_NS}
oc apply -f ./tests/manifests/caikit/"$HF_MODEL_NAME"-isvc"${INF_PROTO}".yaml -n ${TEST_NS}
oc apply -f ./tests/manifests/caikit/"$MICROSOFT_MODEL_NAME"-isvc"${INF_PROTO}".yaml -n ${TEST_NS}