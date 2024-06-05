set -o pipefail
set -o nounset
set -o errtrace

source "$(dirname "$(realpath "$0")")/env.sh"
source "$(dirname "$(realpath "$0")")/utils.sh"

echo
echo "Wait until grpc endpoint is READY"

ISVC_NAMES=(all-minilm-caikit bge-large-en-caikit multilingual-large-caikit)
MODEL_IDS=(all-MiniLM-L12-v2-caikit bge-large-en-v1.5-caikit multilingual-e5-large-caikit)
MODEL_NAMES=(sentence-transformers/all-MiniLM-L12-v2 BAAI/bge-large-en-v1.5 intfloat/multilingual-e5-large)

for i in "${!ISVC_NAMES[@]}"
do
    wait_for_pods_ready "serving.kserve.io/inferenceservice=${ISVC_NAMES[i]}" "${TEST_NS}"
    oc wait --for=condition=ready pod -l serving.kserve.io/inferenceservice=${ISVC_NAMES[i]} -n ${TEST_NS} --timeout=300s
    export ISVC_HOSTNAME=$(oc get isvc "${ISVC_NAMES[i]}" -n $TEST_NS -o jsonpath='{.status.components.predictor.url}' | cut -d'/' -f3)
    echo "Testing ${ISVC_NAMES[i]}"
    python ./tests/test_embeddings-grpc.py -m="${MODEL_IDS[i]}" -n="${MODEL_NAMES[i]}" -i="${ISVC_HOSTNAME}"
    echo
done