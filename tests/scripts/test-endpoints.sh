set -o pipefail
set -o nounset
set -o errtrace

source "$(dirname "$(realpath "$0")")/env.sh"
source "$(dirname "$(realpath "$0")")/utils.sh"

echo
echo "Wait until HTTP runtime is READY"

ISVC_NAMES=(all-minilm-caikit bge-large-en-caikit multilingual-large-caikit)
MODEL_IDS=(all-MiniLM-L12-v2-caikit bge-large-en-v1.5-caikit multilingual-e5-large-caikit)
MODEL_NAMES=(sentence-transformers/all-MiniLM-L12-v2 BAAI/bge-large-en-v1.5 intfloat/multilingual-e5-large)

for i in "${!ISVC_NAMES[@]}"
do
    wait_for_pods_ready "serving.kserve.io/inferenceservice=${ISVC_NAMES[i]}" "${TEST_NS}"
    oc wait --for=condition=ready pod -l serving.kserve.io/inferenceservice=${ISVC_NAMES[i]} -n ${TEST_NS} --timeout=300s
    ISVC_URL=$(oc get isvc ${ISVC_NAMES[i]} -n $TEST_NS -o jsonpath='{.status.url}')
    echo "Testing ${ISVC_NAMES[i]}"
    python ./tests/test_embeddings.py -m="${MODEL_IDS[i]}" -n="${MODEL_NAMES[i]}" -i="${ISVC_URL}"
    echo
done