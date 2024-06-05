export TEST_NS=caikit-embeddings-test
export MINIO_NS=minio
export deploy_odh_operator=false

getKserveNS() {
  if [[ ${TARGET_OPERATOR} == "odh" ]]
  then
    echo "opendatahub"
  else
    echo "redhat-ods-applications"
  fi
}

getOpType() {
  target_op=$1
  if [[ ${target_op} == "odh" ]]
  then
    echo "odh"
  else
    echo "rhods"
  fi
}

getOpNS() {
  target_op=$1
  if [[ ${target_op} == "odh" ]]
  then
    echo "openshift-operators"
  else
    echo "redhat-ods-operator"
  fi
}