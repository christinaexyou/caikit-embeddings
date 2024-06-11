# Copyright 2023 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Using specific tag to avoid newer minio versions that don't currently work
FROM docker.io/minio/minio:RELEASE.2021-06-17T00-10-46Z.hotfix.35a0912ff as minio-examples

EXPOSE 9000

ARG MODEL_DIR=/data1/modelmesh-example-models

USER root

RUN useradd -u 1000 -g 0 modelmesh
RUN mkdir -p ${MODEL_DIR}
RUN chown -R 1000:0 /data1 && \
    chgrp -R 0 /data1 && \
    chmod -R g=u /data1

COPY --chown=1000:0 <YOURMODEL> ${MODEL_DIR}

USER 1000