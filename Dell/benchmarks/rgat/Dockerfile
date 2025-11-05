ARG FROM_IMAGE_NAME=nvcr.io/nvidia/dgl:25.03-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/gnn
ENV WORK_DIR=/workspace/gnn

# make git happy
RUN git config --global user.name "a" && \
    git config --global user.email "a"

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

COPY . .
# required packages not included in DGL container:
RUN pip install -r requirements.txt

ENV WG_LOAD_USE_DIRECTIO=1

ENV PIP_CONSTRAINT=""

CMD /bin/bash
