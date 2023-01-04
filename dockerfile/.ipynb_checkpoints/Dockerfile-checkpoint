FROM nvcr.io/nvidia/tensorflow:22.03-tf2-py3

ENV SHELL /bin/bash

RUN pip install minio kfp kserve

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

EXPOSE 8888

CMD ["bash", "-c", "jupyter lab --allow-root --notebook-dir=/workspace --ip=0.0.0.0 --no-browser --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]