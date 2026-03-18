FROM python:3.8-slim

ENV LAMBDA_TASK_ROOT=/var/task
WORKDIR /var/task

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ca-certificates libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir \
    torch==1.9.0+cpu torchvision==0.10.0+cpu \
    --find-links https://download.pytorch.org/whl/torch_stable.html

RUN python3 -m pip install --no-cache-dir --no-deps facenet-pytorch==2.5.3

COPY requirements.txt /var/task/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /var/task/requirements.txt

RUN mkdir -p /tmp/.cache && chmod -R 777 /tmp/.cache
ENV TORCH_HOME=/tmp/.cache/torch
ENV XDG_CACHE_HOME=/tmp/.cache

COPY fd_lambda.py fr_lambda.py /var/task/

ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]
CMD ["fd_lambda.handler"]
