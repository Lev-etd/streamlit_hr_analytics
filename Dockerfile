FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN set -ex && mkdir /app
WORKDIR /app

COPY ./requirements.txt /requirements.txt
COPY code/ ./
RUN python -m pip install -r /requirements.txt
RUN apt-get update && apt-get install -y \
curl gsfonts libsndfile1
