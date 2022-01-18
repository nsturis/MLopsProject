# syntax=docker/dockerfile:1
FROM python:3.9-bullseye

RUN apt update && \
   apt install --no-install-recommends -y build-essential gcc && \
   apt clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/nsturis/MLopsProject.git
WORKDIR /MLopsProject
RUN make requirements
RUN dvc pull
RUN make data
RUN make train