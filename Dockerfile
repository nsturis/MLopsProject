# syntax=docker/dockerfile:1
FROM ubuntu:18.04
COPY . /src
CMD dvc pull
RUN make data
RUN make train

