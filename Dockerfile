# syntax=docker/dockerfile:1
FROM gcr.io/google.com/cloudsdktool/cloud-sdk:latest
SHELL ["/bin/sh", "-c"]

RUN apt update && \
   apt install --no-install-recommends -y build-essential gcc && \
   apt clean && rm -rf /var/lib/apt/lists/*

RUN apt update
RUN apt-get -y install python3-venv
RUN pip3 install virtualenv
RUN git clone https://github.com/nsturis/MLopsProject.git
WORKDIR /MLopsProject
COPY onyx-glider-337908-7017e9498613.json cloud_api_key.json
ENV VIRUTALENV=env
RUN python3 -m venv ${VIRUTALENV}
ENV PATH="${VIRUTALENV}/bin:$PATH"
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]