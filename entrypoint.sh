#!/bin/sh
pwd
export GOOGLE_APPLICATION_CREDENTIALS="google_api_key.json"
gcloud auth activate-service-account dtu-hpc@onyx-glider-337908.iam.gserviceaccount.com --key-file=google_api_key.json
dvc pull
make data
make train