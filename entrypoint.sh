#!/bin/sh
make requirements
gcloud auth activate-service-account dtu-hpc@onyx-glider-337908.iam.gserviceaccount.com --key-file=cloud_api_key.json 
dvc pull
make data
make train