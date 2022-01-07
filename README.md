It's raining cats and dogs
==============================

Our exam project for 02476 Machine Learning Operations will utilize the computer vision library `kornia` in object detection of cats or dogs. The overall goal of the project is to create a Kornia [transformer](https://kornia.readthedocs.io/en/latest/applications/image_classification.html) which will be trained on the [cats-vs-dogs](https://www.kaggle.com/shaunthesheep/microsoft-catsvsdogs-dataset) dataset from Kaggle. 

Initially, we will train a deep learning model on the cats-vs-dogs dataset, but our approach makes it easy to upscale the scope of the project, e.g. by adding more pictures of different animals or adding pictures with no objects. The dataset is 886 MB and consists of 12500 cat and 12500 dog pictures in different scales and resolutions. In addition to the Kornia transformers used for our deep learning model, we will utilize Kornia augmentations to get more data for our model.

For our project, we have used `cookiecutter` to organize our repository as noted by the project organization below. To store large amounts of data, we will use `dvc`, specifically for remote storage of our dataset and weight files obtained after training the Kornia transformer. To store config files we will use `hydra`, which we will incorporate with `wandb`, which we will use to log our model training and hyperparameter tuning. We hope to make use of PyTorch-Lightning in our project if compatible with Kornia.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
