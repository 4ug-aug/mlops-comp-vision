Project in MLOPS
==============================


1. **Overall project goal:** This project is about setting up a proper MLOps pipeline to train, test, deploy and monitor deep learning models that performs image classification.
2. **Framework:** As we are going to perform image classification the Pytorch image models (timm) framwork is well suited for our task. The timm framwork provides a lot of utilities in regards to image classification such as pretrained models and training pipelines. In addtion we utilize the pytorch lightning framwork to easen the training process of our model
3. **Intended use of framework:** In many domains of image classification the tasks share some high level common features, this allows for the utilization of models trained on other image domains. The timm framwork provides a lot of pretrained models, and our intend is to try and use some of these.
4. **Data:** The scope of this project is to perform image classification on a dataset of moths. This dataset consist of 224x224 colored images with 100 different classes. There are a total of 12500 training images and 500 test and validation images and the classes are equally distributed. We only plan to work with a subset of the data since it makes everything easier to work with, since the overall scope is not the performance of the model.
5. **Deep learning models:** We plan to use deep convolutional neural networks as they are most commonly used for image classification.


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
