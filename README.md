TwitterSARai
==============================

This project implements and deploys a classifier for VPD tweets based on machine
learning algorithms from Python's sklearn package. Currently, the TwitterSARai project
both serves and depends on the [AzTwitterSAR](https://github.com/lp-code/AzTwitterSAR)
project:

![TwitterSAR architecture](docs/twittersar_architecture.png)

AzTwitterSAR does timer-based querying of the Twitter API for new tweets from the
relevant account. It also does it's own keyword-based assessment of the tweets,
which results in many false positives. This is where TwitterSARai comes in: based
on the AzTwitterSAR-preselected tweets, TwitterSARai implements a machine-learning
model that filters the tweets more narrowly, so that only actually interesting ones
are delivered to the Slack channel for user notification. TwitterSARai returns its
classification result to AzTwitterSAR, which then dispatches the message to Slack,
if applicable.

Model
-----
The modelling uses sklearn classifiers and does a simple grid search covering
both preprocessing and model configurations. Since the AzTwitterSAR-preprocessed
tweets are approximately balanced between "interesting" and "not interesting",
the f1-score is used as metric. For the best model, the ROC curve is used to
establish a probability score threshold that is on the safe side, i.e. we rather
allow for false positives than false negatives (we don't want to miss out on
interesting tweets).  

Data
----
The machine learning model is based on a data set of tweets starting in December 2016.
At that point, the Twitter account was 
[Hordaland PD](https://twitter.com/hordalandpoliti). In November 2017, the new account
of [Vest PD](https://twitter.com/politivest)
was started. The data is not checked into the git repository. Data and pipeline
versioning is based on [DVC](https://dvc.org/), whose meta-information files are
version-controlled by git and therefore present in the repo.

Deployment
----------
TwitterSARai is deployed as an Azure Function. Since a Function App can only host
Functions of a single programming language, TwitterSARai lives in its own app rather
than together with AzTwitterSAR. 
Typical usage of the classifier is covered by the free tier for Functions.

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
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
