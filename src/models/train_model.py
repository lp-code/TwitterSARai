"""
Build a model to filter tweets that have been flagged as potentially
interesting by the keyword matching algorithm.
"""
import json
import logging
import pickle
import sys
from pathlib import Path
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

import click

from model_utils import ClfSwitcher, TextPreprocessor


_project_dir = Path(__file__).resolve().parents[2]
_data_dir = _project_dir / "data"

def print_confusion_matrix(y_true, y_pred):
    print(pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True))

def print_model_metrics(y_actual, y_predicted):
    print(metrics.classification_report(y_actual, y_predicted,
                                        target_names=["0", "1"]))

    print_confusion_matrix(y_actual, y_predicted)

def write_metrics_file(
        y_train_actual, y_train_predicted, y_test_actual, y_test_predicted
):
    metrics_data = dict(
        f1_train=metrics.f1_score(y_train_actual, y_train_predicted),
        f1_test=metrics.f1_score(y_test_actual, y_test_predicted),
    )
    with open(_project_dir / "metrics.json", "w") as jsonfile:
        json.dump(metrics_data, jsonfile)


def plot_roc(roc_auc, fpr, tpr, thresholds):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr) #, label='naive Bayes')
    for i in range(1, len(fpr), 3):
        x, y, val = fpr[i], tpr[i], thresholds[i]
        plt.annotate(np.round(val, 2), (x, y - 0.04))
        plt.annotate(i, (x, y + 0.04))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve (AUC={roc_auc:4.2f})")
    plt.legend(loc="best")
    with open(_project_dir / "roc.svg", "w") as svg_file:
        plt.savefig(svg_file, format="svg")
    plt.show()


def print_wrong_predictions(description, docs, y_actual, y_pred):
    print(f"\n\n   WRONG PREDICTIONS ({description})")
    for pred, actual, txt in zip(y_pred, y_actual.to_numpy(), docs.to_numpy()):
        if actual != pred:
            print(f"Actual: {actual}, predicted: {pred}, {txt}")


@click.command()
@click.argument(
    "input_filepath",
    default=_data_dir / "processed" / "tweets_processed.csv",
    type=click.Path(exists=True)
)
@click.argument(
    "output_filepath",
    default=_data_dir / "xxx" / "something",
    type=click.Path())
def main(input_filepath, output_filepath):

    df = pd.read_csv(input_filepath)
    print("n_tweets: %d" % len(df))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        df["doc"], df["i_man"], test_size=0.2, random_state=42)

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent
    pipeline = Pipeline([
        ("prep", TextPreprocessor()),
        ("vect", None),  # This stage is set below in the grid search parameters.
        ("clf", ClfSwitcher()),
    ])

    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {
        "prep__remove_punctuation": [True],  # all winning ones have this
        "prep__remove_stopwords": [True],  # all winning ones have this
        "prep__stem": [True],
        "vect": [CountVectorizer(), TfidfVectorizer(max_df=0.95)],
        "vect__ngram_range": [(1, 1), (1, 2)],  # unigrams or bigrams
        "vect__min_df": [3, 5],  # all winning ones have 1 (kind of obvious...)
        "clf__estimator": [
            # LinearSVC(C=1000),  # has no predict_proba function
            # MultinomialNB(alpha=0.01),
            # MultinomialNB(alpha=1),  # winning ones mostly have this
            # BernoulliNB(alpha=xxx), none of these were successful
            ComplementNB(alpha=0.1, norm=False),  # best ComplementNB
            #SGDClassifier(alpha=1e-3, random_state=42, max_iter = 50, loss="log"),
            #SGDClassifier(alpha=1e-3, random_state=42, max_iter=150, loss="log"),
        ],
        "clf__threshold": np.arange(0.03, 0.05, 0.002),
    }
    grid_search = GridSearchCV(pipeline, parameters, scoring="f1", n_jobs=-1, verbose=1)
    grid_search.fit(docs_train, y_train)

    # TASK: print the mean and std for each candidate along with the parameter
    # settings for all the candidates explored by grid search.
    n_candidates = len(grid_search.cv_results_["params"])
    for i in range(n_candidates):
        print(i, "params - %s; mean - %0.2f; std - %0.2f"
                 % (grid_search.cv_results_["params"][i],
                    grid_search.cv_results_["mean_test_score"][i],
                    grid_search.cv_results_["std_test_score"][i]))

    # Predict the outcome on the testing set using the best model among those
    # with the different parameters.
    y_test_pred = grid_search.predict(docs_test)
    # To compare to dev-set metrics:
    y_train_pred = grid_search.predict(docs_train)

    print("BEST MODEL RESULTS (dev):")
    print_model_metrics(y_train, y_train_pred)

    print("BEST MODEL RESULTS (test):")
    print_model_metrics(y_test, y_test_pred)

    # The metrics file is version-controlled and a dvc target.
    write_metrics_file(y_train, y_train_pred, y_test, y_test_pred)

    # AUC and ROC
    prob_test = grid_search.predict_proba(docs_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, prob_test[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    plot_roc(roc_auc, fpr, tpr, thresholds)

    print(f"Best score: {grid_search.best_score_}")
    print(f"Best index: {grid_search.best_index_}")
    print(f"Best estimator: {grid_search.best_estimator_}")
    pprint(grid_search.best_params_)

    # For manual checking of the misclassified instances:
    print_wrong_predictions("training data", docs_train, y_train, y_train_pred)
    print_wrong_predictions("test data", docs_test, y_test, y_test_pred)

    # The following file can be unpickled and the result used to create a DataFrame.
    try:
        with open(_project_dir / "grid_search_results2.pck", "wb") as f:
            pickle.dump(grid_search.cv_results_, f)
    except PermissionError:
        pass

    # Finally, we dump the model; it is used by the inference function in Azure.
    joblib.dump(
        grid_search,
        _project_dir / "models" / "model.joblib"
    )


if __name__ == "__main__":
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV
    """
    Create models.
    """
    logger = logging.getLogger(__name__)
    logger.info("Create models.")

    main()
