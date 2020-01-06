"""
Build a model to filter tweets that have been flagged as potentially
interesting by the keyword matching algorithm.
"""
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

import click


_project_dir = Path(__file__).resolve().parents[2]
_data_dir = _project_dir / "data"

def print_confusion_matrix(y_true, y_pred):
    print(pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True))

def print_model_metrics(y_actual, y_predicted):
    print(metrics.classification_report(y_actual, y_predicted,
                                        target_names=["0", "1"]))

    print_confusion_matrix(y_actual, y_predicted)

def write_metrics_file(y_actual, y_predicted):
    metrics_data = dict(f1=metrics.f1_score(y_actual, y_predicted))
    with open(_project_dir / "metrics.json", "w") as jsonfile:
        json.dump(metrics_data, jsonfile)



@click.command()
@click.argument(
    'input_filepath',
    default=_data_dir / "processed" / "tweets_processed.csv",
    type=click.Path(exists=True)
)
@click.argument(
    'output_filepath',
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
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', LinearSVC(C=1000)),
    ])

    # Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
    }
    grid_search = GridSearchCV(pipeline, parameters, scoring="f1", n_jobs=-1)
    grid_search.fit(docs_train, y_train)

    # TASK: print the mean and std for each candidate along with the parameter
    # settings for all the candidates explored by grid search.
    n_candidates = len(grid_search.cv_results_['params'])
    for i in range(n_candidates):
        print(i, 'params - %s; mean - %0.2f; std - %0.2f'
                 % (grid_search.cv_results_['params'][i],
                    grid_search.cv_results_['mean_test_score'][i],
                    grid_search.cv_results_['std_test_score'][i]))

    # Predict the outcome on the testing set using the best model among those
    # with the different parameters.
    y_predicted = grid_search.predict(docs_test)

    print_model_metrics(y_test, y_predicted)

    # The metrics file is version-controlled and a dvc target.
    write_metrics_file(y_test, y_predicted)

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
