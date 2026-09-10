"""Controls for the method: is the analysis still doing what it claims?

    python -m pytest -q          # from the repository root

The tests in test_data.py check the input. These check the analysis itself --
the structural rule that keeps the test set out of training, and the published
numbers the articles tell readers they will reproduce exactly.
"""

import pytest
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.evaluate import folds, held_out_results
from src.load_data import features_and_labels, load_data, validate
from src.model import build_model, split_data

# The numbers printed in article 3. If one of these moves, the prose is wrong,
# and this is where it should be noticed -- not by a reader.
PUBLISHED_CV_ACCURACY = 0.973
PUBLISHED_TEST_ACCURACY = 0.965
PUBLISHED_FALSE_NEGATIVES = 3
PUBLISHED_FALSE_POSITIVES = 1


@pytest.fixture(scope="module")
def split():
    """The data, checked, and split exactly as the analysis splits it."""
    X, y = features_and_labels(validate(load_data()))
    return split_data(X, y)


def test_the_scaler_is_inside_the_pipeline():
    """The leakage guard, made structural.

    Standardise the whole dataset first and the test rows influence the means
    and standard deviations the model trains on. The accuracy that comes out is
    flattering and there is nothing in it to see. Keeping the scaler inside the
    pipeline makes that impossible rather than merely discouraged: it refits on
    each fold's training part, on its own, every time.
    """
    model = build_model()

    assert isinstance(model, Pipeline)
    assert model.steps[0][0] == "standardscaler"


def test_the_split_holds_a_fifth_back_and_keeps_the_mix(split):
    """20% held out, with the same proportion of cancers in both halves.

    Without stratifying, an unlucky random split could hand you a test set with
    far too few malignant samples, and an accuracy that means nothing.
    """
    X_train, X_test, y_train, y_test = split

    assert len(X_train) == 455
    assert len(X_test) == 114
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.01)


def test_the_published_cross_validated_accuracy_still_holds(split):
    """97.3% -- the headline number of article 3."""
    X_train, _, y_train, _ = split

    scores = cross_val_score(build_model(), X_train, y_train, cv=folds())

    assert scores.mean() == pytest.approx(PUBLISHED_CV_ACCURACY, abs=0.005), (
        f"the articles report {PUBLISHED_CV_ACCURACY:.3f}, this run gives {scores.mean():.4f}"
    )


def test_the_published_held_out_result_still_holds(split):
    """96.5% on 114 patients, 3 cancers missed and 1 false alarm.

    The two mistakes are counted separately on purpose. They are not equally
    bad, and a change in the balance between them matters even if the accuracy
    does not move.
    """
    X_train, X_test, y_train, y_test = split
    model = build_model().fit(X_train, y_train)

    results = held_out_results(model, X_test, y_test)

    assert results["accuracy"] == pytest.approx(PUBLISHED_TEST_ACCURACY, abs=0.005)
    assert results["false_negatives"] == PUBLISHED_FALSE_NEGATIVES
    assert results["false_positives"] == PUBLISHED_FALSE_POSITIVES
