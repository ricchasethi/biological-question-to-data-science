"""The model, and the split it is allowed to learn from.

    from src.model import split_data, build_model

    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_model().fit(X_train, y_train)

The seeds are fixed here rather than typed into each notebook cell. The articles
promise readers that their numbers will match exactly, and a seed that lives in
only one cell is a seed nobody knows about.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TEST_SIZE = 0.20
SPLIT_SEED = 42

# The probability above which a sample is called malignant. 0.5 by default, not
# by argument -- see the threshold section of article 3.
THRESHOLD = 0.5

# The band in which the model is not allowed to answer.
#
# Chosen on CROSS-VALIDATED TRAINING predictions, never on the test set. Tuning a
# referral band on held-out patients would be exactly the leak article 3 spends a
# whole section on, committed at the last possible moment. See referral_table()
# in evaluate.py, which is the code that chose it: 0.30-0.70 sits at the bend of
# the curve, where the first few percent of referrals have bought most of the
# errors and further widening starts sending easy cases to a human.
BAND = (0.30, 0.70)


def split_data(X, y):
    """Hold back 20% of the samples, untouched, to test on once.

    stratify=y keeps the same proportion of malignant samples in both halves.
    Without it a random split could hand you a test set with far too few cancers.
    """
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED)


def build_model():
    """Scaling and the classifier, in ONE object.

    This matters more than it looks. The scaler learns its means and standard
    deviations when the pipeline is fitted, so it only ever sees training data --
    inside cross-validation it refits on each fold's training part. Scale the data
    first, by hand, and test rows influence the training transformation. That is a
    leak you cannot see in the final accuracy, so the rule has to be structural.

    Fitting the two together also means the scaler travels with the model when it
    is saved, so predictions can never be made on a scale the model never saw.
    """
    return make_pipeline(
        StandardScaler(),                   # learns means and SDs, from training data only
        LogisticRegression(max_iter=5000),  # enough iterations to converge properly
    )


def decide(probabilities, band=BAND):
    """Three outcomes, not two. "refer" is a real answer, not a failure to answer.

    A model that must answer every case answers the ones it has no business
    answering. Routing its least certain patients to a pathologist does not make
    the model better -- every probability it produces is unchanged -- it makes the
    *system* better, by sending uncertainty to someone qualified to resolve it.

    The cost is real and lands on a person: 3.5% of patients referred is 350 extra
    expert reviews per 10,000 slides. A band the pathologist cannot absorb is not
    a safety feature, it is a queue.
    """
    low, high = band
    return np.array([
        "refer" if low <= p <= high else ("malignant" if p >= THRESHOLD else "benign")
        for p in probabilities
    ])
