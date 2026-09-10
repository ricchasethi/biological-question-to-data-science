"""The model, and the split it is allowed to learn from.

    from src.model import split_data, build_model

    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_model().fit(X_train, y_train)

The seeds are fixed here rather than typed into each notebook cell. The articles
promise readers that their numbers will match exactly, and a seed that lives in
only one cell is a seed nobody knows about.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TEST_SIZE = 0.20
SPLIT_SEED = 42

# The probability above which a sample is called malignant. 0.5 by default, not
# by argument -- see the threshold section of article 3.
THRESHOLD = 0.5


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
