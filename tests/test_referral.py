"""Controls for the referral band: the model's right to say "I don't know".

    python -m pytest -q          # from the repository root

The band is the one hyperparameter in this project chosen by looking at
performance, which makes it the one most at risk of being chosen dishonestly.
These tests pin both halves: that it was picked on training folds, and what it
actually buys on the held-out patients -- including what it fails to buy.
"""

import pytest

from src.evaluate import (
    CANDIDATE_BANDS,
    held_out_results,
    referral_results,
    referral_table,
)
from src.load_data import features_and_labels, load_data, validate
from src.model import BAND, build_model, decide, split_data

# The four patients the band refers, from article 4. Two of them are cancers the
# 0.5 threshold alone would have sent home.
REFERRED = {"862548": 0.576, "9112085": 0.509, "874858": 0.315, "879523": 0.492}

# The cancer the band does NOT catch: called benign with 94% confidence.
CONFIDENT_MISTAKE = 859983


@pytest.fixture(scope="module")
def split():
    X, y = features_and_labels(validate(load_data()))
    return split_data(X, y)


@pytest.fixture(scope="module")
def probabilities(split):
    X_train, X_test, y_train, y_test = split
    model = build_model().fit(X_train, y_train)
    return held_out_results(model, X_test, y_test)["probabilities"], y_test


def test_decide_gives_three_outcomes():
    """Two answers and an abstention, not two answers and a guess."""
    outcomes = decide([0.01, 0.31, 0.49, 0.71, 0.99])

    assert list(outcomes) == ["benign", "refer", "refer", "malignant", "malignant"]


def test_the_band_was_chosen_on_the_training_folds(split):
    """The band's provenance, reproduced.

    Every probability behind this table came from a model that had not seen the
    patient it was scoring, and none of them came from the test set. A referral
    band tuned on held-out patients is the leak of article 3, committed at the
    last possible moment.
    """
    X_train, _, y_train, _ = split

    table = {row["band"]: row for row in referral_table(X_train, y_train)}
    chosen = table[BAND]

    assert chosen["n_referred"] == 17
    assert chosen["errors_caught"] == 8
    assert chosen["accuracy_of_the_rest"] == pytest.approx(0.9909, abs=0.0005)


def test_the_band_in_use_is_one_the_training_table_justifies(split):
    """The constant in model.py has to be a band the evidence actually supports."""
    X_train, _, y_train, _ = split

    assert BAND in CANDIDATE_BANDS

    chosen = {row["band"]: row for row in referral_table(X_train, y_train)}[BAND]
    assert chosen["accuracy_of_the_rest"] > 0.99
    assert chosen["fraction_referred"] < 0.05   # a band a human could absorb


def test_the_band_refers_four_of_the_held_out_patients(probabilities):
    """3.5% of patients, and exactly which ones."""
    probs, y_test = probabilities

    referral = referral_results(probs, y_test)

    assert referral["n_referred"] == 4
    assert referral["fraction_referred"] == pytest.approx(0.035, abs=0.001)
    assert referral["referred_patients"] == pytest.approx(REFERRED, abs=0.001)


def test_referral_turns_three_missed_cancers_into_one(probabilities):
    """The number that matters, stated as consequences rather than accuracy.

    Deciding every patient, the model sends three women home with an untreated
    cancer. Allowed to abstain, it sends one -- and asks a pathologist about four.
    """
    probs, y_test = probabilities

    referral = referral_results(probs, y_test)

    assert referral["cancers_missed_automatic"] == 1
    assert referral["errors_automatic"] == 1
    assert referral["accuracy_automatic"] == pytest.approx(0.991, abs=0.001)
    assert referral["n_automatic"] == 110


def test_a_confident_mistake_is_not_caught(probabilities):
    """The honest limit, kept in the test suite so nobody forgets it.

    Patient 859983 is a cancer the model calls benign at P = 0.061 -- 94%
    confident, and completely wrong. She is nowhere near the band, and no
    widening of it reaches her without referring half the cohort.

    A model can tell you when it is uncertain. It cannot tell you when it is
    confidently wrong.
    """
    probs, y_test = probabilities

    position = list(y_test.index).index(CONFIDENT_MISTAKE)
    probability = probs[position]

    assert y_test.loc[CONFIDENT_MISTAKE] == 1          # she has cancer
    assert probability == pytest.approx(0.061, abs=0.001)
    assert decide([probability])[0] == "benign"        # and the band lets it through
    assert probability < BAND[0]
