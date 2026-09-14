"""The published numbers, and the controls that make them believable.

    from src.evaluate import repeated_cross_validation, permutation_control, held_out_results

A model is an experiment, so it needs controls. Three of them live here:

  blank            predict benign for everyone, and see what accuracy that alone buys
  replicate        the same model on 50 different folds, to see how much it wobbles
  negative control shuffle the labels 1,000 times, and check the model cannot learn nothing
"""

import numpy as np
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    permutation_test_score,
)
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.model import BAND, THRESHOLD, build_model, decide

# Three separate seeds, each fixed. They are different numbers because that is
# what the published notebooks used; changing any of them changes a printed result.
FOLD_SEED = 42         # the plain 5-fold split
REPEAT_SEED = 1        # the 5 x 10 repeated split
PERMUTATION_SEED = 0   # which 1,000 shufflings of the labels are drawn

N_PERMUTATIONS = 1000


def folds(seed=FOLD_SEED):
    """5 folds, stratified so every fold has the same mix of benign and malignant."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def baseline_accuracy(y):
    """The blank: predict benign for everyone and count how often that is right.

    Any model that cannot beat this has learned nothing worth having.
    """
    return 1 - y.mean()


def repeated_cross_validation(X_train, y_train):
    """The replicate: 5 folds x 10 repeats = 50 estimates of accuracy.

    One cross-validation gives one number, and that number depends on how the
    folds happened to fall. Repeating it shows how much the estimate itself moves.
    Returns the 50 scores; take .mean() and .std() of them.
    """
    repeated = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=REPEAT_SEED)
    return cross_val_score(build_model(), X_train, y_train, cv=repeated, scoring="accuracy")


def permutation_control(X_train, y_train):
    """The negative control: destroy the link between measurements and diagnosis.

    Shuffling the labels leaves the data with no signal in it at all. Cross-validate
    1,000 times on shuffled labels and you learn what accuracy this pipeline reaches
    when there is genuinely nothing to find. The real score has to sit far outside
    that spread, or it means nothing.

    Returns (observed score, the 1,000 shuffled scores, p-value).
    """
    return permutation_test_score(
        build_model(), X_train, y_train,
        scoring="accuracy", cv=folds(),
        n_permutations=N_PERMUTATIONS, random_state=PERMUTATION_SEED, n_jobs=-1,
    )


def wilson_interval(successes, n, z=1.96):
    """A 95% confidence interval for a proportion.

    The Wilson score interval, not the textbook one, because the textbook formula
    misbehaves when the proportion is near 0 or 1 -- which is exactly where a good
    classifier lives.
    """
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return centre - half_width, centre + half_width


def held_out_results(model, X_test, y_test):
    """Evaluate the fitted model on the held-out samples. Once.

    Returns a dictionary of everything the articles report, so that the numbers in
    the prose and the numbers in the code cannot drift apart.
    """
    probabilities = model.predict_proba(X_test)[:, 1]   # column 1 = P(malignant)
    predictions = (probabilities >= THRESHOLD).astype(int)

    # The four outcomes. Naming them is worth the line: fn is a cancer sent home.
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_test, predictions).ravel()

    correct = true_neg + true_pos
    accuracy_low, accuracy_high = wilson_interval(correct, len(y_test))
    recall_low, recall_high = wilson_interval(true_pos, true_pos + false_neg)

    return {
        "n_samples": len(y_test),
        "true_negatives": int(true_neg),
        "false_positives": int(false_pos),
        "false_negatives": int(false_neg),
        "true_positives": int(true_pos),
        "accuracy": correct / len(y_test),
        "accuracy_ci": (accuracy_low, accuracy_high),
        "precision": true_pos / (true_pos + false_pos),
        "recall": true_pos / (true_pos + false_neg),
        "recall_ci": (recall_low, recall_high),
        "specificity": true_neg / (true_neg + false_pos),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "baseline": baseline_accuracy(y_test),
        "probabilities": probabilities,
        "predictions": predictions,
    }


# --- the referral band ------------------------------------------------------

# The bands considered when choosing where the model should stop answering.
CANDIDATE_BANDS = [(0.40, 0.60), (0.35, 0.65), (0.30, 0.70), (0.20, 0.80), (0.10, 0.90)]


def referral_table(X_train, y_train, bands=CANDIDATE_BANDS):
    """How the band was chosen -- on TRAINING folds, so the test set stays unspent.

    The band is a hyperparameter. Article 3 was emphatic that anything chosen by
    looking at the test set has spent it, and a referral band tuned on held-out
    patients would be that leak committed at the last possible moment. So this
    uses cross-validated predictions on the training half only: every probability
    here was produced by a model that had not seen that patient.

    Returns one row per candidate band. The curve rises steeply then flattens --
    the first few percent of referrals buy most of the errors, and after that you
    are sending easy cases to a human for a shrinking return.
    """
    probabilities = cross_val_predict(
        build_model(), X_train, y_train, cv=folds(), method="predict_proba",
    )[:, 1]
    truth = np.asarray(y_train)
    wrong = (probabilities >= THRESHOLD).astype(int) != truth

    rows = []
    for low, high in bands:
        referred = (probabilities >= low) & (probabilities <= high)
        kept = ~referred
        rows.append({
            "band": (low, high),
            "n_referred": int(referred.sum()),
            "fraction_referred": float(referred.mean()),
            "errors_caught": int(wrong[referred].sum()),
            "accuracy_of_the_rest": float((~wrong[kept]).sum() / kept.sum()),
        })
    return rows


def referral_results(probabilities, y_test, band=BAND):
    """What the band does to the held-out patients. Spent once, at the very end.

    Reports the automatic decisions separately from the referrals, because they
    are the only ones the system is actually claiming. An accuracy that quietly
    includes the cases you handed to a human is not an accuracy.
    """
    decisions = decide(probabilities, band)
    referred = decisions == "refer"
    automatic = ~referred

    truth = np.asarray(y_test)
    wrong = (probabilities >= THRESHOLD).astype(int) != truth
    missed = (probabilities < THRESHOLD) & (truth == 1)   # a cancer called benign

    return {
        "band": list(band),
        "n_referred": int(referred.sum()),
        "fraction_referred": float(referred.mean()),
        "n_automatic": int(automatic.sum()),
        "errors_automatic": int(wrong[automatic].sum()),
        "accuracy_automatic": float((~wrong[automatic]).sum() / automatic.sum()),
        "cancers_missed_automatic": int(missed[automatic].sum()),
        "referred_patients": {
            str(patient): round(float(probability), 3)
            for patient, probability in zip(y_test.index[referred], probabilities[referred])
        },
    }
