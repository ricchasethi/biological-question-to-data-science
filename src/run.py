"""Run the whole analysis, start to finish, with one command.

    python -m src.run
    make run

Run it from the repository root. It takes about 15 seconds.

This is the step that turns an analysis into something another person can use.
A notebook needs someone to open it and click the cells in the right order, and
it remembers things you did twenty minutes ago. This does not. It starts from the
raw file every time, in a fixed order, and prints the same report.

It does NOT download anything. If the data is missing or has changed it stops and
tells you to run src/fetch_data.py, because quietly analysing a different file is
worse than not running at all.
"""

import sys

from src.controls import controls_that_moved, read_controls
from src.evaluate import (
    baseline_accuracy,
    permutation_control,
    repeated_cross_validation,
    held_out_results,
)
from src.fetch_data import find_problems, read_checksums
from src.load_data import load_data, validate, features_and_labels
from src.manifest import write_manifest
from src.model import THRESHOLD, build_model, split_data


def rule(title):
    """A heading, so the report can be read at a glance."""
    print(f"\n{title}")
    print("-" * len(title))


def main():
    # 1. The input. Check it before anything else touches it.
    problems = find_problems(read_checksums())
    if problems:
        print("STOP. The raw data is not what this analysis was checked against:")
        for problem in problems:
            print("   ", problem)
        print("\nRun:  python -m src.fetch_data")
        return 1

    # 2. The data, checked against the contract before anything is computed from
    #    it, and the split. The test set is put aside here and not looked at
    #    again until step 5.
    try:
        df = validate(load_data())
    except ValueError as failure:
        print("STOP.", failure)
        return 1

    X, y = features_and_labels(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    rule("DATA")
    print(f"  {len(df)} samples, {X.shape[1]} measurements each")
    print(f"  {int(y.sum())} malignant, {int((1 - y).sum())} benign"
          f"   (prevalence {y.mean():.3f})")
    print(f"  train {len(X_train)}, held out {len(X_test)}")
    print("  data contract passed")

    # 3. The model. Scaling and the classifier are fitted together, on the
    #    training half only.
    model = build_model().fit(X_train, y_train)

    rule("MODEL")
    print("  Logistic regression, all 30 standardised features, fitted inside a")
    print("  pipeline that scales using training statistics only.")
    print(f"  training accuracy {model.score(X_train, y_train):.3f}")

    # 4. The controls. These are what make the number in step 5 believable.
    cv_scores = repeated_cross_validation(X_train, y_train)
    observed, null_scores, p_value = permutation_control(X_train, y_train)

    rule("CONTROLS")
    print(f"  blank      predict benign for everyone     {baseline_accuracy(y_test):.3f}")
    print(f"  replicate  5 x 10 cross-validation         {cv_scores.mean():.3f}"
          f" +/- {cv_scores.std():.3f}")
    print(f"  negative   1,000 shuffled-label runs       {null_scores.mean():.3f}"
          f" +/- {null_scores.std():.3f}")
    print(f"             best a shuffle ever reached     {null_scores.max():.3f}")
    print(f"             p-value for the real model      {p_value:.3f}")

    # 5. The held-out samples. Evaluated once, at the end, on purpose.
    results = held_out_results(model, X_test, y_test)

    rule(f"HELD-OUT TEST SET  ({results['n_samples']} samples, threshold {THRESHOLD})")
    print(f"  accuracy    {results['accuracy']:.3f}"
          f"   95% CI {results['accuracy_ci'][0]:.3f} to {results['accuracy_ci'][1]:.3f}")
    print(f"  recall      {results['recall']:.3f}"
          f"   95% CI {results['recall_ci'][0]:.3f} to {results['recall_ci'][1]:.3f}")
    print(f"  precision   {results['precision']:.3f}")
    print(f"  ROC-AUC     {results['roc_auc']:.3f}")
    print(f"  baseline    {results['baseline']:.3f}   (predict benign for everyone)")
    print()
    print(f"  false negatives  {results['false_negatives']}"
          f"   a cancer called benign, and sent home untreated")
    print(f"  false positives  {results['false_positives']}"
          f"   a benign sample called malignant, and worked up")

    # 6. The control patients. Three samples whose probabilities must not move.
    #    They are read last because they need the fitted model, and they gate the
    #    manifest: a run whose controls have drifted is not a run worth recording.
    readings = read_controls(model, X)

    rule("CONTROL PATIENTS  (read on every run)")
    for patient, reading in readings.items():
        mark = "ok" if reading["passed"] else "MOVED"
        print(f"  {patient}   expected {reading['expected']:.6f}"
              f"   read {reading['observed']:.6f}   {mark}")

    moved = controls_that_moved(readings)
    if moved:
        print("\nSTOP. Control samples have moved:", ", ".join(str(p) for p in moved))
        print("Every number above came off the same pipeline, so none of them can be")
        print("trusted. No manifest was written. Find what changed before rerunning.")
        return 1

    # 7. The record. Written last, and only once everything above has held.
    path = write_manifest(model, cv_scores, results, readings)

    rule("RECORDED")
    print(f"  {path}")
    print("  input, code, environment, model and controls, all named by checksum")

    rule("LIMITS")
    print("  One institution, early-1990s instruments, 569 samples, no patient context.")
    print("  A teaching dataset. NOTHING HERE IS FOR CLINICAL USE.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
