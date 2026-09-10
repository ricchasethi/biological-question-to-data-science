"""Three patients whose predicted probability must not move.

    from src.controls import read_controls, controls_that_moved

    readings = read_controls(model, X)

Every assay plate carries controls: a sample known to be positive, a sample known
to be negative, and -- if you are being careful -- one that sits near the
decision boundary, because that is where a drifting assay shows itself first.
You do not run them because you are interested in those three samples. You run
them because they tell you whether to believe the other ninety.

This is the same idea for a model. These three patients are read on every run,
and their probabilities are recorded to six decimal places. If one of them moves,
something upstream has changed -- the data, a library, a seed, a preprocessing
step -- and every other number from that run is suspect.

A test tells you the code is right. A control tells you *this run* was right.
"""

# All three are in the held-out test set, and they were chosen by looking at that
# set once, for this purpose. They are not a performance measurement -- they are a
# fingerprint of the whole pipeline, and any single number that moves is a signal.
CONTROLS = {
    859711: {
        "diagnosis": "benign",
        "expected": 0.000000,   # the lowest probability in the test set
        "why": "negative control: the sample this model is most certain is benign",
    },
    915143: {
        "diagnosis": "malignant",
        "expected": 1.000000,   # the highest probability in the test set
        "why": "positive control: the sample this model is most certain is malignant",
    },
    879523: {
        "diagnosis": "malignant",
        "expected": 0.491610,   # 0.008 below the 0.5 threshold
        "why": "borderline control: a cancer this model calls benign, and only just",
    },
}

# Six decimal places. The model never returns exactly 0 or 1 -- the true readings
# are 2.4e-08 and 0.9999999999 -- so what is being asserted here is agreement to
# one part in a million, which is far tighter than any real change would be.
TOLERANCE = 1e-6


def read_controls(model, X):
    """Predict the three control patients and compare with what they should be.

    Returns {patient id: {expected, observed, passed}}, ready to go straight into
    the run manifest.
    """
    readings = {}
    for patient, control in CONTROLS.items():
        observed = float(model.predict_proba(X.loc[[patient]])[0, 1])
        readings[patient] = {
            "expected": control["expected"],
            "observed": round(observed, 6),
            "passed": abs(observed - control["expected"]) <= TOLERANCE,
        }
    return readings


def controls_that_moved(readings):
    """The patient ids whose probability is no longer what it was. Empty is good."""
    return [patient for patient, reading in readings.items() if not reading["passed"]]
