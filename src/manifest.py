"""The label on the reagent: one small file recording what produced a number.

    from src.manifest import write_manifest

    path = write_manifest(model, cv_scores, results, controls)

Six months from now someone will point at a figure and ask where it came from.
"I ran the notebook" is not an answer. This file is: it names the input by
checksum, the code by commit, the environment by version, the model by checksum,
and the three control readings that say the machinery was working at the time.

One manifest per run, in results/, named for the moment it ran. Nothing here is
ever overwritten -- a run record you can edit is not a record. The manifest is
written LAST, only after the controls have passed, so a file in results/ means a
run that was worth keeping.

This is the same discipline as a laboratory notebook, and it exists for the same
reason: the result is not the whole story, and the story is what you need later.
"""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib

from src.controls import TOLERANCE
from src.evaluate import FOLD_SEED, N_PERMUTATIONS, PERMUTATION_SEED, REPEAT_SEED
from src.fetch_data import sha256
from src.load_data import DATA_FILE
from src.model import SPLIT_SEED, THRESHOLD

RESULTS = Path("results")
MODELS = Path("models")
MODEL_FILE = MODELS / "model.joblib"

# The packages that can change a number if they change. Recording the whole
# environment is requirements.lock.txt's job; this is the short list you would
# actually check first.
RECORDED_PACKAGES = ["pandas", "numpy", "scipy", "scikit-learn", "joblib"]


def code_version():
    """The commit this ran from, and whether the working tree was clean.

    A commit alone is not enough. Code that has been edited but not committed
    does not exist anywhere except this laptop, so a manifest naming a commit
    while the tree is dirty is quietly claiming more than it can prove.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "uncommitted_changes": None,
                "note": "not a git repository -- this run cannot be traced to any code"}

    changed = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True,
    ).stdout.strip()
    return {"commit": commit, "uncommitted_changes": bool(changed)}


def save_model(model):
    """Write the fitted pipeline to models/, and return its checksum.

    The scaler travels inside the pipeline, so what is saved here is the whole
    transformation -- the means and standard deviations included. A model saved
    without its scaler is a model that can be fed data on a scale it never saw.
    """
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    return sha256(MODEL_FILE)


def build_manifest(model, cv_scores, results, controls):
    """Assemble everything worth knowing about this run, as a plain dictionary.

    The five things worth versioning together get a section each: the input, the
    code, the environment, the model, and the settings that decide what any of it
    means.
    """
    return {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {
            "file": DATA_FILE,
            "sha256": sha256(Path(DATA_FILE)),
        },
        "code": code_version(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": {name: version(name) for name in RECORDED_PACKAGES},
        },
        "model": {
            "file": str(MODEL_FILE),
            "sha256": save_model(model),
            "description": "StandardScaler + LogisticRegression, 30 features",
        },
        "settings": {
            "split_seed": SPLIT_SEED,
            "fold_seed": FOLD_SEED,
            "repeat_seed": REPEAT_SEED,
            "permutation_seed": PERMUTATION_SEED,
            "n_permutations": N_PERMUTATIONS,
            "threshold": THRESHOLD,
        },
        "controls": {
            "tolerance": TOLERANCE,
            "readings": {str(patient): reading for patient, reading in controls.items()},
        },
        "results": {
            "cv_scheme": "5-fold stratified, 10 repeats",
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_sd": round(float(cv_scores.std()), 4),
            "n_test": int(results["n_samples"]),
            "test_accuracy": round(float(results["accuracy"]), 4),
            "test_recall": round(float(results["recall"]), 4),
            "test_precision": round(float(results["precision"]), 4),
            "test_roc_auc": round(float(results["roc_auc"]), 4),
            "true_negatives": int(results["true_negatives"]),
            "false_positives": int(results["false_positives"]),
            "false_negatives": int(results["false_negatives"]),
            "true_positives": int(results["true_positives"]),
        },
    }


def write_manifest(model, cv_scores, results, controls):
    """Write the manifest for this run and return where it went.

    The name carries the timestamp and the first eight characters of the input
    checksum, so a directory listing already tells you when a run happened and
    whether two runs read the same file.
    """
    manifest = build_manifest(model, cv_scores, results, controls)

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS / f"run-{stamp}-{manifest['input']['sha256'][:8]}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
