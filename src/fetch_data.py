"""Get the raw data, and prove it is the right file.

    python -m src.fetch_data            # download and extract what is missing, then check
    python -m src.fetch_data --verify   # only check what is already here

Run it from the repository root.

An analysis that downloads its own input has no fixed input. The notebook cell
this replaces overwrote data/wdbc.data every time it ran, so nothing recorded
which version produced the published numbers.

So the raw files are treated the way a laboratory treats a sample: they arrive
once, they are labelled with a checksum, and they are read-only afterwards.
"""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

# Where the data came from. The source is part of the data's identity, so it is
# written down here rather than typed into a notebook cell.
URL = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"

DATA = Path("data")


def sha256(path):
    """The fingerprint of a file: 64 characters that change if any byte changes.

    Two files with the same name can hold different data. Two files with the
    same SHA-256 cannot.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_checksums():
    """Read data/checksums.sha256 into {path: fingerprint}.

    Each line looks like:   d606af41...  data/wdbc.data
    Lines starting with # are comments and are skipped.
    """
    checksums = {}
    for line in (DATA / "checksums.sha256").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            fingerprint, name = line.split()
            checksums[Path(name)] = fingerprint
    return checksums


def download_and_extract():
    """Fetch the archive if we do not have it, and unpack only what is missing."""
    DATA.mkdir(exist_ok=True)
    archive = DATA / "wdbc.zip"

    # Download once. Re-running this should not hit someone else's server again.
    if not archive.exists():
        print("downloading", URL)
        urllib.request.urlretrieve(URL, archive)

    with zipfile.ZipFile(archive) as zipped:
        for name in zipped.namelist():
            if not (DATA / name).exists():
                print("extracting", name)
                zipped.extract(name, DATA)


def find_problems(checksums):
    """Return a list of what is wrong with the raw files. Empty list means all good."""
    problems = []
    for path, expected in checksums.items():
        if not path.exists():
            problems.append(f"{path} is missing")
        else:
            found = sha256(path)
            if found != expected:
                problems.append(
                    f"{path} has changed\n"
                    f"      expected {expected}\n"
                    f"      found    {found}"
                )
    return problems


def main():
    if "--verify" not in sys.argv:
        download_and_extract()

    checksums = read_checksums()
    problems = find_problems(checksums)

    # Stop rather than analyse the wrong file. A crash gets noticed; a plausible
    # result computed from unexpected data does not.
    if problems:
        print("\nSTOP. The raw data is not what this analysis was checked against:")
        for problem in problems:
            print("   ", problem)
        print("\nThe published results came from the recorded files, so anything")
        print("computed from these would not be comparable. To start again, delete")
        print("data/wdbc.data and data/wdbc.names and re-run this script.")
        return 1

    for path in checksums:
        path.chmod(0o444)  # read-only for everyone, including you. Deleting it is deliberate.
        print(f"verified {path}  {path.stat().st_size:,} bytes  read-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
