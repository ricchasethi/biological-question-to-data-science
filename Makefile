# One command each, so nobody has to remember the arguments.
#
#   make          get the data if needed, run the controls, then run the analysis
#   make data     download and extract the raw data, then check it
#   make verify   check the raw data without touching the network
#   make test     run the controls on the code (about 3 seconds)
#   make run      run the analysis and print the report
#   make predict  score a batch of new samples:  make predict FILE=batch.csv

PYTHON = .venv/bin/python

.PHONY: all data verify test run predict

# The tests come before the analysis on purpose. If a control has moved, the
# report that follows it is not worth reading.
all: data test run

data:
	$(PYTHON) -m src.fetch_data

verify:
	$(PYTHON) -m src.fetch_data --verify

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) -m src.run

# Scoring new samples, which is the one thing `make run` cannot do: it needs a
# file, and there is no sensible default for whose patients to diagnose.
predict:
ifndef FILE
	$(error give it a batch to score:  make predict FILE=path/to/batch.csv)
endif
	$(PYTHON) -m src.predict $(FILE)
