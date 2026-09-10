# One command each, so nobody has to remember the arguments.
#
#   make          get the data if needed, run the controls, then run the analysis
#   make data     download and extract the raw data, then check it
#   make verify   check the raw data without touching the network
#   make test     run the controls on the code (about 3 seconds)
#   make run      run the analysis and print the report

PYTHON = .venv/bin/python

.PHONY: all data verify test run

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
