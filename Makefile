.PHONY: help setup lint test data baseline train-raw train-order evaluate report all clean

PY ?= python

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## install the package and dev tooling
	$(PY) -m pip install -e ".[dev]"

lint:  ## static checks
	ruff check src tests

test:  ## unit tests
	pytest

data:  ## simulate vibration records for every operating condition
	bfd data

baseline:  ## envelope features + gradient boosting
	bfd baseline

train-raw:  ## CNN on the raw waveform
	bfd train --rep raw

train-order:  ## CNN on the order-tracked waveform
	bfd train --rep order

evaluate:  ## score every CNN on every operating condition
	bfd evaluate

report:  ## collect metrics and figures into reports/
	bfd report

all: data baseline train-raw train-order evaluate report  ## full reproduction

clean:
	rm -rf data models reports/*.json reports/*.png reports/RESULTS.md
